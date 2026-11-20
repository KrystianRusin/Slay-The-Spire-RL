"""The learner: owns the master copy of the policy, trains it, and serves learner.proto.

Each round, it collects one rollout per actor for the current version. Once it
has LEARNER_NUM_ACTORS of them, a background thread concatenates them, runs one
PPO update, and publishes the next version to the coordinator at
COORDINATOR_ADDR. Actors wait there for that version before playing their next
rollout, so the learner sends its weights once however many actors there are.

After every update it saves a checkpoint to LEARNER_CHECKPOINT_DIR (if set),
and on startup it resumes from the newest one there.

Env: COORDINATOR_ADDR (host:port), LEARNER_PORT [50051], LEARNER_NUM_ACTORS [1],
LEARNER_N_STEPS [from TrainingConfig], LEARNER_CHECKPOINT_DIR [unset: no checkpoints].
"""
import dataclasses
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent import futures

import grpc
import numpy as np

from model import checkpoint
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.model_utils import train
from model.policy import TrainingConfig, make_model
from rpc import learner_pb2 as pb
from rpc import learner_pb2_grpc
from rpc.publisher import WeightsPublisher
from rpc.codec import (BUFFER_FIELDS, GRPC_OPTIONS, config_to_proto, encode_state_dict,
                       rollout_from_proto, state_dict_digest)

def log(message):
    """One line to stdout in a single write, so lines from different threads don't interleave."""
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def check_matches_model(rollout, model):
    """Raise ValueError if a rollout's arrays can't be trained on by model."""
    for key, space in model.observation_space.spaces.items():
        array = rollout["observations"].get(key)
        if array is None or array.shape[1:] != space.shape:
            raise ValueError(f"observations[{key}] doesn't match the shape {space.shape}")
    for field in BUFFER_FIELDS:
        expected = (1,) if field == "actions" else ()
        if rollout[field].shape[1:] != expected:
            raise ValueError(f"{field} has shape {rollout[field].shape}")
    if rollout["action_masks"].shape[1:] != (model.action_space.n,):
        raise ValueError(f"action_masks has shape {rollout['action_masks'].shape}")


def merge_rollouts(rollouts, model):
    """One buffer holding every actor's steps, ready to train on."""
    total = sum(rollout["n_steps"] for rollout in rollouts)
    buffer = CustomRolloutBuffer(
        buffer_size=total,
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1,
    )
    for key in buffer.observations:
        buffer.observations[key] = np.concatenate([rollout["observations"][key] for rollout in rollouts])
    for field, attribute in BUFFER_FIELDS.items():
        setattr(buffer, attribute, np.concatenate([rollout[field] for rollout in rollouts]))
    buffer.action_masks = np.concatenate([rollout["action_masks"] for rollout in rollouts])
    buffer.pos = total
    buffer.full = True
    return buffer


class LearnerService(learner_pb2_grpc.LearnerServicer):
    def __init__(self, config, publisher, num_actors=1, checkpoint_dir=None):
        """publisher: anything with publish(pb.Weights), normally a WeightsPublisher."""
        self.config = config
        self.publisher = publisher
        self.num_actors = num_actors
        self.checkpoint_dir = checkpoint_dir
        self.model = make_model(config)
        self.policy = self.model.policy
        self.version = 0
        self.run_id = uuid.uuid4().hex[:12]  # tells actors this learner process from earlier ones
        resumed_from = self._resume() if checkpoint_dir else None
        self._config = config_to_proto(config)
        self._lock = threading.Lock()  # gRPC serves calls on a thread pool
        self._round_full = threading.Condition(self._lock)
        self._pool = {}  # actor_id -> its rollout for the current version
        self.accepted = 0
        self.rejected = 0
        self.last_stats = None
        self._round_started = time.monotonic()  # when the current version was published
        self._first_arrival = None  # when the current round's first rollout came in
        if resumed_from:
            log(f"learner: resumed from {resumed_from}")
        log(f"learner: starting at v{self.version}, sha={state_dict_digest(self.policy.state_dict())}, "
            f"waiting for {num_actors} actor(s) per round, run_id={self.run_id}")
        self.publisher.publish(self._weights_message())

    def _resume(self):
        """Load the newest checkpoint, if any. Returns its path, or None for a fresh start."""
        state = checkpoint.load_latest(self.checkpoint_dir)
        if state is None:
            return None
        # Carrying on under different settings would mix two runs into one.
        checkpoint.check_config(state["config"], self.config)
        checkpoint.restore(self.model, state)
        self.version = state["version"]
        return state["path"]

    def _weights_message(self):
        """The current weights as a message for the coordinator."""
        return pb.Weights(version=self.version, state_dict=encode_state_dict(self.policy.state_dict()),
                          config=self._config, run_id=self.run_id)

    def SubmitRollout(self, request, context):
        try:
            rollout = rollout_from_proto(request)
            if rollout["action_masks"] is not None:
                check_matches_model(rollout, self.model)
        except ValueError as e:
            with self._lock:
                return self._reject(request, f"malformed rollout: {e}")

        with self._lock:
            if rollout["action_masks"] is None:
                # log_probs are only meaningful with the masks they were computed over
                return self._reject(request, "rollout has no action masks")
            if rollout["policy_version"] != self.version:
                return self._reject(request, f"made with v{rollout['policy_version']}, "
                                             f"learner is on v{self.version}")
            if rollout["actor_id"] in self._pool:
                return self._reject(request, f"already submitted for v{self.version}")
            if len(self._pool) >= self.num_actors:
                return self._reject(request, f"round for v{self.version} is already full")

            if not self._pool:
                self._first_arrival = time.monotonic()
            self._last_arrival = time.monotonic()
            self._pool[rollout["actor_id"]] = rollout
            self.accepted += 1
            log(f"learner: rollout #{self.accepted} from actor {rollout['actor_id']}, "
                  f"version {rollout['policy_version']}, {rollout['n_steps']} steps, "
                  f"reward sum={rollout['rewards'].sum():.1f}, "
                  f"episodes ended={int(rollout['dones'].sum())}, "
                  f"legal actions/step={rollout['action_masks'].sum(axis=1).mean():.1f}, "
                  f"{request.ByteSize() / 1e6:.1f} MB, "
                  f"round {len(self._pool)}/{self.num_actors}")
            if len(self._pool) == self.num_actors:
                self._round_full.notify()
        return pb.SubmitAck(accepted=True, run_id=self.run_id)

    def _reject(self, request, reason):
        """Refuse a rollout. The caller holds the lock."""
        self.rejected += 1
        log(f"learner: REJECTED rollout from actor {request.actor_id}: {reason}")
        return pb.SubmitAck(accepted=False, reason=reason, run_id=self.run_id)

    def start_training(self):
        threading.Thread(target=self._training_loop, name="trainer", daemon=True).start()

    def _training_loop(self):
        try:
            while True:
                with self._lock:
                    while len(self._pool) < self.num_actors:
                        self._round_full.wait()
                    rollouts = list(self._pool.values())
                self._train_round(rollouts)
        except Exception:
            # A learner that can't train must not keep accepting rollouts as if it could.
            traceback.print_exc()
            os._exit(1)

    def _train_round(self, rollouts):
        # Outside the lock, so rollouts are still answered during the update: the
        # pool is full, so they're refused until the version below changes.
        started = time.monotonic()
        stats = train(self.model, merge_rollouts(rollouts, self.model))
        if self.checkpoint_dir:
            # Saved before it is published: once an actor can play with a version,
            # a restarted learner resumes at that version, not the one before.
            checkpoint.save_checkpoint(self.checkpoint_dir, self.version + 1, self.model, self.config)

        with self._lock:
            self.version += 1
            self._pool = {}
            self.last_stats = stats
            version = self.version
            now = time.monotonic()
            round_seconds = now - self._round_started  # publish to publish (round 1: from startup)
            spread = self._last_arrival - self._first_arrival
            self._round_started = now
        self.publisher.publish(self._weights_message())
        log(f"learner: trained v{version - 1} -> v{version} on {stats.steps} steps from "
              f"{len(rollouts)} actor(s) in {time.monotonic() - started:.1f}s, "
              f"sha={state_dict_digest(self.policy.state_dict())}, "
              f"policy_loss={stats.policy_loss:.4f}, value_loss={stats.value_loss:.2f}, "
              f"entropy_loss={stats.entropy_loss:.4f}, approx_kl={stats.approx_kl:.4f}, "
              f"first |ratio-1| mean={stats.first_ratio_error_mean:.6f} "
              f"max={stats.first_ratio_error_max:.6f}, "
              f"returns mean={stats.returns_mean:.2f} std={stats.returns_std:.2f}, "
              f"explained_variance={stats.explained_variance:.3f}")
        log(f"learner: round for v{version - 1} took {round_seconds:.1f}s, "
              f"{stats.steps / round_seconds:.2f} steps/s overall, "
              f"rollouts arrived over {spread:.1f}s")


def serve(service, port):
    """Start a gRPC server for service on port. Returns the server, already running."""
    # Enough threads for every actor to submit at once.
    workers = futures.ThreadPoolExecutor(max_workers=service.num_actors + 2)
    server = grpc.server(workers, options=GRPC_OPTIONS)
    learner_pb2_grpc.add_LearnerServicer_to_server(service, server)
    bound = server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server, bound


def main():
    coordinator_address = os.environ.get("COORDINATOR_ADDR")
    if not coordinator_address:
        sys.exit("learner: COORDINATOR_ADDR must be set: actors get the weights from the coordinator")
    port = int(os.environ.get("LEARNER_PORT", "50051"))
    num_actors = int(os.environ.get("LEARNER_NUM_ACTORS") or 1)
    config = TrainingConfig()
    if os.environ.get("LEARNER_N_STEPS"):
        config = dataclasses.replace(config, n_steps=int(os.environ["LEARNER_N_STEPS"]))

    try:
        service = LearnerService(config, WeightsPublisher(coordinator_address), num_actors,
                                 os.environ.get("LEARNER_CHECKPOINT_DIR") or None)
    except checkpoint.ConfigMismatch as e:
        log(f"learner: refusing to resume, the newest checkpoint has a different config: {e}\n"
            f"learner: to start a new run, move the checkpoint folder away")
        sys.exit(1)
    service.start_training()
    server, _ = serve(service, port)
    log(f"learner: listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
