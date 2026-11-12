"""The learner: owns the master copy of the policy, trains it, and serves learner.proto.

Each round, it collects one rollout per actor for the current version. Once it
has LEARNER_NUM_ACTORS of them, a background thread concatenates them, runs one
PPO update, and publishes the next version. Actors wait for that version with a
long poll (GetWeights with min_version) before playing their next rollout.

Env: LEARNER_PORT [50051], LEARNER_NUM_ACTORS [1], LEARNER_N_STEPS [from TrainingConfig].
"""
import dataclasses
import os
import sys
import threading
import time
import traceback
from concurrent import futures

import grpc
import numpy as np

from model.custom_rollout_buffer import CustomRolloutBuffer
from model.model_utils import train
from model.policy import TrainingConfig, make_model
from rpc import learner_pb2 as pb
from rpc import learner_pb2_grpc
from rpc.codec import (BUFFER_FIELDS, GRPC_OPTIONS, config_to_proto, encode_state_dict,
                       rollout_from_proto, state_dict_digest)

def log(message):
    """One line to stdout in a single write, so lines from different threads don't interleave."""
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


# The longest GetWeights holds a call open waiting for min_version. Short enough
# to stay well inside the client's call timeout; the client just asks again.
LONG_POLL_SECONDS = 30


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
    def __init__(self, config, num_actors=1):
        self.config = config
        self.num_actors = num_actors
        self.model = make_model(config)
        self.policy = self.model.policy
        self.version = 0
        self._weights = encode_state_dict(self.policy.state_dict())
        self._config = config_to_proto(config)
        self._lock = threading.Lock()  # gRPC serves calls on a thread pool
        self._round_full = threading.Condition(self._lock)
        self._new_version = threading.Condition(self._lock)
        self.long_poll_seconds = LONG_POLL_SECONDS
        self._pool = {}  # actor_id -> its rollout for the current version
        self.accepted = 0
        self.rejected = 0
        self.last_stats = None
        self._round_started = time.monotonic()  # when the current version was published
        self._first_arrival = None  # when the current round's first rollout came in
        log(f"learner: serving weights v{self.version}, "
              f"sha={state_dict_digest(self.policy.state_dict())}, {len(self._weights) / 1e6:.2f} MB, "
              f"waiting for {num_actors} actor(s) per round")

    def GetWeights(self, request, context):
        with self._lock:
            # Releases the lock while waiting, so rollouts and training carry on.
            self._new_version.wait_for(lambda: self.version >= request.min_version,
                                       timeout=self.long_poll_seconds)
            if self.version < request.min_version:
                return pb.Weights(version=self.version)  # not yet: no weights, ask again
            log(f"learner: actor {request.actor_id} fetched weights v{self.version}")
            return pb.Weights(version=self.version, state_dict=self._weights, config=self._config)

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
        return pb.SubmitAck(accepted=True)

    def _reject(self, request, reason):
        """Refuse a rollout. The caller holds the lock."""
        self.rejected += 1
        log(f"learner: REJECTED rollout from actor {request.actor_id}: {reason}")
        return pb.SubmitAck(accepted=False, reason=reason)

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
        # Outside the lock, so actors can still fetch weights during the update. The
        # full pool refuses new rollouts, and GetWeights serves the cached bytes of
        # the old version, which only change below, together with the version number.
        started = time.monotonic()
        stats = train(self.model, merge_rollouts(rollouts, self.model))
        weights = encode_state_dict(self.policy.state_dict())

        with self._lock:
            self.version += 1
            self._weights = weights
            self._pool = {}
            self.last_stats = stats
            version = self.version
            self._new_version.notify_all()  # wake every actor waiting for this version
            now = time.monotonic()
            round_seconds = now - self._round_started  # publish to publish (round 1: from startup)
            spread = self._last_arrival - self._first_arrival
            self._round_started = now
        log(f"learner: trained v{version - 1} -> v{version} on {stats.steps} steps from "
              f"{len(rollouts)} actor(s) in {time.monotonic() - started:.1f}s, "
              f"sha={state_dict_digest(self.policy.state_dict())}, "
              f"policy_loss={stats.policy_loss:.4f}, value_loss={stats.value_loss:.2f}, "
              f"entropy_loss={stats.entropy_loss:.4f}, approx_kl={stats.approx_kl:.4f}, "
              f"first |ratio-1| mean={stats.first_ratio_error_mean:.6f} "
              f"max={stats.first_ratio_error_max:.6f}")
        log(f"learner: round for v{version - 1} took {round_seconds:.1f}s, "
              f"{stats.steps / round_seconds:.2f} steps/s overall, "
              f"rollouts arrived over {spread:.1f}s")


def serve(service, port):
    """Start a gRPC server for service on port. Returns the server, already running."""
    # Every actor waiting for new weights holds a worker thread, so there must be
    # spare ones for the SubmitRollout that completes the round.
    workers = futures.ThreadPoolExecutor(max_workers=service.num_actors + 8)
    server = grpc.server(workers, options=GRPC_OPTIONS)
    learner_pb2_grpc.add_LearnerServicer_to_server(service, server)
    bound = server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server, bound


def main():
    port = int(os.environ.get("LEARNER_PORT", "50051"))
    num_actors = int(os.environ.get("LEARNER_NUM_ACTORS") or 1)
    config = TrainingConfig()
    if os.environ.get("LEARNER_N_STEPS"):
        config = dataclasses.replace(config, n_steps=int(os.environ["LEARNER_N_STEPS"]))

    service = LearnerService(config, num_actors)
    service.start_training()
    server, _ = serve(service, port)
    log(f"learner: listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
