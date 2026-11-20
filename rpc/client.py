"""The actor's side of learner.proto: weights come from the coordinator, rollouts go to the learner."""
import time

import grpc

from rpc import learner_pb2 as pb
from rpc import learner_pb2_grpc
from rpc.codec import GRPC_OPTIONS, config_from_proto, decode_state_dict, rollout_to_proto

# How long one call may take. Longer than the coordinator's long poll (30 s),
# and while a service is unreachable a call waits this long for it to come back.
CALL_TIMEOUT = 60

# The service is down or restarting: wait and try again rather than crash.
RETRY_CODES = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)
RETRY_FIRST_SECONDS = 1
RETRY_MAX_SECONDS = 10


class LearnerRestartedWithNewConfig(Exception):
    """The learner came back with a different network or settings than this actor built."""


class LearnerClient:
    """Fetches weights from the coordinator and submits rollouts to the learner.

    Has the same put() as the experience queue run_environment expects, so the
    actor hands it finished rollouts without knowing they leave the process.

    Calls ride out either service going away: they are retried until it is back.
    """

    def __init__(self, coordinator_address, learner_address, actor_id):
        self.actor_id = actor_id
        self.coordinator = learner_pb2_grpc.ActorsStub(
            grpc.insecure_channel(coordinator_address, options=GRPC_OPTIONS))
        self.stub = learner_pb2_grpc.LearnerStub(grpc.insecure_channel(learner_address, options=GRPC_OPTIONS))
        self.version = None  # the version of the weights last fetched
        self.config = None  # the config they came with
        self.run_id = ""  # the learner process they came from
        self.last_accepted = False  # whether the last submitted rollout was accepted

    def _call(self, method, request, service):
        """method(request), retried with backoff for as long as service is unreachable."""
        delay = RETRY_FIRST_SECONDS
        while True:
            try:
                # wait_for_ready: while the service is down, wait for it instead of failing at once.
                return method(request, timeout=CALL_TIMEOUT, wait_for_ready=True)
            except grpc.RpcError as e:
                if e.code() not in RETRY_CODES:
                    raise
                print(f"{service} unreachable ({e.code().name}), retrying in {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, RETRY_MAX_SECONDS)

    def get_weights(self, min_version=0):
        """Fetch the learner's policy from the coordinator, waiting until it is at least min_version.

        If the learner restarted since the last fetch, returns its current
        weights straight away, whatever min_version was. Returns
        (version, config, state_dict).
        """
        while True:
            request = pb.GetWeightsRequest(actor_id=self.actor_id, min_version=min_version,
                                           run_id=self.run_id)
            reply = self._call(self.coordinator.GetWeights, request, "coordinator")
            if reply.HasField("config"):
                break
            # The coordinator held the call as long as it would; ask again.
        config = config_from_proto(reply.config)
        if self.run_id and reply.run_id != self.run_id:
            print(f"learner restarted (run {self.run_id} -> {reply.run_id}), "
                  f"now on v{reply.version}", flush=True)
        if self.config is not None and config != self.config:
            # This actor's network was built for the old config; the new weights won't fit.
            raise LearnerRestartedWithNewConfig(f"was {self.config}, now {config}")
        self.version, self.config, self.run_id = reply.version, config, reply.run_id
        return reply.version, config, decode_state_dict(reply.state_dict)

    def put(self, rollout):
        """Submit one finished rollout, tagged with the version it was played with."""
        if self.version is None:
            raise RuntimeError("get_weights() must be called before submitting rollouts")
        message = rollout_to_proto(rollout, self.actor_id, self.version)
        ack = self._call(self.stub.SubmitRollout, message, "learner")
        self.last_accepted = ack.accepted
        if ack.accepted:
            # If the learner restarted, this rollout is in the new one's round, so
            # waiting for the next version is right: don't treat it as lost.
            self.run_id = ack.run_id
        status = "accepted" if ack.accepted else f"REJECTED ({ack.reason})"
        print(f"rollout submitted: {message.n_steps} steps, version {self.version}, {status}", flush=True)
        return ack

    def next_weights(self):
        """The weights to play the next rollout with. Blocks until they exist.

        After an accepted rollout that's the version after the one it was played
        with, which the learner publishes once every actor's rollout is in. After a
        rejected one, or if the learner restarted, it's whatever the learner has
        now. The config can't change underneath the actor, so only the weights
        are returned.
        """
        min_version = self.version + 1 if self.last_accepted else 0
        _, _, state_dict = self.get_weights(min_version)
        return state_dict
