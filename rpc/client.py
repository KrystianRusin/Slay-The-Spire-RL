"""The actor's side of learner.proto."""
import grpc

from rpc import learner_pb2 as pb
from rpc import learner_pb2_grpc
from rpc.codec import GRPC_OPTIONS, config_from_proto, decode_state_dict, rollout_to_proto

# How long to wait for the learner on a call; it may still be starting up.
CALL_TIMEOUT = 120


class LearnerClient:
    """Fetches weights from the learner and submits rollouts to it.

    Has the same put() as the experience queue run_environment expects, so the
    actor hands it finished rollouts without knowing they leave the process.
    """

    def __init__(self, address, actor_id):
        self.actor_id = actor_id
        self.channel = grpc.insecure_channel(address, options=GRPC_OPTIONS)
        self.stub = learner_pb2_grpc.LearnerStub(self.channel)
        self.version = None  # the version of the weights last fetched
        self.last_accepted = False  # whether the last submitted rollout was accepted

    def get_weights(self, min_version=0):
        """Fetch the learner's policy, waiting until it has at least min_version.

        Returns (version, config, state_dict).
        """
        while True:
            request = pb.GetWeightsRequest(actor_id=self.actor_id, min_version=min_version)
            reply = self.stub.GetWeights(request, timeout=CALL_TIMEOUT, wait_for_ready=True)
            if reply.version >= min_version:
                break
            # The learner held the call as long as it would; ask again.
        self.version = reply.version
        return reply.version, config_from_proto(reply.config), decode_state_dict(reply.state_dict)

    def put(self, rollout):
        """Submit one finished rollout, tagged with the version it was played with."""
        if self.version is None:
            raise RuntimeError("get_weights() must be called before submitting rollouts")
        message = rollout_to_proto(rollout, self.actor_id, self.version)
        ack = self.stub.SubmitRollout(message, timeout=CALL_TIMEOUT, wait_for_ready=True)
        self.last_accepted = ack.accepted
        status = "accepted" if ack.accepted else f"REJECTED ({ack.reason})"
        print(f"rollout submitted: {message.n_steps} steps, version {self.version}, {status}", flush=True)
        return ack

    def next_weights(self):
        """The weights to play the next rollout with. Blocks until they exist.

        After an accepted rollout that's the version after the one it was played
        with, which the learner publishes once every actor's rollout is in. After a
        rejected one, it's whatever the learner has now. The config is fixed for the
        learner's lifetime, so only the weights are returned.
        """
        min_version = self.version + 1 if self.last_accepted else 0
        _, _, state_dict = self.get_weights(min_version)
        return state_dict
