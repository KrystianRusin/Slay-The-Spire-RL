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

    def get_weights(self):
        """Fetch the learner's current policy. Returns (version, config, state_dict)."""
        reply = self.stub.GetWeights(pb.GetWeightsRequest(actor_id=self.actor_id),
                                     timeout=CALL_TIMEOUT, wait_for_ready=True)
        self.version = reply.version
        return reply.version, config_from_proto(reply.config), decode_state_dict(reply.state_dict)

    def put(self, rollout):
        """Submit one finished rollout, tagged with the version it was played with."""
        if self.version is None:
            raise RuntimeError("get_weights() must be called before submitting rollouts")
        message = rollout_to_proto(rollout, self.actor_id, self.version)
        ack = self.stub.SubmitRollout(message, timeout=CALL_TIMEOUT, wait_for_ready=True)
        status = "accepted" if ack.accepted else f"REJECTED ({ack.reason})"
        print(f"rollout submitted: {message.n_steps} steps, version {self.version}, {status}", flush=True)
        return ack
