"""The learner: owns the master copy of the policy and serves learner.proto.

Phase 1 / M3: a stub. It builds the model once from a TrainingConfig (version
0), hands those weights and the config to any actor that asks, and accepts
rollouts made with the current version. It only counts and logs them: training
comes in M4.

Env: LEARNER_PORT [50051].
"""
import os
import threading
from concurrent import futures

import grpc

from model.policy import TrainingConfig, make_model
from rpc import learner_pb2 as pb
from rpc import learner_pb2_grpc
from rpc.codec import (GRPC_OPTIONS, config_to_proto, encode_state_dict, rollout_from_proto,
                       state_dict_digest)


class LearnerService(learner_pb2_grpc.LearnerServicer):
    def __init__(self, config):
        self.config = config
        self.model = make_model(config)
        self.policy = self.model.policy
        self.version = 0
        self._weights = encode_state_dict(self.policy.state_dict())
        self._config = config_to_proto(config)
        self._lock = threading.Lock()  # gRPC serves calls on a thread pool
        self.accepted = 0
        self.rejected = 0
        print(f"learner: serving weights v{self.version}, "
              f"sha={state_dict_digest(self.policy.state_dict())}, {len(self._weights) / 1e6:.2f} MB", flush=True)

    def GetWeights(self, request, context):
        with self._lock:
            print(f"learner: actor {request.actor_id} fetched weights v{self.version}", flush=True)
            return pb.Weights(version=self.version, state_dict=self._weights, config=self._config)

    def SubmitRollout(self, request, context):
        try:
            rollout = rollout_from_proto(request)
        except ValueError as e:
            return self._reject(request, f"malformed rollout: {e}")

        with self._lock:
            if rollout["policy_version"] != self.version:
                return self._reject(request, f"made with v{rollout['policy_version']}, "
                                             f"learner is on v{self.version}")
            self.accepted += 1
            print(f"learner: rollout #{self.accepted} from actor {rollout['actor_id']}, "
                  f"version {rollout['policy_version']}, {rollout['n_steps']} steps, "
                  f"reward sum={rollout['rewards'].sum():.1f}, "
                  f"episodes ended={int(rollout['dones'].sum())}, "
                  f"{request.ByteSize() / 1e6:.1f} MB", flush=True)
        return pb.SubmitAck(accepted=True)

    def _reject(self, request, reason):
        self.rejected += 1
        print(f"learner: REJECTED rollout from actor {request.actor_id}: {reason}", flush=True)
        return pb.SubmitAck(accepted=False, reason=reason)


def serve(service, port):
    """Start a gRPC server for service on port. Returns the server, already running."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=GRPC_OPTIONS)
    learner_pb2_grpc.add_LearnerServicer_to_server(service, server)
    bound = server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server, bound


def main():
    port = int(os.environ.get("LEARNER_PORT", "50051"))
    server, _ = serve(LearnerService(TrainingConfig()), port)
    print(f"learner: listening on port {port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
