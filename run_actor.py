"""Run one actor: play the game through the middleman and hand off rollouts.

With LEARNER_ADDR set, the actor fetches its starting weights from the learner,
submits each finished rollout to it over gRPC, and then waits for the learner's
next version before playing the next rollout. Without it,
rollouts go to a stand-in that logs and drops them (M2), so the actor can be
tested with no learner running.

The learner also sends the TrainingConfig (rollout length, gamma, network
shape), so the actor has no settings of its own. With no learner it uses the
defaults in model/policy.py.

Env: LEARNER_ADDR (host:port), STS_MIDDLEMAN_PORT [9999], STS_ACTOR_ID [hostname].
"""
import os
import socket
import time

from environment.run_env import run_environment
from model.policy import TrainingConfig


class LoggingSink:
    """Stands in for the learner: log each rollout, keep nothing."""

    def __init__(self):
        self.count = 0
        self.started = time.monotonic()

    def put(self, rollout):
        self.count += 1
        steps = rollout.pos
        elapsed = time.monotonic() - self.started
        print(f"rollout {self.count}: {steps} steps, "
              f"reward sum={rollout.rewards[:steps].sum():.1f}, "
              f"episodes ended={int(rollout.dones[:steps].sum())}, "
              f"{steps * self.count / elapsed:.2f} steps/s overall", flush=True)


def main():
    actor_id = os.environ.get("STS_ACTOR_ID") or socket.gethostname()
    learner_address = os.environ.get("LEARNER_ADDR")
    config = TrainingConfig()
    initial_state_dict = None
    next_weights = None

    if learner_address:
        # Imported here so the no-learner mode doesn't need the generated gRPC code.
        from rpc.client import LearnerClient
        from rpc.codec import state_dict_digest

        sink = LearnerClient(learner_address, actor_id)
        version, config, initial_state_dict = sink.get_weights()
        print(f"actor {actor_id}: loaded weights v{version} from {learner_address}, "
              f"sha={state_dict_digest(initial_state_dict)}, {config}", flush=True)

        def next_weights():
            started = time.monotonic()
            state_dict = sink.next_weights()
            print(f"actor {actor_id}: loaded weights v{sink.version}, "
                  f"sha={state_dict_digest(state_dict)}, waited {time.monotonic() - started:.1f}s", flush=True)
            return state_dict
    else:
        print(f"actor {actor_id}: no LEARNER_ADDR, rollouts are logged and dropped", flush=True)
        sink = LoggingSink()

    run_environment(
        env_id=0,
        port=int(os.environ.get("STS_MIDDLEMAN_PORT", "9999")),
        experience_queue=sink,
        config=config,
        initial_state_dict=initial_state_dict,
        next_weights=next_weights,
    )


if __name__ == "__main__":
    main()
