"""Run one actor against the middleman with its own local policy and no learner.

Phase 1 / M2: proves the real RL actor can play inside the game container. Each
finished rollout goes to a stand-in for the learner, which logs it and drops it.
M3 replaces the stand-in with a gRPC client to the real learner.

Env: STS_MIDDLEMAN_PORT [9999], STS_N_STEPS (rollout length) [2048].
"""
import os
import time

from environment.run_env import run_environment


class LoggingSink:
    """Stands in for the learner's experience queue: log each rollout, keep nothing."""

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


if __name__ == "__main__":
    run_environment(
        env_id=0,
        port=int(os.environ.get("STS_MIDDLEMAN_PORT", "9999")),
        experience_queue=LoggingSink(),
        n_steps=int(os.environ.get("STS_N_STEPS", "2048")),
    )
