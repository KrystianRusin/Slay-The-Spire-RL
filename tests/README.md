# Tests

No GPU and no running copy of the game: the suite works from committed
game-state fixtures, and the RPC tests start the real Go coordinator binary
that the image carries. Run it in the runtime image (see docs/runbook.md):

```bash
docker compose build learner coordinator
docker run --rm --user 0 --entrypoint bash \
  -v ./tests:/opt/sts/app/tests:ro -v ./pytest.ini:/opt/sts/app/pytest.ini:ro \
  -w /opt/sts/app sts2-game:dev \
  -c "pip install -q pytest >/dev/null 2>&1; python3 -m pytest -q tests"
```

Or from a virtual environment: `pip install -r requirements-dev.txt`, then
`pytest` (the coordinator tests then need the Go binary built locally).

## What covers what

- `test_sts2_env.py`: the observation encoder, legal actions and rewards,
  against the host fixtures.
- `test_sts2_actor.py`: the actor loop against a scripted host: which steps
  reach the rollout buffer, refusals, selection picks, cut-off runs.
- `test_sts2_live.py`: the real-game adapter (mod states in, mod actions out).
- `test_rollout_buffer.py`, `test_masked_log_probs.py`, `test_ppo_update.py`:
  the PPO plumbing.
- `test_rpc.py`, `test_coordinator.py`, `test_checkpoint.py`,
  `test_learner_restart.py`, `test_rollout_handoff.py`: actor/learner/
  coordinator over gRPC, checkpoints and restarts.

## Fixtures

`fixtures/sts2/` holds the first payload of each decision type the headless
host sent during random-bot runs. Its README covers how to refresh them.
