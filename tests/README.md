# Tests

No database, no GPU and no running copy of the game: the suite works from
committed game-state fixtures. From a clean checkout, with the virtual
environment active:

```bash
pip install -r requirements-dev.txt
pytest
```

`requirements-dev.txt` pulls in `requirements.txt`, so that is the only install
step.

## The expected failure

`test_observations_conform_to_the_declared_space` is marked
`xfail(strict=True)`: observations do not yet fit the space the environment
declares for them. When they do, the test fails by *passing* and the marker
should be deleted.

Its report names every component that is out of bounds:

```bash
pytest tests/test_observation_conformance.py --runxfail
```

## Fixtures

`fixtures/game_states/` holds Communication Mod payloads covering combat, the
card reward, map, shop and rest screens, game over, and the pre-run main menu.
Its README covers what each one contains and how to capture more by setting
`STS_CAPTURE_DIR` before starting the middleman.
