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

## The expected failures

Two tests are marked `xfail(strict=True)`. Each fails by *passing* once the
work it waits on lands, at which point the marker should be deleted.

`test_observations_conform_to_the_declared_space` - observations do not yet fit
the space the environment declares for them. Its report names every component
that is out of bounds:

```bash
pytest tests/test_observation_conformance.py --runxfail
```

`test_no_handler_builds_more_features_than_the_declared_width` - the
`HAND_SELECT` and `GRID` screen handlers build more features than the 50 the
screen component declares, so `get_screen_observation` truncates them. Fixing
it means widening the component, not rewiring anything.

## Fixtures

`fixtures/game_states/` holds Communication Mod payloads covering combat, the
card reward, map, shop and rest screens, game over, and the pre-run main menu.
Its README covers what each one contains and how to capture more by setting
`STS_CAPTURE_DIR` before starting the middleman.
