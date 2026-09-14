# Tests

No database server, no GPU and no running copy of the game: the suite works
from committed game-state fixtures, and database tests use a throwaway SQLite
file. From a clean checkout, with the virtual environment active:

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

## Concurrent writers against Postgres

The concurrency tests in `test_data_layer.py` and `test_game_registry.py`
always run against SQLite. To run them against Postgres too, point
`TEST_DATABASE_URL` at an empty scratch database. The tests create their tables
there and drop them afterwards, and refuse to run if any of them already exist.

```bash
TEST_DATABASE_URL=postgresql://postgres:changeme@localhost:5432/slay_the_spire_test pytest tests/test_data_layer.py tests/test_game_registry.py
```

## Kafka

`test_broker.py` runs against a real broker and is skipped unless
`TEST_KAFKA_BOOTSTRAP_SERVERS` is set. It needs permission to create and delete
topics.

```bash
docker run -d --name kafka -p 9092:9092 apache/kafka:4.1.1
TEST_KAFKA_BOOTSTRAP_SERVERS=localhost:9092 pytest tests/test_broker.py
```
