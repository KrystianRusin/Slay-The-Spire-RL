"""Log records written as one JSON object per line, carrying the process's identity."""

import io
import json
import logging

import pytest

from observability.logs import JsonFormatter


@pytest.fixture
def written():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="actor", actor_id="3"))
    logger = logging.getLogger("tests.logs")
    logger.addHandler(handler)
    logger.propagate = False
    yield logger, lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    logger.removeHandler(handler)


def test_each_record_is_a_json_line_with_its_level_and_the_actor_it_came_from(written):
    logger, lines = written

    logger.warning("Lost the connection to the game at %s", ("localhost", 9000))

    (line,) = lines()
    assert line["level"] == "WARNING"
    assert line["logger"] == "tests.logs"
    assert line["message"] == "Lost the connection to the game at ('localhost', 9000)"
    assert line["service"] == "actor"
    assert line["actor_id"] == "3"
    assert "time" in line


def test_an_exception_is_kept_on_the_same_line(written):
    logger, lines = written

    try:
        raise ValueError("bad rollout")
    except ValueError:
        logger.exception("Update failed")

    (line,) = lines()
    assert line["level"] == "ERROR"
    assert "ValueError: bad rollout" in line["exception"]
