"""The middleman's capture path, which is how new fixtures get made."""

import json

import middleman_process
from tests.conftest import load_payload


def test_capturing_is_off_unless_the_directory_is_set(tmp_path, monkeypatch):
    """Not merely "writes nowhere we asked": writes nothing at all.

    The version this replaced wrote game_state_<timestamp>.json into the
    working directory unconditionally, so the check runs from an empty cwd.
    """
    monkeypatch.delenv(middleman_process.CAPTURE_DIR_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    middleman_process.save_game_state(load_payload("rest_site"))

    assert list(tmp_path.iterdir()) == []


def test_a_captured_state_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv(middleman_process.CAPTURE_DIR_VAR, str(tmp_path))
    payload = load_payload("rest_site")

    middleman_process.save_game_state(payload)

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert written[0].name.startswith("rest_"), "captures are named by screen type"
    assert json.loads(written[0].read_text(encoding="utf-8")) == payload


def test_captures_do_not_overwrite_each_other(tmp_path, monkeypatch):
    """Several states arrive per second, so the timestamp alone is not unique."""
    monkeypatch.setenv(middleman_process.CAPTURE_DIR_VAR, str(tmp_path))

    middleman_process.save_game_state(load_payload("rest_site"))
    middleman_process.save_game_state(load_payload("rest_site"))

    assert len(list(tmp_path.glob("*.json"))) == 2


def test_a_pre_run_payload_is_captured_without_a_game_state(tmp_path, monkeypatch):
    monkeypatch.setenv(middleman_process.CAPTURE_DIR_VAR, str(tmp_path))

    middleman_process.save_game_state(load_payload("main_menu"))

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert written[0].name.startswith("no_game_")
