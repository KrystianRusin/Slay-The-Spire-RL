"""Shared test setup: the repository root on sys.path, and loaders for the
committed game-state fixtures.

The fixtures under tests/fixtures/game_states are Communication Mod payloads
exactly as they arrive over the socket, so the whole observation pipeline can
be exercised without Slay the Spire running. See that directory's README for
what each one covers and how to capture more.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "game_states"

# The screens the agent actually encounters, and the fixture that covers each.
# Keyed by fixture name so a test can ask for one by name rather than by index.
SCREEN_COVERAGE = {
    "combat_with_monsters": "NONE",  # in combat, no screen up
    "combat_reward": "COMBAT_REWARD",
    "card_reward": "CARD_REWARD",
    "map_screen": "MAP",
    "shop_screen": "SHOP_SCREEN",
    "rest_site": "REST",
    "game_over": "GAME_OVER",
    "main_menu": None,  # no game_state at all: before a run starts
}


def fixture_names():
    """Every committed fixture, sorted so test ids are stable."""
    return sorted(path.stem for path in FIXTURE_DIR.glob("*.json"))


def load_payload(name):
    """Load one fixture by name, without the .json suffix.

    What comes back is the outer socket payload - available_commands and the
    rest - not the inner game_state.
    """
    path = FIXTURE_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(params=fixture_names())
def named_payload(request):
    """Each fixture in turn, as a (name, payload) pair."""
    return request.param, load_payload(request.param)


@pytest.fixture(scope="session")
def env_class():
    """The environment class itself, so a test can build one per payload.

    Imported lazily: slay_the_spire_env pulls in TensorFlow through the
    tokenizers, which costs several seconds. Ticket 03 removes that.
    """
    from slay_the_spire_env import SlayTheSpireEnv

    return SlayTheSpireEnv


@pytest.fixture(scope="session")
def observation_space(env_class):
    """The declared observation space. Identical for every instance."""
    return env_class({}).observation_space
