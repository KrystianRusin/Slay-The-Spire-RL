"""Shared test setup: the repository root on sys.path, and loaders for the
committed game-state fixtures.

The fixtures under tests/fixtures/sts2 are payloads exactly as the headless
STS2 host sends them, one per decision type, so the environment can be
exercised without the game. See that directory's README for how to refresh
them.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STS2_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sts2"


def load_sts2(name):
    """One STS2 host payload by decision name, e.g. "combat_play"."""
    with (STS2_FIXTURE_DIR / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def sts2_fixture_names():
    return sorted(path.stem for path in STS2_FIXTURE_DIR.glob("*.json"))


@pytest.fixture(scope="session")
def observation_space():
    """The declared observation space of the environment the policy is built on."""
    from sts2.env import Sts2Env

    return Sts2Env().observation_space
