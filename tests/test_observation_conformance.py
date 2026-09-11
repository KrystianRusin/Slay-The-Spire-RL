"""Does an observation built from a game state fit the space we declare?

Today it does not, and that is the point: this module is the written-down
specification for ticket 05, and the unlisted-name test below is the one for
ticket 03. Both are marked xfail so the rest of the suite stays meaningful
while those tickets are open. The markers are strict, so each test fails again
- telling you to delete the marker - as soon as the defect it describes is
fixed.

To read a failure in full, ignoring the markers:

    pytest tests/test_observation_conformance.py --runxfail

Note what conformance cannot see. Ticket 04's dead wiring leaves the deck
component all zeros, which is perfectly in bounds, and the screen component
falls through to a default handler that is also in bounds. Those are ticket
04's own criteria to test; a green conformance run does not imply they are
fixed.
"""

import copy

import pytest

from tests.conformance import check_observation, format_violations, structural_only
from tests.conftest import fixture_names, load_payload


def build_observation(env_class, payload):
    return env_class(payload).flatten_observation(payload)


def test_observations_are_structurally_sound(named_payload, env_class, observation_space):
    """Right components, right shapes, no NaN or infinity. True today."""
    name, payload = named_payload
    observation = build_observation(env_class, payload)

    violations = structural_only(check_observation(name, observation, observation_space))

    assert not violations, format_violations(violations)


@pytest.mark.xfail(
    strict=True,
    reason="Observations do not yet conform: ticket 05 (normalization), with "
           "the token ranges it inherits from ticket 03. Run with --runxfail "
           "to see which components are out of bounds.",
)
def test_observations_conform_to_the_declared_space(env_class, observation_space):
    """Every component of every fixture, checked in one pass.

    Deliberately not parametrized per fixture: the report is far more useful
    when it names every offending component across the whole fixture set at
    once, and a single test can carry a single strict xfail marker.
    """
    violations = []
    for name in fixture_names():
        payload = load_payload(name)
        observation = build_observation(env_class, payload)
        violations.extend(check_observation(name, observation, observation_space))

    assert not violations, format_violations(violations)


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 03: an unrecognised name raises IndexError instead of "
           "falling back to the unknown id.",
)
def test_an_unlisted_card_name_does_not_crash_the_pipeline(env_class):
    """What a modded card, or one added by a game update, does to us.

    The vocabulary is hard-coded, so any name outside it takes the fallback
    path - which currently raises. Built by editing a fixture rather than by
    committing one, so the crash stays inside this test instead of taking the
    rest of the suite down with it.
    """
    payload = copy.deepcopy(load_payload("combat_with_monsters"))
    payload["game_state"]["combat_state"]["hand"][0]["name"] = "Sozu's Revenge"

    build_observation(env_class, payload)
