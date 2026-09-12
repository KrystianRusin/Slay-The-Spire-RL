"""Does an observation built from a game state fit the space we declare?

Not yet, so the check is marked xfail. Run with --runxfail to see the report
naming every component that is out of bounds.
"""

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
