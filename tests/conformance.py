"""Check an observation against the space the environment declares for it.

Kept separate from the tests so that the report reads the same wherever it is
raised from. `check_observation` never raises: it returns every violation it
finds, so a failure names all the offending components at once rather than
stopping at the first.
"""

from dataclasses import dataclass

import numpy as np

# Problems that are about the shape of the observation rather than the values
# in it. These already hold today, so a test can assert on them alone and stay
# green while the value problems wait on tickets 03-05.
STRUCTURAL_PROBLEMS = frozenset({
    "missing", "undeclared", "wrong shape", "non-finite values",
})


@dataclass(frozen=True)
class Violation:
    fixture: str
    component: str
    problem: str
    detail: str

    def __str__(self):
        return f"{self.fixture} / {self.component}: {self.problem} - {self.detail}"


def structural_only(violations):
    return [v for v in violations if v.problem in STRUCTURAL_PROBLEMS]


def _worst(values, offenders, furthest):
    """Locate the most extreme offending entry: its index and its value.

    `furthest` is np.argmin for values under a low bound, np.argmax for values
    over a high one. Non-offending entries are pushed to the opposite extreme
    so they can never win.
    """
    ignore = np.inf if furthest is np.argmin else -np.inf
    index = np.unravel_index(furthest(np.where(offenders, values, ignore)), values.shape)
    return tuple(int(i) for i in index), float(values[index])


def check_component(fixture, name, values, space):
    """Every way `values` fails to be a member of `space`, as Violations."""
    violations = []

    if values.shape != space.shape:
        violations.append(Violation(
            fixture, name, "wrong shape",
            f"built {values.shape}, declared {space.shape}",
        ))
        return violations  # Nothing further can be compared elementwise.

    if values.dtype != space.dtype:
        violations.append(Violation(
            fixture, name, "wrong dtype",
            f"built {values.dtype}, declared {space.dtype}",
        ))

    finite = np.isfinite(values)
    if not finite.all():
        count = int((~finite).sum())
        violations.append(Violation(
            fixture, name, "non-finite values",
            f"{count} of {values.size} entries are NaN or infinite",
        ))
        return violations  # Bounds are meaningless once NaN is in the array.

    low, high = np.broadcast_to(space.low, values.shape), np.broadcast_to(space.high, values.shape)

    below = values < low
    if below.any():
        index, value = _worst(values, below, np.argmin)
        violations.append(Violation(
            fixture, name, "below the declared low",
            f"{int(below.sum())} of {values.size} entries; "
            f"worst is {value:g} at index {index}, declared low {float(low[index]):g}",
        ))

    above = values > high
    if above.any():
        index, value = _worst(values, above, np.argmax)
        violations.append(Violation(
            fixture, name, "above the declared high",
            f"{int(above.sum())} of {values.size} entries; "
            f"worst is {value:g} at index {index}, declared high {float(high[index]):g}",
        ))

    return violations


def check_observation(fixture, observation, space):
    """Check every component of one observation. Returns a list of Violations."""
    violations = []

    missing = set(space.spaces) - set(observation)
    for name in sorted(missing):
        violations.append(Violation(fixture, name, "missing", "not present in the observation"))

    unexpected = set(observation) - set(space.spaces)
    for name in sorted(unexpected):
        violations.append(Violation(fixture, name, "undeclared", "not present in the observation space"))

    for name in sorted(set(observation) & set(space.spaces)):
        values = np.asarray(observation[name])
        violations.extend(check_component(fixture, name, values, space.spaces[name]))

    return violations


def format_violations(violations):
    """A report naming each offending component, one per line."""
    if not violations:
        return "no violations"

    components = sorted({(v.fixture, v.component) for v in violations})
    header = (
        f"{len(violations)} violation(s) across {len(components)} "
        f"fixture/component pair(s):"
    )
    return "\n".join([header] + [f"  {v}" for v in violations])
