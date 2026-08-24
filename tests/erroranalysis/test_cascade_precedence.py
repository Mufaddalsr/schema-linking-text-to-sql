"""The executed cascade must match the documented precedence."""

from __future__ import annotations

import pytest

from schema_linking.erroranalysis.rules import CASCADE
from schema_linking.erroranalysis.taxonomy import CAUSES_FOR_SHAPE, Cause, Shape


def test_every_shape_has_a_cascade():
    assert set(CASCADE) == set(Shape)


@pytest.mark.parametrize("shape", list(Shape))
def test_rule_order_matches_documented_cause_order(shape):
    """CASCADE order, de-duplicated, must equal CAUSES_FOR_SHAPE minus UNRESOLVED.

    Several rules may share a cause (the gate has three); what must hold is
    that the *sequence of distinct causes* is the documented one.
    """
    executed: list[Cause] = []
    for rule in CASCADE[shape]:
        if not executed or executed[-1] is not rule.cause:
            executed.append(rule.cause)
    documented = [c for c in CAUSES_FOR_SHAPE[shape] if c is not Cause.UNRESOLVED]
    assert executed == documented


@pytest.mark.parametrize("shape", list(Shape))
def test_every_rule_declares_a_cause(shape):
    for rule in CASCADE[shape]:
        assert isinstance(rule.cause, Cause), rule


@pytest.mark.parametrize("shape", list(Shape))
def test_rule_names_are_unique_within_a_shape(shape):
    names = [r.__name__ for r in CASCADE[shape]]
    assert len(names) == len(set(names))


def test_name_collision_precedes_sibling():
    """Explicitly guarded because both can fire on the same element."""
    causes = [r.cause for r in CASCADE[Shape.SPUR]]
    assert causes.index(Cause.NAME_COLLISION) < causes.index(Cause.SIBLING)


def test_gate_precedes_every_miss_attribution_rule():
    causes = [r.cause for r in CASCADE[Shape.MISS]]
    last_gate = max(i for i, c in enumerate(causes) if c is Cause.GOLD_DEFECT)
    assert all(c is Cause.GOLD_DEFECT for c in causes[: last_gate + 1])
