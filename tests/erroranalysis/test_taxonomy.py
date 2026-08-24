"""Taxonomy invariants: the enums are the single source of truth."""

from __future__ import annotations

import pytest

from schema_linking.erroranalysis.taxonomy import (
    CAUSES_FOR_SHAPE,
    CAUSE_DEFINITIONS,
    SHAPE_DEFINITIONS,
    Cause,
    Element,
    ErrorInstance,
    Shape,
)


def test_element_canonicalises_table_name():
    assert Element.table_el("  Singer ") == Element("table", "singer", "")


def test_element_canonicalises_column_pair():
    el = Element.column_el("Singer", "Singer_ID")
    assert el == Element("column", "singer", "singer_id")
    assert el.level == "column"


def test_element_is_hashable_and_frozen():
    el = Element.table_el("singer")
    assert {el, Element.table_el("SINGER")} == {el}
    with pytest.raises(AttributeError):
        el.table = "other"  # type: ignore[misc]


def test_every_cause_is_reachable_from_exactly_one_shape():
    """No cause may be orphaned, and none may be shared across shapes.

    UNRESOLVED is the sole exception: it is the residual for every shape.
    """
    assigned = [c for causes in CAUSES_FOR_SHAPE.values() for c in causes]
    shared = {c for c in assigned if assigned.count(c) > 1}
    assert shared == {Cause.UNRESOLVED}
    assert set(assigned) == set(Cause)


def test_gold_defect_is_first_for_miss():
    """The gate outranks every attribution rule."""
    assert CAUSES_FOR_SHAPE[Shape.MISS][0] is Cause.GOLD_DEFECT


def test_unresolved_is_last_for_every_shape():
    for shape, causes in CAUSES_FOR_SHAPE.items():
        assert causes[-1] is Cause.UNRESOLVED, shape


def test_error_instance_carries_shape_and_element():
    err = ErrorInstance(
        question_id=0,
        db_id="concert_singer",
        method="lexical",
        tier="tier1",
        element=Element.column_el("singer", "Singer_ID"),
        shape=Shape.SPUR,
    )
    assert err.element.column == "singer_id"
    assert err.shape is Shape.SPUR


def test_every_shape_has_a_definition():
    assert set(SHAPE_DEFINITIONS) == set(Shape)


def test_every_cause_has_a_definition():
    assert set(CAUSE_DEFINITIONS) == set(Cause)
