"""Axis-1 enumeration: MISS / SPUR / HALL, and nothing else."""

from __future__ import annotations

import pytest

from schema_linking.erroranalysis.census import (
    CENSUS_COLUMNS,
    METHODS,
    enumerate_errors,
    errors_to_frame,
)
from schema_linking.erroranalysis.facts import build_case_facts
from schema_linking.erroranalysis.taxonomy import Element, Shape


def _facts(mini_schema, *, gold1, gold2, pred, question="How many singers?"):
    return build_case_facts(
        question_id=7,
        question=question,
        gold_sql="SELECT count(*) FROM singer",
        schema=mini_schema,
        gold_tier1_raw=gold1,
        gold_tier2_raw=gold2,
        predicted_raw=pred,
        hardness="easy",
    )


def test_perfect_prediction_yields_no_errors(mini_schema):
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer"], "columns": []},
    )
    assert enumerate_errors(facts, "lexical", "tier1") == []


def test_missing_gold_table_is_a_miss(mini_schema):
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": [], "columns": []},
    )
    (err,) = enumerate_errors(facts, "lexical", "tier1")
    assert err.shape is Shape.MISS
    assert err.element == Element.table_el("singer")
    assert err.tier == "tier1"
    assert err.method == "lexical"
    assert err.question_id == 7


def test_extra_existing_column_is_spurious(mini_schema):
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer"], "columns": [["singer", "Country"]]},
    )
    (err,) = enumerate_errors(facts, "lexical", "tier1")
    assert err.shape is Shape.SPUR
    assert err.element == Element.column_el("singer", "Country")


def test_element_absent_from_schema_is_hallucinated_not_spurious(mini_schema):
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer", "orchestra"], "columns": []},
    )
    shapes = {e.element: e.shape for e in enumerate_errors(facts, "llm_forward", "tier1")}
    assert shapes == {Element.table_el("orchestra"): Shape.HALL}


def test_column_of_a_hallucinated_table_is_also_hallucinated(mini_schema):
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer"], "columns": [["orchestra", "Year"]]},
    )
    (err,) = enumerate_errors(facts, "llm_forward", "tier1")
    assert err.shape is Shape.HALL


def test_tier2_uses_the_tier2_gold_set(mini_schema):
    """A join table that is Tier-2-only is a MISS on tier2 and clean on tier1."""
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer", "concert"], "columns": []},
        pred={"tables": ["singer"], "columns": []},
    )
    assert enumerate_errors(facts, "graph", "tier1") == []
    (err,) = enumerate_errors(facts, "graph", "tier2")
    assert err.element == Element.table_el("concert")
    assert err.shape is Shape.MISS


def test_every_error_is_exactly_one_shape(mini_schema):
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": [["singer", "Name"]]},
        gold2={"tables": ["singer"], "columns": [["singer", "Name"]]},
        pred={"tables": ["concert", "orchestra"], "columns": [["concert", "Name"]]},
    )
    errors = enumerate_errors(facts, "embedding", "tier1")
    assert len({e.element for e in errors}) == len(errors)
    assert {e.shape for e in errors} == {Shape.MISS, Shape.SPUR, Shape.HALL}


def test_methods_are_in_scope_order():
    assert METHODS == (
        "lexical",
        "embedding",
        "llm_forward",
        "llm_backward",
        "llm_bidirectional",
        "graph",
    )


def test_frame_has_the_documented_columns(mini_schema):
    facts = _facts(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": [], "columns": []},
    )
    errors = enumerate_errors(facts, "lexical", "tier1")
    frame = errors_to_frame(errors, {7: facts})
    assert list(frame.columns) == list(CENSUS_COLUMNS)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.shape_code == "MISS"
    assert row.cause == ""
    assert row.level == "table"
    assert row.element == "singer"
    assert row.hardness == "easy"
    assert row.n_tables == 2
    assert row.n_columns == 6


def test_frame_of_no_errors_still_has_the_columns(mini_schema):
    frame = errors_to_frame([], {})
    assert list(frame.columns) == list(CENSUS_COLUMNS)
    assert frame.empty
