"""Axis-1 enumeration: MISS / SPUR / HALL, and nothing else."""

from __future__ import annotations

import pandas as pd

from schema_linking.erroranalysis.census import (
    CENSUS_COLUMNS,
    METHODS,
    add_schema_size_bin,
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


def _size_row(db_id: str, n_columns: int, *, question_id: int = 0) -> dict:
    """A minimal census row carrying only what `add_schema_size_bin` needs."""
    return {
        "question_id": question_id,
        "db_id": db_id,
        "method": "lexical",
        "tier": "tier1",
        "level": "table",
        "element": "t",
        "shape_code": "MISS",
        "cause": "",
        "rule_name": "",
        "evidence": "",
        "hardness": "easy",
        "n_tables": 1,
        "n_columns": n_columns,
        "schema_size_bin": "",
    }


def test_schema_size_bin_is_computed_over_distinct_db_pairs_not_rows():
    """Regression guard for the documented correctness property: binning
    must run over distinct (db_id, n_columns) pairs, not over error rows.

    ``db_a`` contributes 100 rows, all with the smallest n_columns; the
    other three databases contribute one row each, each strictly larger.
    Correctly db-weighted, the four *distinct* values (1, 2, 3, 4) split
    evenly across the four quartiles, so every database lands in its own
    bin. If binning were done over rows instead, db_a's 100 duplicate rows
    would dominate the quantile edges and collapse every database into a
    single bin -- this is verified directly against the (pre-fix)
    row-unaware computation in the task report.
    """
    rows = [_size_row("db_a", 1, question_id=i) for i in range(100)]
    rows.append(_size_row("db_b", 2, question_id=200))
    rows.append(_size_row("db_c", 3, question_id=201))
    rows.append(_size_row("db_d", 4, question_id=202))
    frame = pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))

    result = add_schema_size_bin(frame)

    bin_by_db = result.drop_duplicates("db_id").set_index("db_id")["schema_size_bin"]
    assert bin_by_db["db_a"] == "Q1_smallest"
    assert bin_by_db["db_d"] == "Q4_largest"
    assert len(set(bin_by_db)) == 4


def test_schema_size_bin_is_one_consistent_value_per_db_id():
    rows = [
        _size_row("db_a", 5, question_id=0),
        _size_row("db_a", 5, question_id=1),
        _size_row("db_b", 10, question_id=2),
        _size_row("db_c", 15, question_id=3),
        _size_row("db_d", 20, question_id=4),
    ]
    frame = pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))

    result = add_schema_size_bin(frame)

    per_db_bin_counts = result.groupby("db_id")["schema_size_bin"].nunique()
    assert (per_db_bin_counts == 1).all()


def test_schema_size_bin_preserves_census_columns_order():
    rows = [
        _size_row("db_a", 1),
        _size_row("db_b", 2),
        _size_row("db_c", 3),
        _size_row("db_d", 4),
    ]
    frame = pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))

    result = add_schema_size_bin(frame)

    assert list(result.columns) == list(CENSUS_COLUMNS)


def test_schema_size_bin_all_databases_same_n_columns_does_not_raise():
    """The exact crash this guards: the pre-fix implementation called
    ``pd.qcut(..., labels=[4 fixed labels], duplicates="drop")``, which
    raises ``ValueError: Bin labels must be one fewer than the number of
    bin edges`` whenever every value is identical, because all quantile
    edges collide and collapse to a single bin. Confirmed to raise against
    the pre-fix code directly (see task-6-report.md)."""
    rows = [_size_row(f"db_{i}", 5, question_id=i) for i in range(6)]
    frame = pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))

    result = add_schema_size_bin(frame)

    assert set(result["schema_size_bin"]) == {"Q1_smallest"}


def test_schema_size_bin_few_distinct_values_does_not_raise():
    """Only two distinct n_columns values among four databases: the same
    fixed-4-labels crash as above, triggered with fewer duplicate values.
    Confirmed to raise against the pre-fix code directly (see
    task-6-report.md). Endpoints must still read smallest/largest."""
    rows = [
        _size_row("db_a", 1, question_id=0),
        _size_row("db_b", 1, question_id=1),
        _size_row("db_c", 2, question_id=2),
        _size_row("db_d", 2, question_id=3),
    ]
    frame = pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))

    result = add_schema_size_bin(frame)

    bin_by_db = result.drop_duplicates("db_id").set_index("db_id")["schema_size_bin"]
    assert bin_by_db["db_a"] == "Q1_smallest"
    assert bin_by_db["db_d"].endswith("_largest")
    assert bin_by_db["db_a"] != bin_by_db["db_d"]
