"""Tests for src/schema_linking/evaluator.py — sections 1–5.

Eight hand-built cases for :func:`per_query_metrics`, F-beta correctness
checks, canonicalisation tests, four `filter_hallucinated` cases, and
an end-to-end :func:`evaluate` test on a 5-query / 2-schema synthetic
dataset whose expected metrics are computed by hand in the test file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from schema_linking.evaluator import (
    EvalResult,
    _canonicalise_column,
    _canonicalise_table,
    evaluate,
    fbeta,
    filter_hallucinated,
    per_query_metrics,
    write_results,
)
from schema_linking.schema_parser import Column, Schema, Table


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("singer", "singer"),
        ("Singer", "singer"),
        ("SINGER", "singer"),
        ("singer ", "singer"),
        ("  Singer  ", "singer"),
        ("\tsinger\n", "singer"),
    ],
)
def test_canonicalise_table_lowercases_and_strips(
    inp: str, expected: str
) -> None:
    assert _canonicalise_table(inp) == expected


def test_canonicalise_column_returns_tuple() -> None:
    got = _canonicalise_column("Singer", "Name")
    assert got == ("singer", "name")
    assert isinstance(got, tuple)


def test_canonicalise_column_normalises_each_half() -> None:
    assert _canonicalise_column("  Singer  ", " NAME ") == ("singer", "name")


def test_canonicalise_column_result_is_hashable() -> None:
    # Tuples of strings are hashable; this is what makes column refs
    # usable as set elements for metric computation.
    s = {_canonicalise_column("Singer", "Name")}
    assert ("singer", "name") in s


# ---------------------------------------------------------------------------
# per_query_metrics — 8 hand-built edge-case combinations
# ---------------------------------------------------------------------------


# Each row: (label, predicted, gold, expected_dict).
CASES: list[tuple[str, set, set, dict]] = [
    (
        "empty_empty",
        set(), set(),
        {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "srr_hit": True,
            "predicted_count": 0, "gold_count": 0,
        },
    ),
    (
        "empty_pred__gold_a",
        set(), {"a"},
        {
            "tp": 0, "fp": 0, "fn": 1,
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
            "srr_hit": False,
            "predicted_count": 0, "gold_count": 1,
        },
    ),
    (
        "pred_a__empty_gold",
        {"a"}, set(),
        {
            "tp": 0, "fp": 1, "fn": 0,
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "srr_hit": True,
            "predicted_count": 1, "gold_count": 0,
        },
    ),
    (
        "exact_match_a",
        {"a"}, {"a"},
        {
            "tp": 1, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "srr_hit": True,
            "predicted_count": 1, "gold_count": 1,
        },
    ),
    (
        "pred_a__gold_ab__under",
        {"a"}, {"a", "b"},
        {
            "tp": 1, "fp": 0, "fn": 1,
            "precision": 1.0, "recall": 0.5, "f1": 2 / 3,
            "srr_hit": False,
            "predicted_count": 1, "gold_count": 2,
        },
    ),
    (
        "pred_ab__gold_a__over",
        {"a", "b"}, {"a"},
        {
            "tp": 1, "fp": 1, "fn": 0,
            "precision": 0.5, "recall": 1.0, "f1": 2 / 3,
            "srr_hit": True,
            "predicted_count": 2, "gold_count": 1,
        },
    ),
    (
        "exact_match_ab",
        {"a", "b"}, {"a", "b"},
        {
            "tp": 2, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "srr_hit": True,
            "predicted_count": 2, "gold_count": 2,
        },
    ),
    (
        "pred_abc__gold_ab__overshoot",
        {"a", "b", "c"}, {"a", "b"},
        {
            "tp": 2, "fp": 1, "fn": 0,
            "precision": 2 / 3, "recall": 1.0, "f1": 0.8,
            "srr_hit": True,
            "predicted_count": 3, "gold_count": 2,
        },
    ),
]


@pytest.mark.parametrize("label,predicted,gold,expected", CASES, ids=[c[0] for c in CASES])
def test_per_query_metrics_case(
    label: str, predicted: set, gold: set, expected: dict
) -> None:
    got = per_query_metrics(predicted, gold)
    assert set(got.keys()) == set(expected.keys()), (
        f"key set mismatch for {label}: got={set(got)}, want={set(expected)}"
    )
    for key, want in expected.items():
        if isinstance(want, float):
            assert got[key] == pytest.approx(want), (
                f"{label}.{key}: got={got[key]!r}, want≈{want!r}"
            )
        else:
            assert got[key] == want, (
                f"{label}.{key}: got={got[key]!r}, want={want!r}"
            )


def test_per_query_metrics_works_with_tuple_elements() -> None:
    """The same primitive must handle column refs (table, column tuples)."""
    pred = {("singer", "name"), ("singer", "age")}
    gold = {("singer", "name")}
    got = per_query_metrics(pred, gold)
    assert got["tp"] == 1
    assert got["fp"] == 1
    assert got["precision"] == 0.5
    assert got["recall"] == 1.0
    assert got["srr_hit"] is True


# ---------------------------------------------------------------------------
# fbeta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,r,expected",
    [
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (0.8, 0.8, 0.8),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.6, 0.4, 2 * 0.6 * 0.4 / (0.6 + 0.4)),  # textbook harmonic mean
        (0.9, 0.1, 2 * 0.9 * 0.1 / (0.9 + 0.1)),
    ],
)
def test_f1_matches_harmonic_mean(p: float, r: float, expected: float) -> None:
    assert fbeta(p, r, 1.0) == pytest.approx(expected)


def test_f6_high_recall_low_precision_is_close_to_one() -> None:
    """F6 with high recall and low precision is dominated by recall."""
    assert fbeta(0.5, 1.0, 6.0) > 0.95


def test_f6_high_precision_low_recall_is_low() -> None:
    """The flip case: high precision can't rescue F6 when recall is poor."""
    assert fbeta(1.0, 0.5, 6.0) < 0.55


def test_f6_weights_recall_about_36_times_more_than_precision() -> None:
    """Sanity: F6 with P=ε, R=1.0 stays high. Symmetric flip stays low."""
    f6_recall_only = fbeta(0.01, 1.0, 6.0)
    f6_precision_only = fbeta(1.0, 0.01, 6.0)
    # Recall-dominant case should be roughly two orders of magnitude
    # higher than the precision-dominant case.
    assert f6_recall_only > 0.25
    assert f6_precision_only < 0.02
    assert f6_recall_only / max(f6_precision_only, 1e-9) > 10.0


def test_fbeta_zero_when_both_inputs_zero() -> None:
    assert fbeta(0.0, 0.0, 1.0) == 0.0
    assert fbeta(0.0, 0.0, 6.0) == 0.0
    assert fbeta(0.0, 0.0, 0.5) == 0.0


def test_fbeta_zero_when_denominator_zero_with_beta_zero() -> None:
    """β = 0 reduces F to P, but with R = 0 the denominator is zero;
    return 0 rather than crash."""
    assert fbeta(0.5, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Helpers for sections 4 & 5
# ---------------------------------------------------------------------------


def _table(name: str, cols: list[str]) -> Table:
    return Table(
        name=name,
        original_name=name,
        columns=[
            Column(
                name=c,
                original_name=c,
                type="number",
                table_name=name,
                is_primary_key=False,
            )
            for c in cols
        ],
    )


def _schema(db_id: str, tables: list[Table]) -> Schema:
    return Schema(db_id=db_id, tables=tables, foreign_keys=[])


# Two schemas reused by the synthetic dataset.
_SCHEMA_A = _schema("A", [_table("tA", ["x", "y"]), _table("tB", ["z"])])
_SCHEMA_B = _schema("B", [_table("tC", ["p", "q"])])
_SCHEMAS = {"A": _SCHEMA_A, "B": _SCHEMA_B}


# ---------------------------------------------------------------------------
# Section 4 — filter_hallucinated
# ---------------------------------------------------------------------------


def test_filter_keeps_known_table_and_column() -> None:
    pred = {"db_id": "A", "tables": ["tA"], "columns": [["tA", "x"]]}
    filtered, halluc = filter_hallucinated(pred, _SCHEMA_A)
    assert filtered["tables"] == ["tA"]
    assert filtered["columns"] == [["tA", "x"]]
    assert halluc["tables"] == []
    assert halluc["columns"] == []


def test_filter_removes_unknown_table() -> None:
    pred = {
        "db_id": "A",
        "tables": ["tA", "FakeTable"],
        "columns": [["tA", "x"]],
    }
    filtered, halluc = filter_hallucinated(pred, _SCHEMA_A)
    assert filtered["tables"] == ["tA"]
    assert halluc["tables"] == ["FakeTable"]
    assert halluc["columns"] == []


def test_filter_removes_column_whose_table_is_unknown() -> None:
    pred = {
        "db_id": "A",
        "tables": ["tA"],
        "columns": [["tA", "x"], ["FakeTable", "x"]],
    }
    filtered, halluc = filter_hallucinated(pred, _SCHEMA_A)
    assert filtered["columns"] == [["tA", "x"]]
    assert halluc["columns"] == [["FakeTable", "x"]]


def test_filter_removes_column_whose_table_exists_but_column_does_not() -> None:
    """The case that distinguishes the column rule from the table rule."""
    pred = {
        "db_id": "A",
        "tables": ["tA"],
        "columns": [["tA", "x"], ["tA", "no_such_column"]],
    }
    filtered, halluc = filter_hallucinated(pred, _SCHEMA_A)
    assert filtered["columns"] == [["tA", "x"]]
    assert halluc["columns"] == [["tA", "no_such_column"]]
    assert halluc["tables"] == []  # table tA itself is fine


def test_filter_is_case_insensitive_for_matching_but_preserves_case() -> None:
    pred = {
        "db_id": "A",
        "tables": ["TA"],            # different case than schema's "tA"
        "columns": [["TA", "X"]],
    }
    filtered, halluc = filter_hallucinated(pred, _SCHEMA_A)
    assert filtered["tables"] == ["TA"]  # original case preserved
    assert filtered["columns"] == [["TA", "X"]]
    assert halluc["tables"] == [] and halluc["columns"] == []


# ---------------------------------------------------------------------------
# Section 5 — evaluate, synthetic 5-query / 2-schema dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_eval_inputs() -> dict:
    """Five queries with hand-computed expected metrics.

    Schema A: tA{x, y}, tB{z}
    Schema B: tC{p, q}

    qid 0  perfect           pred = gold = {tA} / {(tA, x)}
    qid 1  partial recall     pred misses tB / (tB, z)
    qid 2  with 2 hallucinations: FakeTable, (tC, fake_col)
    qid 3  vacuous            empty pred, empty gold
    qid 4  perfect, gold has only table, no columns
    """
    gold = {
        0: {"db_id": "A", "tables": ["tA"], "columns": [["tA", "x"]]},
        1: {
            "db_id": "A",
            "tables": ["tA", "tB"],
            "columns": [["tA", "x"], ["tB", "z"]],
        },
        2: {"db_id": "B", "tables": ["tC"], "columns": [["tC", "p"]]},
        3: {"db_id": "A", "tables": [], "columns": []},
        4: {"db_id": "B", "tables": ["tC"], "columns": []},
    }
    predictions = {
        0: {"db_id": "A", "tables": ["tA"], "columns": [["tA", "x"]]},
        1: {"db_id": "A", "tables": ["tA"], "columns": [["tA", "x"]]},
        2: {
            "db_id": "B",
            "tables": ["tC", "FakeTable"],
            "columns": [["tC", "p"], ["tC", "fake_col"]],
        },
        3: {"db_id": "A", "tables": [], "columns": []},
        4: {"db_id": "B", "tables": ["tC"], "columns": []},
    }
    hardness = {0: "easy", 1: "medium", 2: "hard", 3: "easy", 4: "extra"}
    return {
        "predictions": predictions,
        "gold": gold,
        "schemas": _SCHEMAS,
        "hardness": hardness,
    }


def _row(df, **filters):
    """Return the single row matching every filter; assert exactly one."""
    mask = None
    for k, v in filters.items():
        m = df[k] == v
        mask = m if mask is None else (mask & m)
    matches = df[mask]
    assert len(matches) == 1, (
        f"expected 1 row for {filters}, got {len(matches)}:\n{matches}"
    )
    return matches.iloc[0]


def test_evaluate_returns_evalresult(synthetic_eval_inputs: dict) -> None:
    result = evaluate(
        **synthetic_eval_inputs, method_name="m", tier_name="t"
    )
    assert isinstance(result, EvalResult)
    assert len(result.per_query) == 5
    assert not result.aggregated.empty


def test_evaluate_per_query_table_metrics(synthetic_eval_inputs: dict) -> None:
    result = evaluate(
        **synthetic_eval_inputs, method_name="m", tier_name="t"
    )
    pq = result.per_query.set_index("question_id")

    # qid 0: pred = gold = {tA}, perfect.
    assert pq.loc[0, "table_tp"] == 1
    assert pq.loc[0, "table_fp"] == 0
    assert pq.loc[0, "table_fn"] == 0
    assert pq.loc[0, "table_precision"] == pytest.approx(1.0)
    assert pq.loc[0, "table_recall"] == pytest.approx(1.0)
    assert pq.loc[0, "table_f1"] == pytest.approx(1.0)
    assert bool(pq.loc[0, "table_srr_hit"]) is True

    # qid 1: pred = {tA}, gold = {tA, tB}.
    assert pq.loc[1, "table_tp"] == 1
    assert pq.loc[1, "table_fn"] == 1
    assert pq.loc[1, "table_precision"] == pytest.approx(1.0)
    assert pq.loc[1, "table_recall"] == pytest.approx(0.5)
    assert pq.loc[1, "table_f1"] == pytest.approx(2 / 3)
    assert bool(pq.loc[1, "table_srr_hit"]) is False

    # qid 2: pred has FakeTable (hallucinated); filtered pred = {tC}.
    assert pq.loc[2, "table_tp"] == 1
    assert pq.loc[2, "table_fp"] == 0
    assert pq.loc[2, "table_hallucinated"] == 1
    assert pq.loc[2, "column_hallucinated"] == 1

    # qid 3: vacuous, perfect.
    assert pq.loc[3, "table_tp"] == 0
    assert pq.loc[3, "table_precision"] == pytest.approx(1.0)
    assert pq.loc[3, "table_recall"] == pytest.approx(1.0)
    assert pq.loc[3, "hallucination_rate"] == pytest.approx(0.0)


def test_evaluate_hallucination_rate_per_query(
    synthetic_eval_inputs: dict,
) -> None:
    result = evaluate(
        **synthetic_eval_inputs, method_name="m", tier_name="t"
    )
    pq = result.per_query.set_index("question_id")
    # qid 2 is the only query with hallucinations: 2 hallucinated out of 4 predicted.
    assert pq.loc[2, "hallucination_rate"] == pytest.approx(0.5)
    for qid in (0, 1, 3, 4):
        assert pq.loc[qid, "hallucination_rate"] == pytest.approx(0.0)


# ---------- aggregated frame: every "all" cell hand-checked ----------


def test_evaluate_aggregated_all_tables_macro(
    synthetic_eval_inputs: dict,
) -> None:
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    row = _row(
        result.aggregated,
        level="tables",
        hardness="all",
        aggregation="macro",
    )
    assert row["method"] == "m"
    assert row["tier"] == "t"
    assert row["n_queries"] == 5
    # macro P = mean(1,1,1,1,1) = 1.0; R = mean(1,0.5,1,1,1) = 0.9
    # F1 = mean(1, 2/3, 1, 1, 1) = (4 + 2/3)/5 = 14/15
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(0.9)
    assert row["f1"] == pytest.approx(14 / 15)
    assert row["f6"] == pytest.approx(fbeta(1.0, 0.9, 6.0))
    assert row["srr"] == pytest.approx(0.8)  # 4 of 5 hits
    assert row["hallucination_rate"] == pytest.approx(0.1)


def test_evaluate_aggregated_all_tables_micro(
    synthetic_eval_inputs: dict,
) -> None:
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    row = _row(
        result.aggregated,
        level="tables",
        hardness="all",
        aggregation="micro",
    )
    # tables pooled: TP=4 (qids 0,1,2,4), FP=0, FN=1 (tB missing in qid 1)
    assert row["n_queries"] == 5
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(0.8)
    assert row["f1"] == pytest.approx(8 / 9)
    assert row["f6"] == pytest.approx(fbeta(1.0, 0.8, 6.0))
    assert row["srr"] == pytest.approx(0.8)
    assert row["hallucination_rate"] == pytest.approx(0.1)


def test_evaluate_aggregated_all_columns_macro(
    synthetic_eval_inputs: dict,
) -> None:
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    row = _row(
        result.aggregated,
        level="columns",
        hardness="all",
        aggregation="macro",
    )
    # column per-query P,R,F1,srr identical to tables for this dataset
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(0.9)
    assert row["f1"] == pytest.approx(14 / 15)
    assert row["srr"] == pytest.approx(0.8)


def test_evaluate_aggregated_all_columns_micro(
    synthetic_eval_inputs: dict,
) -> None:
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    row = _row(
        result.aggregated,
        level="columns",
        hardness="all",
        aggregation="micro",
    )
    # columns pooled: TP=3 (qids 0,1,2), FP=0, FN=1 (tB.z missing in qid 1)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(0.75)
    assert row["f1"] == pytest.approx(6 / 7)
    assert row["f6"] == pytest.approx(fbeta(1.0, 0.75, 6.0))


def test_evaluate_aggregated_per_hardness_query_counts(
    synthetic_eval_inputs: dict,
) -> None:
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    # Each (level, agg) slice should report the right n_queries per bucket.
    for level in ("tables", "columns"):
        for agg in ("macro", "micro"):
            assert _row(
                result.aggregated, hardness="easy", level=level, aggregation=agg
            )["n_queries"] == 2  # qids 0, 3
            assert _row(
                result.aggregated, hardness="medium", level=level, aggregation=agg
            )["n_queries"] == 1
            assert _row(
                result.aggregated, hardness="hard", level=level, aggregation=agg
            )["n_queries"] == 1
            assert _row(
                result.aggregated, hardness="extra", level=level, aggregation=agg
            )["n_queries"] == 1


def test_evaluate_aggregated_column_set_matches_contract(
    synthetic_eval_inputs: dict,
) -> None:
    """The aggregated frame must have the locked column schema."""
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    expected = {
        "method", "tier", "level", "hardness", "aggregation",
        "n_queries", "precision", "recall", "f1", "f6", "srr",
        "hallucination_rate",
    }
    assert set(result.aggregated.columns) == expected


def test_evaluate_aggregated_row_count(
    synthetic_eval_inputs: dict,
) -> None:
    """5 buckets (all + easy/medium/hard/extra) × 2 levels × 2 aggregations = 20."""
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    assert len(result.aggregated) == 20


# ---------- robustness rules ----------


def test_evaluate_skips_prediction_only_qids(
    synthetic_eval_inputs: dict, caplog: pytest.LogCaptureFixture
) -> None:
    """qid in predictions but not in gold should be skipped with a warning."""
    inp = dict(synthetic_eval_inputs)
    inp["predictions"] = {
        **inp["predictions"],
        99: {"db_id": "A", "tables": ["tA"], "columns": []},
    }
    with caplog.at_level(logging.WARNING, logger="schema_linking.evaluator"):
        result = evaluate(**inp, method_name="m", tier_name="t")
    assert 99 not in set(result.per_query["question_id"])
    assert any(
        "99" in rec.getMessage() and "predictions" in rec.getMessage()
        for rec in caplog.records
    )


def test_evaluate_treats_missing_predictions_as_empty(
    synthetic_eval_inputs: dict, caplog: pytest.LogCaptureFixture
) -> None:
    """qid in gold but not in predictions should evaluate as empty pred."""
    inp = dict(synthetic_eval_inputs)
    new_preds = dict(inp["predictions"])
    del new_preds[0]  # drop qid 0 from predictions; gold still has it
    inp["predictions"] = new_preds
    with caplog.at_level(logging.INFO, logger="schema_linking.evaluator"):
        result = evaluate(**inp, method_name="m", tier_name="t")
    pq = result.per_query.set_index("question_id")
    # qid 0's gold = {tA} / {(tA, x)}, empty pred → P=1, R=0, srr=False
    assert pq.loc[0, "table_predicted_count"] == 0
    assert pq.loc[0, "table_precision"] == pytest.approx(1.0)
    assert pq.loc[0, "table_recall"] == pytest.approx(0.0)
    assert bool(pq.loc[0, "table_srr_hit"]) is False


def test_evaluate_unknown_hardness_bucket(
    synthetic_eval_inputs: dict,
) -> None:
    """A qid missing from the hardness dict lands in the ``unknown`` bucket."""
    inp = dict(synthetic_eval_inputs)
    inp["hardness"] = {0: "easy"}  # only qid 0 has a hardness
    result = evaluate(**inp, method_name="m", tier_name="t")
    assert "unknown" in set(result.aggregated["hardness"])
    unknown_row = _row(
        result.aggregated, hardness="unknown", level="tables", aggregation="macro"
    )
    assert unknown_row["n_queries"] == 4  # qids 1, 2, 3, 4


def test_evaluate_skips_query_with_unknown_db_id(
    synthetic_eval_inputs: dict, caplog: pytest.LogCaptureFixture
) -> None:
    """A qid whose gold ``db_id`` has no schema should be skipped."""
    inp = dict(synthetic_eval_inputs)
    new_gold = dict(inp["gold"])
    new_gold[5] = {"db_id": "no_such_db", "tables": ["t"], "columns": []}
    new_preds = dict(inp["predictions"])
    new_preds[5] = {"db_id": "no_such_db", "tables": ["t"], "columns": []}
    inp["gold"], inp["predictions"] = new_gold, new_preds
    with caplog.at_level(logging.WARNING, logger="schema_linking.evaluator"):
        result = evaluate(**inp, method_name="m", tier_name="t")
    assert 5 not in set(result.per_query["question_id"])


# ---------------------------------------------------------------------------
# write_results
# ---------------------------------------------------------------------------


def test_write_results_concatenates_multiple_methods(
    synthetic_eval_inputs: dict, tmp_path: Path
) -> None:
    """Two methods on the same inputs should produce one combined CSV per frame."""
    result_a = evaluate(**synthetic_eval_inputs, method_name="alpha", tier_name="tier1")
    result_b = evaluate(**synthetic_eval_inputs, method_name="beta", tier_name="tier1")

    agg_path = tmp_path / "out" / "aggregated.csv"
    pq_path = tmp_path / "out" / "per_query.csv"
    write_results([result_a, result_b], agg_path, pq_path)
    assert agg_path.is_file() and pq_path.is_file()

    agg = pd.read_csv(agg_path)
    pq = pd.read_csv(pq_path)

    # Both methods land in both frames.
    assert set(agg["method"]) == {"alpha", "beta"}
    assert set(pq["method"]) == {"alpha", "beta"}

    # Row count sanity: aggregated == sum of the two individual results' rows.
    assert len(agg) == len(result_a.aggregated) + len(result_b.aggregated)
    assert len(pq) == len(result_a.per_query) + len(result_b.per_query)


def test_write_results_empty_list_writes_empty_csvs(tmp_path: Path) -> None:
    """An empty input list still produces files (with the aggregated header)."""
    agg_path = tmp_path / "agg.csv"
    pq_path = tmp_path / "pq.csv"
    write_results([], agg_path, pq_path)

    assert agg_path.is_file() and pq_path.is_file()
    agg = pd.read_csv(agg_path)
    # Aggregated keeps the locked column schema even when there are no rows.
    expected_cols = {
        "method", "tier", "level", "hardness", "aggregation",
        "n_queries", "precision", "recall", "f1", "f6", "srr",
        "hallucination_rate",
    }
    assert set(agg.columns) == expected_cols
    assert len(agg) == 0


def test_write_results_creates_parent_dirs(
    synthetic_eval_inputs: dict, tmp_path: Path
) -> None:
    """Nested non-existent parent directories should be created."""
    result = evaluate(**synthetic_eval_inputs, method_name="m", tier_name="t")
    agg_path = tmp_path / "deep" / "nested" / "dir" / "agg.csv"
    pq_path = tmp_path / "another" / "deep" / "path" / "pq.csv"
    write_results([result], agg_path, pq_path)
    assert agg_path.is_file() and pq_path.is_file()
