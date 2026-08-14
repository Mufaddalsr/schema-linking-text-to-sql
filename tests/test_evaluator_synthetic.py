"""Adversarial and edge-case synthetic scenarios for the evaluator.

Six hand-built ``(gold, prediction)`` pairs that cover the canonical
failure modes (perfect, empty-prediction, fully-vacuous,
hallucinations, pure over-prediction, pure under-prediction); a
property test that ``prediction = gold ⇒ all metrics = 1.0`` over 20
random gold sets; and an F6-weighting scenario showing recall
dominates precision in the recall-weighted score.

Synthetic data only — this file never touches real Spider data.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from schema_linking.evaluator import EvalResult, evaluate, fbeta
from schema_linking.schema_parser import Column, Schema, Table


# ---------------------------------------------------------------------------
# Schema fixture: 3 tables × 4 columns
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_schema() -> Schema:
    """3 tables (``a``, ``b``, ``c``), each with 4 columns (``c0``..``c3``)."""

    def make_table(name: str) -> Table:
        return Table(
            name=name,
            original_name=name,
            columns=[
                Column(
                    name=f"c{i}",
                    original_name=f"c{i}",
                    type="number",
                    table_name=name,
                    is_primary_key=False,
                )
                for i in range(4)
            ],
        )

    return Schema(
        db_id="synth",
        tables=[make_table(t) for t in ("a", "b", "c")],
        foreign_keys=[],
    )


def _single_query_eval(
    gold_entry: dict[str, Any],
    pred_entry: dict[str, Any],
    schema: Schema,
) -> EvalResult:
    """Wrap a single (gold, prediction) pair into an evaluate() call."""
    return evaluate(
        predictions={0: pred_entry},
        gold={0: gold_entry},
        schemas={"synth": schema},
        hardness={0: "easy"},
        method_name="m",
        tier_name="t",
    )


# ---------------------------------------------------------------------------
# Six adversarial / edge-case scenarios
# ---------------------------------------------------------------------------


def test_scenario_1_perfect_prediction(synthetic_schema: Schema) -> None:
    """gold == pred → P = R = F1 = 1.0, SRR = True on both levels."""
    gold = {
        "db_id": "synth",
        "tables": ["a"],
        "columns": [["a", "c0"], ["a", "c1"]],
    }
    pred = {
        "db_id": "synth",
        "tables": ["a"],
        "columns": [["a", "c0"], ["a", "c1"]],
    }
    result = _single_query_eval(gold, pred, synthetic_schema)
    pq = result.per_query.iloc[0]
    for level in ("table", "column"):
        assert pq[f"{level}_precision"] == pytest.approx(1.0)
        assert pq[f"{level}_recall"] == pytest.approx(1.0)
        assert pq[f"{level}_f1"] == pytest.approx(1.0)
        assert bool(pq[f"{level}_srr_hit"]) is True
    assert pq["hallucination_rate"] == 0.0


def test_scenario_2_empty_pred_nonempty_gold(synthetic_schema: Schema) -> None:
    """∅ prediction with non-empty gold: P = 1.0 (vacuous), R = 0, F1 = 0, SRR = False."""
    gold = {"db_id": "synth", "tables": ["a"], "columns": [["a", "c0"]]}
    pred = {"db_id": "synth", "tables": [], "columns": []}
    result = _single_query_eval(gold, pred, synthetic_schema)
    pq = result.per_query.iloc[0]
    for level in ("table", "column"):
        assert pq[f"{level}_precision"] == pytest.approx(1.0)
        assert pq[f"{level}_recall"] == pytest.approx(0.0)
        assert pq[f"{level}_f1"] == pytest.approx(0.0)
        assert bool(pq[f"{level}_srr_hit"]) is False


def test_scenario_3_empty_pred_empty_gold(synthetic_schema: Schema) -> None:
    """Fully vacuous: P = R = F1 = 1.0, SRR = True (∅ ⊆ ∅)."""
    gold = {"db_id": "synth", "tables": [], "columns": []}
    pred = {"db_id": "synth", "tables": [], "columns": []}
    result = _single_query_eval(gold, pred, synthetic_schema)
    pq = result.per_query.iloc[0]
    for level in ("table", "column"):
        assert pq[f"{level}_precision"] == pytest.approx(1.0)
        assert pq[f"{level}_recall"] == pytest.approx(1.0)
        assert pq[f"{level}_f1"] == pytest.approx(1.0)
        assert bool(pq[f"{level}_srr_hit"]) is True
    assert pq["hallucination_rate"] == 0.0


def test_scenario_4_hallucinated_tables_filtered_before_metric(
    synthetic_schema: Schema,
) -> None:
    """Hallucinated table_z is excluded from P/R but counted in
    ``hallucination_rate``."""
    gold = {"db_id": "synth", "tables": ["a"], "columns": [["a", "c0"]]}
    pred = {
        "db_id": "synth",
        "tables": ["a", "table_z"],          # table_z is not in the schema
        "columns": [["a", "c0"]],
    }
    result = _single_query_eval(gold, pred, synthetic_schema)
    pq = result.per_query.iloc[0]
    # Hallucination accounting: 1 hallucinated out of 3 raw predicted items.
    assert pq["table_hallucinated"] == 1
    assert pq["column_hallucinated"] == 0
    assert pq["hallucination_rate"] == pytest.approx(1 / 3)
    # After filtering, prediction matches gold exactly → P = R = 1.
    for level in ("table", "column"):
        assert pq[f"{level}_precision"] == pytest.approx(1.0)
        assert pq[f"{level}_recall"] == pytest.approx(1.0)
        assert bool(pq[f"{level}_srr_hit"]) is True


def test_scenario_5_pure_over_prediction(synthetic_schema: Schema) -> None:
    """gold ∪ {one extra schema-valid column}: recall = 1, precision < 1, SRR = True.

    No hallucinations (the extra is a real schema column, just not in gold).
    """
    gold = {"db_id": "synth", "tables": ["a"], "columns": [["a", "c0"]]}
    pred = {
        "db_id": "synth",
        "tables": ["a"],
        "columns": [["a", "c0"], ["a", "c1"]],   # (a, c1) is real but not in gold
    }
    result = _single_query_eval(gold, pred, synthetic_schema)
    pq = result.per_query.iloc[0]
    assert pq["hallucination_rate"] == pytest.approx(0.0)
    assert pq["column_hallucinated"] == 0
    # Columns: TP=1, FP=1 → P=0.5, R=1, SRR=True (gold ⊆ pred).
    assert pq["column_precision"] == pytest.approx(0.5)
    assert pq["column_recall"] == pytest.approx(1.0)
    assert pq["column_f1"] == pytest.approx(2 / 3)
    assert bool(pq["column_srr_hit"]) is True
    # Tables level untouched.
    assert pq["table_precision"] == pytest.approx(1.0)
    assert pq["table_recall"] == pytest.approx(1.0)
    assert bool(pq["table_srr_hit"]) is True


def test_scenario_6_pure_under_prediction(synthetic_schema: Schema) -> None:
    """gold − {one element}: recall < 1, SRR = False, precision = 1."""
    gold = {
        "db_id": "synth",
        "tables": ["a", "b"],
        "columns": [["a", "c0"], ["b", "c0"]],
    }
    pred = {
        "db_id": "synth",
        "tables": ["a"],                 # missing 'b'
        "columns": [["a", "c0"]],        # missing (b, c0)
    }
    result = _single_query_eval(gold, pred, synthetic_schema)
    pq = result.per_query.iloc[0]
    for level in ("table", "column"):
        # TP=1, FP=0, FN=1 → P=1, R=0.5; gold not subset of pred → SRR=False.
        assert pq[f"{level}_precision"] == pytest.approx(1.0)
        assert pq[f"{level}_recall"] == pytest.approx(0.5)
        assert pq[f"{level}_f1"] == pytest.approx(2 / 3)
        assert bool(pq[f"{level}_srr_hit"]) is False


# ---------------------------------------------------------------------------
# Property test: prediction = gold ⇒ all metrics = 1.0
# ---------------------------------------------------------------------------


def test_property_perfect_prediction_for_20_random_gold_sets(
    synthetic_schema: Schema,
) -> None:
    """For any (gold, gold) pair, every metric must be 1.0 and
    hallucination_rate must be 0 — regardless of cardinality.

    Sampling: 20 random gold sets with sizes uniformly drawn from
    the entire schema (3 tables × 4 columns)."""
    rng = random.Random(1234)
    table_pool = ["a", "b", "c"]

    gold: dict[int, dict[str, Any]] = {}
    pred: dict[int, dict[str, Any]] = {}
    hardness: dict[int, str] = {}

    for qid in range(20):
        n_tables = rng.randint(0, len(table_pool))
        tables = rng.sample(table_pool, n_tables)
        col_pool = [(t, f"c{i}") for t in tables for i in range(4)]
        n_cols = rng.randint(0, len(col_pool)) if col_pool else 0
        cols = rng.sample(col_pool, n_cols) if n_cols else []

        entry: dict[str, Any] = {
            "db_id": "synth",
            "tables": list(tables),
            "columns": [list(c) for c in cols],
        }
        gold[qid] = entry
        # Independent deep copy so an in-place mutation in evaluator
        # internals couldn't accidentally make this test pass.
        pred[qid] = {
            "db_id": "synth",
            "tables": list(entry["tables"]),
            "columns": [list(c) for c in entry["columns"]],
        }
        hardness[qid] = "easy"

    result = evaluate(
        pred,
        gold,
        {"synth": synthetic_schema},
        hardness,
        method_name="m",
        tier_name="t",
    )

    pq = result.per_query
    # Every per-query numeric metric is exactly 1.0.
    for col in (
        "table_precision",
        "table_recall",
        "table_f1",
        "column_precision",
        "column_recall",
        "column_f1",
    ):
        bad = pq.index[pq[col] != 1.0].tolist()
        assert not bad, f"non-1.0 in {col} at qids {bad}: values={pq.loc[bad, col].tolist()}"
    assert pq["table_srr_hit"].all()
    assert pq["column_srr_hit"].all()
    assert (pq["hallucination_rate"] == 0.0).all()

    # Aggregated rows: same invariant.
    for _, row in result.aggregated.iterrows():
        assert row["precision"] == pytest.approx(1.0), row.to_dict()
        assert row["recall"] == pytest.approx(1.0), row.to_dict()
        assert row["f1"] == pytest.approx(1.0), row.to_dict()
        assert row["srr"] == pytest.approx(1.0), row.to_dict()
        assert row["hallucination_rate"] == pytest.approx(0.0), row.to_dict()


# ---------------------------------------------------------------------------
# F6 weighting: recall dominates
# ---------------------------------------------------------------------------


def test_f6_user_specified_values_favour_recall_heavy() -> None:
    """User-spec values: P=0.99, R=0.50 vs P=0.50, R=0.99.

    Closed-form:
      F6(0.99, 0.50) = 37·0.99·0.5 / (36·0.99 + 0.5)  ≈ 0.507
      F6(0.50, 0.99) = 37·0.50·0.99 / (36·0.50 + 0.99) ≈ 0.965

    F6 weights recall β² = 36× more than precision, so the recall-heavy
    case dominates by a factor of ~1.9.
    """
    f6_precision_heavy = fbeta(0.99, 0.50, 6.0)
    f6_recall_heavy = fbeta(0.50, 0.99, 6.0)
    assert f6_precision_heavy == pytest.approx(0.507, abs=0.005)
    assert f6_recall_heavy == pytest.approx(0.965, abs=0.005)
    assert f6_recall_heavy > f6_precision_heavy
    assert f6_recall_heavy / f6_precision_heavy > 1.5


def test_f6_evaluator_scenario_recall_method_beats_precision_method(
    synthetic_schema: Schema,
) -> None:
    """Two methods on the same gold:

    * Method A — high precision, low recall (P=1.0, R=0.5).
    * Method B — low precision, high recall (P=0.5, R=1.0).

    F6(A) = fbeta(1.0, 0.5, 6) ≈ 0.507; F6(B) = fbeta(0.5, 1.0, 6) ≈ 0.974.
    """
    gold = {
        0: {
            "db_id": "synth",
            "tables": ["a"],
            "columns": [["a", c] for c in ("c0", "c1", "c2", "c3")],
        }
    }
    # A predicts 2 of 4 → TP=2, FP=0, FN=2 → P=1, R=0.5
    pred_a = {
        0: {
            "db_id": "synth",
            "tables": ["a"],
            "columns": [["a", "c0"], ["a", "c1"]],
        }
    }
    # B predicts all 4 + 4 extras → TP=4, FP=4, FN=0 → P=0.5, R=1
    pred_b = {
        0: {
            "db_id": "synth",
            "tables": ["a", "b"],
            "columns": [["a", c] for c in ("c0", "c1", "c2", "c3")]
            + [["b", c] for c in ("c0", "c1", "c2", "c3")],
        }
    }

    schemas = {"synth": synthetic_schema}
    hardness = {0: "easy"}
    result_a = evaluate(pred_a, gold, schemas, hardness, "A", "t")
    result_b = evaluate(pred_b, gold, schemas, hardness, "B", "t")

    def _col_f6(result: EvalResult) -> float:
        rows = result.aggregated
        match = rows[
            (rows["level"] == "columns")
            & (rows["aggregation"] == "macro")
            & (rows["hardness"] == "all")
        ]
        assert len(match) == 1
        return float(match.iloc[0]["f6"])

    f6_a = _col_f6(result_a)
    f6_b = _col_f6(result_b)

    assert f6_a == pytest.approx(fbeta(1.0, 0.5, 6.0), abs=1e-6)
    assert f6_b == pytest.approx(fbeta(0.5, 1.0, 6.0), abs=1e-6)
    assert f6_b > f6_a
    assert f6_b > 0.9   # recall-heavy F6 lives near 1
    assert f6_a < 0.6   # precision-heavy F6 is much lower
