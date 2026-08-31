"""Identity-property tests on real Spider dev gold.

Feeding the gold back to :func:`evaluate` as a prediction must score
1.0 across every metric and 0 hallucinations. A failure here is a
STOP-and-debug signal — it means canonicalisation, set construction,
hallucination filtering, or aggregation is broken in a way that no
synthetic test will surface.

Two additional sanity tests:

* Empty predictions against Taniguchi gold — exercises the vacuous
  precision rule and recall fraction.
* All-hallucinated predictions — exercises the hallucination filter
  on real schemas at scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from schema_linking.data_loader import load_spider_questions
from schema_linking.evaluator import evaluate
from schema_linking.schema_parser import Schema, load_schemas
from schema_linking.utils.difficulty import difficulty_for_examples


# ---------- helpers ----------


def _load_gold(path: str) -> dict[int, dict[str, Any]]:
    """Read a gold JSON file and coerce its string qids back to int."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(qid): entry for qid, entry in raw.items()}


def _agg_row(df: pd.DataFrame, *, level: str, aggregation: str, hardness: str) -> pd.Series:
    sub = df[
        (df["level"] == level)
        & (df["aggregation"] == aggregation)
        & (df["hardness"] == hardness)
    ]
    assert len(sub) == 1, f"expected one row for level={level} agg={aggregation} h={hardness}; got {len(sub)}"
    return sub.iloc[0]


# ---------- module-scope fixtures ----------


@pytest.fixture(scope="module")
def real_schemas() -> dict[str, Schema]:
    return load_schemas()


@pytest.fixture(scope="module")
def hardness_map() -> dict[int, str]:
    return difficulty_for_examples(load_spider_questions("dev"))


@pytest.fixture(scope="module")
def taniguchi_gold() -> dict[int, dict[str, Any]]:
    return _load_gold("data/processed/gold_links_dev_mentioned.json")


@pytest.fixture(scope="module")
def sqlglot_tier2_gold() -> dict[int, dict[str, Any]]:
    return _load_gold("data/processed/gold_links_dev_all_sql.json")


# ---------- Test 1: Taniguchi gold ≡ itself ----------


def test_taniguchi_gold_identity_scores_all_ones(
    taniguchi_gold: dict[int, dict[str, Any]],
    real_schemas: dict[str, Schema],
    hardness_map: dict[int, str],
) -> None:
    """Taniguchi gold fed back as predictions must score exactly 1.0 on
    every aggregated row.

    A failure with diff > 1e-3 in any cell is a STOP signal — something
    is broken in canonicalisation, set construction, or aggregation.
    """
    result = evaluate(
        predictions=taniguchi_gold,
        gold=taniguchi_gold,
        schemas=real_schemas,
        hardness=hardness_map,
        method_name="identity",
        tier_name="tier1",
    )
    assert not result.aggregated.empty
    # Exact equality, not pytest.approx — any drift indicates a bug.
    for _, row in result.aggregated.iterrows():
        assert row["precision"] == 1.0, row.to_dict()
        assert row["recall"] == 1.0, row.to_dict()
        assert row["f1"] == 1.0, row.to_dict()
        assert row["f6"] == 1.0, row.to_dict()
        assert row["srr"] == 1.0, row.to_dict()
        assert row["hallucination_rate"] == 0.0, row.to_dict()


# ---------- Test 2: sqlglot Tier-2 gold ≡ itself ----------


def test_sqlglot_tier2_gold_identity_scores_all_ones(
    sqlglot_tier2_gold: dict[int, dict[str, Any]],
    real_schemas: dict[str, Schema],
    hardness_map: dict[int, str],
) -> None:
    """Same identity test for the sqlglot Tier-2 gold."""
    result = evaluate(
        predictions=sqlglot_tier2_gold,
        gold=sqlglot_tier2_gold,
        schemas=real_schemas,
        hardness=hardness_map,
        method_name="identity",
        tier_name="tier2",
    )
    assert not result.aggregated.empty
    for _, row in result.aggregated.iterrows():
        assert row["precision"] == 1.0, row.to_dict()
        assert row["recall"] == 1.0, row.to_dict()
        assert row["f1"] == 1.0, row.to_dict()
        assert row["f6"] == 1.0, row.to_dict()
        assert row["srr"] == 1.0, row.to_dict()
        assert row["hallucination_rate"] == 0.0, row.to_dict()


# ---------- Test 3: empty predictions ----------


def test_empty_predictions_against_taniguchi_gold(
    taniguchi_gold: dict[int, dict[str, Any]],
    real_schemas: dict[str, Schema],
    hardness_map: dict[int, str],
) -> None:
    """Predict ``{tables: [], columns: []}`` for every query.

    Per the locked edge-case rules (``∅ pred, non-∅ gold → P = 1.0``):

    * ``hallucination_rate == 0`` everywhere — no items predicted.
    * ``precision == 1.0`` for both levels (vacuous: zero false positives).
    * ``recall``  = fraction of queries whose gold at that level is empty
      (vacuous R = 1) — all other queries contribute R = 0.
    * ``srr``     = same fraction (``∅ ⊆ ∅`` but ``∅ ⊉ non-empty``).
    """
    empty_preds: dict[int, dict[str, Any]] = {
        qid: {"db_id": g["db_id"], "tables": [], "columns": []}
        for qid, g in taniguchi_gold.items()
    }
    result = evaluate(
        predictions=empty_preds,
        gold=taniguchi_gold,
        schemas=real_schemas,
        hardness=hardness_map,
        method_name="empty",
        tier_name="tier1",
    )

    n_total = len(taniguchi_gold)
    n_empty_tables = sum(1 for g in taniguchi_gold.values() if not g["tables"])
    n_empty_cols = sum(1 for g in taniguchi_gold.values() if not g["columns"])

    # 1) Hallucination rate is 0 across every aggregated row.
    for _, row in result.aggregated.iterrows():
        assert row["hallucination_rate"] == 0.0, row.to_dict()

    # 2) Sanity: at least one query in Taniguchi dev should have an empty
    #    gold column set (e.g. ``Count the number of templates.``).
    assert n_empty_cols > 0, (
        "Expected ≥1 empty-gold-columns query in Taniguchi dev — if this "
        "fires, either the gold file was regenerated wrong or this test's "
        "logic needs revising."
    )

    # 3) Precision is vacuous-1 for both levels under "all" / macro.
    tables_macro = _agg_row(
        result.aggregated, level="tables", aggregation="macro", hardness="all"
    )
    cols_macro = _agg_row(
        result.aggregated, level="columns", aggregation="macro", hardness="all"
    )
    assert tables_macro["precision"] == pytest.approx(1.0)
    assert cols_macro["precision"] == pytest.approx(1.0)

    # 4) Recall equals the empty-gold fraction at each level.
    assert tables_macro["recall"] == pytest.approx(n_empty_tables / n_total)
    assert cols_macro["recall"] == pytest.approx(n_empty_cols / n_total)

    # 5) SRR matches the same proportion (vacuous subset rule).
    assert tables_macro["srr"] == pytest.approx(n_empty_tables / n_total)
    assert cols_macro["srr"] == pytest.approx(n_empty_cols / n_total)


# ---------- Test 4: all-hallucinated predictions ----------


def test_all_hallucinated_predictions_against_taniguchi_gold(
    taniguchi_gold: dict[int, dict[str, Any]],
    real_schemas: dict[str, Schema],
    hardness_map: dict[int, str],
) -> None:
    """Predict one fabricated table name for every query.

    Every raw prediction is hallucinated, so ``hallucination_rate ==
    1.0`` for every aggregated row. After filtering, the prediction
    becomes empty, so the vacuous-precision rule applies and
    ``precision == 1.0`` for both levels. This is the locked behaviour
    — hallucination is reported as its own metric rather than being
    folded into precision.
    """
    fake_table = "ZZZ_THIS_TABLE_DOES_NOT_EXIST_ZZZ"
    fake_preds: dict[int, dict[str, Any]] = {
        qid: {"db_id": g["db_id"], "tables": [fake_table], "columns": []}
        for qid, g in taniguchi_gold.items()
    }
    result = evaluate(
        predictions=fake_preds,
        gold=taniguchi_gold,
        schemas=real_schemas,
        hardness=hardness_map,
        method_name="all_hallucinated",
        tier_name="tier1",
    )

    # Every row: every query predicted exactly 1 fake → 1/1 = 1.0.
    for _, row in result.aggregated.iterrows():
        assert row["hallucination_rate"] == pytest.approx(1.0), row.to_dict()

    # Filtered pred is empty for every query → vacuous P = 1.0.
    tables_macro = _agg_row(
        result.aggregated, level="tables", aggregation="macro", hardness="all"
    )
    cols_macro = _agg_row(
        result.aggregated, level="columns", aggregation="macro", hardness="all"
    )
    assert tables_macro["precision"] == pytest.approx(1.0)
    assert cols_macro["precision"] == pytest.approx(1.0)
