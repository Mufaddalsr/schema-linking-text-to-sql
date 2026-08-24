"""Chapter tables. Each is a pure function of the classified census."""

from __future__ import annotations

import pandas as pd
import pytest

from schema_linking.erroranalysis.census import CENSUS_COLUMNS
from schema_linking.erroranalysis.views import (
    cause_by_hardness,
    cause_by_method,
    cause_by_schema_size,
    gold_defects,
    shape_by_method,
)


def _row(**kw):
    base = {
        "question_id": 1,
        "db_id": "d",
        "method": "lexical",
        "tier": "tier1",
        "level": "column",
        "element": "singer.name",
        "shape_code": "MISS",
        "cause": "UNFORCED",
        "rule_name": "lexically_available",
        "evidence": "lexical=95",
        "hardness": "easy",
        "n_tables": 2,
        "n_columns": 6,
        "schema_size_bin": "Q1_smallest",
    }
    base.update(kw)
    return base


@pytest.fixture
def census():
    return pd.DataFrame(
        [
            _row(),
            _row(cause="PARAPHRASE", element="singer.country"),
            _row(shape_code="SPUR", cause="SIBLING", element="singer.id"),
            _row(method="graph", cause="UNVERBALISED", hardness="hard"),
            _row(method="graph", shape_code="HALL", cause="FABRICATED", element="x.y"),
            _row(tier="tier2", cause="JOIN-ONLY", element="concert"),
        ],
        columns=list(CENSUS_COLUMNS),
    )


@pytest.fixture
def bases():
    """Gold and predicted element counts per (method, tier)."""
    return pd.DataFrame(
        [
            {"method": "lexical", "tier": "tier1", "n_gold": 10, "n_predicted": 20},
            {"method": "lexical", "tier": "tier2", "n_gold": 12, "n_predicted": 20},
            {"method": "graph", "tier": "tier1", "n_gold": 10, "n_predicted": 8},
            {"method": "graph", "tier": "tier2", "n_gold": 12, "n_predicted": 8},
        ]
    )


def test_shape_by_method_counts_each_shape(census):
    table = shape_by_method(census)
    row = table[(table.method == "lexical") & (table.tier == "tier1")].iloc[0]
    assert row.MISS == 2
    assert row.SPUR == 1
    assert row.HALL == 0


def test_shape_by_method_covers_every_method_tier_pair(census):
    table = shape_by_method(census)
    assert len(table) == len(census.groupby(["method", "tier"]))


def test_cause_by_method_uses_gold_base_for_misses(census, bases):
    table = cause_by_method(census, bases, tier="tier1")
    row = table[(table.method == "lexical") & (table.cause == "UNFORCED")].iloc[0]
    assert row.n == 1
    assert row.base == 10
    assert row.rate == pytest.approx(0.1)
    assert row.base_kind == "gold_elements"


def test_cause_by_method_uses_prediction_base_for_spurious(census, bases):
    table = cause_by_method(census, bases, tier="tier1")
    row = table[(table.method == "lexical") & (table.cause == "SIBLING")].iloc[0]
    assert row.base == 20
    assert row.rate == pytest.approx(0.05)
    assert row.base_kind == "predicted_elements"


def test_cause_by_method_uses_prediction_base_for_hallucinations(census, bases):
    table = cause_by_method(census, bases, tier="tier1")
    row = table[(table.method == "graph") & (table.cause == "FABRICATED")].iloc[0]
    assert row.base == 8
    assert row.base_kind == "predicted_elements"


def test_cause_by_method_filters_to_the_requested_tier(census, bases):
    table = cause_by_method(census, bases, tier="tier1")
    assert "JOIN-ONLY" not in set(table.cause)


def test_cause_by_hardness_reports_within_hardness_shares(census):
    table = cause_by_hardness(census, tier="tier1")
    easy = table[table.hardness == "easy"]
    assert easy.share.sum() == pytest.approx(1.0)


def test_cause_by_schema_size_reports_within_bin_shares(census):
    table = cause_by_schema_size(census, tier="tier1")
    assert table.groupby("schema_size_bin").share.sum().round(9).eq(1.0).all()


def test_gold_defects_lists_only_gate_rows():
    frame = pd.DataFrame(
        [
            _row(cause="GOLD-DEFECT", rule_name="missed_by_most_methods",
                 evidence="element=singer.name; n_methods_missing=6; confirmed=pending"),
            _row(cause="UNFORCED"),
        ],
        columns=list(CENSUS_COLUMNS),
    )
    table = gold_defects(frame)
    assert len(table) == 1
    assert table.iloc[0].rule_name == "missed_by_most_methods"
    assert table.iloc[0].needs_confirmation is True


def test_gold_defects_marks_self_evidencing_rules_as_confirmed():
    frame = pd.DataFrame(
        [_row(cause="GOLD-DEFECT", rule_name="gold_element_not_in_schema")],
        columns=list(CENSUS_COLUMNS),
    )
    assert bool(gold_defects(frame).iloc[0].needs_confirmation) is False


def test_gold_defects_deduplicates_across_methods():
    """One defective gold element is one defect, not six."""
    frame = pd.DataFrame(
        [
            _row(cause="GOLD-DEFECT", rule_name="gold_element_not_in_schema", method=m)
            for m in ("lexical", "embedding", "graph")
        ],
        columns=list(CENSUS_COLUMNS),
    )
    table = gold_defects(frame)
    assert len(table) == 1
    assert table.iloc[0].n_methods_affected == 3
