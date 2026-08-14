"""Tests for ``schema_linking.run_evaluation`` (append_results, compare_methods_mcnemar).

Synthetic only — no Spider data or config touched. ``evaluate_method`` is
not unit-tested here (like ``run_linker.run_lexical_on_dev``, it's a thin
glue function over real ``load_config`` / ``load_schemas`` / dev data);
it's exercised by actually running it, not by a synthetic test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from schema_linking.run_evaluation import append_results, compare_methods_mcnemar
from schema_linking.utils.statistical import mcnemar_srr


class TestAppendResults:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        rows = pd.DataFrame(
            [{"method": "lexical", "tier": "tier1", "level": "tables", "f1": 0.5}]
        )
        append_results(rows, out, key_cols=["method", "tier", "level"])

        assert out.is_file()
        written = pd.read_csv(out)
        assert len(written) == 1
        assert written.iloc[0]["f1"] == pytest.approx(0.5)

    def test_rerun_replaces_same_key_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        first = pd.DataFrame(
            [{"method": "lexical", "tier": "tier1", "level": "tables", "f1": 0.5}]
        )
        append_results(first, out, key_cols=["method", "tier", "level"])

        second = pd.DataFrame(
            [{"method": "lexical", "tier": "tier1", "level": "tables", "f1": 0.9}]
        )
        append_results(second, out, key_cols=["method", "tier", "level"])

        written = pd.read_csv(out)
        assert len(written) == 1  # replaced, not duplicated
        assert written.iloc[0]["f1"] == pytest.approx(0.9)

    def test_rerun_preserves_unrelated_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        first = pd.DataFrame(
            [
                {"method": "lexical", "tier": "tier1", "level": "tables", "f1": 0.5},
                {"method": "lexical", "tier": "tier2", "level": "tables", "f1": 0.4},
            ]
        )
        append_results(first, out, key_cols=["method", "tier", "level"])

        second = pd.DataFrame(
            [{"method": "lexical", "tier": "tier1", "level": "tables", "f1": 0.99}]
        )
        append_results(second, out, key_cols=["method", "tier", "level"])

        written = pd.read_csv(out).sort_values("tier").reset_index(drop=True)
        assert len(written) == 2
        tier1_row = written[written["tier"] == "tier1"].iloc[0]
        tier2_row = written[written["tier"] == "tier2"].iloc[0]
        assert tier1_row["f1"] == pytest.approx(0.99)
        assert tier2_row["f1"] == pytest.approx(0.4)  # untouched


def _per_query_row(
    qid: int, method: str, tier: str, table_hit: bool, column_hit: bool
) -> dict:
    return {
        "question_id": qid,
        "method": method,
        "tier": tier,
        "table_srr_hit": table_hit,
        "column_srr_hit": column_hit,
    }


@pytest.fixture
def synthetic_per_query_csv(tmp_path: Path) -> Path:
    """10 queries x 2 methods x 2 tiers, with a hand-designed disagreement pattern."""
    rows = []
    for qid in range(10):
        # method A hits tables on evens, B hits on odds -> pure disagreement.
        a_table_hit = qid % 2 == 0
        b_table_hit = qid % 2 == 1
        # Both always agree on columns (all hits) -> no discordant pairs.
        rows.append(_per_query_row(qid, "A", "tier1", a_table_hit, True))
        rows.append(_per_query_row(qid, "B", "tier1", b_table_hit, True))
        rows.append(_per_query_row(qid, "A", "tier2", a_table_hit, True))
        rows.append(_per_query_row(qid, "B", "tier2", b_table_hit, True))
    path = tmp_path / "main_per_query.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestCompareMethodsMcnemar:
    def test_returns_one_row_per_tier_and_element_type(
        self, synthetic_per_query_csv: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "mcnemar.csv"
        result = compare_methods_mcnemar(
            "A", "B", per_query_path=synthetic_per_query_csv, output_path=out
        )

        assert len(result) == 4  # 2 tiers x 2 element types
        assert set(result["tier"]) == {"tier1", "tier2"}
        assert set(result["element_type"]) == {"table", "column"}
        assert set(result.columns) == {
            "method_a",
            "method_b",
            "tier",
            "element_type",
            "n_a_only",
            "n_b_only",
            "n_both",
            "n_neither",
            "statistic",
            "p_value",
        }

    def test_writes_csv_to_output_path(
        self, synthetic_per_query_csv: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "mcnemar.csv"
        compare_methods_mcnemar(
            "A", "B", per_query_path=synthetic_per_query_csv, output_path=out
        )
        assert out.is_file()
        reloaded = pd.read_csv(out)
        assert len(reloaded) == 4

    def test_matches_direct_mcnemar_srr_call(
        self, synthetic_per_query_csv: Path, tmp_path: Path
    ) -> None:
        per_query = pd.read_csv(synthetic_per_query_csv)
        tier1 = per_query[per_query["tier"] == "tier1"]
        expected = mcnemar_srr(
            tier1[tier1["method"] == "A"], tier1[tier1["method"] == "B"], "table"
        )

        result = compare_methods_mcnemar(
            "A", "B", per_query_path=synthetic_per_query_csv, output_path=tmp_path / "mcnemar.csv"
        )
        row = result[(result["tier"] == "tier1") & (result["element_type"] == "table")].iloc[0]

        assert row["n_a_only"] == expected["n_a_only"]
        assert row["n_b_only"] == expected["n_b_only"]
        assert row["n_both"] == expected["n_both"]
        assert row["n_neither"] == expected["n_neither"]
        assert row["p_value"] == pytest.approx(expected["p_value"])

    def test_pure_disagreement_pattern_has_zero_agreement_counts(
        self, synthetic_per_query_csv: Path, tmp_path: Path
    ) -> None:
        # By construction, table hits never agree (n_both = n_neither = 0)
        # and column hits always agree (n_a_only = n_b_only = 0).
        result = compare_methods_mcnemar(
            "A", "B", per_query_path=synthetic_per_query_csv, output_path=tmp_path / "mcnemar.csv"
        )
        table_row = result[(result["tier"] == "tier1") & (result["element_type"] == "table")].iloc[0]
        column_row = result[(result["tier"] == "tier1") & (result["element_type"] == "column")].iloc[0]

        assert table_row["n_both"] == 0
        assert table_row["n_neither"] == 0
        assert table_row["n_a_only"] == 5
        assert table_row["n_b_only"] == 5

        assert column_row["n_a_only"] == 0
        assert column_row["n_b_only"] == 0
        assert column_row["n_both"] == 10

    def test_skips_tier_missing_one_method(self, tmp_path: Path) -> None:
        rows = [
            _per_query_row(0, "A", "tier1", True, True),
            _per_query_row(0, "B", "tier1", True, True),
            _per_query_row(0, "A", "tier2", True, True),
            # "B" has no tier2 rows at all.
        ]
        path = tmp_path / "partial.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        result = compare_methods_mcnemar(
            "A", "B", per_query_path=path, output_path=tmp_path / "mcnemar.csv"
        )
        assert set(result["tier"]) == {"tier1"}
        assert len(result) == 2  # table + column, tier1 only
