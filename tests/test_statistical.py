"""Tests for src/schema_linking/utils/statistical.py.

Three hand-built scenarios for :func:`mcnemar_srr`:

1. A clearly beats B — large discordant counts, asymptotic path.
2. Identical predictions — p = 1.0, statistic = 0.
3. Small sample (n=10) — exact path, statistic = ``min(n_a_only, n_b_only)``.

Plus an element_type="column" sanity check.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest

from schema_linking.utils.statistical import mcnemar_srr


def _per_query_frame(
    a_hits: list[bool],
    *,
    element_type: Literal["table", "column"] = "table",
) -> pd.DataFrame:
    """Build a minimal `per_query` frame carrying just `question_id` and
    the SRR column for one element type."""
    col = f"{element_type}_srr_hit"
    return pd.DataFrame(
        {
            "question_id": list(range(len(a_hits))),
            col: a_hits,
        }
    )


# ---------- 1. A clearly beats B -------------------------------------------


def test_a_clearly_beats_b() -> None:
    """100 queries: A correct on 80, B correct on 50, 40 both correct.

    Expected paired counts:
      n_both = 40, n_a_only = 40, n_b_only = 10, n_neither = 10.

    Discordant = 50 ≥ 25 → asymptotic chi-square (with continuity
    correction). Manual chi² = (|40 - 10| - 1)² / (40 + 10) = 841/50
    ≈ 16.82; the corresponding p-value is ≈ 4e-5.
    """
    a_hits = [True] * 80 + [False] * 20
    b_hits = (
        [True] * 40         # qids 0..39   — both correct
        + [False] * 40      # qids 40..79  — A only
        + [True] * 10       # qids 80..89  — B only
        + [False] * 10      # qids 90..99  — neither
    )
    df_a = _per_query_frame(a_hits)
    df_b = _per_query_frame(b_hits)

    result = mcnemar_srr(df_a, df_b, element_type="table")

    assert result["n_both"] == 40
    assert result["n_a_only"] == 40
    assert result["n_b_only"] == 10
    assert result["n_neither"] == 10
    assert result["p_value"] < 0.001
    # Asymptotic chi-square with Edwards' correction ≈ 16.82.
    assert result["statistic"] == pytest.approx(16.82, abs=0.01)


# ---------- 2. Identical predictions ---------------------------------------


def test_identical_predictions() -> None:
    """When A == B on every query, discordant counts are zero, statistic
    is zero, and the test produces p = 1."""
    a_hits = [True] * 60 + [False] * 40
    df_a = _per_query_frame(a_hits)
    df_b = df_a.copy()

    result = mcnemar_srr(df_a, df_b, element_type="table")

    assert result["n_a_only"] == 0
    assert result["n_b_only"] == 0
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0


# ---------- 3. Small sample, exact path ------------------------------------


def test_small_sample_uses_exact_path() -> None:
    """n=10 with 6 discordant pairs (< 25) routes through the exact path.

    Construction:

    ============  ===  ===  ====
    qid           A    B    cell
    ============  ===  ===  ====
    0, 1, 2        T    T   both
    3, 4, 5, 6, 7  T    F   A only (5 queries)
    8              F    T   B only (1 query)
    9              F    F   neither
    ============  ===  ===  ====

    Discordant pairs = 5 + 1 = 6 < 25. statsmodels' exact path returns
    ``statistic = min(n_a_only, n_b_only) = 1``. The two-sided binomial
    test with n=6, k=1, p=0.5 has p = 2·∑_{i≤1} C(6,i)·0.5⁶ = 14/64
    = 0.21875.
    """
    a_hits = [True] * 8 + [False] * 2
    b_hits = (
        [True] * 3         # qids 0..2 — both
        + [False] * 5      # qids 3..7 — A only
        + [True] * 1       # qid 8     — B only
        + [False] * 1      # qid 9     — neither
    )
    df_a = _per_query_frame(a_hits)
    df_b = _per_query_frame(b_hits)

    result = mcnemar_srr(df_a, df_b, element_type="table")

    assert result["n_both"] == 3
    assert result["n_a_only"] == 5
    assert result["n_b_only"] == 1
    assert result["n_neither"] == 1
    # Exact-path signature: statistic is the smaller off-diagonal count.
    assert result["statistic"] == 1.0
    # Two-sided binomial p with n=6, k=1, p=0.5.
    assert result["p_value"] == pytest.approx(0.21875, abs=1e-6)


def test_exact_threshold_boundary_discordant_24_uses_exact() -> None:
    """24 discordant pairs is the largest count that still routes through
    the exact binomial path (threshold is ``< 25``)."""
    # 12 A-only, 12 B-only, 50 both, 50 neither → 124 queries total.
    a_hits = [True] * 12 + [False] * 12 + [True] * 50 + [False] * 50
    b_hits = [False] * 12 + [True] * 12 + [True] * 50 + [False] * 50
    df_a = _per_query_frame(a_hits)
    df_b = _per_query_frame(b_hits)
    result = mcnemar_srr(df_a, df_b, element_type="table")
    assert result["n_a_only"] + result["n_b_only"] == 24
    # Exact path: statistic == min(off-diagonals) = 12, an integer count.
    assert result["statistic"] == 12.0


def test_exact_threshold_boundary_discordant_25_uses_asymptotic() -> None:
    """25 discordant pairs crosses into the asymptotic path.

    Construction uses asymmetric off-diagonals (5 vs 20) so the chi²
    value (≈ 7.84 with Edwards' correction) is unambiguously distinct
    from ``min(n_a_only, n_b_only) = 5`` (the exact-path signature).
    """
    a_hits = [True] * 5 + [False] * 20 + [True] * 50 + [False] * 50
    b_hits = [False] * 5 + [True] * 20 + [True] * 50 + [False] * 50
    df_a = _per_query_frame(a_hits)
    df_b = _per_query_frame(b_hits)
    result = mcnemar_srr(df_a, df_b, element_type="table")
    assert result["n_a_only"] + result["n_b_only"] == 25
    assert result["n_a_only"] == 5
    assert result["n_b_only"] == 20
    # Asymptotic chi² with Edwards' correction = (|5-20|-1)² / 25 = 196/25 = 7.84.
    # Exact path would return min(5, 20) = 5.
    assert result["statistic"] == pytest.approx(7.84, abs=0.01)


# ---------- 4. element_type plumbing ---------------------------------------


def test_element_type_column_uses_column_column() -> None:
    """element_type='column' should read column_srr_hit, not table_srr_hit."""
    qids = list(range(10))
    df_a = pd.DataFrame(
        {
            "question_id": qids,
            # Different values to make the test fail loudly if the wrong
            # column is being read.
            "table_srr_hit": [False] * 10,
            "column_srr_hit": [True] * 10,
        }
    )
    df_b = pd.DataFrame(
        {
            "question_id": qids,
            "table_srr_hit": [True] * 10,
            "column_srr_hit": [True] * 10,
        }
    )

    result = mcnemar_srr(df_a, df_b, element_type="column")

    # Both A and B hit on every query for the column SRR → all `n_both`.
    assert result["n_both"] == 10
    assert result["n_a_only"] == 0
    assert result["n_b_only"] == 0
    assert result["p_value"] == 1.0


# ---------- 5. Inner-join behaviour for mismatched qid sets ----------------


def test_qids_not_present_in_both_are_dropped() -> None:
    """If a qid appears in only one input, McNemar can't pair it; drop it."""
    df_a = pd.DataFrame(
        {"question_id": [0, 1, 2, 3], "table_srr_hit": [True, True, True, False]}
    )
    df_b = pd.DataFrame(
        {"question_id": [2, 3, 4, 5], "table_srr_hit": [True, False, True, True]}
    )
    # Intersection: {2, 3}; A says (T, F), B says (T, F) → both agree.
    result = mcnemar_srr(df_a, df_b, element_type="table")
    assert result["n_both"] + result["n_neither"] == 2
    assert result["n_a_only"] == 0
    assert result["n_b_only"] == 0
