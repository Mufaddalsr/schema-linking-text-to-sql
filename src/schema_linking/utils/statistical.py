"""Statistical tests for paired schema-linking comparisons.

For two methods evaluated on the same query set, the question
``did method A do better than method B on SRR?`` reduces to a paired
binary-outcomes test: per query, A either hits or misses; same for B.
McNemar's test operates on the 2×2 paired-outcomes table and ignores
queries where both methods agree — those carry no signal about the
direction of any difference.

This module exposes :func:`mcnemar_srr`, which takes the ``per_query``
DataFrames from :func:`evaluator.evaluate` and runs McNemar's test on
the per-query ``{element_type}_srr_hit`` booleans.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

__all__ = ["mcnemar_srr"]

# Standard rule of thumb (Agresti 2002, *Categorical Data Analysis*):
# below this many discordant pairs, prefer the exact binomial test;
# at or above it, the asymptotic chi-square is reliable.
_EXACT_THRESHOLD = 25


def mcnemar_srr(
    per_query_a: pd.DataFrame,
    per_query_b: pd.DataFrame,
    element_type: Literal["table", "column"],
) -> dict[str, float]:
    """McNemar's test on paired SRR hits between two methods.

    The 2×2 contingency table built from the joined per-query SRR
    booleans is::

                         B correct      B incorrect
        A correct          n_both         n_a_only
        A incorrect       n_b_only        n_neither

    Off-diagonal counts (``n_a_only`` + ``n_b_only``) carry the signal;
    queries where both methods hit or both miss are ignored by the
    test.

    Parameters
    ----------
    per_query_a, per_query_b
        :attr:`EvalResult.per_query` DataFrames filtered to one method
        each (and matching tier). Must contain ``question_id`` and
        ``{element_type}_srr_hit`` columns. Queries present in one but
        not the other are dropped via inner join.
    element_type
        ``"table"`` or ``"column"`` — selects which SRR column to test.

    Returns
    -------
    dict[str, float]
        Keys:

        * ``n_both``, ``n_a_only``, ``n_b_only``, ``n_neither`` — the
          four paired counts (ints).
        * ``statistic`` — for the exact path,
          ``min(n_a_only, n_b_only)`` (i.e. the smaller off-diagonal
          count, which is what the binomial test statistic boils down
          to). For the asymptotic path, the McNemar chi-square value
          with Edwards' continuity correction.
        * ``p_value`` — two-sided p-value.

    Notes
    -----
    Test selection: if ``n_a_only + n_b_only < 25`` the exact two-sided
    binomial test is used (``statsmodels.mcnemar(exact=True)``);
    otherwise the asymptotic chi-square with Edwards' continuity
    correction is used. 25 is the standard rule of thumb (Agresti 2002).
    """
    srr_col = f"{element_type}_srr_hit"

    merged = per_query_a[["question_id", srr_col]].merge(
        per_query_b[["question_id", srr_col]],
        on="question_id",
        suffixes=("_a", "_b"),
    )
    a_hit = merged[f"{srr_col}_a"].astype(bool)
    b_hit = merged[f"{srr_col}_b"].astype(bool)

    n_both = int((a_hit & b_hit).sum())
    n_a_only = int((a_hit & ~b_hit).sum())
    n_b_only = int((~a_hit & b_hit).sum())
    n_neither = int((~a_hit & ~b_hit).sum())

    table = [[n_both, n_a_only], [n_b_only, n_neither]]
    use_exact = (n_a_only + n_b_only) < _EXACT_THRESHOLD
    result = mcnemar(table, exact=use_exact)

    return {
        "n_both": n_both,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_neither": n_neither,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
