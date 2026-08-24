"""Threshold sensitivity analysis for the anchoring causes.

``UNFORCED``, ``AMBIG-LOST``, ``PARAPHRASE`` and ``UNVERBALISED`` are
separated by two cut-offs. A single tuned value would make those four counts
an artefact of the tuning, so the chapter reports the whole grid and the
operating point's stability within it.

Every other cause is threshold-independent by construction; they appear in
the sweep with a flat share, which is itself a useful check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pandas as pd

from schema_linking.erroranalysis.facts import CaseFacts
from schema_linking.erroranalysis.rules import CascadeContext, classify_census

LEXICAL_GRID: tuple[int, ...] = (50, 60, 70, 80, 90)
"""rapidfuzz ``partial_ratio`` cut-offs to sweep."""

SEMANTIC_GRID: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75)
"""Cosine-similarity cut-offs to sweep."""


def sweep_thresholds(
    census: pd.DataFrame,
    facts_by_method: Mapping[str, Mapping[int, CaseFacts]],
    ctx: CascadeContext,
    lexical_grid: Sequence[int] = LEXICAL_GRID,
    semantic_grid: Sequence[float] = SEMANTIC_GRID,
) -> pd.DataFrame:
    """Re-classify the census at every threshold combination.

    Parameters
    ----------
    census
        An *unclassified* census frame. Re-classifying an already-coded frame
        works too — ``classify_census`` overwrites the three code columns.
    facts_by_method
        Built once; unchanged across cells.
    ctx
        Template context. Only ``ctx.cfg``'s two thresholds vary.

    Returns
    -------
    pandas.DataFrame
        Long form: one row per (lexical_threshold, semantic_threshold, cause).
    """
    rows = []
    for lex in lexical_grid:
        for sem in semantic_grid:
            cell_ctx = replace(
                ctx,
                cfg=replace(
                    ctx.cfg, lexical_threshold=int(lex), semantic_threshold=float(sem)
                ),
            )
            coded = classify_census(census, facts_by_method, cell_ctx)
            counts = coded.cause.value_counts()
            for cause, n in counts.items():
                rows.append(
                    {
                        "lexical_threshold": int(lex),
                        "semantic_threshold": float(sem),
                        "cause": str(cause),
                        "n": int(n),
                        "share": float(n) / len(coded),
                    }
                )
    return pd.DataFrame(rows)


def stability_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """How much each cause's share moves across the full grid.

    ``sweep_thresholds`` omits a row for ``(cell, cause)`` whenever the
    cause has zero rows in that cell — ``value_counts`` never emits zero
    counts. Left as-is, aggregating only the rows that exist understates
    ``range`` for any cause absent from at least one cell: its true minimum
    share there is ``0.0``, not the smallest *observed* non-zero share. A
    cause that only ever appears in a single cell would then report
    ``range == 0.0`` — the smallest possible value — and sort to the
    bottom of the table as the most "stable" cause, when a share that goes
    from 0 in every other cell to something nonzero in that one cell is in
    fact maximally volatile. This function zero-fills every ``(cell,
    cause)`` combination absent from ``sweep`` before computing
    ``min_share``/``max_share``, so ``range`` reflects the true swing
    across the whole grid.

    Parameters
    ----------
    sweep
        Output of :func:`sweep_thresholds`: one row per
        ``(lexical_threshold, semantic_threshold, cause)`` triple that
        actually occurred, with a ``share`` column.

    Returns
    -------
    pandas.DataFrame
        One row per cause, with columns ``cause``, ``min_share``,
        ``max_share``, ``range`` (``max_share - min_share``, computed after
        zero-filling absent cells) and ``n_cells_present`` (the number of
        grid cells where the cause actually has a nonzero row — *not*
        inflated by the zero-fill; it still means "cells this cause was
        observed in"). Sorted by ``range`` descending, so the most
        threshold-sensitive causes sort first. Any cause with
        ``n_cells_present`` below the total number of grid cells is, by
        construction, absent from at least one cell, so its ``min_share``
        is always ``0.0``.
    """
    cells = sweep[["lexical_threshold", "semantic_threshold"]].drop_duplicates()
    causes = pd.DataFrame({"cause": sweep["cause"].unique()})
    full_grid = cells.merge(causes, how="cross")
    filled = full_grid.merge(
        sweep[["lexical_threshold", "semantic_threshold", "cause", "share"]],
        on=["lexical_threshold", "semantic_threshold", "cause"],
        how="left",
    )
    filled["share"] = filled["share"].fillna(0.0)

    n_cells_present = (
        sweep.groupby("cause")["share"].size().rename("n_cells_present")
    )
    grouped = filled.groupby("cause", as_index=False).agg(
        min_share=("share", "min"),
        max_share=("share", "max"),
    )
    grouped = grouped.merge(n_cells_present, on="cause", how="left")
    return grouped.assign(
        range=grouped["max_share"] - grouped["min_share"]
    ).sort_values("range", ascending=False, ignore_index=True)
