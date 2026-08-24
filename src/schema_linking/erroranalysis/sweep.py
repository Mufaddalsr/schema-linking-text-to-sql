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
    """How much each cause's share moves across the grid.

    A cause with a near-zero ``range`` is threshold-independent and can be
    reported without qualification. A wide ``range`` must be reported with
    the sweep alongside it.
    """
    grouped = sweep.groupby("cause", as_index=False).agg(
        min_share=("share", "min"),
        max_share=("share", "max"),
        n_cells_present=("share", "size"),
    )
    return grouped.assign(
        range=grouped["max_share"] - grouped["min_share"]
    ).sort_values("range", ascending=False, ignore_index=True)
