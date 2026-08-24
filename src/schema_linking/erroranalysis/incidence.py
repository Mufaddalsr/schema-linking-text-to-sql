"""Which methods found which gold element.

The census is organised by method; this module transposes it. Each row is a
gold element and each method is a column, so the questions that matter for
RQ4 become one-liners: what did every method miss (suspect the gold), what
did exactly one method find (that is the method's real contribution), and
where does a method stand alone in failing.

No sampling design can produce this table — it needs every method's verdict
on every gold element.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from schema_linking.erroranalysis.census import METHODS
from schema_linking.erroranalysis.facts import elements_from_record

if TYPE_CHECKING:
    from schema_linking.erroranalysis.loading import Corpus


def build_incidence(
    corpus: "Corpus",
    tier: str = "tier1",
    methods: Sequence[str] = METHODS,
) -> pd.DataFrame:
    """Gold element x method incidence for one tier.

    Parameters
    ----------
    corpus
        Loaded split: supplies gold tiers and per-method predictions.
    tier
        ``"tier1"`` or ``"tier2"``.
    methods
        Method names to build columns for, in column order.

    Returns
    -------
    pandas.DataFrame
        Columns: ``question_id``, ``db_id``, ``level``, ``element``, one
        boolean column per method (named after the method), and
        ``n_found``, the number of methods that found the element.
    """
    gold_by_qid = corpus.gold_tier1 if tier == "tier1" else corpus.gold_tier2
    rows = []
    for qid, gold_record in gold_by_qid.items():
        predicted = {
            m: elements_from_record(corpus.predictions[m][qid]) for m in methods
        }
        for el in sorted(
            elements_from_record(gold_record), key=lambda e: (e.level, e.render())
        ):
            found = {m: bool(el in predicted[m]) for m in methods}
            rows.append(
                {
                    "question_id": qid,
                    "db_id": gold_record["db_id"],
                    "level": el.level,
                    "element": el.render(),
                    **found,
                    "n_found": sum(found.values()),
                }
            )
    return pd.DataFrame(rows)


def hard_cases(incidence: pd.DataFrame, max_found: int = 1) -> pd.DataFrame:
    """Gold elements found by at most ``max_found`` methods.

    ``n_found == 0`` are gold-defect candidates: if no method finds an
    element, the annotation is as likely at fault as six independent
    methods. ``n_found == 1`` are the discriminating cases — the only gold
    elements where the methods genuinely disagree.

    Parameters
    ----------
    incidence
        Output of :func:`build_incidence`.
    max_found
        Inclusive upper bound on ``n_found`` to keep.

    Returns
    -------
    pandas.DataFrame
        ``incidence`` filtered to the rare-find rows, plus a ``found_by``
        column naming the finder (empty string when ``n_found == 0``).
    """
    methods = [c for c in incidence.columns if incidence[c].dtype == bool]
    subset = incidence[incidence.n_found <= max_found].copy()
    subset["found_by"] = [
        ", ".join(m for m in methods if row[m]) for _, row in subset.iterrows()
    ]
    return subset.sort_values(
        ["n_found", "question_id", "element"], ignore_index=True
    )


def method_contrast(incidence: pd.DataFrame) -> pd.DataFrame:
    """Per method: how often it stands alone, in success and in failure.

    Parameters
    ----------
    incidence
        Output of :func:`build_incidence`.

    Returns
    -------
    pandas.DataFrame
        Columns ``method``, ``n_found``, ``n_missed``, ``n_unique_finds``
        (only this method found it), ``n_unique_misses`` (every other
        method found it and only this one missed), and the two as shares of
        that method's totals.
    """
    methods = [c for c in incidence.columns if incidence[c].dtype == bool]
    n_methods = len(methods)
    rows = []
    for m in methods:
        found = incidence[m]
        unique_finds = int(((incidence.n_found == 1) & found).sum())
        unique_misses = int(((incidence.n_found == n_methods - 1) & ~found).sum())
        n_found = int(found.sum())
        n_missed = int((~found).sum())
        rows.append(
            {
                "method": m,
                "n_found": n_found,
                "n_missed": n_missed,
                "n_unique_finds": unique_finds,
                "n_unique_misses": unique_misses,
                "unique_find_share": unique_finds / n_found if n_found else 0.0,
                "unique_miss_share": unique_misses / n_missed if n_missed else 0.0,
            }
        )
    return pd.DataFrame(rows)
