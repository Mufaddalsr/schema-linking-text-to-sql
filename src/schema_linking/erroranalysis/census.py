"""Enumerate the error census: one row per erroneous link.

Axis 1 only. Cause assignment is layered on by
:mod:`schema_linking.erroranalysis.rules`, and is deliberately separate so
this enumeration can be proved equal to ``main_per_query.csv`` before any
judgement enters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from schema_linking.erroranalysis.facts import CaseFacts
from schema_linking.erroranalysis.taxonomy import Element, ErrorInstance, Shape

METHODS: tuple[str, ...] = (
    "lexical",
    "embedding",
    "llm_forward",
    "llm_backward",
    "llm_bidirectional",
    "graph",
)
"""The six methods in scope §3 order (A, B, C, D, E, G)."""

TIERS: tuple[str, ...] = ("tier1", "tier2")

CENSUS_COLUMNS: tuple[str, ...] = (
    "question_id",
    "db_id",
    "method",
    "tier",
    "level",
    "element",
    "shape_code",
    "cause",
    "rule_name",
    "evidence",
    "hardness",
    "n_tables",
    "n_columns",
    "schema_size_bin",
)
"""Column order of ``census.csv``.

``shape_code`` rather than ``shape`` because ``DataFrame.shape`` is taken —
``row.shape`` would silently return the frame's dimensions.
"""


def _exists(element: Element, facts: CaseFacts) -> bool:
    """Whether ``element`` is present in the case's database schema."""
    if element.level == "table":
        return element.table in facts.index.tables
    return element in facts.index.columns


def enumerate_errors(
    facts: CaseFacts,
    method: str,
    tier: str,
) -> list[ErrorInstance]:
    """Every erroneous link for one (question, method, tier).

    Parameters
    ----------
    facts
        The case bundle. ``facts.predicted`` is this method's prediction.
    method
        One of :data:`METHODS`.
    tier
        ``"tier1"`` or ``"tier2"``.

    Returns
    -------
    list[ErrorInstance]
        Misses first, then spurious and hallucinated predictions. Each
        element appears at most once: ``HALL`` pre-empts ``SPUR``.
    """
    gold = facts.gold_for(tier)
    errors = [
        ErrorInstance(
            question_id=facts.question_id,
            db_id=facts.db_id,
            method=method,
            tier=tier,
            element=el,
            shape=Shape.MISS,
        )
        for el in sorted(gold - facts.predicted, key=lambda e: (e.level, e.render()))
    ]
    for el in sorted(
        facts.predicted - gold, key=lambda e: (e.level, e.render())
    ):
        errors.append(
            ErrorInstance(
                question_id=facts.question_id,
                db_id=facts.db_id,
                method=method,
                tier=tier,
                element=el,
                shape=Shape.SPUR if _exists(el, facts) else Shape.HALL,
            )
        )
    return errors


def errors_to_frame(
    errors: Sequence[ErrorInstance],
    facts_by_qid: Mapping[int, CaseFacts],
) -> pd.DataFrame:
    """Render error instances as the long census frame.

    ``cause``, ``rule_name``, ``evidence`` and ``schema_size_bin`` are left
    empty; they are filled by :mod:`rules` and by
    :func:`add_schema_size_bin` respectively.
    """
    rows = []
    for err in errors:
        facts = facts_by_qid[err.question_id]
        rows.append(
            {
                "question_id": err.question_id,
                "db_id": err.db_id,
                "method": err.method,
                "tier": err.tier,
                "level": err.element.level,
                "element": err.element.render(),
                "shape_code": str(err.shape),
                "cause": "",
                "rule_name": "",
                "evidence": "",
                "hardness": facts.hardness,
                "n_tables": facts.n_tables,
                "n_columns": facts.n_columns,
                "schema_size_bin": "",
            }
        )
    return pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))


def add_schema_size_bin(frame: pd.DataFrame) -> pd.DataFrame:
    """Add quartile bins over ``n_columns``, computed across databases.

    Binning is over the set of distinct ``(db_id, n_columns)`` pairs, not
    over error rows — otherwise a database with many errors would drag the
    quartile boundaries toward itself.
    """
    per_db = frame[["db_id", "n_columns"]].drop_duplicates()
    labels = ["Q1_smallest", "Q2", "Q3", "Q4_largest"]
    per_db = per_db.assign(
        schema_size_bin=pd.qcut(
            per_db["n_columns"], q=4, labels=labels, duplicates="drop"
        ).astype(str)
    )
    return frame.drop(columns=["schema_size_bin"]).merge(
        per_db[["db_id", "schema_size_bin"]], on="db_id", how="left"
    )[list(CENSUS_COLUMNS)]
