"""Error taxonomy: two orthogonal axes over the error census.

Axis 1 (:class:`Shape`) is deterministic — it is read off the confusion
matrix and needs no judgement. Axis 2 (:class:`Cause`) is assigned by the
ordered cascade in :mod:`schema_linking.erroranalysis.rules`.

This module is the single source of truth for both axes. Operational
definitions live in :data:`SHAPE_DEFINITIONS` and :data:`CAUSE_DEFINITIONS`
(module-level dicts, not member docstrings — enum member ``__doc__`` is
``None`` at runtime, so it cannot carry text). ``codebook.py`` renders
``outputs/error_analysis/codebook.md`` from those dicts, so a definition is
never maintained in two places.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from schema_linking.evaluator import _canonicalise_column, _canonicalise_table

Level = Literal["table", "column"]
Evidence = dict[str, str | int | float]


class Shape(StrEnum):
    """Which side of the confusion matrix an error sits on."""

    MISS = "MISS"
    SPUR = "SPUR"
    HALL = "HALL"


SHAPE_DEFINITIONS: dict[Shape, str] = {
    Shape.MISS: (
        "Element is in the tier's gold set and absent from the prediction."
    ),
    Shape.SPUR: (
        "Element is predicted, exists in the schema, and is not gold for "
        "this tier."
    ),
    Shape.HALL: inspect.cleandoc(
        """
        Element is predicted and does not exist in the database schema.

        Takes precedence over `SPUR`: a predicted element absent from the schema is never also counted as spurious.
        """
    ),
}


class Cause(StrEnum):
    """Why the error happened. Assigned by first-match on an ordered cascade."""

    GOLD_DEFECT = "GOLD-DEFECT"
    JOIN_ONLY = "JOIN-ONLY"
    IMPLICIT_AGG = "IMPLICIT-AGG"
    UNVERBALISED = "UNVERBALISED"
    PARAPHRASE = "PARAPHRASE"
    AMBIG_LOST = "AMBIG-LOST"
    UNFORCED = "UNFORCED"
    TIER_ARTEFACT = "TIER-ARTEFACT"
    NAME_COLLISION = "NAME-COLLISION"
    SIBLING = "SIBLING"
    QUESTION_ANCHORED = "QUESTION-ANCHORED"
    UNANCHORED = "UNANCHORED"
    WRONG_DB = "WRONG-DB"
    MALFORMED = "MALFORMED"
    FABRICATED = "FABRICATED"
    UNRESOLVED = "UNRESOLVED"


CAUSE_DEFINITIONS: dict[Cause, str] = {
    Cause.GOLD_DEFECT: inspect.cleandoc(
        """
        The gold annotation, not the method, is at fault.

        Fires when the gold element does not exist in the database schema; or it is Tier-1 gold that never appears in the gold SQL string; or it is gold, present in the schema, and missed by at least five of the six methods. The first two clauses exclude automatically; the third only flags for manual confirmation (design §7.3).
        """
    ),
    Cause.JOIN_ONLY: inspect.cleandoc(
        """
        In Tier-2 gold and absent from Tier-1 gold.

        The question never referred to the element, so the method was not asked to find it from the question alone. Fires on Tier-2 only.
        """
    ),
    Cause.IMPLICIT_AGG: (
        "The element is the `*` selector, or is required only by an "
        "aggregate with no independent mention in the question."
    ),
    Cause.UNVERBALISED: inspect.cleandoc(
        """
        Nothing in the question points at the element.

        No question span matches the element name lexically (below the fuzzy threshold) and none matches it semantically (below the embedding threshold).
        """
    ),
    Cause.PARAPHRASE: inspect.cleandoc(
        """
        A semantically matching question span exists but no lexical one.

        Synonym or paraphrase resolution was needed and failed. This is v1's `LM`, made precise.
        """
    ),
    Cause.AMBIG_LOST: inspect.cleandoc(
        """
        Right name, wrong table.

        The element name lexically matches a question span, and the method predicted a same-canonical-named element belonging to a different table.
        """
    ),
    Cause.UNFORCED: (
        "The element name lexically matches a question span and the "
        "method missed it with no competing prediction. An unforced "
        "error."
    ),
    Cause.TIER_ARTEFACT: inspect.cleandoc(
        """
        The element is gold in the other tier.

        The method is correct; the tier is strict. 543 of lexical's 3,156 Tier-1 column false positives fall here (design §1.4).
        """
    ),
    Cause.NAME_COLLISION: inspect.cleandoc(
        """
        The element shares a canonical name with a gold element belonging to a different table.

        Outranks `SIBLING` because it is the more specific condition: both can hold at once when the collided-with table is itself gold.
        """
    ),
    Cause.SIBLING: inspect.cleandoc(
        """
        Over-generation in the right neighbourhood.

        The element is a column of a table that is in gold, or a table that is foreign-key-adjacent to a gold table.
        """
    ),
    Cause.QUESTION_ANCHORED: (
        "The element name lexically matches a question span but the "
        "element is not gold in either tier. The question mentions it; "
        "the gold does not need it."
    ),
    Cause.UNANCHORED: (
        "Free-floating over-prediction: no lexical or semantic anchor in "
        "the question, and no relation to any gold element."
    ),
    Cause.WRONG_DB: (
        "The predicted name exists in the schema of a different Spider "
        "database."
    ),
    Cause.MALFORMED: (
        "The predicted entry is structurally invalid — unparseable, "
        "wrong arity, or empty."
    ),
    Cause.FABRICATED: "The predicted name appears in no Spider schema at all.",
    Cause.UNRESOLVED: (
        "No rule matched. Exported to `residual_review.csv` for human "
        "adjudication; never assigned silently."
    ),
}


CAUSES_FOR_SHAPE: dict[Shape, tuple[Cause, ...]] = {
    Shape.MISS: (
        Cause.GOLD_DEFECT,
        Cause.JOIN_ONLY,
        Cause.IMPLICIT_AGG,
        Cause.AMBIG_LOST,
        Cause.UNFORCED,
        Cause.PARAPHRASE,
        Cause.UNVERBALISED,
        Cause.UNRESOLVED,
    ),
    Shape.SPUR: (
        Cause.TIER_ARTEFACT,
        Cause.NAME_COLLISION,
        Cause.SIBLING,
        Cause.QUESTION_ANCHORED,
        Cause.UNANCHORED,
        Cause.UNRESOLVED,
    ),
    Shape.HALL: (
        Cause.MALFORMED,
        Cause.WRONG_DB,
        Cause.FABRICATED,
        Cause.UNRESOLVED,
    ),
}
"""Cascade order per shape. Position in the tuple *is* the precedence."""


@dataclass(frozen=True, slots=True)
class Element:
    """A schema element in canonical (lowercased, stripped) form.

    Attributes
    ----------
    level
        ``"table"`` or ``"column"``.
    table
        Canonical table name.
    column
        Canonical column name, or ``""`` when ``level == "table"``.
    """

    level: Level
    table: str
    column: str

    @classmethod
    def table_el(cls, name: str) -> "Element":
        """Build a table element, canonicalising ``name``."""
        return cls("table", _canonicalise_table(name), "")

    @classmethod
    def column_el(cls, table: str, column: str) -> "Element":
        """Build a column element, canonicalising both halves."""
        t, c = _canonicalise_column(table, column)
        return cls("column", t, c)

    def render(self) -> str:
        """Human-readable form: ``singer`` or ``singer.singer_id``."""
        return self.table if self.level == "table" else f"{self.table}.{self.column}"


@dataclass(frozen=True, slots=True)
class ErrorInstance:
    """One erroneous link — the unit of analysis.

    Attributes
    ----------
    question_id
        Spider dev index, matching ``main_per_query.csv``.
    db_id
        Spider database identifier.
    method
        One of the six method names.
    tier
        ``"tier1"`` or ``"tier2"``.
    element
        The schema element that was missed, spurious, or hallucinated.
    shape
        Axis-1 code.
    """

    question_id: int
    db_id: str
    method: str
    tier: str
    element: Element
    shape: Shape
