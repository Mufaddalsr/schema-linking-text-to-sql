"""The ordered cause cascade.

Each rule is a pure predicate over one :class:`ErrorInstance` and its
:class:`CaseFacts`, plus a :class:`CascadeContext` carrying the few facts
that are cross-case (how many methods missed an element; which elements the
gold SQL actually uses). :func:`classify` walks the shape's rule tuple and
returns the first non-``None`` verdict.

Precedence is data, not control flow: it lives in
``taxonomy.CAUSES_FOR_SHAPE`` and is asserted by
``tests/erroranalysis/test_cascade_precedence.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from schema_linking.erroranalysis.facts import CaseFacts
from schema_linking.erroranalysis.taxonomy import (
    Cause,
    Element,
    ErrorInstance,
    Evidence,
    Shape,
)
from schema_linking.utils.config import ErrorAnalysisConfig


@dataclass(frozen=True, slots=True)
class Verdict:
    """A cause assignment plus the facts that justified it.

    Attributes
    ----------
    cause
        The assigned :class:`Cause`.
    rule_name
        Name of the rule function that fired. ``""`` for ``UNRESOLVED``.
    evidence
        Rule-specific facts, serialised into the census's ``evidence``
        column. Keep it small and human-readable — this is what you read
        when adjudicating.
    """

    cause: Cause
    rule_name: str
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class CascadeContext:
    """Cross-case information the cascade needs.

    Attributes
    ----------
    cfg
        Thresholds from ``config.yaml``.
    missed_by_count
        ``{(tier, question_id, element): n_methods_that_missed_it}``. Built
        once over the whole census; drives the gate's third clause.
    gold_sql_elements
        ``{question_id: frozenset[Element]}`` — elements the gold SQL string
        actually references, via ``utils.sql_parsing``. Drives the gate's
        second clause.
    """

    cfg: ErrorAnalysisConfig
    missed_by_count: Mapping[tuple[str, int, Element], int]
    gold_sql_elements: Mapping[int, frozenset[Element]]


class Rule(Protocol):
    """A cascade rule."""

    cause: Cause

    def __call__(
        self, err: ErrorInstance, facts: CaseFacts, ctx: CascadeContext
    ) -> Verdict | None:
        """Return a :class:`Verdict` if the rule fires, else ``None``."""
        ...


def _rule(cause: Cause):
    """Decorator tagging a function as a cascade rule for ``cause``."""

    def wrap(fn):
        fn.cause = cause
        return fn

    return wrap


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@_rule(Cause.GOLD_DEFECT)
def gold_element_not_in_schema(err, facts, ctx):
    """Gold names an element the database does not have."""
    if err.shape is not Shape.MISS:
        return None
    present = (
        err.element.table in facts.index.tables
        if err.element.level == "table"
        else err.element in facts.index.columns
    )
    if present:
        return None
    return Verdict(
        Cause.GOLD_DEFECT,
        "gold_element_not_in_schema",
        {"element": err.element.render(), "db_id": facts.db_id},
    )


@_rule(Cause.GOLD_DEFECT)
def tier1_gold_absent_from_sql(err, facts, ctx):
    """Tier-1 claims the question mentions an element the gold SQL never uses.

    Tier-1 comes from Taniguchi's human annotation and Tier-2 from the SQL,
    so this clause is meaningful on Tier-1 only.
    """
    if err.shape is not Shape.MISS or err.tier != "tier1":
        return None
    sql_elements = ctx.gold_sql_elements.get(err.question_id)
    if sql_elements is None or err.element in sql_elements:
        return None
    return Verdict(
        Cause.GOLD_DEFECT,
        "tier1_gold_absent_from_sql",
        {"element": err.element.render(), "gold_sql": facts.gold_sql[:200]},
    )


@_rule(Cause.GOLD_DEFECT)
def missed_by_most_methods(err, facts, ctx):
    """Nearly every method missed this gold element — suspect the gold.

    Flags only. ``evidence["confirmed"]`` stays ``"pending"`` until manually
    checked against the gold SQL (design §7.3), because this clause uses the
    methods to audit the gold and cannot then be independent evidence about
    them.
    """
    if err.shape is not Shape.MISS:
        return None
    n = ctx.missed_by_count.get((err.tier, err.question_id, err.element), 0)
    if n < ctx.cfg.gold_defect_min_methods:
        return None
    return Verdict(
        Cause.GOLD_DEFECT,
        "missed_by_most_methods",
        {
            "element": err.element.render(),
            "n_methods_missing": n,
            "confirmed": "pending",
        },
    )


# ---------------------------------------------------------------------------
# Calibration-spike rules
# ---------------------------------------------------------------------------


@_rule(Cause.JOIN_ONLY)
def tier2_only_gold(err, facts, ctx):
    """The missed element is Tier-2 gold and not Tier-1 gold."""
    if err.shape is not Shape.MISS or err.tier != "tier2":
        return None
    if err.element in facts.gold_tier1:
        return None
    return Verdict(
        Cause.JOIN_ONLY, "tier2_only_gold", {"element": err.element.render()}
    )


@_rule(Cause.TIER_ARTEFACT)
def gold_in_the_other_tier(err, facts, ctx):
    """The spurious element is gold under the other tier."""
    if err.shape is not Shape.SPUR:
        return None
    other = facts.other_tier(err.tier)
    if err.element not in facts.gold_for(other):
        return None
    return Verdict(
        Cause.TIER_ARTEFACT,
        "gold_in_the_other_tier",
        {"element": err.element.render(), "gold_in": other},
    )


@_rule(Cause.SIBLING)
def adjacent_to_gold(err, facts, ctx):
    """Over-generation in the right neighbourhood.

    A spurious column belonging to a gold table, or a spurious table one
    foreign-key edge from a gold table.
    """
    if err.shape is not Shape.SPUR:
        return None
    gold = facts.gold_for(err.tier)
    gold_tables = {e.table for e in gold}
    if err.element.level == "column" and err.element.table in gold_tables:
        return Verdict(
            Cause.SIBLING,
            "adjacent_to_gold",
            {"element": err.element.render(), "relation": "column_of_gold_table"},
        )
    if err.element.level == "table":
        neighbours = facts.index.fk_adjacent.get(err.element.table, frozenset())
        if neighbours & gold_tables:
            return Verdict(
                Cause.SIBLING,
                "adjacent_to_gold",
                {
                    "element": err.element.render(),
                    "relation": "fk_adjacent_to_gold_table",
                },
            )
    return None


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

# SPIKE_ONLY: Tasks 9-11 extend these tuples to the full cascade.
CASCADE: dict[Shape, tuple[Rule, ...]] = {
    Shape.MISS: (
        gold_element_not_in_schema,
        tier1_gold_absent_from_sql,
        missed_by_most_methods,
        tier2_only_gold,
    ),
    Shape.SPUR: (
        gold_in_the_other_tier,
        adjacent_to_gold,
    ),
    Shape.HALL: (),
}

_UNRESOLVED = Verdict(Cause.UNRESOLVED, "", {})


def classify(
    err: ErrorInstance, facts: CaseFacts, ctx: CascadeContext
) -> Verdict:
    """Assign a cause by walking the cascade for ``err``'s shape.

    Returns
    -------
    Verdict
        The first rule that fires, or an ``UNRESOLVED`` verdict.
    """
    for rule in CASCADE[err.shape]:
        verdict = rule(err, facts, ctx)
        if verdict is not None:
            return verdict
    return _UNRESOLVED
