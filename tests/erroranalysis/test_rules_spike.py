"""The three calibration-spike rules: TIER-ARTEFACT, JOIN-ONLY, SIBLING."""

from __future__ import annotations

from schema_linking.erroranalysis.facts import build_case_facts
from schema_linking.erroranalysis.rules import CascadeContext, classify
from schema_linking.erroranalysis.taxonomy import (
    Cause,
    Element,
    ErrorInstance,
    Shape,
)
from schema_linking.utils.config import ErrorAnalysisConfig


def _ctx():
    return CascadeContext(
        cfg=ErrorAnalysisConfig(), missed_by_count={}, gold_sql_elements={}
    )


def _case(mini_schema, *, gold1, gold2, pred):
    return build_case_facts(
        question_id=1,
        question="Show the names of singers at each concert",
        gold_sql="SELECT s.Name FROM singer s JOIN concert c ON s.Singer_ID = c.Singer_ID",
        schema=mini_schema,
        gold_tier1_raw=gold1,
        gold_tier2_raw=gold2,
        predicted_raw=pred,
        hardness="medium",
    )


def _err(facts, element, shape, tier="tier1"):
    return ErrorInstance(
        question_id=1,
        db_id=facts.db_id,
        method="lexical",
        tier=tier,
        element=element,
        shape=shape,
    )


def test_spurious_element_that_is_tier2_gold_is_a_tier_artefact(mini_schema):
    """The 17.2% case: a correctly-found join column penalised on Tier-1."""
    el = Element.column_el("concert", "Singer_ID")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": [["singer", "Name"]]},
        gold2={
            "tables": ["singer", "concert"],
            "columns": [["singer", "Name"], ["concert", "Singer_ID"]],
        },
        pred={"tables": ["singer"], "columns": [["singer", "Name"], ["concert", "Singer_ID"]]},
    )
    verdict = classify(_err(facts, el, Shape.SPUR), facts, _ctx())
    assert verdict.cause is Cause.TIER_ARTEFACT
    assert verdict.evidence["gold_in"] == "tier2"


def test_tier_artefact_also_fires_the_other_way(mini_schema):
    """A Tier-1-only element predicted and scored against Tier-2."""
    el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": [["singer", "Country"]]},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer"], "columns": [["singer", "Country"]]},
    )
    verdict = classify(_err(facts, el, Shape.SPUR, tier="tier2"), facts, _ctx())
    assert verdict.cause is Cause.TIER_ARTEFACT
    assert verdict.evidence["gold_in"] == "tier1"


def test_missed_tier2_only_element_is_join_only(mini_schema):
    el = Element.table_el("concert")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer", "concert"], "columns": []},
        pred={"tables": ["singer"], "columns": []},
    )
    verdict = classify(_err(facts, el, Shape.MISS, tier="tier2"), facts, _ctx())
    assert verdict.cause is Cause.JOIN_ONLY


def test_join_only_never_fires_on_tier1(mini_schema):
    """Tier-1 has no join-only elements by construction."""
    el = Element.table_el("concert")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer", "concert"], "columns": []},
        gold2={"tables": ["singer", "concert"], "columns": []},
        pred={"tables": ["singer"], "columns": []},
    )
    assert classify(_err(facts, el, Shape.MISS), facts, _ctx()).cause is not Cause.JOIN_ONLY


def test_spurious_column_of_a_gold_table_is_a_sibling(mini_schema):
    el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": [["singer", "Singer_ID"]]},
        gold2={"tables": ["singer"], "columns": [["singer", "Singer_ID"]]},
        pred={"tables": ["singer"], "columns": [["singer", "Singer_ID"], ["singer", "Country"]]},
    )
    verdict = classify(_err(facts, el, Shape.SPUR), facts, _ctx())
    assert verdict.cause is Cause.SIBLING
    assert verdict.evidence["relation"] == "column_of_gold_table"


def test_spurious_fk_adjacent_table_is_a_sibling(mini_schema):
    el = Element.table_el("concert")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer", "concert"], "columns": []},
    )
    verdict = classify(_err(facts, el, Shape.SPUR), facts, _ctx())
    assert verdict.cause is Cause.SIBLING
    assert verdict.evidence["relation"] == "fk_adjacent_to_gold_table"


def test_unrelated_spurious_element_is_unresolved_in_the_spike(mini_schema):
    """Task 10 will code this UNANCHORED; the spike must leave it UNRESOLVED."""
    el = Element.column_el("concert", "Concert_ID")
    facts = _case(
        mini_schema,
        gold1={"tables": [], "columns": []},
        gold2={"tables": [], "columns": []},
        pred={"tables": [], "columns": [["concert", "Concert_ID"]]},
    )
    verdict = classify(_err(facts, el, Shape.SPUR), facts, _ctx())
    assert verdict.cause is Cause.UNRESOLVED
    assert verdict.rule_name == ""
