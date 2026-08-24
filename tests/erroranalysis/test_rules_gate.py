"""The GOLD-DEFECT gate outranks every attribution rule."""

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


def _ctx(missed_by_count=None, gold_sql_elements=None, **cfg_kw):
    return CascadeContext(
        cfg=ErrorAnalysisConfig(**cfg_kw),
        missed_by_count=missed_by_count or {},
        gold_sql_elements=gold_sql_elements or {},
    )


def _case(mini_schema, *, gold1, gold2, pred, question="How many singers?", sql="SELECT count(*) FROM singer"):
    return build_case_facts(
        question_id=1,
        question=question,
        gold_sql=sql,
        schema=mini_schema,
        gold_tier1_raw=gold1,
        gold_tier2_raw=gold2,
        predicted_raw=pred,
        hardness="easy",
    )


def _miss(facts, element, tier="tier1"):
    return ErrorInstance(
        question_id=facts.question_id,
        db_id=facts.db_id,
        method="lexical",
        tier=tier,
        element=element,
        shape=Shape.MISS,
    )


def test_gold_element_absent_from_schema_is_a_gold_defect(mini_schema):
    facts = _case(
        mini_schema,
        gold1={"tables": ["orchestra"], "columns": []},
        gold2={"tables": ["orchestra"], "columns": []},
        pred={"tables": ["singer"], "columns": []},
    )
    verdict = classify(_miss(facts, Element.table_el("orchestra")), facts, _ctx())
    assert verdict.cause is Cause.GOLD_DEFECT
    assert verdict.rule_name == "gold_element_not_in_schema"


def test_tier1_gold_absent_from_gold_sql_is_a_gold_defect(mini_schema):
    """Tier-1 claims the question mentions concert.Name, but the SQL never uses it."""
    el = Element.column_el("concert", "Name")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": [["concert", "Name"]]},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer"], "columns": []},
    )
    ctx = _ctx(gold_sql_elements={1: frozenset({Element.table_el("singer")})})
    verdict = classify(_miss(facts, el), facts, ctx)
    assert verdict.cause is Cause.GOLD_DEFECT
    assert verdict.rule_name == "tier1_gold_absent_from_sql"


def test_the_sql_clause_does_not_fire_on_tier2(mini_schema):
    """Tier-2 is derived FROM the SQL, so the clause is meaningless there."""
    el = Element.column_el("concert", "Name")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": [["concert", "Name"]]},
        pred={"tables": ["singer"], "columns": []},
    )
    ctx = _ctx(gold_sql_elements={1: frozenset({Element.table_el("singer")})})
    verdict = classify(_miss(facts, el, tier="tier2"), facts, ctx)
    assert verdict.cause is not Cause.GOLD_DEFECT


def test_missed_by_five_of_six_methods_is_flagged(mini_schema):
    el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": [["singer", "Country"]]},
        gold2={"tables": ["singer"], "columns": [["singer", "Country"]]},
        pred={"tables": ["singer"], "columns": []},
    )
    ctx = _ctx(
        missed_by_count={("tier1", 1, el): 5},
        gold_sql_elements={1: frozenset({el, Element.table_el("singer")})},
    )
    verdict = classify(_miss(facts, el), facts, ctx)
    assert verdict.cause is Cause.GOLD_DEFECT
    assert verdict.rule_name == "missed_by_most_methods"
    assert verdict.evidence["confirmed"] == "pending"


def test_missed_by_four_of_six_is_not_flagged(mini_schema):
    el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": [["singer", "Country"]]},
        gold2={"tables": ["singer"], "columns": [["singer", "Country"]]},
        pred={"tables": ["singer"], "columns": []},
    )
    ctx = _ctx(
        missed_by_count={("tier1", 1, el): 4},
        gold_sql_elements={1: frozenset({el, Element.table_el("singer")})},
    )
    assert classify(_miss(facts, el), facts, ctx).cause is not Cause.GOLD_DEFECT


def test_gate_does_not_apply_to_spurious_predictions(mini_schema):
    """A spurious element absent from the schema is HALL, never GOLD-DEFECT."""
    facts = _case(
        mini_schema,
        gold1={"tables": ["singer"], "columns": []},
        gold2={"tables": ["singer"], "columns": []},
        pred={"tables": ["singer"], "columns": [["singer", "Country"]]},
    )
    err = ErrorInstance(
        question_id=1,
        db_id="concert_singer",
        method="lexical",
        tier="tier1",
        element=Element.column_el("singer", "Country"),
        shape=Shape.SPUR,
    )
    assert classify(err, facts, _ctx()).cause is not Cause.GOLD_DEFECT
