"""The SPUR half of the cascade."""

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


def _ctx(**kw):
    defaults = dict(
        cfg=ErrorAnalysisConfig(lexical_threshold=70, semantic_threshold=0.55),
        missed_by_count={},
        gold_sql_elements={},
        all_schema_tables={},
        all_schema_columns={},
    )
    defaults.update(kw)
    return CascadeContext(**defaults)


def _case(mini_schema, *, gold, pred, lex=None, question="q"):
    return build_case_facts(
        question_id=1,
        question=question,
        gold_sql="SELECT 1",
        schema=mini_schema,
        gold_tier1_raw=gold,
        gold_tier2_raw=gold,
        predicted_raw=pred,
        hardness="easy",
        lexical_scores=lex or {},
    )


def _spur(facts, element):
    return ErrorInstance(
        question_id=1,
        db_id=facts.db_id,
        method="embedding",
        tier="tier1",
        element=element,
        shape=Shape.SPUR,
    )


def test_same_name_different_table_is_a_name_collision(mini_schema):
    el = Element.column_el("concert", "Name")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Name"]]},
        pred={"tables": ["singer"], "columns": [["singer", "Name"], ["concert", "Name"]]},
    )
    verdict = classify(_spur(facts, el), facts, _ctx())
    assert verdict.cause is Cause.NAME_COLLISION
    assert verdict.evidence["collides_with"] == "singer.name"


def test_name_collision_outranks_sibling(mini_schema):
    """concert.Name is both a name collision and a column of a gold table.

    NAME-COLLISION is the more specific condition and must win.
    """
    el = Element.column_el("concert", "Name")
    facts = _case(
        mini_schema,
        gold={
            "tables": ["singer", "concert"],
            "columns": [["singer", "Name"]],
        },
        pred={
            "tables": ["singer", "concert"],
            "columns": [["singer", "Name"], ["concert", "Name"]],
        },
    )
    assert classify(_spur(facts, el), facts, _ctx()).cause is Cause.NAME_COLLISION


def test_lexically_matching_non_gold_element_is_question_anchored(mini_schema):
    """The question says 'country' but the gold does not need it."""
    el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold={"tables": [], "columns": []},
        pred={"tables": [], "columns": [["singer", "Country"]]},
        lex={el: 100},
        question="List singers by country",
    )
    verdict = classify(_spur(facts, el), facts, _ctx())
    assert verdict.cause is Cause.QUESTION_ANCHORED
    assert verdict.evidence["lexical"] == 100


def test_unrelated_unanchored_element_is_unanchored(mini_schema):
    el = Element.column_el("concert", "Concert_ID")
    facts = _case(
        mini_schema,
        gold={"tables": [], "columns": []},
        pred={"tables": [], "columns": [["concert", "Concert_ID"]]},
        lex={el: 10},
        question="How many singers are there?",
    )
    assert classify(_spur(facts, el), facts, _ctx()).cause is Cause.UNANCHORED


def test_no_spur_can_remain_unresolved(mini_schema):
    """UNANCHORED is terminal for SPUR."""
    el = Element.column_el("concert", "Concert_ID")
    for score in (0, 69, 70, 100):
        facts = _case(
            mini_schema,
            gold={"tables": [], "columns": []},
            pred={"tables": [], "columns": [["concert", "Concert_ID"]]},
            lex={el: score},
        )
        assert classify(_spur(facts, el), facts, _ctx()).cause is not Cause.UNRESOLVED
