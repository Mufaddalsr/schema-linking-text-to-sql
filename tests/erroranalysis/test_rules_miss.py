"""The MISS half of the cascade — the anchoring decision table."""

from __future__ import annotations

import pytest

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
        cfg=ErrorAnalysisConfig(lexical_threshold=70, semantic_threshold=0.55),
        missed_by_count={},
        gold_sql_elements={},
    )


def _case(mini_schema, *, gold, pred, lex, sem, question="q", sql="SELECT 1"):
    """A case where ``gold`` is identical on both tiers, so JOIN-ONLY cannot fire."""
    return build_case_facts(
        question_id=1,
        question=question,
        gold_sql=sql,
        schema=mini_schema,
        gold_tier1_raw=gold,
        gold_tier2_raw=gold,
        predicted_raw=pred,
        hardness="easy",
        lexical_scores=lex,
        semantic_scores=sem,
    )


def _miss(facts, element):
    return ErrorInstance(
        question_id=1,
        db_id=facts.db_id,
        method="embedding",
        tier="tier1",
        element=element,
        shape=Shape.MISS,
    )


def test_star_column_is_implicit_agg(mini_schema):
    el = Element.column_el("singer", "*")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "*"]]},
        pred={"tables": ["singer"], "columns": []},
        lex={},
        sem={},
    )
    assert classify(_miss(facts, el), facts, _ctx()).cause is Cause.IMPLICIT_AGG


def test_lexically_available_with_competitor_is_ambig_lost(mini_schema):
    """Question says 'name'; the method predicted concert.Name, gold is singer.Name."""
    gold_el = Element.column_el("singer", "Name")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Name"]]},
        pred={"tables": ["concert"], "columns": [["concert", "Name"]]},
        lex={gold_el: 100},
        sem={gold_el: 0.9},
        question="What is the name?",
    )
    verdict = classify(_miss(facts, gold_el), facts, _ctx())
    assert verdict.cause is Cause.AMBIG_LOST
    assert verdict.evidence["predicted_instead"] == "concert.name"


def test_lexically_available_without_competitor_is_unforced(mini_schema):
    gold_el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Country"]]},
        pred={"tables": ["singer"], "columns": []},
        lex={gold_el: 95},
        sem={gold_el: 0.9},
        question="Which country?",
    )
    verdict = classify(_miss(facts, gold_el), facts, _ctx())
    assert verdict.cause is Cause.UNFORCED
    assert verdict.evidence["lexical"] == 95


def test_semantic_only_is_paraphrase(mini_schema):
    gold_el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Country"]]},
        pred={"tables": ["singer"], "columns": []},
        lex={gold_el: 20},
        sem={gold_el: 0.8},
        question="Where is the artist from?",
    )
    verdict = classify(_miss(facts, gold_el), facts, _ctx())
    assert verdict.cause is Cause.PARAPHRASE
    assert verdict.evidence["semantic"] == pytest.approx(0.8)


def test_neither_anchor_is_unverbalised(mini_schema):
    gold_el = Element.column_el("singer", "Singer_ID")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Singer_ID"]]},
        pred={"tables": ["singer"], "columns": []},
        lex={gold_el: 15},
        sem={gold_el: 0.1},
        question="How many are there?",
    )
    assert classify(_miss(facts, gold_el), facts, _ctx()).cause is Cause.UNVERBALISED


def test_missing_score_is_treated_as_no_anchor(mini_schema):
    """A gold element absent from both score maps must not crash the cascade."""
    gold_el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Country"]]},
        pred={"tables": ["singer"], "columns": []},
        lex={},
        sem={},
    )
    assert classify(_miss(facts, gold_el), facts, _ctx()).cause is Cause.UNVERBALISED


def test_threshold_is_inclusive(mini_schema):
    """A score exactly at the threshold counts as anchored."""
    gold_el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Country"]]},
        pred={"tables": ["singer"], "columns": []},
        lex={gold_el: 70},
        sem={gold_el: 0.0},
    )
    assert classify(_miss(facts, gold_el), facts, _ctx()).cause is Cause.UNFORCED


def test_ambig_lost_requires_a_same_name_prediction_not_just_any(mini_schema):
    """Predicting an unrelated column must not count as the competitor."""
    gold_el = Element.column_el("singer", "Country")
    facts = _case(
        mini_schema,
        gold={"tables": ["singer"], "columns": [["singer", "Country"]]},
        pred={"tables": ["concert"], "columns": [["concert", "Concert_ID"]]},
        lex={gold_el: 95},
        sem={gold_el: 0.9},
    )
    assert classify(_miss(facts, gold_el), facts, _ctx()).cause is Cause.UNFORCED


def test_no_miss_can_remain_unresolved(mini_schema):
    """The decision table is exhaustive over MISS."""
    gold_el = Element.column_el("singer", "Country")
    for lex_score, sem_score in [(0, 0.0), (100, 0.0), (0, 1.0), (100, 1.0)]:
        facts = _case(
            mini_schema,
            gold={"tables": ["singer"], "columns": [["singer", "Country"]]},
            pred={"tables": ["singer"], "columns": []},
            lex={gold_el: lex_score},
            sem={gold_el: sem_score},
        )
        verdict = classify(_miss(facts, gold_el), facts, _ctx())
        assert verdict.cause is not Cause.UNRESOLVED, (lex_score, sem_score)
