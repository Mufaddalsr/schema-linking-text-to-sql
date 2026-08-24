"""Hallucination causes: MALFORMED, WRONG-DB, FABRICATED."""

from __future__ import annotations

from schema_linking.erroranalysis.facts import SchemaIndex, build_case_facts
from schema_linking.erroranalysis.rules import CascadeContext, classify
from schema_linking.erroranalysis.taxonomy import (
    Cause,
    Element,
    ErrorInstance,
    Shape,
)
from schema_linking.utils.config import ErrorAnalysisConfig


def _ctx(all_schemas):
    indices = {db: SchemaIndex.build(s) for db, s in all_schemas.items()}
    return CascadeContext(
        cfg=ErrorAnalysisConfig(),
        missed_by_count={},
        gold_sql_elements={},
        all_schema_tables={db: i.tables for db, i in indices.items()},
        all_schema_columns={db: i.columns for db, i in indices.items()},
    )


def _case(mini_schema, pred):
    return build_case_facts(
        question_id=1,
        question="q",
        gold_sql="SELECT 1",
        schema=mini_schema,
        gold_tier1_raw={"tables": ["singer"], "columns": []},
        gold_tier2_raw={"tables": ["singer"], "columns": []},
        predicted_raw=pred,
        hardness="easy",
    )


def _hall(facts, element):
    return ErrorInstance(
        question_id=1,
        db_id=facts.db_id,
        method="llm_backward",
        tier="tier1",
        element=element,
        shape=Shape.HALL,
    )


def test_table_from_another_db_is_wrong_db(mini_schema, all_schemas):
    el = Element.table_el("flights")
    facts = _case(mini_schema, {"tables": ["singer", "flights"], "columns": []})
    verdict = classify(_hall(facts, el), facts, _ctx(all_schemas))
    assert verdict.cause is Cause.WRONG_DB
    assert verdict.evidence["found_in"] == "flight_1"


def test_column_from_another_db_is_wrong_db(mini_schema, all_schemas):
    el = Element.column_el("flights", "Origin")
    facts = _case(mini_schema, {"tables": ["singer"], "columns": [["flights", "Origin"]]})
    assert classify(_hall(facts, el), facts, _ctx(all_schemas)).cause is Cause.WRONG_DB


def test_invented_name_is_fabricated(mini_schema, all_schemas):
    el = Element.table_el("orchestra")
    facts = _case(mini_schema, {"tables": ["singer", "orchestra"], "columns": []})
    assert classify(_hall(facts, el), facts, _ctx(all_schemas)).cause is Cause.FABRICATED


def test_empty_name_is_malformed(mini_schema, all_schemas):
    el = Element.table_el("")
    facts = _case(mini_schema, {"tables": ["singer", ""], "columns": []})
    assert classify(_hall(facts, el), facts, _ctx(all_schemas)).cause is Cause.MALFORMED


def test_column_with_empty_half_is_malformed(mini_schema, all_schemas):
    el = Element.column_el("singer", "")
    facts = _case(mini_schema, {"tables": ["singer"], "columns": [["singer", ""]]})
    assert classify(_hall(facts, el), facts, _ctx(all_schemas)).cause is Cause.MALFORMED


def test_no_hall_can_remain_unresolved(mini_schema, all_schemas):
    for el in (
        Element.table_el("orchestra"),
        Element.table_el("flights"),
        Element.table_el(""),
        Element.column_el("nowhere", "nothing"),
    ):
        facts = _case(mini_schema, {"tables": ["singer"], "columns": []})
        assert classify(_hall(facts, el), facts, _ctx(all_schemas)).cause is not Cause.UNRESOLVED
