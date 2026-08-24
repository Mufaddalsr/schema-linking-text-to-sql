"""Applying the cascade across a whole census frame."""

from __future__ import annotations

import pandas as pd

from schema_linking.erroranalysis.census import (
    CENSUS_COLUMNS,
    enumerate_errors,
    errors_to_frame,
)
from schema_linking.erroranalysis.facts import build_case_facts
from schema_linking.erroranalysis.rules import (
    CascadeContext,
    classify_census,
    coverage_report,
)
from schema_linking.utils.config import ErrorAnalysisConfig


def _facts(mini_schema):
    return build_case_facts(
        question_id=1,
        question="Show the names of singers at each concert",
        gold_sql="SELECT Name FROM singer JOIN concert ON singer.Singer_ID = concert.Singer_ID",
        schema=mini_schema,
        gold_tier1_raw={"tables": ["singer"], "columns": [["singer", "Name"]]},
        gold_tier2_raw={
            "tables": ["singer", "concert"],
            "columns": [["singer", "Name"], ["concert", "Singer_ID"]],
        },
        predicted_raw={
            "tables": ["singer"],
            "columns": [["singer", "Name"], ["singer", "Country"]],
        },
        hardness="medium",
    )


def _ctx():
    return CascadeContext(
        cfg=ErrorAnalysisConfig(), missed_by_count={}, gold_sql_elements={}
    )


def _census(mini_schema):
    facts = _facts(mini_schema)
    errors = [
        e
        for tier in ("tier1", "tier2")
        for e in enumerate_errors(facts, "lexical", tier)
    ]
    frame = errors_to_frame(errors, {1: facts})
    return frame, {"lexical": {1: facts}}


def test_classify_census_preserves_row_count_and_columns(mini_schema):
    frame, facts_by_method = _census(mini_schema)
    coded = classify_census(frame, facts_by_method, _ctx())
    assert len(coded) == len(frame)
    assert list(coded.columns) == list(CENSUS_COLUMNS)


def test_every_row_gets_a_cause(mini_schema):
    frame, facts_by_method = _census(mini_schema)
    coded = classify_census(frame, facts_by_method, _ctx())
    assert (coded.cause != "").all()


def test_sibling_and_join_only_are_assigned(mini_schema):
    frame, facts_by_method = _census(mini_schema)
    coded = classify_census(frame, facts_by_method, _ctx())
    by_element = dict(zip(coded.element, coded.cause, strict=True))
    assert by_element["singer.country"] == "SIBLING"
    assert by_element["concert"] == "JOIN-ONLY"


def test_evidence_is_serialised_as_a_readable_string(mini_schema):
    frame, facts_by_method = _census(mini_schema)
    coded = classify_census(frame, facts_by_method, _ctx())
    row = coded[coded.element == "singer.country"].iloc[0]
    assert isinstance(row.evidence, str)
    assert "column_of_gold_table" in row.evidence


def test_coverage_report_shares_sum_to_one(mini_schema):
    frame, facts_by_method = _census(mini_schema)
    coded = classify_census(frame, facts_by_method, _ctx())
    report = coverage_report(coded)
    assert report["share"].sum() == 1.0
    assert set(report.columns) == {"shape_code", "cause", "n", "share"}


def test_coverage_report_is_empty_frame_safe():
    empty = pd.DataFrame(columns=list(CENSUS_COLUMNS))
    report = coverage_report(empty)
    assert report.empty
