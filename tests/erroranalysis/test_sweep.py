"""Threshold sweep mechanics."""

from __future__ import annotations

import pandas as pd

from schema_linking.erroranalysis.census import enumerate_errors, errors_to_frame
from schema_linking.erroranalysis.facts import build_case_facts
from schema_linking.erroranalysis.rules import CascadeContext
from schema_linking.erroranalysis.sweep import (
    LEXICAL_GRID,
    SEMANTIC_GRID,
    stability_summary,
    sweep_thresholds,
)
from schema_linking.erroranalysis.taxonomy import Element
from schema_linking.utils.config import ErrorAnalysisConfig


def _setup(mini_schema):
    el = Element.column_el("singer", "Country")
    facts = build_case_facts(
        question_id=1,
        question="Which country is the artist from?",
        gold_sql="SELECT Country FROM singer",
        schema=mini_schema,
        gold_tier1_raw={"tables": ["singer"], "columns": [["singer", "Country"]]},
        gold_tier2_raw={"tables": ["singer"], "columns": [["singer", "Country"]]},
        predicted_raw={"tables": ["singer"], "columns": []},
        hardness="easy",
        lexical_scores={el: 75},
        semantic_scores={el: 0.6},
    )
    errors = enumerate_errors(facts, "lexical", "tier1")
    frame = errors_to_frame(errors, {1: facts})
    ctx = CascadeContext(
        cfg=ErrorAnalysisConfig(), missed_by_count={}, gold_sql_elements={}
    )
    return frame, {"lexical": {1: facts}}, ctx


def test_sweep_has_one_row_per_cell_per_cause(mini_schema):
    frame, facts, ctx = _setup(mini_schema)
    sweep = sweep_thresholds(frame, facts, ctx, (60, 80), (0.5, 0.7))
    assert set(sweep.columns) == {
        "lexical_threshold",
        "semantic_threshold",
        "cause",
        "n",
        "share",
    }
    assert sweep.groupby(["lexical_threshold", "semantic_threshold"]).ngroups == 4


def test_lowering_the_lexical_threshold_moves_the_cause(mini_schema):
    """At lexical 60 the element is anchored (UNFORCED); at 80 it is not."""
    frame, facts, ctx = _setup(mini_schema)
    sweep = sweep_thresholds(frame, facts, ctx, (60, 80), (0.5,))
    at_60 = set(sweep[sweep.lexical_threshold == 60].cause)
    at_80 = set(sweep[sweep.lexical_threshold == 80].cause)
    assert at_60 == {"UNFORCED"}
    assert at_80 == {"PARAPHRASE"}


def test_shares_sum_to_one_per_cell(mini_schema):
    frame, facts, ctx = _setup(mini_schema)
    sweep = sweep_thresholds(frame, facts, ctx, LEXICAL_GRID, SEMANTIC_GRID)
    totals = sweep.groupby(["lexical_threshold", "semantic_threshold"])["share"].sum()
    assert (totals.round(9) == 1.0).all()


def test_stability_summary_reports_range_per_cause(mini_schema):
    frame, facts, ctx = _setup(mini_schema)
    sweep = sweep_thresholds(frame, facts, ctx, (60, 80), (0.5, 0.7))
    summary = stability_summary(sweep)
    assert set(summary.columns) == {
        "cause",
        "min_share",
        "max_share",
        "range",
        "n_cells_present",
    }
    assert (summary["range"] >= 0).all()


def test_stability_summary_zero_fills_cells_where_a_cause_is_absent():
    """A cause present in one cell of several must not read as stable.

    ``sweep_thresholds`` never emits a row for ``(cell, cause)`` when the
    cause has zero rows in that cell (``value_counts`` omits zero counts).
    ``RARE`` occurs in only 1 of the 3 cells below, with a large share
    there; its true minimum share across the grid is 0.0, not the smallest
    *observed* share. Before the zero-fill fix, ``stability_summary``
    computed ``min_share``/``max_share`` only over the rows that exist,
    so a cause present in a single cell reported ``range == 0.0`` — the
    most "stable" possible value — which is the opposite of the truth.
    """
    sweep = pd.DataFrame(
        [
            {
                "lexical_threshold": 50,
                "semantic_threshold": 0.5,
                "cause": "RARE",
                "n": 9,
                "share": 0.9,
            },
            {
                "lexical_threshold": 50,
                "semantic_threshold": 0.5,
                "cause": "COMMON",
                "n": 1,
                "share": 0.1,
            },
            {
                "lexical_threshold": 60,
                "semantic_threshold": 0.5,
                "cause": "COMMON",
                "n": 10,
                "share": 1.0,
            },
            {
                "lexical_threshold": 70,
                "semantic_threshold": 0.5,
                "cause": "COMMON",
                "n": 10,
                "share": 1.0,
            },
        ]
    )
    summary = stability_summary(sweep)
    rare = summary[summary.cause == "RARE"].iloc[0]
    assert rare.n_cells_present == 1
    assert rare.min_share == 0.0
    assert rare.max_share == 0.9
    assert rare.range == 0.9


def test_stability_summary_min_share_is_zero_whenever_a_cause_is_absent_somewhere(
    mini_schema,
):
    """Invariant: absent from any cell implies a true minimum of 0.0.

    Over the full documented grid, this fixture's single error row is
    coded as exactly one cause per cell (25 cells total), so every cause
    that appears is necessarily missing from at least one other cell —
    none can have ``n_cells_present == 25``. Each must therefore report
    ``min_share == 0.0``.
    """
    frame, facts, ctx = _setup(mini_schema)
    sweep = sweep_thresholds(frame, facts, ctx, LEXICAL_GRID, SEMANTIC_GRID)
    summary = stability_summary(sweep)
    total_cells = len(LEXICAL_GRID) * len(SEMANTIC_GRID)
    partial = summary[summary["n_cells_present"] < total_cells]
    assert not partial.empty
    assert (partial["min_share"] == 0.0).all()


def test_grids_are_the_documented_ones():
    assert LEXICAL_GRID == (50, 60, 70, 80, 90)
    assert SEMANTIC_GRID == (0.35, 0.45, 0.55, 0.65, 0.75)
