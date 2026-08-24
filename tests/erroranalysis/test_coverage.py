"""Phase-2 calibration: measure residual size on the real corpus.

These do not assert a target — the residual size is the unknown the
checkpoint exists to discover. They assert only that the measurement runs
and is internally consistent, and they print the report for the reviewer.
"""

from __future__ import annotations

import pytest

from schema_linking.erroranalysis.census import build_census
from schema_linking.erroranalysis.loading import load_corpus
from schema_linking.erroranalysis.rules import (
    build_context,
    classify_census,
    coverage_report,
)
from schema_linking.erroranalysis.scoring import NullSemanticScorer
from schema_linking.utils.config import load_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def coded():
    cfg = load_config()
    corpus = load_corpus("dev")
    census = build_census(corpus, NullSemanticScorer(), cfg)
    ctx = build_context(corpus, census, cfg.error_analysis)
    from schema_linking.erroranalysis.census import METHODS, build_facts

    facts_by_method = {
        m: build_facts(corpus, m, NullSemanticScorer(), cfg) for m in METHODS
    }
    return classify_census(census, facts_by_method, ctx)


def test_full_census_is_classified(coded):
    # 24130 is the census size independently proven by
    # test_census_consistency.py::test_grand_total_is_24130 against
    # main_per_query.csv. The brief hardcoded 24086 here; that number is
    # stale (classify_census is row-count-preserving, so this must equal
    # the census size, not a smaller figure).
    assert len(coded) == 24130
    assert (coded.cause != "").all()


def test_report_the_residual(coded, capsys):
    """Prints the phase-2 numbers. Read them; they gate Tasks 10-12."""
    report = coverage_report(coded)
    with capsys.disabled():
        print("\n=== spike coverage ===")
        print(report.to_string(index=False))
        residual = report[report.cause == "UNRESOLVED"]["share"].sum()
        print(f"\nresidual share: {residual:.1%}")
    assert 0.0 <= report["share"].sum() <= 1.0 + 1e-9


def test_tier_artefact_matches_the_design_measurement(coded):
    """Design §1.4: 543 of lexical's Tier-1 column FPs are Tier-2 gold."""
    n = len(
        coded[
            (coded.method == "lexical")
            & (coded.tier == "tier1")
            & (coded.level == "column")
            & (coded.cause == "TIER-ARTEFACT")
        ]
    )
    assert n == 543
