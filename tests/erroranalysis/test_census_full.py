"""The complete cascade over the real corpus, with real embeddings."""

from __future__ import annotations

import pandas as pd
import pytest

from schema_linking.erroranalysis.census import METHODS, build_census, build_facts
from schema_linking.erroranalysis.loading import load_corpus
from schema_linking.erroranalysis.rules import (
    build_context,
    classify_census,
    coverage_report,
)
from schema_linking.erroranalysis.scoring import EmbeddingSemanticScorer
from schema_linking.erroranalysis.taxonomy import Cause
from schema_linking.utils.config import load_config
from schema_linking.utils.embeddings import SchemaEncoder

pytestmark = pytest.mark.integration

CENSUS_TOTAL = 24130
"""Rows in the full classified census (controller ruling R14).

The brief's original figure, 24,086, omitted the 44 hallucinated elements:
``outputs/results/main_per_query.csv`` excludes hallucinations from its FP
columns and reports them separately, so the plan's count under-totalled.
The true census size, independently proven against that CSV by
``test_census_consistency.py::test_grand_total_is_24130``, is 24,130.
"""

NEVER_FIRING_CAUSES: frozenset[Cause] = frozenset(
    {
        # Zero `*` columns exist in either gold tier, and zero appear
        # across all six prediction files (verified directly over the
        # corpus). IMPLICIT-AGG's implemented predicate (`aggregate_only`)
        # only matches the `*` selector — the aggregate-without-mention
        # clause was never implemented (ruling R16; see
        # taxonomy.CAUSE_DEFINITIONS[Cause.IMPLICIT_AGG]) — so with no `*`
        # columns present, the rule has nothing to match.
        Cause.IMPLICIT_AGG,
        # All six methods emitted structurally valid predictions: no empty
        # or half-empty identifiers anywhere in the corpus, so MALFORMED's
        # predicate is never satisfied.
        Cause.MALFORMED,
        # All 44 hallucinations are invented names that match no table or
        # column in any other Spider database (verified directly), so
        # WRONG-DB never outranks FABRICATED.
        Cause.WRONG_DB,
    }
)
"""Causes allowed to have zero rows in the real census (controller ruling R15).

Each is a genuine property of the dataset, not a cascade defect — see the
per-cause comments above for the supporting evidence. Extending this set
must be a deliberate, evidenced act: any *other* cause with zero observed
rows should still fail ``test_every_cause_in_the_taxonomy_is_observed``,
which is precisely the signal that test exists to preserve. See
``test_allowlisted_causes_really_do_have_zero_rows`` for the companion
check that keeps this allowlist honest.
"""


@pytest.fixture(scope="module")
def coded():
    cfg = load_config()
    corpus = load_corpus("dev")
    encoder = SchemaEncoder(
        model_name=cfg.embedding.model_name,
        revision=cfg.embedding.revision,
        cache_dir=cfg.embedding.cache_dir,
    )
    scorer = EmbeddingSemanticScorer(
        encoder, corpus.schemas, cfg.embedding.cache_dir / "questions"
    )
    census = build_census(corpus, scorer, cfg)
    ctx = build_context(corpus, census, cfg.error_analysis)
    facts_by_method = {m: build_facts(corpus, m, scorer, cfg) for m in METHODS}
    return classify_census(census, facts_by_method, ctx)


def test_residual_is_empty(coded):
    """Every terminal rule fires unconditionally, so nothing may be UNRESOLVED.

    A nonzero count means a shape branch is missing its terminal rule.
    """
    residual = coded[coded.cause == "UNRESOLVED"]
    assert residual.empty, residual.head(20).to_string()


def test_grand_total_is_24130(coded):
    """Renamed from the brief's ``test_all_24086_rows_are_coded`` (ruling R14):

    the true census size is 24,130, not the plan's 24,086, so a test named
    after the stale figure would lie about what it asserts.
    """
    assert len(coded) == CENSUS_TOTAL
    assert (coded.cause != "").all()


def test_every_cause_in_the_taxonomy_is_observed(coded):
    """A cause that never fires on 24k real errors is a dead rule — investigate.

    Three causes are allowlisted (controller ruling R15) as genuine dataset
    properties rather than cascade defects; see ``NEVER_FIRING_CAUSES`` for
    the evidence recorded against each. Any *other* never-firing cause must
    still fail this test.
    """
    observed = set(coded.cause.unique())
    expected = {str(c) for c in Cause} - {"UNRESOLVED"}
    expected -= {str(c) for c in NEVER_FIRING_CAUSES}
    assert expected - observed == set(), f"never fired: {expected - observed}"


def test_allowlisted_causes_really_do_have_zero_rows(coded):
    """Keeps ``NEVER_FIRING_CAUSES`` honest.

    If one of the three allowlisted causes starts firing (e.g. a future
    corpus update introduces a `*` column, a malformed prediction, or a
    hallucination that collides with another database), this test fails
    first — flagging the allowlist itself as stale rather than letting
    ``test_every_cause_in_the_taxonomy_is_observed`` silently stay
    over-permissive.
    """
    for cause in NEVER_FIRING_CAUSES:
        n = int((coded.cause == str(cause)).sum())
        assert n == 0, (
            f"{cause} fired {n} times but is allowlisted as never-firing in "
            "NEVER_FIRING_CAUSES — remove it from the allowlist and let "
            "test_every_cause_in_the_taxonomy_is_observed cover it instead"
        )


def test_incidence_matches_the_design_measurement():
    """Design §1.4, measured during design against the real files."""
    from schema_linking.erroranalysis.incidence import build_incidence

    corpus = load_corpus("dev")
    inc = build_incidence(corpus, tier="tier1")
    columns = inc[inc.level == "column"]
    assert len(columns) == 1767
    assert int((columns.n_found == 0).sum()) == 18
    assert int((columns.n_found == 1).sum()) == 23


def test_print_the_final_coverage(coded, capsys):
    with capsys.disabled():
        print("\n=== full cascade coverage ===")
        print(coverage_report(coded).to_string(index=False))
        print("\n=== cause by method (tier1) ===")
        t1 = coded[coded.tier == "tier1"]
        print(pd.crosstab(t1.method, t1.cause).to_string())
