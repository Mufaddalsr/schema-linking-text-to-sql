"""Corpus loading against the real dev split."""

from __future__ import annotations

import pytest

from schema_linking.erroranalysis.census import METHODS
from schema_linking.erroranalysis.loading import load_corpus

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def corpus():
    return load_corpus("dev")


def test_loads_all_dev_questions(corpus):
    assert len(corpus.examples) == 1034


def test_loads_both_gold_tiers_for_every_question(corpus):
    qids = {e.question_id for e in corpus.examples}
    assert set(corpus.gold_tier1) == qids
    assert set(corpus.gold_tier2) == qids


def test_loads_all_six_prediction_files(corpus):
    assert set(corpus.predictions) == set(METHODS)
    for method in METHODS:
        assert len(corpus.predictions[method]) == 1034, method


def test_every_question_has_a_hardness_label(corpus):
    assert set(corpus.hardness) == {e.question_id for e in corpus.examples}
    assert set(corpus.hardness.values()) <= {"easy", "medium", "hard", "extra"}


def test_schema_index_exists_for_every_db(corpus):
    assert {e.db_id for e in corpus.examples} <= set(corpus.indices)


def test_prediction_db_ids_agree_with_spider(corpus):
    """A prediction file whose db_id disagrees would silently corrupt the census."""
    by_qid = {e.question_id: e.db_id for e in corpus.examples}
    for method, preds in corpus.predictions.items():
        mismatched = [q for q, p in preds.items() if p["db_id"] != by_qid[q]]
        assert mismatched == [], f"{method}: {mismatched[:5]}"
