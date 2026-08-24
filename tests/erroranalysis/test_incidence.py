"""Gold-element incidence across methods."""

from __future__ import annotations

import pandas as pd
import pytest

from schema_linking.erroranalysis.incidence import (
    build_incidence,
    hard_cases,
    method_contrast,
)


class _FakeCorpus:
    """Minimal stand-in for Corpus: two questions, three gold elements."""

    def __init__(self, gold, predictions):
        self.gold_tier1 = gold
        self.gold_tier2 = gold
        self.predictions = predictions
        self.examples = []

    def example_by_qid(self):
        return {}


def _corpus():
    gold = {
        1: {"db_id": "d", "tables": ["singer"], "columns": [["singer", "Name"]]},
        2: {"db_id": "d", "tables": ["concert"], "columns": []},
    }
    predictions = {
        # lexical finds everything
        "lexical": {
            1: {"db_id": "d", "tables": ["singer"], "columns": [["singer", "Name"]]},
            2: {"db_id": "d", "tables": ["concert"], "columns": []},
        },
        # embedding misses singer.Name
        "embedding": {
            1: {"db_id": "d", "tables": ["singer"], "columns": []},
            2: {"db_id": "d", "tables": ["concert"], "columns": []},
        },
        # graph misses concert entirely
        "graph": {
            1: {"db_id": "d", "tables": ["singer"], "columns": [["singer", "Name"]]},
            2: {"db_id": "d", "tables": [], "columns": []},
        },
    }
    return _FakeCorpus(gold, predictions)


@pytest.fixture
def incidence():
    return build_incidence(_corpus(), tier="tier1", methods=("lexical", "embedding", "graph"))


def test_one_row_per_gold_element(incidence):
    assert len(incidence) == 3
    assert set(incidence.element) == {"singer", "singer.name", "concert"}


def test_boolean_column_per_method(incidence):
    row = incidence[incidence.element == "singer.name"].iloc[0]
    assert bool(row.lexical) is True
    assert bool(row.embedding) is False
    assert bool(row.graph) is True


def test_n_found_counts_the_methods(incidence):
    by_el = dict(zip(incidence.element, incidence.n_found, strict=True))
    assert by_el["singer"] == 3
    assert by_el["singer.name"] == 2
    assert by_el["concert"] == 2


def test_hard_cases_returns_only_rare_finds(incidence):
    hard = hard_cases(incidence)
    assert set(hard.element) == set()  # nothing is found by 0 or 1 here


def test_hard_cases_finds_an_element_only_one_method_got():
    corpus = _corpus()
    # make lexical the only finder of singer.Name
    corpus.predictions["graph"][1] = {"db_id": "d", "tables": ["singer"], "columns": []}
    inc = build_incidence(corpus, tier="tier1", methods=("lexical", "embedding", "graph"))
    hard = hard_cases(inc)
    assert set(hard.element) == {"singer.name"}
    assert hard.iloc[0].n_found == 1
    assert hard.iloc[0].found_by == "lexical"


def test_method_contrast_reports_unique_finds(incidence):
    corpus = _corpus()
    corpus.predictions["graph"][1] = {"db_id": "d", "tables": ["singer"], "columns": []}
    inc = build_incidence(corpus, tier="tier1", methods=("lexical", "embedding", "graph"))
    contrast = method_contrast(inc)
    lexical = contrast[contrast.method == "lexical"].iloc[0]
    assert lexical.n_unique_finds == 1
    assert lexical.n_unique_misses == 0


def test_method_contrast_reports_unique_misses(incidence):
    contrast = method_contrast(incidence)
    embedding = contrast[contrast.method == "embedding"].iloc[0]
    assert embedding.n_unique_misses == 1  # only embedding missed singer.name


def test_contrast_covers_every_method(incidence):
    contrast = method_contrast(incidence)
    assert set(contrast.method) == {"lexical", "embedding", "graph"}


def test_method_columns_are_boolean_dtype(incidence):
    # hard_cases and method_contrast both find their method columns by
    # dtype == bool; if construction ever produces object dtype instead,
    # they silently see zero method columns and return empty results.
    for method in ("lexical", "embedding", "graph"):
        assert incidence[method].dtype == bool, incidence[method].dtype
