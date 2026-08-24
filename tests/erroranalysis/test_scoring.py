"""Anchoring scores: lexical via rapidfuzz, semantic via injected scorer."""

from __future__ import annotations

import pytest

from schema_linking.erroranalysis.scoring import (
    NullSemanticScorer,
    element_text,
    lexical_scores,
)
from schema_linking.erroranalysis.taxonomy import Element


def test_element_text_for_table_uses_readable_name(mini_schema):
    assert element_text(Element.table_el("singer"), mini_schema) == "singer"


def test_element_text_for_column_matches_encoder_convention(mini_schema):
    """Must match utils/embeddings.py: '<column.name> of <table.name>'."""
    el = Element.column_el("singer", "Singer_ID")
    assert element_text(el, mini_schema) == "singer id of singer"


def test_element_text_raises_on_unknown_element(mini_schema):
    with pytest.raises(KeyError):
        element_text(Element.column_el("singer", "nonexistent"), mini_schema)


def test_exact_mention_scores_100(mini_schema):
    scores = lexical_scores(
        "How many singers are there?", [Element.table_el("singer")], mini_schema
    )
    assert scores[Element.table_el("singer")] == 100


def test_unmentioned_element_scores_low(mini_schema):
    scores = lexical_scores(
        "How many singers are there?", [Element.table_el("concert")], mini_schema
    )
    assert scores[Element.table_el("concert")] < 70


def test_underscored_column_matches_spaced_question_phrase(mini_schema):
    """'singer id' in the question must anchor singer.Singer_ID."""
    el = Element.column_el("singer", "Singer_ID")
    scores = lexical_scores("What is the singer id of Joe?", [el], mini_schema)
    assert scores[el] >= 70


def test_scoring_is_case_insensitive(mini_schema):
    el = Element.table_el("singer")
    lower = lexical_scores("count the singers", [el], mini_schema)[el]
    upper = lexical_scores("COUNT THE SINGERS", [el], mini_schema)[el]
    assert lower == upper


def test_scores_cover_every_requested_element(mini_schema):
    els = [Element.table_el("singer"), Element.column_el("concert", "Name")]
    assert set(lexical_scores("anything", els, mini_schema)) == set(els)


def test_null_semantic_scorer_returns_zero_for_all():
    els = [Element.table_el("singer")]
    assert NullSemanticScorer().score("a question", els) == {els[0]: 0.0}


# --- R12: lexical scoring uses the bare element name, not the qualified
# "<column> of <table>" rendering. See scoring.py module docstring.


def test_lexical_scores_bare_column_name_not_qualified_rendering(mini_schema):
    """The qualified rendering 'country of singer' partial-matches 'singer'
    in a question that never mentions country (measured 62), scoring
    *higher* than a question that actually says 'country' (measured 58) --
    an inversion. Bare-name scoring must not reproduce that inversion:
    country must score higher against a question that mentions it than
    against one that does not (measured 100 vs 43 on this checkout).
    """
    el = Element.column_el("singer", "Country")
    mentioning = lexical_scores(
        "list singers by country", [el], mini_schema
    )[el]
    not_mentioning = lexical_scores(
        "how many singers are there?", [el], mini_schema
    )[el]
    assert mentioning > not_mentioning


def test_short_generic_column_name_does_not_over_fire(mini_schema):
    """A short, generic column name ('name') must not score above the
    default anchoring threshold (70) against a question that never
    mentions it -- the guard against bare scoring over-firing on short
    tokens (measured 50 on this checkout)."""
    el = Element.column_el("singer", "Name")
    score = lexical_scores("how many singers are there?", [el], mini_schema)[el]
    assert score < 70
