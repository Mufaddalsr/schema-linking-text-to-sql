"""Question-to-element anchoring scores.

Two independent signals decide whether the question "points at" a schema
element, and their disagreement is what separates ``PARAPHRASE`` from
``UNVERBALISED``:

* **Lexical** — rapidfuzz ``partial_ratio`` between the element's bare name
  and the question. Surface-form overlap only.
* **Semantic** — cosine similarity between the question's embedding and the
  element's, using the project's pinned encoder.

Two renderings, not one
------------------------
This module builds an element's readable name in two different ways, and
the two must **not** be unified:

* :func:`element_text` (and the internal ``_qualified_readable``) renders a
  column as ``f"{column.name} of {table.name}"``. This matches
  :mod:`schema_linking.utils.embeddings` exactly, so it is what
  :class:`EmbeddingSemanticScorer` uses to line up with the embedding
  linker's cached vectors, and it is the encoder's text tested directly by
  ``test_element_text_for_column_matches_encoder_convention``.
* ``lexical_scores`` (via the internal ``_bare_readable``) renders a column
  as its bare ``column.name``, with no table qualifier.

The qualified rendering is wrong for lexical scoring: rapidfuzz's
``partial_ratio`` treats the element name as a substring probe, and the
``" of <table>"`` suffix can match words in the question that have nothing
to do with the column, inverting the signal. Concretely, on this schema
``"country of singer"`` scores *higher* (62) against a question that never
mentions country ("how many singers are there?") than against one that
does (58, "list singers by country") — the ``" of singer"`` suffix matches
"singers" in the irrelevant question. Scoring the bare name ``"country"``
instead gives 43 and 100 respectively, restoring the correct ordering.
Short/generic bare names (``id``, ``no``, ``name``, ``year``, ``age``) were
checked and stay below the default 70 anchoring threshold against
non-mentioning questions, so bare scoring does not trade the qualified
rendering's false positives for new ones of its own.

A future reader who "unifies" the two renderings would silently
reintroduce the qualified rendering's inversion into the lexical signal —
please don't.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from schema_linking.erroranalysis.taxonomy import Element
from schema_linking.schema_parser import Schema
from schema_linking.utils.embeddings import SchemaEncoder


def _qualified_readable(schema: Schema) -> dict[Element, str]:
    """Map every element of ``schema`` to its encoder-convention rendering.

    Tables render as ``table.name``; columns as
    ``f"{column.name} of {table.name}"``, matching
    :mod:`schema_linking.utils.embeddings` exactly. Used by
    :func:`element_text` and by :class:`EmbeddingSemanticScorer`. Do **not**
    use this for lexical scoring — see the module docstring.
    """
    out: dict[Element, str] = {}
    for table in schema.tables:
        out[Element.table_el(table.original_name)] = table.name
        for column in table.columns:
            out[Element.column_el(table.original_name, column.original_name)] = (
                f"{column.name} of {table.name}"
            )
    return out


def _bare_readable(schema: Schema) -> dict[Element, str]:
    """Map every element of ``schema`` to its bare, unqualified name.

    Tables render as ``table.name``; columns as ``column.name`` alone, with
    no ``" of <table>"`` suffix. Used only by :func:`lexical_scores` — the
    qualified rendering's table suffix can match unrelated words in the
    question and invert the lexical signal (see module docstring). Do not
    use this rendering for the semantic scorer; it must keep matching
    :mod:`schema_linking.utils.embeddings`'s convention via
    ``_qualified_readable``.
    """
    out: dict[Element, str] = {}
    for table in schema.tables:
        out[Element.table_el(table.original_name)] = table.name
        for column in table.columns:
            out[Element.column_el(table.original_name, column.original_name)] = (
                column.name
            )
    return out


def element_text(element: Element, schema: Schema) -> str:
    """Readable rendering of ``element``, matching the embedding encoder.

    Tables render as their Spider ``table_names`` entry; columns as
    ``"<column name> of <table name>"``. This is a convenience for tests
    and diagnostics: it rebuilds the whole schema lookup on every call
    rather than caching it. That's fine — the hot path
    (:func:`lexical_scores`) builds its own index once per call for every
    element it scores, and :class:`~schema_linking.schema_parser.Schema`
    contains list fields, so it cannot be ``lru_cache``d here anyway.

    Raises
    ------
    KeyError
        If ``element`` is not in ``schema`` — callers must screen
        hallucinations out first.
    """
    try:
        return _qualified_readable(schema)[element]
    except KeyError as exc:
        raise KeyError(
            f"{element.render()!r} is not in schema {schema.db_id!r}"
        ) from exc


def lexical_scores(
    question: str,
    elements: Sequence[Element],
    schema: Schema,
) -> dict[Element, int]:
    """rapidfuzz ``partial_ratio`` of each element's bare name against ``question``.

    Parameters
    ----------
    question
        The natural-language question, in any case.
    elements
        Elements to score. All must exist in ``schema``.
    schema
        The database schema, used for bare-name renderings.

    Returns
    -------
    dict[Element, int]
        Score in ``0-100`` for every element in ``elements``.

    Raises
    ------
    KeyError
        If any element in ``elements`` is not in ``schema``.

    Notes
    -----
    ``partial_ratio`` is used rather than ``ratio`` because the element name
    is a short substring of a long question — the same choice the lexical
    linker makes. Underscores become spaces first, so ``Singer_ID`` can
    match the question phrase "singer id".

    Scoring uses the *bare* element name (``column.name`` alone, no table
    qualifier) rather than :func:`element_text`'s qualified rendering — see
    the module docstring for why the qualified form inverts this signal.
    """
    bare = _bare_readable(schema)
    q = question.lower()
    scores: dict[Element, int] = {}
    for el in elements:
        if el not in bare:
            raise KeyError(f"{el.render()!r} is not in schema {schema.db_id!r}")
        name = bare[el].replace("_", " ").lower()
        scores[el] = int(round(fuzz.partial_ratio(name, q)))
    return scores


class NullSemanticScorer:
    """Semantic scorer that always returns ``0.0``.

    Used by the Task 8 calibration spike, whose three rules need no semantic
    signal, and by any test that wants lexical behaviour in isolation.
    """

    def score(
        self, question: str, elements: Sequence[Element]
    ) -> dict[Element, float]:
        """Return ``0.0`` for every element."""
        return {el: 0.0 for el in elements}


class EmbeddingSemanticScorer:
    """Cosine similarity via the project's pinned :class:`SchemaEncoder`.

    Question vectors are cached to disk keyed by the encoder revision and a
    hash of the question text, so a full census run encodes each of the 1,034
    dev questions once.

    Parameters
    ----------
    encoder
        A configured :class:`SchemaEncoder`.
    schemas
        Every schema, keyed by ``db_id``. Element vectors come from the
        encoder's existing schema cache.
    cache_dir
        Directory for the question-vector cache.
    """

    def __init__(
        self,
        encoder: SchemaEncoder,
        schemas: Mapping[str, Schema],
        cache_dir: Path,
    ) -> None:
        self._encoder = encoder
        self._schemas = dict(schemas)
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._element_vectors = self._build_element_vectors()
        self._question_cache: dict[str, np.ndarray] = {}

    def _build_element_vectors(self) -> dict[Element, np.ndarray]:
        """Flatten the encoder's per-schema vectors into an element lookup."""
        encoded = self._encoder.encode_schema(self._schemas)
        out: dict[Element, np.ndarray] = {}
        for db_id, bundle in encoded.items():
            for name, vec in zip(
                bundle["table_names"], bundle["table_vectors"], strict=True
            ):
                out[Element.table_el(name)] = vec
            for (t, c), vec in zip(
                bundle["column_names"], bundle["column_vectors"], strict=True
            ):
                out[Element.column_el(t, c)] = vec
        return out

    def _question_vector(self, question: str) -> np.ndarray:
        """Encode ``question``, memoised in RAM and on disk."""
        if question in self._question_cache:
            return self._question_cache[question]
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        path = self._cache_dir / f"q_{self._encoder.revision}_{digest}.npy"
        if path.is_file():
            vec = np.load(path)
        else:
            vec = self._encoder.encode([question])[0]
            np.save(path, vec)
        self._question_cache[question] = vec
        return vec

    def score(
        self, question: str, elements: Sequence[Element]
    ) -> dict[Element, float]:
        """Cosine similarity of each element against ``question``.

        Vectors from :class:`SchemaEncoder` are already L2-normalised, so the
        dot product is the cosine. Results are clipped to ``[0, 1]``.
        """
        qv = self._question_vector(question)
        out: dict[Element, float] = {}
        for el in elements:
            ev = self._element_vectors.get(el)
            out[el] = 0.0 if ev is None else float(np.clip(qv @ ev, 0.0, 1.0))
        return out
