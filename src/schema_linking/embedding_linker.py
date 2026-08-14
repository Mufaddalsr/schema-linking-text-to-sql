"""Sentence-embedding schema linker (Method B).

Cosine-similarity linking between the question and every table/column's
pre-rendered text (see the rendering rules documented in
``schema_linking.utils.embeddings``). Both question and schema vectors are
L2-normalised by :class:`SchemaEncoder.encode`, so cosine similarity is a
plain dot product — no separate normalisation step is needed here.

Top-k + threshold semantics
----------------------------
An element is predicted iff it is among the ``top_k`` highest-scoring
elements of its kind for that schema **and** its score clears the
configured threshold. The two conditions are independent: ``top_k`` caps
the candidate count regardless of how many clear threshold, and the
threshold can still drop elements that made the top-k cut.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from schema_linking.base import ColumnScore, Prediction, TableScore
from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema
from schema_linking.utils.embeddings import SchemaEncoder

_STRATEGY: str = "cosine"


class EmbeddingLinker:
    """Cosine-similarity schema linker over pre-computed sentence embeddings.

    Parameters
    ----------
    encoder
        A configured :class:`SchemaEncoder`. Used to encode questions at
        prediction time and to pre-encode every schema at construction
        time.
    schemas
        Every ``db_id -> Schema`` this linker may be asked to predict
        against. Encoded once via ``encoder.encode_schema`` and cached in
        ``self._schema_index``.
    table_top_k
        Maximum number of tables predicted per question.
    table_threshold
        Minimum cosine similarity for a table to be predicted.
    column_top_k
        Maximum number of columns predicted per question.
    column_threshold
        Minimum cosine similarity for a column to be predicted.
    """

    name: str = "embedding"

    def __init__(
        self,
        encoder: SchemaEncoder,
        schemas: dict[str, Schema],
        table_top_k: int,
        table_threshold: float,
        column_top_k: int,
        column_threshold: float,
    ) -> None:
        self.encoder = encoder
        self.table_top_k = table_top_k
        self.table_threshold = table_threshold
        self.column_top_k = column_top_k
        self.column_threshold = column_threshold
        self._schema_index = encoder.encode_schema(schemas)

    def predict_one(self, example: SpiderExample, schema: Schema) -> Prediction:
        """Link one example against its schema.

        Returns
        -------
        Prediction
            Predicted tables/columns in original schema case, plus cosine
            scores for each predicted element in ``table_scores`` /
            ``column_scores``.
        """
        question_vec = self.encoder.encode([example.question])[0]
        return self._predict_from_vector(question_vec, schema.db_id)

    def predict_all(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, Prediction]:
        """Link a batch of examples, keyed by ``question_id``.

        All question strings are encoded in a single batch call — encoding
        one question at a time is ~20x slower over a full split.
        """
        question_vecs = self.encoder.encode([ex.question for ex in examples])
        return {
            ex.question_id: self._predict_from_vector(question_vecs[i], ex.db_id)
            for i, ex in enumerate(examples)
        }

    def similarity_matrix(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, dict]:
        """Raw per-example cosine scores, before top-k / threshold filtering.

        Intended for the week-5 threshold/top-k tuning sweep and week-6
        error analysis: cache this to disk once, then apply any
        ``top_k`` / ``threshold`` combination against it without
        re-encoding anything.

        Returns
        -------
        dict
            ``{question_id: {"question": str, "table_names": tuple[str,
            ...], "table_scores": np.ndarray, "column_names":
            tuple[tuple[str, str], ...], "column_scores": np.ndarray}}``.
            ``*_scores`` arrays are row-aligned with their ``*_names``
            tuple.
        """
        question_vecs = self.encoder.encode([ex.question for ex in examples])
        result: dict[int, dict] = {}
        for i, ex in enumerate(examples):
            index = self._schema_index[ex.db_id]
            q_vec = question_vecs[i]
            result[ex.question_id] = {
                "question": ex.question,
                "table_names": index["table_names"],
                "table_scores": np.dot(index["table_vectors"], q_vec),
                "column_names": index["column_names"],
                "column_scores": np.dot(index["column_vectors"], q_vec),
            }
        return result

    def _predict_from_vector(self, question_vec: np.ndarray, db_id: str) -> Prediction:
        index = self._schema_index[db_id]

        tables, table_sims = self._top_k_above_threshold(
            question_vec,
            index["table_vectors"],
            index["table_names"],
            self.table_top_k,
            self.table_threshold,
        )
        columns, column_sims = self._top_k_above_threshold(
            question_vec,
            index["column_vectors"],
            index["column_names"],
            self.column_top_k,
            self.column_threshold,
        )

        return Prediction(
            db_id=db_id,
            tables=tuple(tables),
            columns=tuple(columns),
            table_scores=tuple(
                TableScore(table=name, strategy=_STRATEGY, score=score)
                for name, score in zip(tables, table_sims, strict=True)
            ),
            column_scores=tuple(
                ColumnScore(table=t, column=c, strategy=_STRATEGY, score=score)
                for (t, c), score in zip(columns, column_sims, strict=True)
            ),
        )

    @staticmethod
    def _top_k_above_threshold(
        question_vec: np.ndarray,
        vectors: np.ndarray,
        names: Sequence,
        top_k: int,
        threshold: float,
    ) -> tuple[list, list[float]]:
        """Top-``top_k`` names by cosine score, then filtered by ``threshold``."""
        if vectors.shape[0] == 0:
            return [], []
        scores = np.dot(vectors, question_vec)
        return select_top_k_above_threshold(names, scores, top_k, threshold)


def select_top_k_above_threshold(
    names: Sequence,
    scores: np.ndarray,
    top_k: int,
    threshold: float,
) -> tuple[list, list[float]]:
    """Top-``top_k`` names by precomputed score, then filtered by ``threshold``.

    Pulled out of :class:`EmbeddingLinker` so a grid-search tuner can apply
    many ``(top_k, threshold)`` combinations to an already-computed score
    array without recomputing any dot products — see
    :func:`schema_linking.utils.tuning.tune_embedding`.

    Parameters
    ----------
    names
        Candidate names (table name strings, or ``(table, column)``
        tuples), aligned with ``scores``.
    scores
        Cosine similarity of each candidate against the question, same
        length as ``names``. Empty input returns ``([], [])``.
    top_k
        Maximum number of candidates to keep, ranked by score.
    threshold
        Minimum score for a top-``top_k`` candidate to be kept.
    """
    if scores.size == 0:
        return [], []
    top_indices = np.argsort(-scores)[:top_k]
    kept = [(names[i], float(scores[i])) for i in top_indices if scores[i] >= threshold]
    picked_names = [name for name, _ in kept]
    picked_scores = [score for _, score in kept]
    return picked_names, picked_scores
