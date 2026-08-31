"""Tests for ``schema_linking.embedding_linker.EmbeddingLinker``.

Unit tests use a ``FakeEncoder`` with hand-picked, hand-computed vectors —
no model download, no ``SchemaEncoder``. The one exception is
``TestEmbeddingLinkerIntegration``, which exercises the real pinned BGE
model against real Spider dev examples; it is marked
``@pytest.mark.integration`` and skipped unless ``RUN_INTEGRATION=1`` (see
``tests/conftest.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from schema_linking.base import Linker, Prediction
from schema_linking.data_loader import SpiderExample
from schema_linking.embedding_linker import EmbeddingLinker
from schema_linking.schema_parser import Column, Schema, Table


class FakeEncoder:
    """Stands in for ``SchemaEncoder``: fixed schema index, scripted question vectors."""

    def __init__(self, schema_index: dict, question_vectors: dict[str, np.ndarray]) -> None:
        self._schema_index = schema_index
        self._question_vectors = question_vectors
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.encode_calls.append(list(texts))
        return np.array([self._question_vectors[t] for t in texts])

    def encode_schema(self, schemas: dict[str, Schema]) -> dict:
        return self._schema_index


def _example(question: str, qid: int = 0, db_id: str = "test_db") -> SpiderExample:
    return SpiderExample(
        question_id=qid,
        db_id=db_id,
        question=question,
        query="SELECT 1",
        sql={},
        split="dev",
    )


def _tiny_schema() -> Schema:
    """A schema with 3 tables, 1 column each — matches the fake 3-D index below."""
    tables = []
    for name in ("A", "B", "C"):
        col = Column(
            name=f"{name} col", original_name=f"{name}Col", type="text", table_name=name,
            is_primary_key=False,
        )
        tables.append(Table(name=name, original_name=name, columns=[col]))
    return Schema(db_id="test_db", tables=tables, foreign_keys=[])


def _fake_schema_index() -> dict:
    """One-hot 3-D vectors for tables A/B/C and their sole columns."""
    onehots = np.eye(3, dtype=np.float64)
    return {
        "test_db": {
            "table_names": ("A", "B", "C"),
            "table_vectors": onehots.copy(),
            "column_names": (("A", "ACol"), ("B", "BCol"), ("C", "CCol")),
            "column_vectors": onehots.copy(),
        }
    }


class TestEmbeddingLinkerBasic:
    def test_satisfies_protocol(self) -> None:
        encoder = FakeEncoder(_fake_schema_index(), {})
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas={"test_db": _tiny_schema()},
            table_top_k=3,
            table_threshold=0.5,
            column_top_k=3,
            column_threshold=0.5,
        )
        assert isinstance(linker, Linker)

    def test_predict_one_matches_hand_computed_top_k(self) -> None:
        # Question vector == table A's / column ACol's one-hot exactly:
        # cosine(q, A) = 1.0, cosine(q, B) = cosine(q, C) = 0.0.
        q = "what is a"
        encoder = FakeEncoder(_fake_schema_index(), {q: np.array([1.0, 0.0, 0.0])})
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas={"test_db": _tiny_schema()},
            table_top_k=3,
            table_threshold=0.5,
            column_top_k=3,
            column_threshold=0.5,
        )
        pred = linker.predict_one(_example(q), _tiny_schema())

        assert pred.tables == ("A",)
        assert pred.columns == (("A", "ACol"),)
        assert pred.table_scores is not None
        assert {ts.table: ts.score for ts in pred.table_scores} == {"A": pytest.approx(1.0)}
        assert pred.column_scores is not None
        assert {cs.column: cs.score for cs in pred.column_scores} == {"ACol": pytest.approx(1.0)}


class TestEmbeddingLinkerThresholdAndTopK:
    def test_threshold_cutoff_returns_empty_prediction(self) -> None:
        # Every score is well below an unreachable 0.99 threshold.
        q = "irrelevant question"
        encoder = FakeEncoder(_fake_schema_index(), {q: np.array([0.1, 0.1, 0.1])})
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas={"test_db": _tiny_schema()},
            table_top_k=3,
            table_threshold=0.99,
            column_top_k=3,
            column_threshold=0.99,
        )
        pred = linker.predict_one(_example(q), _tiny_schema())
        assert pred.tables == ()
        assert pred.columns == ()

    def test_top_k_cap_limits_predictions_even_above_threshold(self) -> None:
        # All three tables/columns clear threshold=0.5; top_k=1 keeps only
        # the single highest-scoring one (A: 0.9 > B: 0.8 > C: 0.7).
        q = "a b c all relevant"
        encoder = FakeEncoder(_fake_schema_index(), {q: np.array([0.9, 0.8, 0.7])})
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas={"test_db": _tiny_schema()},
            table_top_k=1,
            table_threshold=0.5,
            column_top_k=1,
            column_threshold=0.5,
        )
        pred = linker.predict_one(_example(q), _tiny_schema())
        assert pred.tables == ("A",)
        assert pred.columns == (("A", "ACol"),)


class TestEmbeddingLinkerStarColumn:
    def test_star_column_never_predicted(self) -> None:
        # Schema objects can never contain a `*` column — schema_parser
        # excludes it at parse time, by design. This guards against
        # a future regression reintroducing it anywhere in the encode /
        # predict path.
        q = "select everything"
        encoder = FakeEncoder(_fake_schema_index(), {q: np.array([1.0, 1.0, 1.0])})
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas={"test_db": _tiny_schema()},
            table_top_k=3,
            table_threshold=0.0,
            column_top_k=3,
            column_threshold=0.0,
        )
        pred = linker.predict_one(_example(q), _tiny_schema())
        assert "*" not in pred.tables
        assert all(col != "*" for _, col in pred.columns)


class TestEmbeddingLinkerPredictAll:
    def test_predict_all_batches_question_encoding(self) -> None:
        questions = ["q0", "q1", "q2"]
        vectors = {
            "q0": np.array([1.0, 0.0, 0.0]),
            "q1": np.array([0.0, 1.0, 0.0]),
            "q2": np.array([0.0, 0.0, 1.0]),
        }
        encoder = FakeEncoder(_fake_schema_index(), vectors)
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas={"test_db": _tiny_schema()},
            table_top_k=1,
            table_threshold=0.5,
            column_top_k=1,
            column_threshold=0.5,
        )
        examples = [_example(q, qid=i) for i, q in enumerate(questions)]
        result = linker.predict_all(examples, {"test_db": _tiny_schema()})

        assert len(encoder.encode_calls) == 1
        assert encoder.encode_calls[0] == questions
        assert set(result.keys()) == {0, 1, 2}
        assert result[0].tables == ("A",)
        assert result[1].tables == ("B",)
        assert result[2].tables == ("C",)
        for pred in result.values():
            assert isinstance(pred, Prediction)

    def test_similarity_matrix_returns_raw_scores_for_every_element(self) -> None:
        q = "q0"
        encoder = FakeEncoder(_fake_schema_index(), {q: np.array([1.0, 0.0, 0.0])})
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas={"test_db": _tiny_schema()},
            table_top_k=1,
            table_threshold=0.99,
            column_top_k=1,
            column_threshold=0.99,
        )
        examples = [_example(q, qid=0)]
        matrix = linker.similarity_matrix(examples, {"test_db": _tiny_schema()})

        entry = matrix[0]
        assert entry["question"] == q
        assert entry["table_names"] == ("A", "B", "C")
        np.testing.assert_allclose(entry["table_scores"], [1.0, 0.0, 0.0])
        assert entry["column_names"] == (("A", "ACol"), ("B", "BCol"), ("C", "CCol"))
        np.testing.assert_allclose(entry["column_scores"], [1.0, 0.0, 0.0])


@pytest.mark.integration
class TestEmbeddingLinkerIntegration:
    def test_predicts_on_real_dev_examples(self) -> None:
        from schema_linking.data_loader import load_spider_questions
        from schema_linking.schema_parser import load_schemas
        from schema_linking.utils.config import load_config
        from schema_linking.utils.embeddings import SchemaEncoder

        config = load_config()
        examples = list(load_spider_questions("dev"))[:10]
        all_schemas = load_schemas()
        needed_schemas = {ex.db_id: all_schemas[ex.db_id] for ex in examples}

        encoder = SchemaEncoder(
            model_name=config.embedding.model_name,
            revision=config.embedding.revision,
            cache_dir=config.embedding.cache_dir,
        )
        linker = EmbeddingLinker(
            encoder=encoder,
            schemas=needed_schemas,
            table_top_k=5,
            table_threshold=0.3,
            column_top_k=10,
            column_threshold=0.3,
        )
        predictions = linker.predict_all(examples, needed_schemas)

        assert set(predictions.keys()) == {ex.question_id for ex in examples}
        for pred in predictions.values():
            assert isinstance(pred, Prediction)
