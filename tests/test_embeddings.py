"""Tests for src/schema_linking/utils/embeddings.py.

Model note
----------
These tests patch ``SchemaEncoder``'s model loader to reuse a single,
module-scoped ``sentence-transformers/all-MiniLM-L6-v2`` instance instead
of downloading the production model (``BAAI/bge-small-en-v1.5``) once per
test. This keeps the suite fast while still exercising the real
``SentenceTransformer.encode(normalize_embeddings=True)`` path for the
unit-norm assertion. Cache-key, cache-hit, and cache-miss behaviour only
depend on the ``model_name`` / ``revision`` strings our wrapper records —
not on which underlying model object is actually loaded — so substituting
the fallback model does not weaken those tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from schema_linking.schema_parser import Column, Schema, Table
from schema_linking.utils.embeddings import SchemaEncoder

_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def real_model() -> SentenceTransformer:
    return SentenceTransformer(_FALLBACK_MODEL)


@pytest.fixture()
def patch_loader(monkeypatch: pytest.MonkeyPatch, real_model: SentenceTransformer) -> None:
    """Make SchemaEncoder.__init__ reuse the pre-downloaded model."""
    monkeypatch.setattr(
        "schema_linking.utils.embeddings.SentenceTransformer",
        lambda model_name, revision=None: real_model,
    )


def _make_encoder(cache_dir: Path, revision: str = "rev-a") -> SchemaEncoder:
    return SchemaEncoder(model_name=_FALLBACK_MODEL, revision=revision, cache_dir=cache_dir)


def _tiny_schema() -> Schema:
    stu_id = Column(
        name="student ID",
        original_name="StuID",
        type="number",
        table_name="Students",
        is_primary_key=True,
    )
    stu_name = Column(
        name="student name",
        original_name="StuName",
        type="text",
        table_name="Students",
        is_primary_key=False,
    )
    table = Table(name="Students", original_name="Students", columns=[stu_id, stu_name])
    return Schema(db_id="test_db", tables=[table], foreign_keys=[])


def test_encode_returns_unit_norm_vectors(tmp_path: Path, patch_loader: None) -> None:
    encoder = _make_encoder(tmp_path)
    vectors = encoder.encode(["a student", "a course enrollment"])

    assert vectors.shape == (2, encoder.dim)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_encode_schema_output_aligned(tmp_path: Path, patch_loader: None) -> None:
    encoder = _make_encoder(tmp_path)
    result = encoder.encode_schema({"test_db": _tiny_schema()})

    entry = result["test_db"]
    assert entry["table_names"] == ("Students",)
    assert entry["column_names"] == (("Students", "StuID"), ("Students", "StuName"))
    assert entry["table_vectors"].shape[0] == len(entry["table_names"])
    assert entry["column_vectors"].shape[0] == len(entry["column_names"])
    assert entry["table_vectors"].shape[1] == encoder.dim
    assert entry["column_vectors"].shape[1] == encoder.dim


def test_cache_hit_avoids_reencoding(
    tmp_path: Path, patch_loader: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = _make_encoder(tmp_path)
    schema = _tiny_schema()
    encoder.encode_schema({"test_db": schema})  # populates the cache

    def _fail_if_called(texts: list[str], batch_size: int = 32) -> np.ndarray:
        raise AssertionError("encode() must not be called on a cache hit")

    monkeypatch.setattr(encoder, "encode", _fail_if_called)

    result = encoder.encode_schema({"test_db": schema})  # should be a cache hit

    assert result["test_db"]["table_names"] == ("Students",)


def test_cache_miss_on_revision_change(tmp_path: Path, patch_loader: None) -> None:
    schema = _tiny_schema()

    encoder_v1 = _make_encoder(tmp_path, revision="rev-a")
    encoder_v1.encode_schema({"test_db": schema})
    assert len(list(tmp_path.glob("*.npz"))) == 1

    encoder_v2 = _make_encoder(tmp_path, revision="rev-b")
    encoder_v2.encode_schema({"test_db": schema})
    assert len(list(tmp_path.glob("*.npz"))) == 2
