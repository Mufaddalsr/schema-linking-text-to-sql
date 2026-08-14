"""Sentence-embedding encoder for schema elements (Method B).

Rendering rules (locked in ``docs/decisions.md``)
--------------------------------------------------
Text fed to the encoder uses Spider's human-readable names, not the
DB-canonical (``original_name``) identifiers — natural-language phrases
embed better with a sentence model than snake_case/camelCase identifiers:

- table text: ``table.name``
- column text: ``f"{column.name} of {table.name}"``

The ``table_names`` / ``column_names`` returned by :meth:`SchemaEncoder.
encode_schema` are ``original_name``s, matching the rest of the codebase's
"predictions in original schema case" convention (``base.Linker``).

Caching
-------
:meth:`SchemaEncoder.encode_schema` caches per ``db_id`` under
``cache_dir`` as a ``.npz`` file (``table_vectors`` + ``column_vectors``),
keyed by ``sha256(model_name + revision + "rendering_v1" + json(schema
metadata))``. The ``"rendering_v1"`` tag busts the cache if the rendering
rules above ever change. Deliberately not pickle-based — cache files hold
only arrays, never arbitrary Python objects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from schema_linking.schema_parser import Schema

_RENDERING_VERSION: str = "rendering_v1"


class SchemaEncoder:
    """Encodes questions and schema elements into normalised embeddings.

    Parameters
    ----------
    model_name
        HuggingFace model id, e.g. ``"BAAI/bge-small-en-v1.5"``.
    revision
        Pinned HuggingFace commit hash for reproducibility.
    cache_dir
        Directory for cached per-``db_id`` schema embeddings. Created if it
        does not already exist.
    """

    def __init__(self, model_name: str, revision: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.revision = revision
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(model_name, revision=revision)
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode ``texts`` into unit-norm embeddings.

        Parameters
        ----------
        texts
            Strings to encode.
        batch_size
            Batch size passed to ``SentenceTransformer.encode``.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), self.dim)``, L2-normalised rows. Empty
            input returns an empty array of the correct width rather than
            calling the model.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_schema(self, schemas: dict[str, Schema]) -> dict[str, dict]:
        """Encode every table and column of each schema, with on-disk caching.

        Parameters
        ----------
        schemas
            Mapping of ``db_id`` to :class:`Schema`.

        Returns
        -------
        dict
            ``{db_id: {"table_names": tuple[str, ...],
            "table_vectors": np.ndarray, "column_names":
            tuple[tuple[str, str], ...], "column_vectors": np.ndarray}}``.
            Names are in original schema case; vectors are row-aligned with
            their respective names tuple.
        """
        return {db_id: self._encode_one_schema(db_id, schema) for db_id, schema in schemas.items()}

    def _encode_one_schema(self, db_id: str, schema: Schema) -> dict:
        table_names = tuple(table.original_name for table in schema.tables)
        table_texts = [table.name for table in schema.tables]
        column_names = tuple(
            (table.original_name, column.original_name)
            for table in schema.tables
            for column in table.columns
        )
        column_texts = [
            f"{column.name} of {table.name}"
            for table in schema.tables
            for column in table.columns
        ]

        cache_path = self._cache_path(db_id, table_names, column_names)
        cached = self._load_cache(cache_path)
        if cached is not None:
            table_vectors, column_vectors = cached
        else:
            table_vectors = self.encode(table_texts)
            column_vectors = self.encode(column_texts)
            self._save_cache(cache_path, table_vectors, column_vectors)

        return {
            "table_names": table_names,
            "table_vectors": table_vectors,
            "column_names": column_names,
            "column_vectors": column_vectors,
        }

    def _cache_path(
        self,
        db_id: str,
        table_names: tuple[str, ...],
        column_names: tuple[tuple[str, str], ...],
    ) -> Path:
        metadata = {
            "db_id": db_id,
            "tables": list(table_names),
            "columns": [list(pair) for pair in column_names],
        }
        payload = self.model_name + self.revision + _RENDERING_VERSION + json.dumps(
            metadata, sort_keys=True
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npz"

    @staticmethod
    def _load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        if not cache_path.is_file():
            return None
        with np.load(cache_path) as data:
            return data["table_vectors"], data["column_vectors"]

    @staticmethod
    def _save_cache(
        cache_path: Path, table_vectors: np.ndarray, column_vectors: np.ndarray
    ) -> None:
        np.savez(cache_path, table_vectors=table_vectors, column_vectors=column_vectors)
