"""Typed loader form ``config.yaml``.

The config holds paths and linker hyperparameters tuned offline.

Path resolution rule
--------------------
Every relative path in the YAML is resolved against the **config file's
directory**.

Linker section
--------------
``linkers.lexical.fuzzy_threshold`` is the rapidfuzz ``partial_ratio``
threshold for the fuzzy strategy. missing keys fall back to the defaults below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_LEXICAL_FUZZY_THRESHOLD_DEFAULT: int = 80

_EMBEDDING_MODEL_NAME_DEFAULT: str = "BAAI/bge-small-en-v1.5"
_EMBEDDING_REVISION_DEFAULT: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
_EMBEDDING_CACHE_DIR_DEFAULT: str = "data/processed/embeddings"

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH: Path = _REPO_ROOT / "config.yaml"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Input-data locations.

    Attributes
    ----------
    spider_dir
        Directory holding raw Spider files (``train_spider.json``,
        ``dev.json``, ``tables.json``, ``database/``).
    processed_dir
        Directory for derived artefacts (gold-link JSON from week 2 onward).
    taniguchi_splits_dir
        Directory holding Taniguchi et al.'s human-annotated schema-linking
        labels (``dev.jsonl`` + ``test.jsonl``, 517 examples each, jointly
        covering Spider's 1034-example dev set). Used as an external
        reference gold-link source.
    """

    spider_dir: Path
    processed_dir: Path
    taniguchi_splits_dir: Path


@dataclass(frozen=True, slots=True)
class OutputsConfig:
    """Output destinations for predictions, results, and run logs."""

    predictions_dir: Path
    results_dir: Path
    logs_dir: Path


@dataclass(frozen=True, slots=True)
class LexicalLinkerConfig:
    """Hyperparameters for the lexical linker.

    Attributes
    ----------
    fuzzy_threshold
        rapidfuzz ``partial_ratio`` cutoff in ``[0, 100]`` for the fuzzy
        strategy. Tuned offline on a train subset.
    """

    fuzzy_threshold: int = _LEXICAL_FUZZY_THRESHOLD_DEFAULT


@dataclass(frozen=True, slots=True)
class LinkersConfig:
    """Linker hyperparameters grouped by method."""

    lexical: LexicalLinkerConfig = field(default_factory=LexicalLinkerConfig)


@dataclass(frozen=True, slots=True)
class TunedEmbeddingConfig:
    """Grid-search-selected top-k/threshold knobs for ``EmbeddingLinker``.

    Produced by ``notebooks/05_embedding_tuning.ipynb`` /
    :func:`schema_linking.utils.tuning.tune_embedding`; see
    ``docs/decisions.md`` for the chosen values and the sweep that picked
    them.
    """

    table_top_k: int
    table_threshold: float
    column_top_k: int
    column_threshold: float


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the sentence-embedding schema encoder (Method B).

    Attributes
    ----------
    model_name
        HuggingFace model id, e.g. ``"BAAI/bge-small-en-v1.5"``.
    revision
        Pinned HuggingFace commit hash. Locked (not a branch name like
        ``"main"``) so the encoder — and therefore the embedding cache — is
        reproducible across runs.
    cache_dir
        Directory for cached schema embeddings, keyed by
        ``(model_name, revision, rendering rules, schema content)``. See
        :class:`schema_linking.utils.embeddings.SchemaEncoder`.
    tuned
        The grid-search-selected ``EmbeddingLinker`` knobs, or ``None`` if
        ``embedding.tuned`` is absent from ``config.yaml`` (e.g. before
        tuning has been run).
    """

    model_name: str = _EMBEDDING_MODEL_NAME_DEFAULT
    revision: str = _EMBEDDING_REVISION_DEFAULT
    cache_dir: Path = Path(_EMBEDDING_CACHE_DIR_DEFAULT)
    tuned: TunedEmbeddingConfig | None = None


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level project configuration."""

    data: DataConfig
    outputs: OutputsConfig
    linkers: LinkersConfig = field(default_factory=LinkersConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


def load_config(path: Path | None = None) -> Config:
    """Load and validate ``config.yaml``.

    Parameters
    ----------
    path
        Path to the YAML config. Defaults to ``<repo_root>/config.yaml``
        (resolved from this module's location, not the CWD).

    Returns
    -------
    Config
        Fully resolved configuration. All path fields are absolute
        :class:`pathlib.Path` objects.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If a required section or key is missing from the YAML.
    """
    cfg_path = (path if path is not None else DEFAULT_CONFIG_PATH).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    base = cfg_path.parent

    data_raw = raw["data"]
    outputs_raw = raw["outputs"]

    data = DataConfig(
        spider_dir=_resolve(base, data_raw["spider_dir"]),
        processed_dir=_resolve(base, data_raw["processed_dir"]),
        taniguchi_splits_dir=_resolve(base, data_raw["taniguchi_splits_dir"]),
    )
    outputs = OutputsConfig(
        predictions_dir=_resolve(base, outputs_raw["predictions_dir"]),
        results_dir=_resolve(base, outputs_raw["results_dir"]),
        logs_dir=_resolve(base, outputs_raw["logs_dir"]),
    )
    linkers = _parse_linkers(raw.get("linkers"))
    embedding = _parse_embedding(base, raw.get("embedding"))
    return Config(data=data, outputs=outputs, linkers=linkers, embedding=embedding)


def _parse_linkers(raw: Any) -> LinkersConfig:
    """Parse the optional ``linkers`` YAML section.

    Missing section or missing keys fall back to dataclass defaults.
    """
    if not isinstance(raw, dict):
        return LinkersConfig()
    lexical_raw = raw.get("lexical") or {}
    return LinkersConfig(
        lexical=LexicalLinkerConfig(
            fuzzy_threshold=int(
                lexical_raw.get("fuzzy_threshold", _LEXICAL_FUZZY_THRESHOLD_DEFAULT)
            ),
        ),
    )


def _parse_embedding(base: Path, raw: Any) -> EmbeddingConfig:
    """Parse the optional ``embedding`` YAML section.

    Missing section or missing keys fall back to dataclass defaults.
    ``tuned`` stays ``None`` unless ``embedding.tuned`` is present.
    """
    if not isinstance(raw, dict):
        raw = {}
    return EmbeddingConfig(
        model_name=raw.get("model_name", _EMBEDDING_MODEL_NAME_DEFAULT),
        revision=raw.get("revision", _EMBEDDING_REVISION_DEFAULT),
        cache_dir=_resolve(base, raw.get("cache_dir", _EMBEDDING_CACHE_DIR_DEFAULT)),
        tuned=_parse_tuned_embedding(raw.get("tuned")),
    )


def _parse_tuned_embedding(raw: Any) -> TunedEmbeddingConfig | None:
    if not isinstance(raw, dict):
        return None
    return TunedEmbeddingConfig(
        table_top_k=int(raw["table_top_k"]),
        table_threshold=float(raw["table_threshold"]),
        column_top_k=int(raw["column_top_k"]),
        column_threshold=float(raw["column_threshold"]),
    )


def _resolve(base: Path, value: str) -> Path:
    """Resolve ``value`` against ``base`` and return an absolute :class:`Path`.

    Absolute ``value`` strings pass through; relative ones are joined to
    ``base`` and then resolved.
    """
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()
