"""Load one split's questions, schemas, gold tiers and predictions once.

The census touches every (question, method, tier) triple, so every input is
read a single time into a :class:`Corpus` and reused. Schema indices are
built per database, not per question.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schema_linking.data_loader import SpiderExample, load_spider_questions
from schema_linking.erroranalysis.facts import SchemaIndex
from schema_linking.schema_parser import Schema, load_schemas
from schema_linking.utils.config import Config, load_config
from schema_linking.utils.difficulty import difficulty_for_examples

GOLD_TIER1_FILENAME: str = "gold_links_{split}_mentioned.json"
GOLD_TIER2_FILENAME: str = "gold_links_{split}_all_sql.json"
PREDICTION_FILENAME: str = "{method}_{split}.json"


@dataclass(frozen=True, slots=True)
class Corpus:
    """Every input the census needs for one split.

    Attributes
    ----------
    split
        ``"dev"`` or ``"train"``.
    examples
        Spider examples in file order; ``question_id`` matches the tuple
        index. A tuple, matching
        :func:`schema_linking.data_loader.load_spider_questions`'s return
        type.
    schemas
        Every Spider schema, keyed by ``db_id``.
    indices
        :class:`SchemaIndex` per ``db_id``, built once.
    gold_tier1, gold_tier2
        ``{question_id: {"db_id":..., "tables": [...], "columns": [[t, c], ...]}}``.
    predictions
        ``{method: {question_id: prediction_record}}``.
    hardness
        ``{question_id: "easy" | "medium" | "hard" | "extra"}``.
    """

    split: str
    examples: tuple[SpiderExample, ...]
    schemas: dict[str, Schema]
    indices: dict[str, SchemaIndex]
    gold_tier1: dict[int, dict[str, Any]]
    gold_tier2: dict[int, dict[str, Any]]
    predictions: dict[str, dict[int, dict[str, Any]]]
    hardness: dict[int, str]

    def example_by_qid(self) -> dict[int, SpiderExample]:
        """Index the examples by ``question_id``."""
        return {e.question_id: e for e in self.examples}


def _load_keyed_json(path: Path) -> dict[int, dict[str, Any]]:
    """Read a ``{question_id: record}`` JSON file with integer keys.

    The files are written with string keys; they are converted here so every
    downstream consumer works with ``int`` question ids, matching
    :class:`~schema_linking.data_loader.SpiderExample` and
    ``main_per_query.csv``.
    """
    with path.open(encoding="utf-8") as f:
        raw: Mapping[str, Any] = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_corpus(split: str = "dev", config: Config | None = None) -> Corpus:
    """Load everything needed to build the census for ``split``.

    Parameters
    ----------
    split
        ``"dev"`` or ``"train"``.
    config
        Optional :class:`~schema_linking.utils.config.Config`. If ``None``,
        loads the default config from ``<repo_root>/config.yaml``.

    Returns
    -------
    Corpus
        Every input the census needs, read once.

    Raises
    ------
    FileNotFoundError
        If a gold-tier or prediction file is missing. Fails loud rather than
        silently censusing fewer than the six methods.
    """
    from schema_linking.erroranalysis.census import METHODS

    cfg = config if config is not None else load_config()
    examples = load_spider_questions(split, cfg)  # type: ignore[arg-type]
    schemas = load_schemas(cfg.data.spider_dir / "tables.json")
    indices = {db_id: SchemaIndex.build(s) for db_id, s in schemas.items()}

    processed = cfg.data.processed_dir
    gold_tier1 = _load_keyed_json(processed / GOLD_TIER1_FILENAME.format(split=split))
    gold_tier2 = _load_keyed_json(processed / GOLD_TIER2_FILENAME.format(split=split))

    predictions: dict[str, dict[int, dict[str, Any]]] = {}
    for method in METHODS:
        path = cfg.outputs.predictions_dir / PREDICTION_FILENAME.format(
            method=method, split=split
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing prediction file: {path}")
        predictions[method] = _load_keyed_json(path)

    return Corpus(
        split=split,
        examples=examples,
        schemas=schemas,
        indices=indices,
        gold_tier1=gold_tier1,
        gold_tier2=gold_tier2,
        predictions=predictions,
        hardness=difficulty_for_examples(examples),
    )
