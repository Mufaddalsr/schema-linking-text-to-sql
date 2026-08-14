"""Load Spider train/dev examples into typed records.

Paths are resolved via :mod:`schema_linking.utils.config`; nothing in this
module is CWD-dependent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from schema_linking.utils.config import Config, load_config

Split = Literal["train", "dev"]

_SPLIT_FILES: dict[Split, str] = {
    "train": "train_spider.json",
    "dev": "dev.json",
}


@dataclass(frozen=True, slots=True)
class SpiderExample:
    """A single Spider question record.

    Attributes
    ----------
    question_id
        Zero-based row index of the example within its source JSON file.
    db_id
        Spider database identifier (e.g. ``"concert_singer"``).
    question
        Natural-language question.
    query
        Raw gold SQL string.
    sql
        Spider's pre-parsed SQL dict (``select``, ``from``, ``where``, ...)
        kept as-is. Parsing into our own representation is deferred to the
        gold-link extractor.
    split
        Source split this record came from. Useful when callers concatenate
        train + dev and need to track origin.
    """

    question_id: int
    db_id: str
    question: str
    query: str
    sql: dict[str, Any]
    split: Split


def load_spider_questions(
    split: Split,
    config: Config | None = None,
) -> tuple[SpiderExample, ...]:
    """Load Spider examples for the given split.

    Parameters
    ----------
    split
        ``"train"`` (loads ``train_spider.json``, 7000 examples) or
        ``"dev"`` (loads ``dev.json``, 1034 examples).
    config
        Optional :class:`Config`. If ``None``, loads the default config from
        ``<repo_root>/config.yaml``. Pass an explicit config in tests or
        when working off a non-standard data location.

    Returns
    -------
    tuple[SpiderExample, ...]
        Examples in source-file order. ``question_id`` equals the tuple index.

    Raises
    ------
    ValueError
        If ``split`` is not ``"train"`` or ``"dev"``, or if any record has
        an empty ``db_id``, ``question``, or ``query`` field. The error
        message names the offending row index.
    FileNotFoundError
        If the expected JSON file is missing.
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be 'train' or 'dev', got {split!r}")

    cfg = config if config is not None else load_config()
    path = cfg.data.spider_dir / _SPLIT_FILES[split]

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    examples: list[SpiderExample] = []
    for i, ex in enumerate(raw):
        if not ex.get("db_id"):
            raise ValueError(f"empty db_id at {split} index {i}")
        if not ex.get("question"):
            raise ValueError(f"empty question at {split} index {i}")
        if not ex.get("query"):
            raise ValueError(f"empty query at {split} index {i}")
        examples.append(
            SpiderExample(
                question_id=i,
                db_id=ex["db_id"],
                question=ex["question"],
                query=ex["query"],
                sql=ex["sql"],
                split=split,
            )
        )
    return tuple(examples)
