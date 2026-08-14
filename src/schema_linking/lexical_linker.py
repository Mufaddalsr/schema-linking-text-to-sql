"""Raw lexical schema linker — substring, token-overlap, and fuzzy strategies.

This is the simplest baseline in the six-method comparison. Each schema
element (table or column) is checked independently against the question
with three strategies; the element is predicted if **any** strategy fires.
The "best" strategy (highest score, with substring > token > fuzzy
breaking ties at 100) is recorded for diagnostics.

Independence
------------
Tables and columns are evaluated independently. Predicted columns are
**not** filtered by predicted tables — that filtering belongs to a future
method variant if at all. This baseline reflects the rawest possible
string-matching signal.

Strategies
----------
1. **Substring** — normalised element name appears as a substring of the
   normalised question. Normalisation lowercases, replaces ``_`` with
   space, strips/collapses whitespace, and strips punctuation from the
   question (replaced with space).
2. **Token overlap** — every token of the element name (split on
   whitespace/underscore) appears in the question's token set (split on
   non-alphanumeric).
3. **Fuzzy** — ``rapidfuzz.fuzz.partial_ratio`` on the normalised pair
   clears ``fuzzy_threshold``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rapidfuzz import fuzz

from schema_linking.base import (
    ColumnScore,
    Prediction,
    TableScore,
)
from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema
from schema_linking.utils.config import load_config

_FUZZY_THRESHOLD_FALLBACK: int = 80

_PUNCT_TO_SPACE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RUN = re.compile(r"\s+")
_NAME_SPLIT = re.compile(r"[\s_]+")
_QUESTION_SPLIT = re.compile(r"[^a-z0-9]+")

_SUBSTRING = "substring"
_TOKEN = "token"
_FUZZY = "fuzzy"


def _normalise_name(name: str) -> str:
    return _WHITESPACE_RUN.sub(" ", name.lower().replace("_", " ")).strip()


def _normalise_question(question: str) -> str:
    return _WHITESPACE_RUN.sub(" ", _PUNCT_TO_SPACE.sub(" ", question.lower())).strip()


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(t for t in _NAME_SPLIT.split(name.lower()) if t)


def _question_tokens(question: str) -> frozenset[str]:
    return frozenset(t for t in _QUESTION_SPLIT.split(question.lower()) if t)


def _load_threshold_from_config() -> int:
    """Read ``linkers.lexical.fuzzy_threshold`` from config, with fallback."""
    try:
        return load_config().linkers.lexical.fuzzy_threshold
    except FileNotFoundError:
        return _FUZZY_THRESHOLD_FALLBACK


class LexicalLinker:
    """Substring + token-overlap + fuzzy lexical linker.

    Parameters
    ----------
    name
        Short identifier for this linker instance; used downstream when
        naming output files.
    fuzzy_threshold
        Minimum :func:`rapidfuzz.fuzz.partial_ratio` score (0–100) for the
        fuzzy strategy to fire. If ``None`` (the default), the value is
        read from ``config.yaml`` at ``linkers.lexical.fuzzy_threshold``;
        if config cannot be loaded, falls back to ``80``.
    """

    def __init__(
        self,
        name: str = "lexical",
        fuzzy_threshold: int | None = None,
    ) -> None:
        self.name = name
        self.fuzzy_threshold = (
            fuzzy_threshold
            if fuzzy_threshold is not None
            else _load_threshold_from_config()
        )

    def predict_one(self, example: SpiderExample, schema: Schema) -> Prediction:
        """Link one example against its schema.

        Returns
        -------
        Prediction
            Predicted tables and columns in original schema case, plus
            per-element ``table_scores`` / ``column_scores`` recording
            which strategy fired and at what score.
        """
        q_norm = _normalise_question(example.question)
        q_tokens = _question_tokens(example.question)

        tables: list[str] = []
        table_scores: list[TableScore] = []
        columns: list[tuple[str, str]] = []
        column_scores: list[ColumnScore] = []

        for table in schema.tables:
            hit = self._match(table.original_name, q_norm, q_tokens)
            if hit is not None:
                strategy, score = hit
                tables.append(table.original_name)
                table_scores.append(
                    TableScore(
                        table=table.original_name,
                        strategy=strategy,
                        score=score,
                    )
                )

            for col in table.columns:
                chit = self._match(col.original_name, q_norm, q_tokens)
                if chit is not None:
                    cstrategy, cscore = chit
                    columns.append((table.original_name, col.original_name))
                    column_scores.append(
                        ColumnScore(
                            table=table.original_name,
                            column=col.original_name,
                            strategy=cstrategy,
                            score=cscore,
                        )
                    )

        return Prediction(
            db_id=schema.db_id,
            tables=tuple(tables),
            columns=tuple(columns),
            table_scores=tuple(table_scores),
            column_scores=tuple(column_scores),
        )

    def predict_all(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, Prediction]:
        """Link a batch of examples, keyed by ``question_id``."""
        return {
            ex.question_id: self.predict_one(ex, schemas[ex.db_id])
            for ex in examples
        }

    def _match(
        self,
        element_name: str,
        q_norm: str,
        q_tokens: frozenset[str],
    ) -> tuple[str, float] | None:
        """Run all three strategies; return ``(strategy, score)`` of best hit.

        Precedence on ties (substring and token both score 100): substring
        wins, then token, then fuzzy.
        """
        name_norm = _normalise_name(element_name)
        if not name_norm:
            return None

        if name_norm in q_norm:
            return _SUBSTRING, 100.0

        tokens = _name_tokens(element_name)
        if tokens and all(t in q_tokens for t in tokens):
            return _TOKEN, 100.0

        score = fuzz.partial_ratio(name_norm, q_norm)
        if score >= self.fuzzy_threshold:
            return _FUZZY, float(score)

        return None
