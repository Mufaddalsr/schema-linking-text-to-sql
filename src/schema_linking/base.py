"""Shared types for schema linkers and predictions.

This module defines the contract every Week-4+ linker must satisfy and the
canonical on-disk shape for predictions.

Two-shape rule
--------------
:class:`Prediction` carries both **canonical** fields (``db_id``, ``tables``,
``columns``) and **diagnostic** fields (``scores``, ``strategy``,
``token_cost``). Only the canonical fields are written to JSON; diagnostics
stay in memory for error analysis and never touch the saved file.

Hashable predictions
--------------------
Sequence fields are stored as tuples (not lists) so that :class:`Prediction`
remains hashable. The :func:`to_json` helper converts tuples back to lists
to match the cross-tier JSON shape:
``{db_id, tables: [str], columns: [[table, col]]}``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema


@dataclass(frozen=True, slots=True)
class TableScore:
    """Diagnostic score for one predicted table.

    Attributes
    ----------
    table
        Table name in original schema case.
    strategy
        Short identifier of which internal branch fired (e.g.
        ``"substring"``, ``"token"``, ``"fuzzy"``, ``"exact"``).
    score
        Strategy-specific confidence in ``[0, 100]``.
    """

    table: str
    strategy: str
    score: float


@dataclass(frozen=True, slots=True)
class ColumnScore:
    """Diagnostic score for one predicted column.

    Attributes
    ----------
    table
        Parent table name in original schema case.
    column
        Column name in original schema case.
    strategy
        See :class:`TableScore`.
    score
        See :class:`TableScore`.
    """

    table: str
    column: str
    strategy: str
    score: float


@dataclass(frozen=True, slots=True)
class Prediction:
    """A single linker prediction for one question.

    Attributes
    ----------
    db_id
        Spider database identifier (e.g. ``"concert_singer"``).
    tables
        Predicted tables, stored in original schema case (Spider's
        ``table_names_original``). Tuple — not list — so the record is
        hashable.
    columns
        Predicted columns as ``(table, column)`` pairs, both in original
        schema case. Tuple of tuples for hashability.
    table_scores
        Optional per-predicted-table diagnostic records. ``None`` when the
        linker does not produce scores. Not written to the saved JSON.
    column_scores
        Optional per-predicted-column diagnostic records. Not written to
        the saved JSON.
    token_cost
        Optional token count consumed by an LLM linker for this question.
        Diagnostic only.
    extra
        Optional method-specific diagnostic bundle (e.g. the LLM forward
        linker's per-sample parse/aggregation stats). Diagnostic only, never
        written to the saved JSON.
    """

    db_id: str
    tables: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]
    table_scores: tuple[TableScore, ...] | None = None
    column_scores: tuple[ColumnScore, ...] | None = None
    token_cost: int | None = None
    extra: dict[str, Any] | None = None


@runtime_checkable
class Linker(Protocol):
    """Contract for a schema linker.

    Implementations may be lexical, embedding-based, LLM-based, or
    graph-based. They must produce predictions in original schema case;
    the evaluator handles case canonicalisation at compare time.
    """

    def predict_one(
        self, example: SpiderExample, schema: Schema
    ) -> Prediction:
        """Link a single example against its schema."""
        ...

    def predict_all(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, Prediction]:
        """Link a batch of examples, keyed by ``question_id``."""
        ...


def to_json(prediction: Prediction) -> dict[str, Any]:
    """Convert a :class:`Prediction` to its canonical JSON dict.

    Parameters
    ----------
    prediction
        The prediction to serialise.

    Returns
    -------
    dict
        ``{"db_id": str, "tables": list[str], "columns": list[list[str]]}``.
        Diagnostic fields (``scores``, ``strategy``, ``token_cost``) are
        deliberately omitted to keep saved files small and tier-agnostic.
    """
    return {
        "db_id": prediction.db_id,
        "tables": list(prediction.tables),
        "columns": [list(pair) for pair in prediction.columns],
    }


def from_predictions_to_dict(
    predictions: Mapping[int, Prediction],
) -> dict[int, dict[str, Any]]:
    """Build the file-ready ``{qid: {db_id, tables, columns}}`` mapping.

    Parameters
    ----------
    predictions
        Mapping of ``question_id`` to :class:`Prediction`.

    Returns
    -------
    dict
        Same keys as ``predictions``; each value is the canonical
        :func:`to_json` dict. Suitable for direct ``json.dump``.
    """
    return {qid: to_json(pred) for qid, pred in predictions.items()}
