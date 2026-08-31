"""Hyperparameter tuning helpers for the lexical and embedding linkers.

Exposes :func:`tune_fuzzy_threshold` (sweeps
:class:`~schema_linking.lexical_linker.LexicalLinker`'s
``fuzzy_threshold``) and :func:`tune_embedding` (grid-searches
:class:`~schema_linking.embedding_linker.EmbeddingLinker`'s four
top-k/threshold knobs).

Train-only policy
-----------------
Tuning runs on **train** examples and **sqlglot-derived Tier-2** gold
links. Spider dev is never touched here — that's the held-out evaluation
set for the thesis report. Callers are expected to pass an already-sliced
train subset (typically a 1000-example seeded sample, see
``notebooks/03a_lexical_tuning.ipynb`` and ``notebooks/
05_embedding_tuning.ipynb``).

Selection rule (lexical)
-------------------------
Best threshold = argmax of macro F1 at ``element_type``. The other
element-level F1 is recorded as a tie-breaker (descending) so two
thresholds tied on the primary metric resolve deterministically.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from schema_linking.base import from_predictions_to_dict
from schema_linking.data_loader import SpiderExample
from schema_linking.embedding_linker import select_top_k_above_threshold
from schema_linking.evaluator import evaluate
from schema_linking.lexical_linker import LexicalLinker
from schema_linking.schema_parser import Schema
from schema_linking.utils.embeddings import SchemaEncoder

DEFAULT_CANDIDATES: list[int] = [70, 75, 80, 85, 90, 95]

_TIER_NAME = "tier2"

# Embedding grid — LOCKED. 4 x 5 x 4 x 5 = 400 configurations. Do not
# expand without explicit approval.
TABLE_TOP_K_GRID: tuple[int, ...] = (1, 2, 3, 5)
TABLE_THRESHOLD_GRID: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)
COLUMN_TOP_K_GRID: tuple[int, ...] = (3, 5, 8, 12)
COLUMN_THRESHOLD_GRID: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)

_GOLD_TIER_TO_TIER_NAME: dict[str, str] = {
    "mentioned": "tier1",
    "all_sql_used": "tier2",
}


def tune_fuzzy_threshold(
    examples: list[SpiderExample],
    gold: dict[int, dict[str, Any]],
    schemas: dict[str, Schema],
    candidates: list[int] = DEFAULT_CANDIDATES,
    element_type: Literal["table", "column"] = "table",
) -> tuple[int, pd.DataFrame]:
    """Sweep ``fuzzy_threshold`` and pick the value with the best macro F1.

    Parameters
    ----------
    examples
        Train examples to score against. Pass an already-sliced subset
        — this function does not sample.
    gold
        Gold links in evaluator format: ``{qid: {"db_id", "tables",
        "columns"}}``. Tier-2 (sqlglot-derived) is the intended source.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id`` and every
        ``gold[qid]["db_id"]``.
    candidates
        Threshold values to try. Defaults to ``[70, 75, 80, 85, 90, 95]``.
    element_type
        Which level drives the choice. ``"table"`` (default) maximises
        macro F1 on tables and uses column F1 as the tie-breaker;
        ``"column"`` swaps them.

    Returns
    -------
    (best_threshold, sweep_table)
        ``best_threshold`` is an ``int`` from ``candidates``. ``sweep_table``
        is a ``pandas.DataFrame`` with one row per candidate and columns
        ``fuzzy_threshold``, ``table_precision``, ``table_recall``,
        ``table_f1``, ``column_precision``, ``column_recall``,
        ``column_f1``.
    """
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        linker = LexicalLinker(fuzzy_threshold=threshold)
        predictions = from_predictions_to_dict(
            linker.predict_all(examples, schemas)
        )
        result = evaluate(
            predictions=predictions,
            gold=gold,
            schemas=schemas,
            hardness={},
            method_name=f"lexical_thr{threshold}",
            tier_name=_TIER_NAME,
        )
        rows.append(
            {
                "fuzzy_threshold": threshold,
                **_macro_row(result.aggregated, "tables", "table"),
                **_macro_row(result.aggregated, "columns", "column"),
            }
        )

    sweep = pd.DataFrame(rows)
    best = _pick_best(sweep, element_type)
    return best, sweep


def _macro_row(
    aggregated: pd.DataFrame, level: str, prefix: str
) -> dict[str, float]:
    """Pull the macro/all P/R/F1 for one level out of the aggregated frame."""
    mask = (
        (aggregated["level"] == level)
        & (aggregated["aggregation"] == "macro")
        & (aggregated["hardness"] == "all")
    )
    row = aggregated[mask].iloc[0]
    return {
        f"{prefix}_precision": float(row["precision"]),
        f"{prefix}_recall": float(row["recall"]),
        f"{prefix}_f1": float(row["f1"]),
    }


def _pick_best(sweep: pd.DataFrame, element_type: Literal["table", "column"]) -> int:
    """Argmax on ``{element_type}_f1`` with the other level's F1 as tie-break."""
    primary = f"{element_type}_f1"
    secondary = "column_f1" if element_type == "table" else "table_f1"
    ranked = sweep.sort_values(
        [primary, secondary], ascending=[False, False], kind="stable"
    )
    return int(ranked.iloc[0]["fuzzy_threshold"])


def tune_embedding(
    examples: list[SpiderExample],
    gold: dict[int, dict[str, Any]],
    schemas: dict[str, Schema],
    encoder: SchemaEncoder,
    gold_tier: Literal["mentioned", "all_sql_used"],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Grid-search ``EmbeddingLinker``'s four knobs and pick the best mean F1.

    Sweeps the fixed 4 x 5 x 4 x 5 = 400-point grid (see
    ``TABLE_TOP_K_GRID`` / ``TABLE_THRESHOLD_GRID`` / ``COLUMN_TOP_K_GRID``
    / ``COLUMN_THRESHOLD_GRID`` — locked, do not expand without explicit
    approval). Kept fast by computing each expensive step exactly once and
    re-using it across all 400 configurations:

    1. Schema embeddings — ``encoder.encode_schema`` (already disk-cached
       across runs, see ``utils/embeddings.py``).
    2. Question embeddings — one batched ``encoder.encode`` call.
    3. The full cosine similarity matrix (every question against every
       table/column of its schema).

    Each of the 400 configs then only re-applies a top-k/threshold mask to
    the precomputed matrix (:func:`schema_linking.embedding_linker.
    select_top_k_above_threshold`) — no re-encoding, no re-computing dot
    products.

    Parameters
    ----------
    examples
        Train examples to score against. Pass an already-sliced subset —
        this function does not sample (see ``notebooks/
        05_embedding_tuning.ipynb`` for the 1000-example, seed=42 subset
        used for the thesis).
    gold
        Gold links in evaluator format: ``{qid: {"db_id", "tables",
        "columns"}}``.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id`` and every
        ``gold[qid]["db_id"]``.
    encoder
        A configured :class:`~schema_linking.utils.embeddings.
        SchemaEncoder`.
    gold_tier
        Which gold tier ``gold`` was extracted with — ``"all_sql_used"``
        (sqlglot Tier-2, the tuning default per the selection rule below)
        or ``"mentioned"`` (Tier-1). Only affects the ``tier`` label
        passed through to :func:`~schema_linking.evaluator.evaluate`; the
        gold set actually scored against is whatever ``gold`` contains.

    Returns
    -------
    (best_config, sweep_table)
        ``best_config`` is ``{"table_top_k", "table_threshold",
        "column_top_k", "column_threshold"}``. ``sweep_table`` is a
        400-row ``pandas.DataFrame`` with those four columns plus
        ``table_precision``, ``table_recall``, ``table_f1``,
        ``column_precision``, ``column_recall``, ``column_f1``,
        ``mean_f1``, ``mean_recall``.

    Selection rule (locked)
    ------------------------
    Argmax ``mean_f1 = (table_f1 + column_f1) / 2`` (macro, on the
    sqlglot Tier-2 train gold — see module docstring for why train uses
    Tier-2, not Taniguchi Tier-1). Ties broken by higher ``mean_recall``
    (recall-favouring for downstream use).
    """
    tier_name = _GOLD_TIER_TO_TIER_NAME[gold_tier]

    schema_index = encoder.encode_schema(schemas)
    question_vecs = encoder.encode([ex.question for ex in examples])

    raw: dict[int, dict[str, Any]] = {}
    for i, ex in enumerate(examples):
        index = schema_index[ex.db_id]
        q_vec = question_vecs[i]
        raw[ex.question_id] = {
            "db_id": ex.db_id,
            "table_names": index["table_names"],
            "table_scores": _cosine_scores(index["table_vectors"], q_vec),
            "column_names": index["column_names"],
            "column_scores": _cosine_scores(index["column_vectors"], q_vec),
        }

    rows: list[dict[str, Any]] = []
    for table_top_k in TABLE_TOP_K_GRID:
        for table_threshold in TABLE_THRESHOLD_GRID:
            for column_top_k in COLUMN_TOP_K_GRID:
                for column_threshold in COLUMN_THRESHOLD_GRID:
                    predictions = {
                        qid: _predict_from_raw(
                            entry,
                            table_top_k,
                            table_threshold,
                            column_top_k,
                            column_threshold,
                        )
                        for qid, entry in raw.items()
                    }
                    result = evaluate(
                        predictions=predictions,
                        gold=gold,
                        schemas=schemas,
                        hardness={},
                        method_name="embedding_tune",
                        tier_name=tier_name,
                    )
                    table_row = _macro_row(result.aggregated, "tables", "table")
                    column_row = _macro_row(result.aggregated, "columns", "column")
                    mean_f1 = (table_row["table_f1"] + column_row["column_f1"]) / 2.0
                    mean_recall = (
                        table_row["table_recall"] + column_row["column_recall"]
                    ) / 2.0
                    rows.append(
                        {
                            "table_top_k": table_top_k,
                            "table_threshold": table_threshold,
                            "column_top_k": column_top_k,
                            "column_threshold": column_threshold,
                            **table_row,
                            **column_row,
                            "mean_f1": mean_f1,
                            "mean_recall": mean_recall,
                        }
                    )

    sweep = pd.DataFrame(rows)
    best_config = _pick_best_embedding_config(sweep)
    return best_config, sweep


def _cosine_scores(vectors: np.ndarray, question_vec: np.ndarray) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.dot(vectors, question_vec)


def _predict_from_raw(
    entry: dict[str, Any],
    table_top_k: int,
    table_threshold: float,
    column_top_k: int,
    column_threshold: float,
) -> dict[str, Any]:
    """Apply one (top_k, threshold) config to a precomputed score entry."""
    tables, _ = select_top_k_above_threshold(
        entry["table_names"], entry["table_scores"], table_top_k, table_threshold
    )
    columns, _ = select_top_k_above_threshold(
        entry["column_names"], entry["column_scores"], column_top_k, column_threshold
    )
    return {
        "db_id": entry["db_id"],
        "tables": tables,
        "columns": [list(c) for c in columns],
    }


def _pick_best_embedding_config(sweep: pd.DataFrame) -> dict[str, float | int]:
    """Argmax ``mean_f1``, ties broken by higher ``mean_recall``."""
    ranked = sweep.sort_values(
        ["mean_f1", "mean_recall"], ascending=[False, False], kind="stable"
    )
    top = ranked.iloc[0]
    return {
        "table_top_k": int(top["table_top_k"]),
        "table_threshold": float(top["table_threshold"]),
        "column_top_k": int(top["column_top_k"]),
        "column_threshold": float(top["column_threshold"]),
    }
