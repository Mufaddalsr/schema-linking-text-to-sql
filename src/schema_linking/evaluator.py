"""Evaluation primitives and end-to-end evaluator for schema linking.

Sections
--------
1. Canonicalisation helpers (private, comparison-boundary only).
2. :func:`per_query_metrics` — TP / FP / FN, P / R / F1, SRR hit, counts.
3. :func:`fbeta` — generic F-beta; F6 is ``fbeta(p, r, 6.0)``.
4. :func:`filter_hallucinated` — partition a prediction into
   schema-consistent and schema-inconsistent halves.
5. :func:`evaluate` / :class:`EvalResult` — end-to-end runner that
   produces the locked aggregated and per-query DataFrames.
"""

from __future__ import annotations

import logging
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, TypeVar

import pandas as pd

from schema_linking.schema_parser import Schema

logger = logging.getLogger(__name__)

__all__ = [
    "per_query_metrics",
    "fbeta",
    "filter_hallucinated",
    "EvalResult",
    "evaluate",
    "write_results",
]

T = TypeVar("T", bound=Hashable)


# ---------------------------------------------------------------------------
# Section 1 — canonicalisation
# ---------------------------------------------------------------------------


def _normalise(identifier: str) -> str:
    """Single source of truth for identifier canonicalisation."""
    return identifier.strip().lower()


def _canonicalise_table(name: str) -> str:
    """Canonical form of a table identifier for matching.

    Lowercase + whitespace-stripped. Original case is preserved
    everywhere except inside set intersections.
    """
    return _normalise(name)


def _canonicalise_column(table: str, column: str) -> tuple[str, str]:
    """Canonical form of a ``(table, column)`` reference.

    Both halves are normalised independently. Returned as a tuple so
    column references remain hashable for set semantics.
    """
    return (_normalise(table), _normalise(column))


# ---------------------------------------------------------------------------
# Section 2 — per-query metrics
# ---------------------------------------------------------------------------


def per_query_metrics(
    predicted: AbstractSet[T],
    gold: AbstractSet[T],
) -> dict[str, float | int]:
    """Compute TP / FP / FN / P / R / F1 / SRR-hit for one query.

    Parameters
    ----------
    predicted
        Predicted set of items (table names, ``(table, column)`` tuples,
        or any other hashable). Callers are expected to canonicalise
        before passing — see :func:`_canonicalise_table` and
        :func:`_canonicalise_column`.
    gold
        Gold set with the same element type.

    Returns
    -------
    dict[str, float | int]
        Keys: ``tp``, ``fp``, ``fn`` (ints); ``precision``, ``recall``,
        ``f1`` (floats in ``[0, 1]``); ``srr_hit`` (bool — True iff
        ``gold ⊆ predicted``); ``predicted_count``, ``gold_count``
        (ints).

    Edge cases (locked, see ``docs/decisions.md``)
    ---------------------------------------------
    * ``predicted = ∅``, ``gold = ∅`` → ``P = R = F1 = 1.0``,
      ``srr_hit = True`` (vacuously correct).
    * ``predicted = ∅``, ``gold ≠ ∅`` → ``P = 1.0`` (no false
      positives), ``R = 0.0``, ``F1 = 0.0``, ``srr_hit = False``.
    * ``predicted ≠ ∅``, ``gold = ∅`` → ``P = 0.0``, ``R = 1.0`` (all
      zero golds recalled), ``F1 = 0.0``, ``srr_hit = True``
      (``∅ ⊆ predicted``).
    * Otherwise the textbook formulas.

    These rules differ from sklearn (which raises
    ``UndefinedMetricWarning`` on zero-denominator P or R and defaults
    to 0). In schema linking, "predicting nothing for a query with
    nothing to predict" is a correct answer, not undefined; treating
    it as 0 would systematically depress macro-averaged numbers.
    """
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    predicted_count = len(predicted)
    gold_count = len(gold)

    if not predicted and not gold:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "srr_hit": True,
            "predicted_count": 0,
            "gold_count": 0,
        }
    if not predicted:
        return {
            "tp": 0,
            "fp": 0,
            "fn": fn,
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "srr_hit": False,
            "predicted_count": 0,
            "gold_count": gold_count,
        }
    if not gold:
        return {
            "tp": 0,
            "fp": fp,
            "fn": 0,
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "srr_hit": True,
            "predicted_count": predicted_count,
            "gold_count": 0,
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "srr_hit": gold <= predicted,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
    }


# ---------------------------------------------------------------------------
# Section 3 — F-beta
# ---------------------------------------------------------------------------


def fbeta(precision: float, recall: float, beta: float) -> float:
    """Generic F-beta.

    ``F_β = (1 + β²) · P · R / (β² · P + R)``.

    Edge cases
    ----------
    * If ``P == 0`` and ``R == 0`` → returns ``0.0``.
    * If ``β² · P + R == 0`` (only possible when both P and R are zero
      for β > 0; can also fire with ``β = 0`` and ``R = 0``) → returns
      ``0.0``.

    Parameters
    ----------
    precision, recall
        In ``[0, 1]``.
    beta
        Recall-weighting parameter. ``β = 1`` reproduces F1 (harmonic
        mean). ``β = 6`` is the thesis-headline F6, weighting recall
        ≈ ``β² = 36`` times more than precision.
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


# ---------------------------------------------------------------------------
# Section 4 — hallucination detection
# ---------------------------------------------------------------------------


def filter_hallucinated(
    predicted: dict[str, Any],
    schema: Schema,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a prediction into schema-consistent and hallucinated halves.

    Parameters
    ----------
    predicted
        Prediction with shape ``{"db_id": str, "tables": list[str],
        "columns": list[[str, str]]}``. Original case is preserved in
        both output dicts; canonicalisation is used internally only to
        decide membership in the schema.
    schema
        The :class:`Schema` for ``predicted["db_id"]``.

    Returns
    -------
    (filtered, hallucinated)
        Two dicts with the same shape as ``predicted``. ``filtered``
        contains items the schema knows about; ``hallucinated`` is what
        was removed. ``hallucinated["tables"]`` collects unknown
        tables; ``hallucinated["columns"]`` collects column refs whose
        table is missing *or* whose column doesn't exist on that table.
    """
    known_tables: set[str] = {
        _canonicalise_table(t.original_name) for t in schema.tables
    }
    known_columns: set[tuple[str, str]] = {
        _canonicalise_column(t.original_name, c.original_name)
        for t in schema.tables
        for c in t.columns
    }

    kept_tables: list[str] = []
    bad_tables: list[str] = []
    for name in predicted.get("tables") or []:
        if _canonicalise_table(name) in known_tables:
            kept_tables.append(name)
        else:
            bad_tables.append(name)

    kept_cols: list[list[str]] = []
    bad_cols: list[list[str]] = []
    for pair in predicted.get("columns") or []:
        # Accept both list and tuple inputs; emit list for JSON-friendliness.
        t, c = pair[0], pair[1]
        if _canonicalise_column(t, c) in known_columns:
            kept_cols.append([t, c])
        else:
            bad_cols.append([t, c])

    db_id = predicted.get("db_id", schema.db_id)
    return (
        {"db_id": db_id, "tables": kept_tables, "columns": kept_cols},
        {"db_id": db_id, "tables": bad_tables, "columns": bad_cols},
    )


# ---------------------------------------------------------------------------
# Section 5 — end-to-end evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Output of :func:`evaluate`. See ``docs/decisions.md`` for the column
    schemas of both frames."""

    aggregated: pd.DataFrame
    per_query: pd.DataFrame


_AGG_COLUMNS: tuple[str, ...] = (
    "method",
    "tier",
    "level",
    "hardness",
    "aggregation",
    "n_queries",
    "precision",
    "recall",
    "f1",
    "f6",
    "srr",
    "hallucination_rate",
)


def evaluate(
    predictions: dict[int, dict[str, Any]],
    gold: dict[int, dict[str, Any]],
    schemas: dict[str, Schema],
    hardness: dict[int, str],
    method_name: str,
    tier_name: str,
) -> EvalResult:
    """Evaluate a prediction file against a gold file.

    See ``docs/decisions.md`` for the output contract (aggregated and
    per-query column schemas, aggregation rules, robustness rules).

    Parameters
    ----------
    predictions
        Mapping from ``question_id`` to ``{"db_id", "tables", "columns"}``.
    gold
        Same shape as ``predictions``. Gold's set of qids bounds what is
        evaluated; entries in predictions but not in gold are skipped
        with a warning.
    schemas
        Mapping from ``db_id`` to :class:`Schema`. Needed to detect
        hallucinated predictions.
    hardness
        Mapping from ``question_id`` to Spider hardness label
        (``easy`` / ``medium`` / ``hard`` / ``extra``). Missing
        entries are labelled ``"unknown"``.
    method_name, tier_name
        Pass-through to the ``method`` and ``tier`` columns of both
        output frames.

    Returns
    -------
    EvalResult
        ``aggregated`` (one row per slice) and ``per_query`` (one wide
        row per query).
    """
    pred_only = sorted(set(predictions) - set(gold))
    for qid in pred_only:
        logger.warning(
            "qid %s present in predictions but missing from gold — skipping "
            "(predictor-side mismatch)",
            qid,
        )
    gold_only = sorted(set(gold) - set(predictions))
    for qid in gold_only:
        logger.info(
            "qid %s present in gold but missing from predictions — "
            "treating as empty prediction",
            qid,
        )

    per_query_rows: list[dict[str, Any]] = []
    for qid in sorted(gold.keys()):
        gold_entry = gold[qid]
        db_id = gold_entry["db_id"]
        schema = schemas.get(db_id)
        if schema is None:
            logger.warning("no schema for db_id %r at qid %s — skipping", db_id, qid)
            continue

        pred_entry = predictions.get(qid)
        if pred_entry is None:
            pred_entry = {"db_id": db_id, "tables": [], "columns": []}

        n_pred_tables_raw = len(pred_entry.get("tables") or [])
        n_pred_cols_raw = len(pred_entry.get("columns") or [])
        n_pred_total_raw = n_pred_tables_raw + n_pred_cols_raw

        filtered, hallucinated = filter_hallucinated(pred_entry, schema)
        n_hall_tables = len(hallucinated["tables"])
        n_hall_cols = len(hallucinated["columns"])
        n_hall_total = n_hall_tables + n_hall_cols
        hall_rate = n_hall_total / n_pred_total_raw if n_pred_total_raw > 0 else 0.0

        pred_tables: set[str] = {_canonicalise_table(t) for t in filtered["tables"]}
        gold_tables: set[str] = {_canonicalise_table(t) for t in gold_entry["tables"]}
        pred_cols: set[tuple[str, str]] = {
            _canonicalise_column(t, c) for t, c in filtered["columns"]
        }
        gold_cols: set[tuple[str, str]] = {
            _canonicalise_column(t, c) for t, c in gold_entry["columns"]
        }

        table_m = per_query_metrics(pred_tables, gold_tables)
        col_m = per_query_metrics(pred_cols, gold_cols)

        per_query_rows.append(
            {
                "question_id": qid,
                "db_id": db_id,
                "method": method_name,
                "tier": tier_name,
                "hardness": hardness.get(qid, "unknown"),
                "table_tp": table_m["tp"],
                "table_fp": table_m["fp"],
                "table_fn": table_m["fn"],
                "table_precision": table_m["precision"],
                "table_recall": table_m["recall"],
                "table_f1": table_m["f1"],
                "table_srr_hit": table_m["srr_hit"],
                "table_predicted_count": table_m["predicted_count"],
                "table_gold_count": table_m["gold_count"],
                "table_hallucinated": n_hall_tables,
                "column_tp": col_m["tp"],
                "column_fp": col_m["fp"],
                "column_fn": col_m["fn"],
                "column_precision": col_m["precision"],
                "column_recall": col_m["recall"],
                "column_f1": col_m["f1"],
                "column_srr_hit": col_m["srr_hit"],
                "column_predicted_count": col_m["predicted_count"],
                "column_gold_count": col_m["gold_count"],
                "column_hallucinated": n_hall_cols,
                "hallucination_rate": hall_rate,
                "hallucinated_tables_list": ",".join(
                    str(x) for x in hallucinated["tables"]
                ),
                "hallucinated_columns_list": ",".join(
                    f"{t}.{c}" for t, c in hallucinated["columns"]
                ),
            }
        )

    per_query_df = pd.DataFrame(per_query_rows)
    if per_query_df.empty:
        return EvalResult(
            aggregated=pd.DataFrame(columns=list(_AGG_COLUMNS)),
            per_query=per_query_df,
        )

    aggregated_rows = _aggregate_rows(per_query_df, method_name, tier_name)
    aggregated_df = pd.DataFrame(aggregated_rows, columns=list(_AGG_COLUMNS))
    return EvalResult(aggregated=aggregated_df, per_query=per_query_df)


def _aggregate_rows(
    per_query: pd.DataFrame, method_name: str, tier_name: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hardness_order = ["easy", "medium", "hard", "extra", "unknown"]
    present = [h for h in hardness_order if h in per_query["hardness"].unique()]
    buckets = ["all", *present]

    for bucket in buckets:
        sub = (
            per_query if bucket == "all" else per_query[per_query["hardness"] == bucket]
        )
        if sub.empty:
            continue
        for level in ("tables", "columns"):
            for agg in ("macro", "micro"):
                rows.append(
                    _compute_row(sub, level, agg, method_name, tier_name, bucket)
                )
    return rows


def _compute_row(
    sub: pd.DataFrame,
    level: str,
    aggregation: str,
    method_name: str,
    tier_name: str,
    hardness_bucket: str,
) -> dict[str, Any]:
    prefix = "table" if level == "tables" else "column"
    n_queries = int(len(sub))

    if aggregation == "macro":
        precision = float(sub[f"{prefix}_precision"].mean())
        recall = float(sub[f"{prefix}_recall"].mean())
        f1 = float(sub[f"{prefix}_f1"].mean())
    else:
        tp = int(sub[f"{prefix}_tp"].sum())
        fp = int(sub[f"{prefix}_fp"].sum())
        fn = int(sub[f"{prefix}_fn"].sum())
        if tp + fp == 0:
            precision = 1.0  # vacuous: no predictions to be wrong about
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 1.0  # vacuous: nothing gold to miss
        else:
            recall = tp / (tp + fn)
        denom_f1 = 2 * tp + fp + fn
        f1 = (2 * tp / denom_f1) if denom_f1 > 0 else 1.0

    return {
        "method": method_name,
        "tier": tier_name,
        "level": level,
        "hardness": hardness_bucket,
        "aggregation": aggregation,
        "n_queries": n_queries,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f6": fbeta(precision, recall, 6.0),
        "srr": float(sub[f"{prefix}_srr_hit"].mean()),
        "hallucination_rate": float(sub["hallucination_rate"].mean()),
    }


def write_results(
    eval_results: list[EvalResult],
    aggregated_path: Path,
    per_query_path: Path,
) -> None:
    """Concatenate many :class:`EvalResult` and write two CSVs.

    The aggregated frames are stacked along axis 0 with
    ``ignore_index=True`` and written to ``aggregated_path``; same for
    the per-query frames at ``per_query_path``. Parent directories are
    created if missing.

    Empty ``eval_results`` writes an empty CSV with the locked
    aggregated header schema and an empty per-query file (the per-query
    column set is determined by the data, so there's no header to
    pre-write).

    Parameters
    ----------
    eval_results
        Typically one entry per ``(method, tier)`` evaluated.
    aggregated_path, per_query_path
        Output CSV paths.
    """
    aggregated_path = Path(aggregated_path)
    per_query_path = Path(per_query_path)
    aggregated_path.parent.mkdir(parents=True, exist_ok=True)
    per_query_path.parent.mkdir(parents=True, exist_ok=True)

    if eval_results:
        aggregated = pd.concat([r.aggregated for r in eval_results], ignore_index=True)
        per_query = pd.concat([r.per_query for r in eval_results], ignore_index=True)
    else:
        aggregated = pd.DataFrame(columns=list(_AGG_COLUMNS))
        per_query = pd.DataFrame()

    aggregated.to_csv(aggregated_path, index=False)
    per_query.to_csv(per_query_path, index=False)
