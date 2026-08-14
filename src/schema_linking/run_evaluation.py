"""Notebook / ``python -m`` runner for per-method evaluation and paired tests.

This module wires together :mod:`schema_linking.evaluator` and
:mod:`schema_linking.utils.statistical` into the two CSVs that back the
thesis's results chapter, and produces per-method-pair significance tests
on top of them. Like :mod:`schema_linking.run_linker`, this is
intentionally not a true CLI — no ``argparse``, no ``__main__`` hand-off.

Two result tables
------------------
* ``outputs/results/main_results.csv`` — one row per ``(method, tier,
  level)``: the macro, ``hardness="all"`` headline metrics (precision,
  recall, f1, f6, srr, hallucination_rate). Six methods x 2 tiers x 2
  levels = 24 rows at the full thesis scope; built incrementally by
  :func:`evaluate_method`, one method/tier at a time.
* ``outputs/results/main_per_query.csv`` — the full per-query diagnostic
  rows (every method, every tier) that :func:`compare_methods_mcnemar`
  reads to run paired McNemar tests between two methods.

Both are written via :func:`append_results`, which replaces same-key rows
instead of duplicating them — re-running :func:`evaluate_method` for a
``(method, tier)`` pair that's already on disk is therefore idempotent.

Gold source per tier
---------------------
``tier="mentioned"`` evaluates against Taniguchi's human-annotated dev
gold (``data/processed/gold_links_dev_mentioned.json``, tagged
``"tier1"``); ``tier="all_sql_used"`` evaluates against the sqlglot
Tier-2 dev gold (``gold_links_dev_all_sql.json``, tagged ``"tier2"``).
Headline numbers are dev-only — train is reserved for tuning
(``utils/tuning.py``), never for reported results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from schema_linking.data_loader import load_spider_questions
from schema_linking.evaluator import EvalResult, evaluate
from schema_linking.schema_parser import load_schemas
from schema_linking.utils.config import load_config
from schema_linking.utils.difficulty import difficulty_for_examples
from schema_linking.utils.statistical import mcnemar_srr

logger = logging.getLogger(__name__)

_MAIN_RESULTS_FILENAME: str = "main_results.csv"
_MAIN_PER_QUERY_FILENAME: str = "main_per_query.csv"

_GOLD_FILENAME_BY_TIER: dict[str, str] = {
    "mentioned": "gold_links_dev_mentioned.json",
    "all_sql_used": "gold_links_dev_all_sql.json",
}
_TIER_NAME_BY_GOLD_TIER: dict[str, str] = {
    "mentioned": "tier1",
    "all_sql_used": "tier2",
}


def append_results(
    new_rows: pd.DataFrame,
    path: Path,
    key_cols: list[str],
) -> None:
    """Append ``new_rows`` to the CSV at ``path``, replacing same-key rows.

    Any existing row whose ``key_cols`` values match a row in
    ``new_rows`` is dropped before the new rows are written — re-running
    the producer for the same key (e.g. the same ``(method, tier)``) is
    therefore idempotent rather than accumulating duplicate rows.

    Parameters
    ----------
    new_rows
        Rows to add (or use to replace existing rows of the same key).
    path
        CSV path. Created (with parent directories) if it doesn't exist.
    key_cols
        Column names identifying a row's identity, e.g.
        ``["method", "tier", "level"]`` for ``main_results.csv`` or
        ``["method", "tier"]`` for ``main_per_query.csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        existing = pd.read_csv(path)
        new_keys = set(new_rows[key_cols].itertuples(index=False, name=None))
        existing_keys = existing[key_cols].itertuples(index=False, name=None)
        keep_mask = [key not in new_keys for key in existing_keys]
        combined = pd.concat(
            [existing[keep_mask], new_rows], ignore_index=True
        )
    else:
        combined = new_rows.reset_index(drop=True)

    combined.to_csv(path, index=False)


def evaluate_method(
    method_name: str,
    predictions_path: Path | None = None,
    *,
    tier: Literal["mentioned", "all_sql_used"],
    results_path: Path | None = None,
    per_query_path: Path | None = None,
) -> EvalResult:
    """Evaluate one method's dev predictions; append to the main result tables.

    Parameters
    ----------
    method_name
        Short identifier, e.g. ``"lexical"`` / ``"embedding"``. Used as
        the ``method`` column in both output CSVs and, if
        ``predictions_path`` is omitted, to locate the predictions file
        by convention.
    predictions_path
        Path to the method's dev predictions JSON (canonical
        ``{qid: {db_id, tables, columns}}`` shape). Defaults to
        ``<predictions_dir>/{method_name}_dev.json``.
    tier
        Which gold set to score against: ``"mentioned"`` (Taniguchi
        Tier-1, canonical on dev) or ``"all_sql_used"`` (sqlglot Tier-2).
    results_path, per_query_path
        Override the default ``main_results.csv`` / ``main_per_query.csv``
        locations (mainly for tests).

    Returns
    -------
    EvalResult
        The full ``evaluate()`` output, in case a caller wants more than
        the headline rows written to ``main_results.csv``.

    Side effects
    ------------
    Appends 2 rows (tables, columns; macro, hardness="all") to
    ``main_results.csv`` and one row per dev query to
    ``main_per_query.csv``, replacing any existing rows for this
    ``(method, tier)`` — see :func:`append_results`.
    """
    config = load_config()

    if predictions_path is None:
        predictions_path = config.outputs.predictions_dir / f"{method_name}_dev.json"
    gold_path = config.data.processed_dir / _GOLD_FILENAME_BY_TIER[tier]
    tier_name = _TIER_NAME_BY_GOLD_TIER[tier]

    predictions = _load_qid_json(predictions_path)
    gold = _load_qid_json(gold_path)
    schemas = load_schemas()
    hardness = difficulty_for_examples(load_spider_questions("dev"))

    result = evaluate(
        predictions=predictions,
        gold=gold,
        schemas=schemas,
        hardness=hardness,
        method_name=method_name,
        tier_name=tier_name,
    )

    results_path = (
        Path(results_path)
        if results_path is not None
        else config.outputs.results_dir / _MAIN_RESULTS_FILENAME
    )
    per_query_path = (
        Path(per_query_path)
        if per_query_path is not None
        else config.outputs.results_dir / _MAIN_PER_QUERY_FILENAME
    )

    headline = result.aggregated[
        (result.aggregated["aggregation"] == "macro")
        & (result.aggregated["hardness"] == "all")
    ]
    append_results(headline, results_path, key_cols=["method", "tier", "level"])
    append_results(result.per_query, per_query_path, key_cols=["method", "tier"])

    logger.info(
        "evaluate_method: method=%s tier=%s n_queries=%d -> %s, %s",
        method_name,
        tier_name,
        len(result.per_query),
        results_path,
        per_query_path,
    )
    return result


def compare_methods_mcnemar(
    method_a: str,
    method_b: str,
    per_query_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Paired McNemar's test between two methods, for every (tier, element_type).

    Reads ``main_per_query.csv`` (built up by :func:`evaluate_method`),
    filters to ``method_a`` / ``method_b`` for each tier present in the
    file, and runs :func:`~schema_linking.utils.statistical.mcnemar_srr`
    on tables and on columns.

    Parameters
    ----------
    method_a, method_b
        Method names as they appear in the ``method`` column of
        ``per_query_path``.
    per_query_path
        Defaults to ``<results_dir>/main_per_query.csv``.
    output_path
        Defaults to ``<results_dir>/mcnemar_{method_a}_vs_{method_b}.csv``.

    Returns
    -------
    pd.DataFrame
        One row per ``(tier, element_type)`` with columns ``method_a``,
        ``method_b``, ``tier``, ``element_type``, ``n_a_only``,
        ``n_b_only``, ``n_both``, ``n_neither``, ``statistic``,
        ``p_value``. A ``(tier, element_type)`` combo is skipped (with a
        WARNING) if either method has no rows for that tier.
    """
    config = load_config()
    if per_query_path is None:
        per_query_path = config.outputs.results_dir / _MAIN_PER_QUERY_FILENAME
    if output_path is None:
        output_path = (
            config.outputs.results_dir / f"mcnemar_{method_a}_vs_{method_b}.csv"
        )

    per_query = pd.read_csv(Path(per_query_path))

    rows: list[dict[str, Any]] = []
    for tier_name in sorted(per_query["tier"].unique()):
        tier_df = per_query[per_query["tier"] == tier_name]
        a_df = tier_df[tier_df["method"] == method_a]
        b_df = tier_df[tier_df["method"] == method_b]
        if a_df.empty or b_df.empty:
            logger.warning(
                "compare_methods_mcnemar: tier=%s missing rows (a=%d, b=%d) "
                "— skipping",
                tier_name,
                len(a_df),
                len(b_df),
            )
            continue

        for element_type in ("table", "column"):
            stats = mcnemar_srr(a_df, b_df, element_type)
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "tier": tier_name,
                    "element_type": element_type,
                    "n_a_only": stats["n_a_only"],
                    "n_b_only": stats["n_b_only"],
                    "n_both": stats["n_both"],
                    "n_neither": stats["n_neither"],
                    "statistic": stats["statistic"],
                    "p_value": stats["p_value"],
                }
            )

    result_df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    logger.info(
        "compare_methods_mcnemar: %s vs %s -> %d rows -> %s",
        method_a,
        method_b,
        len(result_df),
        output_path,
    )
    return result_df


def _load_qid_json(path: Path) -> dict[int, dict[str, Any]]:
    """Read a ``{qid_str: {...}}`` JSON file, coercing keys back to int."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(qid): entry for qid, entry in raw.items()}
