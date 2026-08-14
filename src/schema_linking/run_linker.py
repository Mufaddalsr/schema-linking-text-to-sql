"""Notebook / ``python -m`` runner for the lexical, embedding, and LLM forward linkers.

This module wires together the data loader, schema parser, linkers, and
JSON/JSONL writers. It is intentionally not a true CLI — there is no
``argparse``, no ``__main__`` hand-off. Call :func:`run_lexical_on_dev`,
:func:`run_embedding_on_dev`, or :func:`run_llm_forward_on_dev` from a
notebook or ``python -c "from schema_linking.run_linker import
run_lexical_on_dev; run_lexical_on_dev()"``.

Output shape
------------
JSON written by :func:`run_lexical` / :func:`run_embedding` /
:func:`run_llm_forward` matches the gold-link convention (see
:mod:`schema_linking.taniguchi_loader`): stringified ``qid`` keys, entries
sorted, ``indent=2``, trailing newline. This keeps prediction and gold
files interchangeable as inputs to the evaluator. :func:`run_embedding` and
:func:`run_llm_forward` additionally write a per-query raw JSONL file — see
their docstrings.

LLM forward cost discipline
-----------------------------
:func:`run_llm_forward_on_dev` is a real, paid API run over the full 1034-
example Spider dev set (~$10 estimated at Haiku 4.5 rates with prompt
caching). Call :func:`dry_run_llm_forward_cost` first — it's a real dry run
on the first few examples (not a token-count heuristic), so its cost
projection is accurate — and confirm the projection before calling
:func:`run_llm_forward_on_dev`. Both share one dedicated cost-capped log
(``outputs/logs/llm_calls_dev_run.jsonl``), so the dry run's small spend
counts toward the same guard as the full run.

Same discipline applies to :func:`run_graph_on_dev` (Method G) — call
:func:`dry_run_graph_cost` first and confirm before the full run; both
share ``outputs/logs/llm_calls_dev_run_graph.jsonl``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from schema_linking.base import from_predictions_to_dict
from schema_linking.data_loader import SpiderExample, load_spider_questions
from schema_linking.embedding_linker import EmbeddingLinker, select_top_k_above_threshold
from schema_linking.graph_linker import GraphLinker
from schema_linking.lexical_linker import LexicalLinker
from schema_linking.llm_linker import BidirectionalLinker, LLMBackwardLinker, LLMForwardLinker
from schema_linking.schema_parser import Schema, load_schemas
from schema_linking.utils.config import load_config
from schema_linking.utils.embeddings import SchemaEncoder
from schema_linking.utils.llm_client import LLMClient
from schema_linking.utils.prompts import BACKWARD_V1, FORWARD_V1, GRAPH_ENDPOINT_V1, render_schema_block

logger = logging.getLogger(__name__)

_DEV_OUTPUT_FILENAME: str = "lexical_dev.json"
_EMBEDDING_DEV_OUTPUT_FILENAME: str = "embedding_dev.json"
_EMBEDDING_DEV_SCORES_FILENAME: str = "embedding_dev_scores.jsonl"

_LLM_FORWARD_DEV_OUTPUT_FILENAME: str = "llm_forward_dev.json"
_LLM_FORWARD_DEV_SAMPLES_FILENAME: str = "llm_forward_dev_samples.jsonl"
_LLM_FORWARD_DEV_RUN_LOG_FILENAME: str = "llm_calls_dev_run.jsonl"
_LLM_FORWARD_MODEL: str = "claude-haiku-4-5-20251001"
_LLM_FORWARD_K_SAMPLES: int = 3
_LLM_FORWARD_TEMPERATURE: float = 0.3
_LLM_FORWARD_MAX_TOKENS: int = 1024
_LLM_FORWARD_COST_CAP_USD: float = 15.0
_PARSE_FAILURE_RATE_TARGET: float = 0.02

_LLM_BACKWARD_DEV_OUTPUT_FILENAME: str = "llm_backward_dev.json"
_LLM_BACKWARD_DEV_SQL_FILENAME: str = "llm_backward_dev_sql.jsonl"
_LLM_BACKWARD_DEV_RUN_LOG_FILENAME: str = "llm_calls_dev_run_backward.jsonl"
_LLM_BACKWARD_MODEL: str = "claude-haiku-4-5-20251001"
_LLM_BACKWARD_TEMPERATURE: float = 0.0
_LLM_BACKWARD_MAX_TOKENS: int = 1024
_LLM_BACKWARD_COST_CAP_USD: float = 5.0

_LLM_BIDIRECTIONAL_DEV_OUTPUT_FILENAME: str = "llm_bidirectional_dev.json"

_GRAPH_DEV_OUTPUT_FILENAME: str = "graph_dev.json"
_GRAPH_DEV_TRACES_FILENAME: str = "graph_dev_traces.jsonl"
_GRAPH_DEV_RUN_LOG_FILENAME: str = "llm_calls_dev_run_graph.jsonl"
_GRAPH_MODEL: str = "claude-haiku-4-5-20251001"
_GRAPH_TEMPERATURE: float = 0.0
_GRAPH_MAX_TOKENS: int = 1024
_GRAPH_COST_CAP_USD: float = 10.0

_COST_REPORT_FILENAME: str = "llm_cost_report.csv"
_COST_LOG_FILENAME_BY_PHASE: dict[str, str] = {
    "prompt_iteration": "llm_calls_prompt_iteration.jsonl",
    "dev_run": _LLM_FORWARD_DEV_RUN_LOG_FILENAME,
    "backward_prompt_iteration": "llm_calls_prompt_iteration.jsonl",
    "backward_dev_run": _LLM_BACKWARD_DEV_RUN_LOG_FILENAME,
    "graph_dev_run": _GRAPH_DEV_RUN_LOG_FILENAME,
}


def run_lexical(
    linker: LexicalLinker,
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    output_path: Path,
) -> None:
    """Predict for every example, save canonical JSON, log basic stats.

    Parameters
    ----------
    linker
        Configured :class:`LexicalLinker` (already holds the tuned
        ``fuzzy_threshold``).
    examples
        Examples to link. Order doesn't matter — the output is keyed
        by ``question_id``.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    output_path
        Where to write the JSON. Parent directories are created.

    Side effects
    ------------
    Writes JSON to ``output_path`` (overwrites if it exists). Logs at
    INFO: example count, avg predicted tables / columns per query, and
    total wall-clock runtime in seconds.
    """
    start = time.perf_counter()
    predictions = linker.predict_all(examples, schemas)
    file_ready = from_predictions_to_dict(predictions)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        str(qid): entry for qid, entry in sorted(file_ready.items())
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, sort_keys=False)
        f.write("\n")

    elapsed = time.perf_counter() - start
    n = len(examples)
    if n > 0:
        avg_tables = sum(len(p.tables) for p in predictions.values()) / n
        avg_columns = sum(len(p.columns) for p in predictions.values()) / n
    else:
        avg_tables = 0.0
        avg_columns = 0.0
    logger.info(
        "lexical: linker=%s threshold=%d n=%d avg_tables=%.2f avg_columns=%.2f "
        "runtime=%.2fs -> %s",
        linker.name,
        linker.fuzzy_threshold,
        n,
        avg_tables,
        avg_columns,
        elapsed,
        output_path,
    )


def run_lexical_on_dev() -> None:
    """Run the lexical linker on Spider dev and save predictions.

    Loads dev examples, the schema collection, and the configured
    ``fuzzy_threshold`` via :func:`load_config`. Writes
    ``<predictions_dir>/lexical_dev.json``.
    """
    config = load_config()
    examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    linker = LexicalLinker()
    output_path = config.outputs.predictions_dir / _DEV_OUTPUT_FILENAME
    run_lexical(linker, examples, schemas, output_path)


def run_embedding(
    linker: EmbeddingLinker,
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    output_path: Path,
    scores_path: Path,
) -> None:
    """Predict for every example; save canonical JSON plus per-query raw scores.

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.embedding_linker.
        EmbeddingLinker` (already holds the tuned top-k/threshold knobs).
    examples
        Examples to link. Order doesn't matter — both outputs are keyed
        by ``question_id``.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    output_path
        Where to write the canonical predictions JSON. Parent
        directories are created.
    scores_path
        Where to write the per-query raw cosine-score JSONL. Parent
        directories are created.

    Side effects
    ------------
    Writes canonical predictions JSON to ``output_path`` and per-query
    raw cosine scores to ``scores_path`` (both overwritten if they
    exist). Logs at INFO: example count, linker config, avg predicted
    tables / columns per query, and total wall-clock runtime.

    Why a scores file
    ------------------
    ``output_path`` only records what was *predicted* (post top-k /
    threshold). Week 9 error analysis needs the *raw* score of every
    schema element regardless of whether it was predicted — e.g. to
    characterise semantic-drift (SD) errors by comparing a missed gold
    element's cosine score against a correct prediction's. Each line of
    ``scores_path`` is ``{"question_id", "db_id", "question",
    "table_scores": {table_name: score}, "column_scores":
    {"table.column": score}}`` covering every table/column in that
    query's schema.

    Both outputs are derived from a single call to
    :meth:`EmbeddingLinker.similarity_matrix`, which batch-encodes every
    question exactly once — not once per output file.
    """
    start = time.perf_counter()
    matrix = linker.similarity_matrix(examples, schemas)
    db_id_by_qid = {ex.question_id: ex.db_id for ex in examples}

    predictions: dict[int, dict[str, Any]] = {}
    score_records: list[dict[str, Any]] = []
    for qid in sorted(matrix):
        entry = matrix[qid]
        db_id = db_id_by_qid[qid]

        tables, _ = select_top_k_above_threshold(
            entry["table_names"],
            entry["table_scores"],
            linker.table_top_k,
            linker.table_threshold,
        )
        columns, _ = select_top_k_above_threshold(
            entry["column_names"],
            entry["column_scores"],
            linker.column_top_k,
            linker.column_threshold,
        )
        predictions[qid] = {
            "db_id": db_id,
            "tables": list(tables),
            "columns": [list(c) for c in columns],
        }

        score_records.append(
            {
                "question_id": qid,
                "db_id": db_id,
                "question": entry["question"],
                "table_scores": {
                    name: float(score)
                    for name, score in zip(entry["table_names"], entry["table_scores"])
                },
                "column_scores": {
                    f"{t}.{c}": float(score)
                    for (t, c), score in zip(entry["column_names"], entry["column_scores"])
                },
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(qid): entry for qid, entry in sorted(predictions.items())}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, sort_keys=False)
        f.write("\n")

    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8") as f:
        for record in score_records:
            f.write(json.dumps(record) + "\n")

    elapsed = time.perf_counter() - start
    n = len(examples)
    if n > 0:
        avg_tables = sum(len(p["tables"]) for p in predictions.values()) / n
        avg_columns = sum(len(p["columns"]) for p in predictions.values()) / n
    else:
        avg_tables = 0.0
        avg_columns = 0.0
    logger.info(
        "embedding: linker=%s table_top_k=%d table_threshold=%.2f "
        "column_top_k=%d column_threshold=%.2f n=%d avg_tables=%.2f "
        "avg_columns=%.2f runtime=%.2fs -> %s (+ %s)",
        linker.name,
        linker.table_top_k,
        linker.table_threshold,
        linker.column_top_k,
        linker.column_threshold,
        n,
        avg_tables,
        avg_columns,
        elapsed,
        output_path,
        scores_path,
    )


def run_embedding_on_dev() -> None:
    """Run the embedding linker on Spider dev; save predictions and scores.

    Loads dev examples, the schema collection, and the tuned
    top-k/threshold config via :func:`load_config`. Writes
    ``<predictions_dir>/embedding_dev.json`` and
    ``<predictions_dir>/embedding_dev_scores.jsonl``.

    Raises
    ------
    ValueError
        If ``config.yaml`` has no ``embedding.tuned`` section — run
        ``notebooks/05_embedding_tuning.ipynb`` first.
    """
    config = load_config()
    tuned = config.embedding.tuned
    if tuned is None:
        raise ValueError(
            "config.yaml has no embedding.tuned section — run "
            "notebooks/05_embedding_tuning.ipynb first"
        )

    examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    encoder = SchemaEncoder(
        model_name=config.embedding.model_name,
        revision=config.embedding.revision,
        cache_dir=config.embedding.cache_dir,
    )
    linker = EmbeddingLinker(
        encoder=encoder,
        schemas=schemas,
        table_top_k=tuned.table_top_k,
        table_threshold=tuned.table_threshold,
        column_top_k=tuned.column_top_k,
        column_threshold=tuned.column_threshold,
    )
    output_path = config.outputs.predictions_dir / _EMBEDDING_DEV_OUTPUT_FILENAME
    scores_path = config.outputs.predictions_dir / _EMBEDDING_DEV_SCORES_FILENAME
    run_embedding(linker, examples, schemas, output_path, scores_path)


def _load_few_shot_with_schema_blocks(
    config: Any, schemas: dict[str, Schema]
) -> list[dict[str, Any]]:
    """Load ``data/processed/few_shot_examples.json`` and enrich each entry
    with a rendered ``schema_block`` — ``LLMForwardLinker`` requires it (see
    ``llm_linker.py`` docstring: it never sees the few-shot examples' own
    schemas, only the current question's)."""
    few_shot_path = config.data.processed_dir / "few_shot_examples.json"
    few_shot = json.loads(few_shot_path.read_text(encoding="utf-8"))
    for ex in few_shot:
        ex["schema_block"] = render_schema_block(schemas[ex["db_id"]])
    return few_shot


def _build_llm_forward_linker(
    config: Any, schemas: dict[str, Schema], log_path: Path
) -> LLMForwardLinker:
    """Wire up the real, locked ``forward_v1`` LLM linker for the dev run."""
    llm_client = LLMClient(
        model=_LLM_FORWARD_MODEL,
        temperature=_LLM_FORWARD_TEMPERATURE,
        max_tokens=_LLM_FORWARD_MAX_TOKENS,
        log_path=log_path,
        cost_cap_usd=_LLM_FORWARD_COST_CAP_USD,
    )
    few_shot = _load_few_shot_with_schema_blocks(config, schemas)
    return LLMForwardLinker(
        llm_client=llm_client,
        prompt=FORWARD_V1,
        few_shot=few_shot,
        k_samples=_LLM_FORWARD_K_SAMPLES,
        aggregation="union",
        extra_metadata={"phase": "dev_run", "prompt_version": FORWARD_V1.version},
    )


def estimate_llm_forward_cost(
    linker: LLMForwardLinker,
    dry_run_examples: list[SpiderExample],
    schemas: dict[str, Schema],
    full_dev_size: int,
) -> dict[str, Any]:
    """Run ``linker`` for real on ``dry_run_examples``; project the full-dev cost.

    Uses real measured token counts/costs from the actual calls, not a
    text-length heuristic — an accurate cost projection needs real numbers,
    including whatever prompt-cache hit rate the dry run happens to see.

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.llm_linker.LLMForwardLinker`.
    dry_run_examples
        A small subset of dev examples to actually call the API on.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    full_dev_size
        Total example count to project the cost onto (e.g. all of dev).

    Returns
    -------
    dict
        ``n_dry_run``, ``dry_run_cost_usd``, ``avg_cost_per_query_usd``,
        ``full_dev_size``, ``projected_total_cost_usd``.
    """
    predictions = {
        ex.question_id: linker.predict_one(ex, schemas[ex.db_id])
        for ex in dry_run_examples
    }
    n = len(dry_run_examples)
    dry_run_cost = sum(p.extra["total_cost_usd"] for p in predictions.values())
    avg_cost_per_query = dry_run_cost / n if n > 0 else 0.0
    projected_total_cost = avg_cost_per_query * full_dev_size

    report = {
        "n_dry_run": n,
        "dry_run_cost_usd": dry_run_cost,
        "avg_cost_per_query_usd": avg_cost_per_query,
        "full_dev_size": full_dev_size,
        "projected_total_cost_usd": projected_total_cost,
    }
    print(
        f"Dry run on {n} examples cost ${dry_run_cost:.4f} "
        f"(${avg_cost_per_query:.5f}/query). Projected cost for the full "
        f"dev set ({full_dev_size} examples): ${projected_total_cost:.2f}."
    )
    logger.info("estimate_llm_forward_cost: %s", report)
    return report


def dry_run_llm_forward_cost(n: int = 10) -> dict[str, Any]:
    """Real dry run on the first ``n`` dev examples; projects the full-dev cost.

    Uses the same dedicated log (and cost cap) as :func:`run_llm_forward_on_dev`
    — the dry run's small spend counts toward that one guard, not a separate
    budget. Call this and confirm the projection before
    :func:`run_llm_forward_on_dev`, which runs the full ~$10 dev set.
    """
    config = load_config()
    all_examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    log_path = config.outputs.logs_dir / _LLM_FORWARD_DEV_RUN_LOG_FILENAME
    linker = _build_llm_forward_linker(config, schemas, log_path)
    return estimate_llm_forward_cost(
        linker, all_examples[:n], schemas, len(all_examples)
    )


def run_llm_forward(
    linker: LLMForwardLinker,
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    output_path: Path,
    samples_path: Path,
) -> None:
    """Predict for every example; save canonical JSON plus per-query raw samples.

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.llm_linker.LLMForwardLinker`.
    examples
        Examples to link. ``predict_all`` groups these by ``db_id``
        internally to maximise prompt-cache hits — order in ``examples``
        doesn't matter.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    output_path
        Canonical predictions JSON (same shape/convention as lexical and
        embedding).
    samples_path
        Per-query raw sample JSONL — one line per query with
        ``question_id, db_id, sample_predictions, n_samples_parsed,
        n_samples_valid, total_input_tokens, total_output_tokens,
        total_cost_usd`` (straight from ``Prediction.extra``), for Week 9
        error analysis.

    Side effects
    ------------
    Writes both files (overwriting if they exist). Prints and logs at INFO:
    example count, total cost, parse failure rate (flagged at WARNING if it
    exceeds the 2% target), avg tables/query, avg columns/query, runtime.
    """
    start = time.perf_counter()
    predictions = linker.predict_all(examples, schemas)

    file_ready = from_predictions_to_dict(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(qid): entry for qid, entry in sorted(file_ready.items())}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, sort_keys=False)
        f.write("\n")

    samples_path = Path(samples_path)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as f:
        for qid in sorted(predictions):
            pred = predictions[qid]
            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "db_id": pred.db_id,
                        "sample_predictions": pred.extra["sample_predictions"],
                        "n_samples_parsed": pred.extra["n_samples_parsed"],
                        "n_samples_valid": pred.extra["n_samples_valid"],
                        "total_input_tokens": pred.extra["total_input_tokens"],
                        "total_output_tokens": pred.extra["total_output_tokens"],
                        "total_cost_usd": pred.extra["total_cost_usd"],
                    }
                )
                + "\n"
            )

    elapsed = time.perf_counter() - start
    n = len(examples)
    total_cost = sum(p.extra["total_cost_usd"] for p in predictions.values())
    total_samples = n * linker.k_samples
    total_parsed = sum(p.extra["n_samples_parsed"] for p in predictions.values())
    parse_failure_rate = (
        1.0 - (total_parsed / total_samples) if total_samples > 0 else 0.0
    )
    avg_tables = sum(len(p.tables) for p in predictions.values()) / n if n > 0 else 0.0
    avg_columns = sum(len(p.columns) for p in predictions.values()) / n if n > 0 else 0.0

    print(f"llm_forward: n={n} runtime={elapsed:.1f}s")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Parse failure rate: {parse_failure_rate:.2%} (target <2%)")
    print(f"Avg tables/query: {avg_tables:.2f}")
    print(f"Avg columns/query: {avg_columns:.2f}")
    if parse_failure_rate > _PARSE_FAILURE_RATE_TARGET:
        logger.warning(
            "run_llm_forward: parse failure rate %.2f%% exceeds the %.0f%% target",
            parse_failure_rate * 100,
            _PARSE_FAILURE_RATE_TARGET * 100,
        )

    logger.info(
        "llm_forward: n=%d avg_tables=%.2f avg_columns=%.2f "
        "parse_failure_rate=%.4f total_cost_usd=%.4f runtime=%.2fs -> %s (+ %s)",
        n, avg_tables, avg_columns, parse_failure_rate, total_cost, elapsed,
        output_path, samples_path,
    )


def run_llm_forward_on_dev() -> None:
    """Run the locked ``forward_v1`` LLM linker (Method C) on all of Spider dev.

    Wires up the real ``LLMClient`` + few-shot examples via config/
    ``data/processed/``, then delegates to :func:`run_llm_forward`. Writes
    ``<predictions_dir>/llm_forward_dev.json`` and
    ``<predictions_dir>/llm_forward_dev_samples.jsonl``.

    This is a real, paid API run over the full dev set (~$10 estimated) —
    call :func:`dry_run_llm_forward_cost` first and confirm the projection
    before calling this.
    """
    config = load_config()
    examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    log_path = config.outputs.logs_dir / _LLM_FORWARD_DEV_RUN_LOG_FILENAME
    linker = _build_llm_forward_linker(config, schemas, log_path)
    output_path = config.outputs.predictions_dir / _LLM_FORWARD_DEV_OUTPUT_FILENAME
    samples_path = config.outputs.predictions_dir / _LLM_FORWARD_DEV_SAMPLES_FILENAME
    run_llm_forward(linker, examples, schemas, output_path, samples_path)


def _load_backward_few_shot_with_schema_blocks(
    config: Any, schemas: dict[str, Schema]
) -> list[dict[str, Any]]:
    """Load ``data/processed/few_shot_examples_backward.json`` and enrich
    each entry with a rendered ``schema_block`` — same convention as
    :func:`_load_few_shot_with_schema_blocks` for the forward linker."""
    few_shot_path = config.data.processed_dir / "few_shot_examples_backward.json"
    few_shot = json.loads(few_shot_path.read_text(encoding="utf-8"))
    for ex in few_shot:
        ex["schema_block"] = render_schema_block(schemas[ex["db_id"]])
    return few_shot


def _build_llm_backward_linker(
    config: Any, schemas: dict[str, Schema], log_path: Path, sql_output_path: Path
) -> LLMBackwardLinker:
    """Wire up the real, locked ``backward_v1`` LLM linker for the dev run."""
    llm_client = LLMClient(
        model=_LLM_BACKWARD_MODEL,
        temperature=_LLM_BACKWARD_TEMPERATURE,
        max_tokens=_LLM_BACKWARD_MAX_TOKENS,
        log_path=log_path,
        cost_cap_usd=_LLM_BACKWARD_COST_CAP_USD,
    )
    few_shot = _load_backward_few_shot_with_schema_blocks(config, schemas)
    return LLMBackwardLinker(
        llm_client=llm_client,
        prompt=BACKWARD_V1,
        few_shot=few_shot,
        sql_output_path=sql_output_path,
        extra_metadata={"phase": "backward_dev_run", "prompt_version": BACKWARD_V1.version},
    )


def estimate_llm_backward_cost(
    linker: LLMBackwardLinker,
    dry_run_examples: list[SpiderExample],
    schemas: dict[str, Schema],
    full_dev_size: int,
) -> dict[str, Any]:
    """Run ``linker`` for real on ``dry_run_examples``; project the full-dev cost.

    Mirrors :func:`estimate_llm_forward_cost`, adapted for
    :class:`~schema_linking.llm_linker.LLMBackwardLinker`'s ``extra`` shape
    (``cost_usd``, not ``total_cost_usd`` — there's no k-samples loop to
    sum over).

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.llm_linker.LLMBackwardLinker`.
    dry_run_examples
        A small subset of dev examples to actually call the API on.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    full_dev_size
        Total example count to project the cost onto (e.g. all of dev).

    Returns
    -------
    dict
        ``n_dry_run``, ``dry_run_cost_usd``, ``avg_cost_per_query_usd``,
        ``full_dev_size``, ``projected_total_cost_usd``.
    """
    predictions = {
        ex.question_id: linker.predict_one(ex, schemas[ex.db_id])
        for ex in dry_run_examples
    }
    n = len(dry_run_examples)
    dry_run_cost = sum(p.extra["cost_usd"] for p in predictions.values())
    avg_cost_per_query = dry_run_cost / n if n > 0 else 0.0
    projected_total_cost = avg_cost_per_query * full_dev_size

    report = {
        "n_dry_run": n,
        "dry_run_cost_usd": dry_run_cost,
        "avg_cost_per_query_usd": avg_cost_per_query,
        "full_dev_size": full_dev_size,
        "projected_total_cost_usd": projected_total_cost,
    }
    print(
        f"Dry run on {n} examples cost ${dry_run_cost:.4f} "
        f"(${avg_cost_per_query:.5f}/query). Projected cost for the full "
        f"dev set ({full_dev_size} examples): ${projected_total_cost:.2f}."
    )
    logger.info("estimate_llm_backward_cost: %s", report)
    return report


def dry_run_llm_backward_cost(n: int = 10) -> dict[str, Any]:
    """Real dry run on the first ``n`` dev examples; projects the full-dev cost.

    Uses the same dedicated log (and cost cap) as
    :func:`run_llm_backward_on_dev` — the dry run's small spend counts
    toward that one guard, not a separate budget. Call this and confirm the
    projection before :func:`run_llm_backward_on_dev`.
    """
    config = load_config()
    all_examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    log_path = config.outputs.logs_dir / _LLM_BACKWARD_DEV_RUN_LOG_FILENAME
    sql_output_path = config.outputs.predictions_dir / _LLM_BACKWARD_DEV_SQL_FILENAME
    linker = _build_llm_backward_linker(config, schemas, log_path, sql_output_path)
    return estimate_llm_backward_cost(
        linker, all_examples[:n], schemas, len(all_examples)
    )


def run_llm_backward(
    linker: LLMBackwardLinker,
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    output_path: Path,
) -> None:
    """Predict for every example; save canonical JSON.

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.llm_linker.LLMBackwardLinker`.
        Its own ``sql_output_path`` (set at construction) is written as a
        side effect of ``predict_all`` — this function does not write it
        again, only the canonical predictions JSON.
    examples
        Examples to link. ``predict_all`` groups these by ``db_id``
        internally to maximise prompt-cache hits — order in ``examples``
        doesn't matter.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    output_path
        Canonical predictions JSON (same shape/convention as the other
        linkers).

    Side effects
    ------------
    Writes ``output_path`` (overwriting if it exists) and, via
    ``linker.predict_all``, ``linker.sql_output_path``. Prints and logs at
    INFO: example count, total cost, parse failure rate (flagged at
    WARNING if it exceeds the 2% target), avg tables/query, avg
    columns/query, runtime.
    """
    start = time.perf_counter()
    predictions = linker.predict_all(examples, schemas)

    file_ready = from_predictions_to_dict(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(qid): entry for qid, entry in sorted(file_ready.items())}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, sort_keys=False)
        f.write("\n")

    elapsed = time.perf_counter() - start
    n = len(examples)
    total_cost = sum(p.extra["cost_usd"] for p in predictions.values())
    n_parse_errors = sum(
        1 for p in predictions.values()
        if any(i["kind"] == "parse_error" for i in p.extra["parse_issues"])
    )
    parse_failure_rate = n_parse_errors / n if n > 0 else 0.0
    avg_tables = sum(len(p.tables) for p in predictions.values()) / n if n > 0 else 0.0
    avg_columns = sum(len(p.columns) for p in predictions.values()) / n if n > 0 else 0.0

    print(f"llm_backward: n={n} runtime={elapsed:.1f}s")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Parse failure rate: {parse_failure_rate:.2%} (target <2%)")
    print(f"Avg tables/query: {avg_tables:.2f}")
    print(f"Avg columns/query: {avg_columns:.2f}")
    if parse_failure_rate > _PARSE_FAILURE_RATE_TARGET:
        logger.warning(
            "run_llm_backward: parse failure rate %.2f%% exceeds the %.0f%% target",
            parse_failure_rate * 100,
            _PARSE_FAILURE_RATE_TARGET * 100,
        )

    logger.info(
        "llm_backward: n=%d avg_tables=%.2f avg_columns=%.2f "
        "parse_failure_rate=%.4f total_cost_usd=%.4f runtime=%.2fs -> %s (+ %s)",
        n, avg_tables, avg_columns, parse_failure_rate, total_cost, elapsed,
        output_path, linker.sql_output_path,
    )


def run_llm_backward_on_dev() -> None:
    """Run the locked ``backward_v1`` LLM linker (Method D) on all of Spider dev.

    Wires up the real ``LLMClient`` + backward few-shot examples via
    config/``data/processed/``, then delegates to :func:`run_llm_backward`.
    Writes ``<predictions_dir>/llm_backward_dev.json`` and (via the linker
    itself) ``<predictions_dir>/llm_backward_dev_sql.jsonl``.

    This is a real, paid API run over the full dev set — call
    :func:`dry_run_llm_backward_cost` first and confirm the projection
    before calling this.
    """
    config = load_config()
    examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    log_path = config.outputs.logs_dir / _LLM_BACKWARD_DEV_RUN_LOG_FILENAME
    sql_output_path = config.outputs.predictions_dir / _LLM_BACKWARD_DEV_SQL_FILENAME
    linker = _build_llm_backward_linker(config, schemas, log_path, sql_output_path)
    output_path = config.outputs.predictions_dir / _LLM_BACKWARD_DEV_OUTPUT_FILENAME
    run_llm_backward(linker, examples, schemas, output_path)


def run_bidirectional(
    linker: BidirectionalLinker,
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    output_path: Path,
) -> None:
    """Predict for every example (no LLM calls — pure set union); save canonical JSON.

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.llm_linker.BidirectionalLinker`.
    examples
        Examples to link.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    output_path
        Canonical predictions JSON (same shape/convention as the other
        linkers).

    Side effects
    ------------
    Writes ``output_path`` (overwriting if it exists). Logs at INFO:
    example count, avg predicted tables/columns per query, runtime
    (expected to be seconds, not the minutes/hours of the LLM-backed
    linkers, since this makes no LLM calls).
    """
    start = time.perf_counter()
    predictions = linker.predict_all(examples, schemas)
    file_ready = from_predictions_to_dict(predictions)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(qid): entry for qid, entry in sorted(file_ready.items())}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, sort_keys=False)
        f.write("\n")

    elapsed = time.perf_counter() - start
    n = len(examples)
    avg_tables = sum(len(p.tables) for p in predictions.values()) / n if n > 0 else 0.0
    avg_columns = sum(len(p.columns) for p in predictions.values()) / n if n > 0 else 0.0
    logger.info(
        "bidirectional: n=%d avg_tables=%.2f avg_columns=%.2f runtime=%.2fs -> %s",
        n, avg_tables, avg_columns, elapsed, output_path,
    )


def run_bidirectional_on_dev() -> None:
    """Run the bidirectional linker (Method E) on all of Spider dev.

    Constructs :class:`~schema_linking.llm_linker.BidirectionalLinker` from
    Method C's and D's already-saved dev predictions
    (``llm_forward_dev.json``, ``llm_backward_dev.json`` — both must
    already exist on disk, i.e. run those two dev runs first). Writes
    ``<predictions_dir>/llm_bidirectional_dev.json``. Makes no LLM calls —
    expect this to run in seconds.
    """
    config = load_config()
    examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    forward_path = config.outputs.predictions_dir / _LLM_FORWARD_DEV_OUTPUT_FILENAME
    backward_path = config.outputs.predictions_dir / _LLM_BACKWARD_DEV_OUTPUT_FILENAME
    linker = BidirectionalLinker(forward_path, backward_path)
    output_path = config.outputs.predictions_dir / _LLM_BIDIRECTIONAL_DEV_OUTPUT_FILENAME
    run_bidirectional(linker, examples, schemas, output_path)


def _load_graph_few_shot_with_schema_blocks(
    config: Any, schemas: dict[str, Schema]
) -> list[dict[str, Any]]:
    """Load ``data/processed/few_shot_examples_graph.json`` and enrich each
    entry with a rendered ``schema_block`` — same convention as
    :func:`_load_few_shot_with_schema_blocks` for the forward linker."""
    few_shot_path = config.data.processed_dir / "few_shot_examples_graph.json"
    few_shot = json.loads(few_shot_path.read_text(encoding="utf-8"))
    for ex in few_shot:
        ex["schema_block"] = render_schema_block(schemas[ex["db_id"]])
    return few_shot


def _build_graph_linker(
    config: Any, schemas: dict[str, Schema], log_path: Path, trace_path: Path
) -> GraphLinker:
    """Wire up the real, locked ``graph_endpoint_v1`` linker for the dev run."""
    llm_client = LLMClient(
        model=_GRAPH_MODEL,
        temperature=_GRAPH_TEMPERATURE,
        max_tokens=_GRAPH_MAX_TOKENS,
        log_path=log_path,
        cost_cap_usd=_GRAPH_COST_CAP_USD,
    )
    few_shot = _load_graph_few_shot_with_schema_blocks(config, schemas)
    return GraphLinker(
        llm_client=llm_client,
        prompt=GRAPH_ENDPOINT_V1,
        few_shot=few_shot,
        schemas=schemas,
        trace_path=trace_path,
        extra_metadata={"phase": "graph_dev_run", "prompt_version": GRAPH_ENDPOINT_V1.version},
    )


def estimate_graph_cost(
    linker: GraphLinker,
    dry_run_examples: list[SpiderExample],
    schemas: dict[str, Schema],
    full_dev_size: int,
) -> dict[str, Any]:
    """Run ``linker`` for real on ``dry_run_examples``; project the full-dev cost.

    Mirrors :func:`estimate_llm_backward_cost`, adapted for
    :class:`~schema_linking.graph_linker.GraphLinker`'s ``extra`` shape
    (``cost_usd``, not ``total_cost_usd`` — Method G is also a single
    deterministic call per question, no k-samples loop to sum over).

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.graph_linker.GraphLinker`.
    dry_run_examples
        A small subset of dev examples to actually call the API on.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    full_dev_size
        Total example count to project the cost onto (e.g. all of dev).

    Returns
    -------
    dict
        ``n_dry_run``, ``dry_run_cost_usd``, ``avg_cost_per_query_usd``,
        ``full_dev_size``, ``projected_total_cost_usd``.
    """
    predictions = {
        ex.question_id: linker.predict_one(ex, schemas[ex.db_id])
        for ex in dry_run_examples
    }
    n = len(dry_run_examples)
    dry_run_cost = sum(p.extra["cost_usd"] for p in predictions.values())
    avg_cost_per_query = dry_run_cost / n if n > 0 else 0.0
    projected_total_cost = avg_cost_per_query * full_dev_size

    report = {
        "n_dry_run": n,
        "dry_run_cost_usd": dry_run_cost,
        "avg_cost_per_query_usd": avg_cost_per_query,
        "full_dev_size": full_dev_size,
        "projected_total_cost_usd": projected_total_cost,
    }
    print(
        f"Dry run on {n} examples cost ${dry_run_cost:.4f} "
        f"(${avg_cost_per_query:.5f}/query). Projected cost for the full "
        f"dev set ({full_dev_size} examples): ${projected_total_cost:.2f}."
    )
    logger.info("estimate_graph_cost: %s", report)
    return report


def dry_run_graph_cost(n: int = 10) -> dict[str, Any]:
    """Real dry run on the first ``n`` dev examples; projects the full-dev cost.

    Uses the same dedicated log (and cost cap) as :func:`run_graph_on_dev` —
    the dry run's small spend counts toward that one guard, not a separate
    budget. Call this and confirm the projection before
    :func:`run_graph_on_dev`, which runs the full 1034-example dev set.
    """
    config = load_config()
    all_examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    log_path = config.outputs.logs_dir / _GRAPH_DEV_RUN_LOG_FILENAME
    trace_path = config.outputs.predictions_dir / _GRAPH_DEV_TRACES_FILENAME
    linker = _build_graph_linker(config, schemas, log_path, trace_path)
    return estimate_graph_cost(linker, all_examples[:n], schemas, len(all_examples))


def run_graph(
    linker: GraphLinker,
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    output_path: Path,
) -> None:
    """Predict for every example; save canonical JSON.

    Parameters
    ----------
    linker
        Configured :class:`~schema_linking.graph_linker.GraphLinker`. Its
        own ``trace_path`` (set at construction) is written as a side
        effect of ``predict_all`` — this function does not write it again,
        only the canonical predictions JSON.
    examples
        Examples to link. ``predict_all`` groups these by ``db_id``
        internally to maximise prompt-cache hits — order in ``examples``
        doesn't matter.
    schemas
        Map ``db_id -> Schema`` covering every ``ex.db_id``.
    output_path
        Canonical predictions JSON (same shape/convention as the other
        linkers).

    Side effects
    ------------
    Writes ``output_path`` (overwriting if it exists) and, via
    ``linker.predict_all``, ``linker.trace_path``. Prints and logs at INFO:
    example count, total cost, failure rate (parse + no-valid-endpoints,
    flagged at WARNING if it exceeds the 2% target), avg tables/query, avg
    columns/query, runtime.
    """
    start = time.perf_counter()
    predictions = linker.predict_all(examples, schemas)

    file_ready = from_predictions_to_dict(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(qid): entry for qid, entry in sorted(file_ready.items())}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, sort_keys=False)
        f.write("\n")

    elapsed = time.perf_counter() - start
    n = len(examples)
    total_cost = sum(p.extra["cost_usd"] for p in predictions.values())
    n_failures = sum(1 for p in predictions.values() if p.extra["failure"] is not None)
    failure_rate = n_failures / n if n > 0 else 0.0
    avg_tables = sum(len(p.tables) for p in predictions.values()) / n if n > 0 else 0.0
    avg_columns = sum(len(p.columns) for p in predictions.values()) / n if n > 0 else 0.0

    print(f"graph: n={n} runtime={elapsed:.1f}s")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Failure rate (parse + no_valid_endpoints): {failure_rate:.2%} (target <2%)")
    print(f"Avg tables/query: {avg_tables:.2f}")
    print(f"Avg columns/query: {avg_columns:.2f}")
    if failure_rate > _PARSE_FAILURE_RATE_TARGET:
        logger.warning(
            "run_graph: failure rate %.2f%% exceeds the %.0f%% target",
            failure_rate * 100,
            _PARSE_FAILURE_RATE_TARGET * 100,
        )

    logger.info(
        "graph: n=%d avg_tables=%.2f avg_columns=%.2f failure_rate=%.4f "
        "total_cost_usd=%.4f runtime=%.2fs -> %s (+ %s)",
        n, avg_tables, avg_columns, failure_rate, total_cost, elapsed,
        output_path, linker.trace_path,
    )


def run_graph_on_dev() -> None:
    """Run the locked ``graph_endpoint_v1`` linker (Method G) on all of Spider dev.

    Wires up the real ``LLMClient`` + graph few-shot examples via config/
    ``data/processed/``, then delegates to :func:`run_graph`. Writes
    ``<predictions_dir>/graph_dev.json`` and (via the linker itself)
    ``<predictions_dir>/graph_dev_traces.jsonl``.

    This is a real, paid API run over the full dev set — call
    :func:`dry_run_graph_cost` first and confirm the projection before
    calling this.
    """
    config = load_config()
    examples = list(load_spider_questions("dev"))
    schemas = load_schemas()
    log_path = config.outputs.logs_dir / _GRAPH_DEV_RUN_LOG_FILENAME
    trace_path = config.outputs.predictions_dir / _GRAPH_DEV_TRACES_FILENAME
    linker = _build_graph_linker(config, schemas, log_path, trace_path)
    output_path = config.outputs.predictions_dir / _GRAPH_DEV_OUTPUT_FILENAME
    run_graph(linker, examples, schemas, output_path)


def write_llm_cost_report(
    log_paths_by_phase: dict[str, Path] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Summarise LLM call costs across Week 6 phases into one CSV.

    Parameters
    ----------
    log_paths_by_phase
        Override ``{phase_name: log_path}`` (mainly for tests). Defaults to
        ``{"iteration": <logs_dir>/llm_calls_prompt_iteration.jsonl,
        "dev_run": <logs_dir>/llm_calls_dev_run.jsonl}``.
    output_path
        Defaults to ``<results_dir>/llm_cost_report.csv``.

    Returns
    -------
    pd.DataFrame
        One row per phase present on disk (a missing log file is skipped
        with a WARNING, not an error — e.g. before the dev run exists yet),
        plus one ``"total"`` row summed across all phases present. Columns:
        ``phase, cost_usd, calls, avg_input_tokens, avg_output_tokens,
        cache_hit_rate``. ``cache_hit_rate`` is computed from summed raw
        token counts (``cache_read_input_tokens / (cache_read_input_tokens
        + cache_creation_input_tokens + input_tokens)``), not by averaging
        already-divided per-call rates.
    """
    config = load_config()
    if log_paths_by_phase is None:
        log_paths_by_phase = {
            phase: config.outputs.logs_dir / filename
            for phase, filename in _COST_LOG_FILENAME_BY_PHASE.items()
        }

    per_phase_raw: dict[str, dict[str, float]] = {}
    for phase, log_path in log_paths_by_phase.items():
        log_path = Path(log_path)
        if not log_path.is_file():
            logger.warning(
                "write_llm_cost_report: no log file for phase=%s (%s) — skipping",
                phase, log_path,
            )
            continue
        lines = [
            line for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            continue
        entries = [json.loads(line) for line in lines]
        matching = [
            e for e in entries if e.get("metadata", {}).get("phase") == phase
        ]
        if not matching:
            logger.warning(
                "write_llm_cost_report: no entries tagged phase=%s in %s — skipping",
                phase, log_path,
            )
            continue
        per_phase_raw[phase] = {
            "calls": len(matching),
            "cost_usd": sum(e["cost_usd"] for e in matching),
            "input_tokens": sum(e["input_tokens"] for e in matching),
            "output_tokens": sum(e["output_tokens"] for e in matching),
            "cache_read_input_tokens": sum(e["cache_read_input_tokens"] for e in matching),
            "cache_creation_input_tokens": sum(
                e["cache_creation_input_tokens"] for e in matching
            ),
        }

    def _row(phase: str, raw: dict[str, float]) -> dict[str, Any]:
        calls = raw["calls"]
        denom = (
            raw["cache_read_input_tokens"]
            + raw["cache_creation_input_tokens"]
            + raw["input_tokens"]
        )
        return {
            "phase": phase,
            "cost_usd": raw["cost_usd"],
            "calls": calls,
            "avg_input_tokens": raw["input_tokens"] / calls if calls > 0 else 0.0,
            "avg_output_tokens": raw["output_tokens"] / calls if calls > 0 else 0.0,
            "cache_hit_rate": raw["cache_read_input_tokens"] / denom if denom > 0 else 0.0,
        }

    rows = [_row(phase, raw) for phase, raw in per_phase_raw.items()]
    if per_phase_raw:
        total_raw = {
            key: sum(raw[key] for raw in per_phase_raw.values())
            for key in (
                "calls", "cost_usd", "input_tokens", "output_tokens",
                "cache_read_input_tokens", "cache_creation_input_tokens",
            )
        }
        rows.append(_row("total", total_raw))

    result_df = pd.DataFrame(rows)
    output_path = (
        Path(output_path) if output_path is not None
        else config.outputs.results_dir / _COST_REPORT_FILENAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    logger.info("write_llm_cost_report: %d phase rows -> %s", len(rows), output_path)
    return result_df
