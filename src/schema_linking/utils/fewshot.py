"""Fixed few-shot example selection for LLM forward prompting (Method C).

Locked strategy: two hand-picked structural patterns from Spider train
(SIMPLE, MULTI_TABLE), not retrieval-based.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from random import Random
from typing import Any

from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema

_DEFAULT_OUTPUT_PATH: Path = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "few_shot_examples.json"
)
_DEFAULT_BACKWARD_OUTPUT_PATH: Path = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "few_shot_examples_backward.json"
)
_DEFAULT_GRAPH_OUTPUT_PATH: Path = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "few_shot_examples_graph.json"
)


def _is_flat_sql(sql: dict[str, Any]) -> bool:
    """No UNION/INTERSECT/EXCEPT and no nested subqueries in FROM or WHERE."""
    if sql.get("intersect") or sql.get("union") or sql.get("except"):
        return False
    table_units = sql.get("from", {}).get("table_units", [])
    if any(unit[0] != "table_unit" for unit in table_units):
        return False
    for cond in sql.get("where", []):
        if isinstance(cond, (list, tuple)) and len(cond) >= 4 and isinstance(cond[3], dict):
            return False  # nested SELECT as a WHERE comparison value
    return True


def _is_simple_pattern(example: SpiderExample, gold_entry: dict[str, Any]) -> bool:
    """Single table, an aggregation, no joins."""
    sql = example.sql
    if not _is_flat_sql(sql):
        return False
    table_units = sql.get("from", {}).get("table_units", [])
    if len(table_units) != 1 or sql.get("from", {}).get("conds"):
        return False
    agg_ids = [agg_id for agg_id, _ in sql.get("select", [False, []])[1]]
    return any(agg_id != 0 for agg_id in agg_ids)


def _is_multi_table_pattern(example: SpiderExample, gold_entry: dict[str, Any]) -> bool:
    """2+ tables joined via FROM, with gold columns spanning 2+ tables."""
    sql = example.sql
    if not _is_flat_sql(sql):
        return False
    if len(sql.get("from", {}).get("table_units", [])) < 2:
        return False
    distinct_tables = {t for t, _ in gold_entry["columns"]}
    return len(distinct_tables) >= 2


_PATTERNS: tuple[tuple[str, Callable[[SpiderExample, dict[str, Any]], bool]], ...] = (
    ("simple", _is_simple_pattern),
    ("multi_table", _is_multi_table_pattern),
)


def _validate_gold_against_schema(
    db_id: str, tables: list[str], columns: list[list[str]], schemas: dict[str, Schema]
) -> None:
    schema = schemas[db_id]
    table_names = {t.original_name for t in schema.tables}
    for t in tables:
        if t not in table_names:
            raise ValueError(f"few-shot candidate references unknown table {t!r} in db {db_id!r}")
    columns_by_table = {t.original_name: {c.original_name for c in t.columns} for t in schema.tables}
    for t, c in columns:
        if t not in columns_by_table or c not in columns_by_table[t]:
            raise ValueError(f"few-shot candidate references unknown column {t}.{c} in db {db_id!r}")


def pick_fewshot_examples(
    train_examples: list[SpiderExample],
    schemas: dict[str, Schema],
    gold: dict[int, dict[str, Any]],
    n: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Pick fixed few-shot examples covering the two locked canonical patterns.

    Parameters
    ----------
    train_examples
        Spider train examples to search.
    schemas
        All Spider schemas, keyed by ``db_id`` — used to sanity-check that
        each candidate's gold tables/columns actually exist in its schema.
    gold
        Tier-1 ("Mentioned") gold links for the train split, keyed by
        **``int``** ``question_id``: ``{question_id: {db_id, tables,
        columns}}`` (cast string keys from a raw ``json.load`` first).
    n
        Must be 2 — only the two patterns above are locked.
    seed
        Seed for the pseudo-random pick among all pattern-matching
        candidates — reproducible without hardcoding a magic ``question_id``.

    Returns
    -------
    list[dict]
        Exactly ``n`` dicts, each ``{question_id, db_id, question, pattern,
        tables, columns}``, JSON-serialisable as-is. Guaranteed distinct
        ``db_id``s.

    Raises
    ------
    ValueError
        If ``n != 2``, if no candidate matches a pattern, or if a
        candidate's gold references a table/column absent from its schema.
    """
    if n != 2:
        raise ValueError(f"pick_fewshot_examples only supports n=2 (locked patterns), got {n}")

    rng = Random(seed)
    picked: list[dict[str, Any]] = []
    used_db_ids: set[str] = set()

    for pattern_name, predicate in _PATTERNS:
        candidates = [
            ex
            for ex in train_examples
            if ex.db_id not in used_db_ids
            and ex.question_id in gold
            and predicate(ex, gold[ex.question_id])
        ]
        if not candidates:
            raise ValueError(f"no train example matches the {pattern_name!r} pattern")
        rng.shuffle(candidates)
        chosen = candidates[0]

        gold_entry = gold[chosen.question_id]
        tables = list(gold_entry["tables"])
        columns = [list(pair) for pair in gold_entry["columns"]]
        _validate_gold_against_schema(chosen.db_id, tables, columns, schemas)

        used_db_ids.add(chosen.db_id)
        picked.append(
            {
                "question_id": chosen.question_id,
                "db_id": chosen.db_id,
                "question": chosen.question,
                "pattern": pattern_name,
                "tables": tables,
                "columns": columns,
            }
        )
    return picked


def save_fewshot_examples(
    examples: list[dict[str, Any]], path: Path = _DEFAULT_OUTPUT_PATH
) -> None:
    """Write picked few-shot examples to disk as JSON, for manual review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)


def to_backward_fewshot_examples(
    fewshot_examples: list[dict[str, Any]], train_examples: list[SpiderExample]
) -> list[dict[str, Any]]:
    """Reformat the forward few-shot picks for backward-style (Method D)
    prompting, by attaching each example's gold SQL.

    The gold SQL comes directly from Spider train (matched by
    ``question_id``) — no LLM in the loop for few-shot generation.

    Parameters
    ----------
    fewshot_examples
        Output of :func:`pick_fewshot_examples` (or the on-disk
        ``few_shot_examples.json``): dicts with ``question_id``, ``db_id``,
        ``question``, ``pattern`` keys (``tables``/``columns`` are dropped —
        Method D doesn't need them).
    train_examples
        Spider train examples to look up gold SQL from.

    Returns
    -------
    list[dict]
        One dict per input example: ``{question_id, db_id, question,
        pattern, gold_sql}``.

    Raises
    ------
    ValueError
        If an example's ``question_id`` has no matching train example.
    """
    query_by_qid = {ex.question_id: ex.query for ex in train_examples}
    out: list[dict[str, Any]] = []
    for ex in fewshot_examples:
        qid = ex["question_id"]
        if qid not in query_by_qid:
            raise ValueError(f"question_id {qid} not found in train_examples")
        out.append(
            {
                "question_id": qid,
                "db_id": ex["db_id"],
                "question": ex["question"],
                "pattern": ex["pattern"],
                "gold_sql": query_by_qid[qid],
            }
        )
    return out


def save_backward_fewshot_examples(
    examples: list[dict[str, Any]], path: Path = _DEFAULT_BACKWARD_OUTPUT_PATH
) -> None:
    """Write backward-reformatted few-shot examples to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)


def to_graph_fewshot_examples(
    fewshot_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reformat the forward few-shot picks for the graph endpoint prompt
    (Method F, ``GRAPH_ENDPOINT_V1``), renaming ``tables`` to ``core_tables``.

    Tier-1 ("Mentioned") gold already excludes join-bridge-only tables and
    columns, so every table in a Tier-1 few-shot's ``tables`` list is
    already a core/endpoint table — no
    filtering is needed, only the key rename the graph prompt expects.

    Parameters
    ----------
    fewshot_examples
        Output of :func:`pick_fewshot_examples` (or the on-disk
        ``few_shot_examples.json``): dicts with ``question_id``, ``db_id``,
        ``question``, ``pattern``, ``tables``, ``columns`` keys.

    Returns
    -------
    list[dict]
        One dict per input example: ``{question_id, db_id, question,
        pattern, core_tables, columns}``.

    Raises
    ------
    ValueError
        If an example has 0 or 4+ core tables, or a column references a
        table outside its ``core_tables``.
    """
    out: list[dict[str, Any]] = []
    for ex in fewshot_examples:
        core_tables = list(ex["tables"])
        if not 1 <= len(core_tables) <= 3:
            raise ValueError(
                f"question_id {ex['question_id']}: expected 1-3 core tables, got {len(core_tables)}"
            )
        columns = [list(pair) for pair in ex["columns"]]
        core_table_set = set(core_tables)
        for table, _ in columns:
            if table not in core_table_set:
                raise ValueError(
                    f"question_id {ex['question_id']}: column table {table!r} not in core_tables"
                )
        out.append(
            {
                "question_id": ex["question_id"],
                "db_id": ex["db_id"],
                "question": ex["question"],
                "pattern": ex["pattern"],
                "core_tables": core_tables,
                "columns": columns,
            }
        )
    return out


def save_graph_fewshot_examples(
    examples: list[dict[str, Any]], path: Path = _DEFAULT_GRAPH_OUTPUT_PATH
) -> None:
    """Write graph-endpoint-reformatted few-shot examples to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
