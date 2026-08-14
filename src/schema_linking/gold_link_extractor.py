"""Gold-link extraction from Spider gold SQL.

Two policies are supported:

* **Tier 2 — "All-SQL-used".** Every table in any ``FROM`` / ``JOIN`` /
  subquery and every column in any clause (``SELECT`` / ``WHERE`` /
  ``GROUP BY`` / ``HAVING`` / ``ORDER BY`` / ``JOIN ON``).
* **Tier 1 — "Mentioned, strict".** Tier 2 minus columns that appear
  *only* in ``JOIN ON``, and minus tables whose only column references
  in the query live in ``JOIN ON`` (pure join-bridge tables).
  ``SELECT *`` (bare or ``t.*``) keeps the referenced table.

Both tiers are built on the shared sqlglot AST walker in
``schema_linking.utils.sql_parsing`` (``strict=True`` — gold SQL should
always resolve against its own schema). See that module's docstring for the
per-edge-case behaviour (aliases, CTEs, subqueries, unions, aggregates, ...).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema
from schema_linking.taniguchi_loader import GoldLinks, save_gold_links
from schema_linking.utils.sql_parsing import (
    _CLAUSE_JOIN_ON,
    _walk,
    extract_schema_references,
)

__all__ = [
    "GoldLinks",
    "save_gold_links",
    "extract_tier1",
    "extract_tier1_all",
    "extract_tier2",
    "extract_tier2_all",
    "ExtractionError",
]

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when extraction cannot complete for a query."""


# ---------- public Tier-2 API ----------


def extract_tier2(example: SpiderExample, schema: Schema) -> GoldLinks:
    """Extract the Tier-2 ("All-SQL-used") gold link set for one example.

    Parameters
    ----------
    example
        Spider example whose ``query`` is parsed by sqlglot.
    schema
        Schema for ``example.db_id``.

    Returns
    -------
    GoldLinks
        Tables and columns sorted lexically, both deduped.

    Raises
    ------
    ExtractionError
        If sqlglot fails to parse ``example.query``.
    """
    refs, issues = extract_schema_references(example.query, schema, strict=True)
    for issue in issues:
        if issue.kind == "parse_error":
            raise ExtractionError(issue.detail)
    return GoldLinks(db_id=example.db_id, tables=refs.tables, columns=refs.columns)


def extract_tier2_all(
    examples: Iterable[SpiderExample],
    schemas: dict[str, Schema],
    *,
    fail_on_parse_error: bool = False,
    error_log_path: Path | None = None,
) -> dict[int, GoldLinks]:
    """Run :func:`extract_tier2` over many examples. See module docstring."""
    return _extract_all(
        examples,
        schemas,
        extract_tier2,
        fail_on_parse_error=fail_on_parse_error,
        error_log_path=error_log_path,
        tier_name="tier2",
    )


# ---------- public Tier-1 API ----------


def extract_tier1(example: SpiderExample, schema: Schema) -> GoldLinks:
    """Extract the Tier-1 ("Mentioned, strict") gold link set.

    Tier-1 = Tier-2 minus:

    * columns whose every appearance is inside a ``JOIN ON`` predicate;
    * tables whose only column references in the query are inside
      ``JOIN ON`` (pure join-bridge tables).

    Tables touched only by a ``SELECT *`` (or ``t.*``) are kept, since
    the wildcard implicitly references their columns.

    Raises
    ------
    ExtractionError
        If sqlglot fails to parse ``example.query``.
    """
    w = _walk(example.query, schema, strict=True)
    for issue in w.issues:
        if issue.kind == "parse_error":
            raise ExtractionError(issue.detail)

    tier1_cols: set[tuple[str, str]] = {
        tc
        for tc, clauses in w.columns_with_clauses.items()
        if clauses - {_CLAUSE_JOIN_ON}
    }
    tables_with_tier1_cols = {t for t, _ in tier1_cols}

    referenced_tables = {t for t, _ in w.columns_with_clauses.keys()}
    no_ref_tables = w.tables - referenced_tables  # e.g. ``SELECT 1 FROM t1``

    tier1_tables = (
        tables_with_tier1_cols | w.star_tables | no_ref_tables
    ) & w.tables

    return GoldLinks(
        db_id=example.db_id,
        tables=tuple(sorted(tier1_tables)),
        columns=tuple(sorted(tier1_cols)),
    )


def extract_tier1_all(
    examples: Iterable[SpiderExample],
    schemas: dict[str, Schema],
    *,
    fail_on_parse_error: bool = False,
    error_log_path: Path | None = None,
) -> dict[int, GoldLinks]:
    """Run :func:`extract_tier1` over many examples."""
    return _extract_all(
        examples,
        schemas,
        extract_tier1,
        fail_on_parse_error=fail_on_parse_error,
        error_log_path=error_log_path,
        tier_name="tier1",
    )


# ---------- batch driver ----------


def _extract_all(
    examples: Iterable[SpiderExample],
    schemas: dict[str, Schema],
    fn,
    *,
    fail_on_parse_error: bool,
    error_log_path: Path | None,
    tier_name: str,
) -> dict[int, GoldLinks]:
    out: dict[int, GoldLinks] = {}
    errors: list[dict[str, object]] = []
    for ex in examples:
        schema = schemas.get(ex.db_id)
        if schema is None:
            err = {
                "question_id": ex.question_id,
                "split": ex.split,
                "db_id": ex.db_id,
                "query": ex.query,
                "error_type": "MissingSchema",
                "error_msg": f"no schema for db_id {ex.db_id!r}",
            }
            if fail_on_parse_error:
                raise ExtractionError(err["error_msg"])
            errors.append(err)
            logger.warning(
                "missing schema for db_id %r (qid %d)",
                ex.db_id,
                ex.question_id,
            )
            continue
        try:
            out[ex.question_id] = fn(ex, schema)
        except ExtractionError as exc:
            err = {
                "question_id": ex.question_id,
                "split": ex.split,
                "db_id": ex.db_id,
                "query": ex.query,
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
            }
            if fail_on_parse_error:
                raise
            errors.append(err)
            logger.warning(
                "extraction failed for qid %d (db %r): %s",
                ex.question_id,
                ex.db_id,
                exc,
            )

    if error_log_path is not None and errors:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with error_log_path.open("w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err) + "\n")

    logger.info(
        "%s: %d ok, %d failed (logged to %s)",
        tier_name,
        len(out),
        len(errors),
        error_log_path,
    )
    return out
