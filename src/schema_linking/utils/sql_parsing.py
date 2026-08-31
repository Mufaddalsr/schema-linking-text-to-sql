"""Shared sqlglot AST walker for resolving SQL against a Spider schema.

Used by gold-link extraction (``gold_link_extractor.py``, always
``strict=True``) and by Method D, which parses LLM-generated SQL against
the same schema (``strict=False``) — hallucinated tables/columns are
a valid Method D prediction, not an error.

The dialect passed to sqlglot is ``"sqlite"``: Spider's gold SQL (and the SQL
LLMs are prompted to emit for Spider databases) is SQLite-flavoured.

Per-edge-case behaviour
------------------------
E1.  ``SELECT *`` — no column added; tables still come from ``FROM`` and
     are tracked as "star tables" so a Tier-1-style consumer can keep them.
E2.  ``COUNT(*)``, ``MAX(col)`` — aggregates wrap an inner expression;
     ``find_all(exp.Column)`` reaches the inner column.
E3.  Aliases (``FROM tbl AS T1``) — alias map carries the resolution.
E4.  ``SELECT … FROM (SELECT … FROM t1)`` — ``find_all`` descends.
E5.  ``WHERE col IN (SELECT col2 FROM t2)`` — same as E4.
E6.  CTEs — CTE names are excluded from the resolved table set; CTE-qualified
     columns fall through to the schema-wide name lookup.
E7.  ``UNION`` / ``INTERSECT`` / ``EXCEPT`` — each branch is visited.
E8.  Self-joins — set semantics dedupe.
E9.  Unknown identifier — recorded as a :class:`ParseIssue`, never raises.
     Dropped when ``strict=True``; included (using its raw SQL-text casing,
     since there is no schema case to fall back to) when ``strict=False``.
E10. ``SUBSTR(col, 1, 3)``, ``DATE(col)`` — the wrapping function is
     ignored; the inner column is collected.

``strict`` flag
----------------
``strict=True`` (gold SQL): unknown tables/columns are dropped and a
:class:`ParseIssue` is recorded — gold SQL should always resolve against its
own schema, so a miss indicates a schema/query mismatch worth flagging but
not worth losing the rest of the extraction over.

``strict=False`` (LLM-generated SQL): unknown tables/columns are kept in the
returned :class:`SchemaReferences` (using the raw casing as written in the
SQL) alongside a :class:`ParseIssue` — the caller decides how to score a
hallucinated reference.

A qualifier that doesn't resolve to any known alias/table (e.g. a
never-aliased or hallucinated table prefix) falls through to the same
schema-wide candidate search used for unqualified columns, in both modes —
this fallback is load-bearing for gold-link output and must not change.
A :class:`ParseIssue` of kind ``"unresolved_alias"`` is still
recorded for diagnostic visibility.

A parse failure (invalid SQL) always returns an empty
:class:`SchemaReferences` plus a single ``"parse_error"`` issue, regardless
of ``strict``. This function never raises.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import SqlglotError

from schema_linking.schema_parser import Schema

__all__ = [
    "SchemaReferences",
    "ParseIssue",
    "extract_schema_references",
    "column_roles",
    "JOIN_ON",
]

_DIALECT = "sqlite"

_CLAUSE_SELECT = "SELECT"
_CLAUSE_WHERE = "WHERE"
_CLAUSE_GROUP_BY = "GROUP_BY"
_CLAUSE_HAVING = "HAVING"
_CLAUSE_ORDER_BY = "ORDER_BY"
_CLAUSE_JOIN_ON = "JOIN_ON"
_CLAUSE_OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class SchemaReferences:
    """Tables and columns a query resolves to, against a given schema.

    Attributes
    ----------
    tables
        Distinct table names, lexically sorted. In original schema case for
        resolved tables; in raw SQL-text case for unresolved ones (only
        possible when the caller passed ``strict=False``).
    columns
        Distinct ``(table, column)`` pairs, lexically sorted. Same casing
        rule as :attr:`tables`.
    """

    tables: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ParseIssue:
    """A single diagnostic recorded while resolving a query against a schema.

    Attributes
    ----------
    kind
        ``"parse_error"`` — sqlglot could not parse the SQL at all.
        ``"unknown_table"`` — a table name not present in the schema.
        ``"unknown_column"`` — a column name not present in its resolved
        (or candidate) table.
        ``"unresolved_alias"`` — a column qualifier that didn't match any
        known alias or table name; the column still falls through to a
        schema-wide candidate search.
        ``"other"`` — reserved for conditions not otherwise classified.
    detail
        Human-readable description of what went wrong.
    """

    kind: Literal[
        "parse_error", "unknown_table", "unknown_column", "unresolved_alias", "other"
    ]
    detail: str


# ---------- schema indexing ----------


@dataclass(frozen=True, slots=True)
class _SchemaIndex:
    tables_by_lower: dict[str, str]
    cols_per_table: dict[str, dict[str, str]]
    tables_for_col: dict[str, list[str]]


def _build_schema_index(schema: Schema) -> _SchemaIndex:
    tables_by_lower: dict[str, str] = {}
    cols_per_table: dict[str, dict[str, str]] = {}
    tables_for_col: dict[str, list[str]] = {}
    for t in schema.tables:
        tables_by_lower[t.original_name.lower()] = t.original_name
        per: dict[str, str] = {}
        for c in t.columns:
            per[c.original_name.lower()] = c.original_name
            tables_for_col.setdefault(c.original_name.lower(), []).append(
                t.original_name
            )
        cols_per_table[t.original_name] = per
    return _SchemaIndex(
        tables_by_lower=tables_by_lower,
        cols_per_table=cols_per_table,
        tables_for_col=tables_for_col,
    )


# ---------- shared internal pass ----------


@dataclass(frozen=True, slots=True)
class _Walked:
    """Result of the shared AST walk used by both Tier-1 and Tier-2 (and,
    with ``strict=False``, Method D)."""

    tables: set[str]
    columns_with_clauses: dict[tuple[str, str], set[str]]
    star_tables: set[str]
    issues: list[ParseIssue]


def _classify_columns(tree: exp.Expression) -> dict[int, str]:
    """Map ``id(Column)`` to its enclosing clause label.

    Two passes: first tag every column inside a ``Join.on`` as
    ``JOIN_ON``; then for the remaining columns, walk up parent pointers
    until we hit the first clause-defining ancestor.
    """
    out: dict[int, str] = {}
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        for c in on.find_all(exp.Column):
            out[id(c)] = _CLAUSE_JOIN_ON

    for c in tree.find_all(exp.Column):
        if id(c) in out:
            continue
        n = c.parent
        while n is not None:
            if isinstance(n, exp.Where):
                out[id(c)] = _CLAUSE_WHERE
                break
            if isinstance(n, exp.Group):
                out[id(c)] = _CLAUSE_GROUP_BY
                break
            if isinstance(n, exp.Having):
                out[id(c)] = _CLAUSE_HAVING
                break
            if isinstance(n, exp.Order):
                out[id(c)] = _CLAUSE_ORDER_BY
                break
            if isinstance(n, exp.Select):
                out[id(c)] = _CLAUSE_SELECT
                break
            n = n.parent
        else:
            out[id(c)] = _CLAUSE_OTHER
    return out


def _select_top_tables(select: exp.Select) -> list[str]:
    """Raw table names directly attached to ``select`` (FROM + its joins).

    Excludes tables nested in subqueries inside the same SELECT.
    """
    names: list[str] = []
    from_node = select.args.get("from_") or select.args.get("from")
    if from_node is not None:
        for t in from_node.find_all(exp.Table):
            names.append(t.name)
    for join in select.args.get("joins") or []:
        for t in join.find_all(exp.Table):
            # Skip tables nested in the ON expression's subqueries.
            if not _is_join_target(t, join):
                continue
            names.append(t.name)
    return names


def _is_join_target(table: exp.Table, join: exp.Join) -> bool:
    """``True`` iff ``table`` is the join's right-hand-side table, not
    something embedded in the join's ON expression."""
    on = join.args.get("on")
    if on is None:
        return True
    n: exp.Expression | None = table
    while n is not None and n is not join:
        if n is on:
            return False
        n = n.parent
    return True


def _collect_star_tables(
    tree: exp.Expression,
    alias_map: dict[str, str],
    cte_names: set[str],
    idx: _SchemaIndex,
) -> set[str]:
    """Tables that are the source of a ``SELECT *`` or ``t.*`` projection.

    These are kept in Tier 1 even when no concrete column reference exists
    for them — the wildcard implicitly references all of their columns.
    """
    result: set[str] = set()
    for select in tree.find_all(exp.Select):
        has_bare_star = any(isinstance(e, exp.Star) for e in select.expressions)
        if has_bare_star:
            for sub in _select_top_tables(select):
                if sub in cte_names:
                    continue
                resolved = idx.tables_by_lower.get(sub.lower())
                if resolved is not None:
                    result.add(resolved)
        for e in select.expressions:
            if isinstance(e, exp.Column) and e.name == "*":
                qualifier = (e.table or "").lower()
                resolved = alias_map.get(qualifier)
                if resolved is not None:
                    result.add(resolved)
    return result


def _first_from_table(
    tree: exp.Expression, cte_names: set[str], idx: _SchemaIndex
) -> str | None:
    """First schema-resolvable table in the outermost ``FROM`` clause."""
    from_node = tree.find(exp.From)
    if from_node is None:
        return None
    for tnode in from_node.find_all(exp.Table):
        if tnode.name in cte_names:
            continue
        resolved = idx.tables_by_lower.get(tnode.name.lower())
        if resolved is not None:
            return resolved
    return None


def _walk(sql: str, schema: Schema, *, strict: bool) -> _Walked:
    """Parse ``sql`` and resolve its table/column references against
    ``schema``. Never raises — parse failures are reported via
    :class:`ParseIssue`."""
    try:
        tree = sqlglot.parse_one(sql, read=_DIALECT)
    except SqlglotError as exc:
        return _Walked(
            tables=set(),
            columns_with_clauses={},
            star_tables=set(),
            issues=[ParseIssue("parse_error", f"sqlglot parse failed: {exc}")],
        )
    if tree is None:
        return _Walked(
            tables=set(),
            columns_with_clauses={},
            star_tables=set(),
            issues=[ParseIssue("parse_error", "sqlglot returned None")],
        )

    issues: list[ParseIssue] = []
    idx = _build_schema_index(schema)
    cte_names = {cte.alias for cte in tree.find_all(exp.CTE) if cte.alias}

    tables_set: set[str] = set()
    alias_map: dict[str, str] = {}
    for tnode in tree.find_all(exp.Table):
        raw_name = tnode.name
        if raw_name in cte_names:
            continue
        resolved = idx.tables_by_lower.get(raw_name.lower())
        if resolved is None:
            issues.append(ParseIssue("unknown_table", f"unknown table {raw_name!r}"))
            if strict:
                continue
            resolved = raw_name
        tables_set.add(resolved)
        if tnode.alias:
            alias_map[tnode.alias.lower()] = resolved
        alias_map[raw_name.lower()] = resolved
        alias_map[resolved.lower()] = resolved

    first_from_table = _first_from_table(tree, cte_names, idx)
    clauses_by_node = _classify_columns(tree)

    columns_with_clauses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cnode in tree.find_all(exp.Column):
        col_name = cnode.name
        if col_name == "*":
            continue
        clause = clauses_by_node.get(id(cnode), _CLAUSE_OTHER)
        qualifier = (cnode.table or "").lower()
        resolved_table: str | None = None
        if qualifier:
            resolved_table = alias_map.get(qualifier)
            if resolved_table is None:
                issues.append(
                    ParseIssue(
                        "unresolved_alias",
                        f"qualifier {qualifier!r} did not resolve to any "
                        "table or alias",
                    )
                )
        if resolved_table is not None:
            col_orig = idx.cols_per_table.get(resolved_table, {}).get(col_name.lower())
            if col_orig is None:
                issues.append(
                    ParseIssue(
                        "unknown_column",
                        f"column {resolved_table}.{col_name} absent from schema",
                    )
                )
                if strict:
                    continue
                col_orig = col_name
            columns_with_clauses[(resolved_table, col_orig)].add(clause)
            continue

        candidates = idx.tables_for_col.get(col_name.lower(), [])
        in_query = [t for t in candidates if t in tables_set]
        chosen: str | None = None
        if len(in_query) == 1:
            chosen = in_query[0]
        elif len(in_query) > 1:
            chosen = first_from_table if first_from_table in in_query else in_query[0]
        elif len(candidates) == 1:
            chosen = candidates[0]
            tables_set.add(chosen)
        elif candidates:
            chosen = (
                first_from_table if first_from_table in candidates else candidates[0]
            )
            tables_set.add(chosen)
        else:
            issues.append(
                ParseIssue("unknown_column", f"column {col_name!r} unresolvable")
            )
            if not strict:
                fallback_table = cnode.table or first_from_table or ""
                columns_with_clauses[(fallback_table, col_name)].add(clause)
            continue
        columns_with_clauses[
            (chosen, idx.cols_per_table[chosen][col_name.lower()])
        ].add(clause)

    star_tables = _collect_star_tables(tree, alias_map, cte_names, idx)
    star_tables &= tables_set  # don't add tables that weren't already in the query

    return _Walked(
        tables=tables_set,
        columns_with_clauses=dict(columns_with_clauses),
        star_tables=star_tables,
        issues=issues,
    )


def extract_schema_references(
    sql: str, schema: Schema, strict: bool = False
) -> tuple[SchemaReferences, list[ParseIssue]]:
    """Resolve ``sql``'s table/column references against ``schema``.

    Parameters
    ----------
    sql
        SQL text to parse (SQLite dialect).
    schema
        Schema to resolve identifiers against.
    strict
        ``True`` (gold SQL): unknown tables/columns are dropped. ``False``
        (LLM-generated SQL, the default): unknown tables/columns are kept,
        using their raw SQL-text casing.

    Returns
    -------
    tuple[SchemaReferences, list[ParseIssue]]
        The resolved, sorted+deduped references, and any diagnostics
        recorded while resolving them. Never raises — a parse failure
        yields an empty :class:`SchemaReferences` plus a ``"parse_error"``
        issue.
    """
    w = _walk(sql, schema, strict=strict)
    refs = SchemaReferences(
        tables=tuple(sorted(w.tables)),
        columns=tuple(sorted(w.columns_with_clauses.keys())),
    )
    return refs, w.issues


#: Public alias for the ``JOIN ON`` clause role (see :func:`column_roles`).
JOIN_ON: str = _CLAUSE_JOIN_ON


def column_roles(
    sql: str, schema: Schema, strict: bool = True
) -> dict[tuple[str, str], frozenset[str]]:
    """Map each resolved column to the set of SQL clauses it appears in.

    This exposes the clause bookkeeping the walker already performs, so
    callers can distinguish a column that only ever joins from one that is
    also selected or filtered. The JO code needs exactly this distinction:
    ``tier2 - tier1`` is *not* the join-only set, because Taniguchi left some ordinary ``WHERE`` / ``ORDER BY``
    columns unannotated.

    Parameters
    ----------
    sql
        SQL text to parse (SQLite dialect).
    schema
        Schema to resolve identifiers against.
    strict
        ``True`` (the default, and correct for gold SQL): drop unknown
        tables/columns. ``False``: keep them with their raw SQL casing.

    Returns
    -------
    dict[tuple[str, str], frozenset[str]]
        ``(table, column) -> {"SELECT", "WHERE", "JOIN_ON", ...}``. A column
        whose value is exactly ``{"JOIN_ON"}`` is join-only. Returns an
        empty mapping when the SQL does not parse.

    Examples
    --------
    >>> roles = column_roles(sql, schema)  # doctest: +SKIP
    >>> join_only = {c for c, r in roles.items() if r == frozenset({JOIN_ON})}
    """
    walked = _walk(sql, schema, strict=strict)
    return {
        col: frozenset(clauses)
        for col, clauses in walked.columns_with_clauses.items()
    }
