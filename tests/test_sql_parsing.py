"""Tests for src/schema_linking/utils/sql_parsing.py.

The per-edge-case tests (E1-E10) use hand-built schemas + synthetic queries
so the assertions remain independent of Spider quirks. All calls use
``strict=True`` (the gold-extraction mode) unless a test is specifically
about ``strict=False`` (LLM-generated-SQL / hallucination) behaviour.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from schema_linking.schema_parser import Column, Schema, Table
from schema_linking.utils.sql_parsing import (
    ParseIssue,
    SchemaReferences,
    extract_schema_references,
)


# ---------- helpers ----------


def _table(name: str, cols: Iterable[str]) -> Table:
    return Table(
        name=name,
        original_name=name,
        columns=[
            Column(
                name=c,
                original_name=c,
                type="number",
                table_name=name,
                is_primary_key=False,
            )
            for c in cols
        ],
    )


def _schema(db_id: str, tables: list[Table]) -> Schema:
    return Schema(db_id=db_id, tables=tables, foreign_keys=[])


@pytest.fixture
def two_table_schema() -> Schema:
    """t1(a, b, shared) + t2(x, y, shared)."""
    return _schema(
        "synth",
        [
            _table("t1", ["a", "b", "shared"]),
            _table("t2", ["x", "y", "shared"]),
        ],
    )


# ---------- E1: SELECT * ----------


def test_E1_select_star_adds_no_columns(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references("SELECT * FROM t1", two_table_schema, strict=True)
    assert g.tables == ("t1",)
    assert g.columns == ()


# ---------- E2: aggregates ----------


def test_E2_count_star_adds_no_columns(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT COUNT(*) FROM t1", two_table_schema, strict=True
    )
    assert g.tables == ("t1",)
    assert g.columns == ()


def test_E2_count_qualified_column(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT COUNT(t1.a) FROM t1", two_table_schema, strict=True
    )
    assert g.tables == ("t1",)
    assert g.columns == (("t1", "a"),)


def test_E2_max_column(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT MAX(a) FROM t1", two_table_schema, strict=True
    )
    assert g.columns == (("t1", "a"),)


# ---------- E3: aliases ----------


def test_E3_alias_resolves_to_underlying_table(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT T1.a, T1.b FROM t1 AS T1", two_table_schema, strict=True
    )
    assert g.tables == ("t1",)
    assert g.columns == (("t1", "a"), ("t1", "b"))


def test_E3_alias_across_join(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT T1.a, T2.x FROM t1 AS T1 JOIN t2 AS T2 ON T1.b = T2.y",
        two_table_schema,
        strict=True,
    )
    assert g.tables == ("t1", "t2")
    assert g.columns == (("t1", "a"), ("t1", "b"), ("t2", "x"), ("t2", "y"))


# ---------- E4: subquery in FROM ----------


def test_E4_subquery_in_from(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT s.a FROM (SELECT a FROM t1) AS s", two_table_schema, strict=True
    )
    assert "t1" in g.tables
    assert ("t1", "a") in g.columns


# ---------- E5: nested SELECT in WHERE ----------


def test_E5_nested_select_in_where(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT a FROM t1 WHERE a IN (SELECT x FROM t2)",
        two_table_schema,
        strict=True,
    )
    assert set(g.tables) == {"t1", "t2"}
    assert ("t1", "a") in g.columns
    assert ("t2", "x") in g.columns


# ---------- E6: CTEs ----------


def test_E6_cte_excludes_cte_name_includes_source(
    two_table_schema: Schema,
) -> None:
    g, _ = extract_schema_references(
        "WITH cte AS (SELECT a, b FROM t1) "
        "SELECT a FROM cte WHERE b > 5",
        two_table_schema,
        strict=True,
    )
    assert g.tables == ("t1",)
    assert set(g.columns) == {("t1", "a"), ("t1", "b")}


# ---------- E7: UNION / INTERSECT / EXCEPT ----------


def test_E7_union_collects_both_branches(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT a FROM t1 UNION SELECT x FROM t2", two_table_schema, strict=True
    )
    assert set(g.tables) == {"t1", "t2"}
    assert set(g.columns) == {("t1", "a"), ("t2", "x")}


def test_E7_intersect_collects_both_branches(
    two_table_schema: Schema,
) -> None:
    g, _ = extract_schema_references(
        "SELECT a FROM t1 INTERSECT SELECT x FROM t2",
        two_table_schema,
        strict=True,
    )
    assert set(g.tables) == {"t1", "t2"}


def test_E7_except_collects_both_branches(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT a FROM t1 EXCEPT SELECT x FROM t2", two_table_schema, strict=True
    )
    assert set(g.tables) == {"t1", "t2"}


# ---------- E8: self-joins ----------


def test_E8_self_join_adds_table_once(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT T1.a, T2.b FROM t1 AS T1 JOIN t1 AS T2 ON T1.a = T2.a",
        two_table_schema,
        strict=True,
    )
    assert g.tables == ("t1",)
    assert set(g.columns) == {("t1", "a"), ("t1", "b")}


# ---------- E9: graceful handling of unknown identifiers ----------


def test_E9_unknown_column_logged_and_skipped(two_table_schema: Schema) -> None:
    g, issues = extract_schema_references(
        "SELECT a, nonexistent FROM t1", two_table_schema, strict=True
    )
    assert ("t1", "a") in g.columns
    assert not any(c[1] == "nonexistent" for c in g.columns)
    assert any(
        i.kind == "unknown_column" and "nonexistent" in i.detail for i in issues
    )


def test_E9_unknown_qualified_column_logged_and_skipped(
    two_table_schema: Schema,
) -> None:
    g, issues = extract_schema_references(
        "SELECT t1.bogus FROM t1", two_table_schema, strict=True
    )
    assert g.tables == ("t1",)
    assert g.columns == ()
    assert any(i.kind == "unknown_column" and "bogus" in i.detail for i in issues)


# ---------- E10: function on column ----------


def test_E10_function_on_column(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT SUBSTR(b, 1, 3) FROM t1", two_table_schema, strict=True
    )
    assert g.tables == ("t1",)
    assert g.columns == (("t1", "b"),)


def test_E10_nested_function_on_column(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT UPPER(TRIM(b)) FROM t1", two_table_schema, strict=True
    )
    assert g.columns == (("t1", "b"),)


# ---------- Ambiguous unqualified column (first-FROM heuristic) ----------


def test_ambiguous_unqualified_attaches_to_first_from(
    two_table_schema: Schema,
) -> None:
    """``shared`` exists in both t1 and t2; without a qualifier it attaches
    to the first FROM table (t1)."""
    g, _ = extract_schema_references(
        "SELECT shared FROM t1 JOIN t2 ON t1.a = t2.x",
        two_table_schema,
        strict=True,
    )
    assert ("t1", "shared") in g.columns
    assert ("t2", "shared") not in g.columns


# ---------- Sorting / dedup invariants ----------


def test_tables_and_columns_are_sorted_and_deduped(
    two_table_schema: Schema,
) -> None:
    g, _ = extract_schema_references(
        "SELECT t1.a, t2.x, t1.b, t2.y FROM t1 JOIN t2 ON t1.a = t2.x "
        "WHERE t1.a = 1",
        two_table_schema,
        strict=True,
    )
    assert list(g.tables) == sorted(set(g.tables))
    assert list(g.columns) == sorted(set(g.columns))


def test_case_insensitive_table_and_column_resolution(
    two_table_schema: Schema,
) -> None:
    g, _ = extract_schema_references(
        "SELECT A, B FROM T1", two_table_schema, strict=True
    )
    assert g.tables == ("t1",)
    assert g.columns == (("t1", "a"), ("t1", "b"))


# ---------- ORDER BY / GROUP BY / HAVING ----------


def test_collects_columns_from_all_clauses(two_table_schema: Schema) -> None:
    g, _ = extract_schema_references(
        "SELECT a FROM t1 WHERE b > 0 GROUP BY a "
        "HAVING SUM(b) > 10 ORDER BY a",
        two_table_schema,
        strict=True,
    )
    assert g.columns == (("t1", "a"), ("t1", "b"))


# ---------- strict=False (Method D / LLM-generated SQL) behaviour ----------


def test_strict_false_includes_hallucinated_column_with_issue(
    two_table_schema: Schema,
) -> None:
    """A real table but a made-up column name: strict=False keeps the
    reference (Method D treats a hallucination as a valid prediction) and
    records why."""
    g, issues = extract_schema_references(
        "SELECT t1.bogus FROM t1", two_table_schema, strict=False
    )
    assert ("t1", "bogus") in g.columns
    assert any(i.kind == "unknown_column" for i in issues)


def test_strict_false_unqualified_unknown_column_attaches_to_first_from_table(
    two_table_schema: Schema,
) -> None:
    """A totally unresolvable, unqualified column (no table has anything
    matching its name) attaches to the query's FROM table under
    strict=False, rather than being left tableless — this is what makes a
    hallucinated column usable as a Method D prediction."""
    g, issues = extract_schema_references(
        "SELECT fake_col FROM t1", two_table_schema, strict=False
    )
    assert ("t1", "fake_col") in g.columns
    assert any(i.kind == "unknown_column" for i in issues)


def test_strict_false_parse_error_returns_empty_refs(
    two_table_schema: Schema,
) -> None:
    """Syntactically broken SQL never raises; it yields empty references
    plus a parse_error issue, regardless of strict."""
    g, issues = extract_schema_references(
        "SELECT (((", two_table_schema, strict=False
    )
    assert g == SchemaReferences(tables=(), columns=())
    assert any(i.kind == "parse_error" for i in issues)
