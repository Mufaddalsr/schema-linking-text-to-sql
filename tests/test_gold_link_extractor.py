"""Tests for src/schema_linking/gold_link_extractor.py.

Walker-level edge cases (E1-E10) live in tests/test_sql_parsing.py against
the shared ``extract_schema_references``. This file covers Tier-1-specific
filtering behaviour, the batch drivers (``extract_tier1_all`` /
``extract_tier2_all``), and integration tests against Spider dev/train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

from schema_linking.data_loader import SpiderExample, load_spider_questions
from schema_linking.gold_link_extractor import (
    ExtractionError,
    extract_tier1,
    extract_tier1_all,
    extract_tier2,
    extract_tier2_all,
)
from schema_linking.schema_parser import Column, Schema, Table, load_schemas


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


def _example(query: str, db_id: str = "synth", qid: int = 0) -> SpiderExample:
    return SpiderExample(
        question_id=qid,
        db_id=db_id,
        question="",
        query=query,
        sql={},
        split="dev",
    )


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


def test_returns_db_id_on_gold_links(two_table_schema: Schema) -> None:
    g = extract_tier2(_example("SELECT * FROM t1"), two_table_schema)
    assert g.db_id == "synth"


# ---------- extract_tier2_all + error handling ----------


def test_extract_tier2_all_skips_bad_queries_and_logs(
    two_table_schema: Schema, tmp_path: Path
) -> None:
    examples = [
        _example("SELECT a FROM t1", qid=0),
        _example("SELECT (((", qid=1),  # parse error
        _example("SELECT x FROM t2", qid=2),
    ]
    schemas = {"synth": two_table_schema}
    log = tmp_path / "errors.jsonl"
    out = extract_tier2_all(examples, schemas, error_log_path=log)
    assert set(out.keys()) == {0, 2}
    assert log.is_file()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["question_id"] == 1
    assert records[0]["error_type"] in {"ExtractionError"}


def test_extract_tier2_all_fail_on_parse_error_raises(
    two_table_schema: Schema,
) -> None:
    examples = [_example("SELECT (((", qid=1)]
    schemas = {"synth": two_table_schema}
    with pytest.raises(ExtractionError):
        extract_tier2_all(examples, schemas, fail_on_parse_error=True)


def test_missing_schema_for_db_id_is_logged_not_raised(
    two_table_schema: Schema, tmp_path: Path
) -> None:
    examples = [_example("SELECT a FROM t1", db_id="missing_db", qid=7)]
    schemas = {"synth": two_table_schema}
    log = tmp_path / "errors.jsonl"
    out = extract_tier2_all(examples, schemas, error_log_path=log)
    assert out == {}
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["error_type"] == "MissingSchema"


# ---------- Integration: Spider dev + train ----------


@pytest.fixture(scope="module")
def real_schemas() -> dict[str, Schema]:
    return load_schemas()


def test_integration_spider_dev_parse_failure_under_one_percent(
    real_schemas: dict[str, Schema],
) -> None:
    dev = load_spider_questions("dev")
    out = extract_tier2_all(dev, real_schemas)
    failed = len(dev) - len(out)
    ratio = failed / len(dev)
    print(f"\nSpider dev: {len(out)}/{len(dev)} extracted, "
          f"{failed} failures ({ratio:.4%})")
    assert ratio < 0.01, f"dev failure rate {ratio:.4%} exceeds 1%"


def test_integration_spider_train_parse_failure_under_two_percent(
    real_schemas: dict[str, Schema],
) -> None:
    train = load_spider_questions("train")
    out = extract_tier2_all(train, real_schemas)
    failed = len(train) - len(out)
    ratio = failed / len(train)
    print(f"\nSpider train: {len(out)}/{len(train)} extracted, "
          f"{failed} failures ({ratio:.4%})")
    assert ratio < 0.02, f"train failure rate {ratio:.4%} exceeds 2%"


# ============================================================
# Tier-1 ("Mentioned, strict") behaviour
# ============================================================


def test_tier1_drops_join_on_only_columns(two_table_schema: Schema) -> None:
    """In ``SELECT a FROM t1 JOIN t2 ON t1.b = t2.x``, ``t1.b`` and ``t2.x``
    only appear in JOIN ON → Tier-1 keeps only ``t1.a`` and drops ``t2``."""
    ex = _example("SELECT a FROM t1 JOIN t2 ON t1.b = t2.x")
    t1 = extract_tier1(ex, two_table_schema)
    t2 = extract_tier2(ex, two_table_schema)
    assert t1.tables == ("t1",)
    assert t1.columns == (("t1", "a"),)
    assert set(t2.tables) == {"t1", "t2"}
    assert {("t1", "b"), ("t2", "x")}.issubset(set(t2.columns))


def test_tier1_keeps_column_used_in_both_select_and_join_on(
    two_table_schema: Schema,
) -> None:
    """If a column appears in both JOIN ON and elsewhere, Tier-1 keeps it."""
    ex = _example("SELECT t1.b FROM t1 JOIN t2 ON t1.b = t2.x")
    g = extract_tier1(ex, two_table_schema)
    assert ("t1", "b") in g.columns
    assert ("t2", "x") not in g.columns
    assert g.tables == ("t1",)


def test_tier1_keeps_where_columns(two_table_schema: Schema) -> None:
    ex = _example("SELECT a FROM t1 JOIN t2 ON t1.b = t2.x WHERE t2.y > 5")
    g = extract_tier1(ex, two_table_schema)
    assert ("t2", "y") in g.columns
    assert set(g.tables) == {"t1", "t2"}


def test_tier1_keeps_group_having_order_columns(
    two_table_schema: Schema,
) -> None:
    ex = _example(
        "SELECT a, COUNT(*) FROM t1 JOIN t2 ON t1.b = t2.x "
        "GROUP BY a HAVING COUNT(t2.y) > 1 ORDER BY a"
    )
    g = extract_tier1(ex, two_table_schema)
    assert ("t1", "a") in g.columns
    assert ("t2", "y") in g.columns
    # JOIN-only columns are dropped
    assert ("t1", "b") not in g.columns
    assert ("t2", "x") not in g.columns
    assert set(g.tables) == {"t1", "t2"}


def test_tier1_select_star_keeps_from_table(two_table_schema: Schema) -> None:
    """`SELECT * FROM t1` has no concrete column refs; t1 is kept via
    the star-tables rule."""
    g = extract_tier1(_example("SELECT * FROM t1"), two_table_schema)
    assert g.tables == ("t1",)
    assert g.columns == ()


def test_tier1_select_star_with_join_keeps_both_tables(
    two_table_schema: Schema,
) -> None:
    """``SELECT * FROM t1 JOIN t2 ON …`` — bare ``*`` implicitly references
    all FROM/JOIN tables, so both survive Tier-1."""
    ex = _example("SELECT * FROM t1 JOIN t2 ON t1.b = t2.x")
    g = extract_tier1(ex, two_table_schema)
    assert set(g.tables) == {"t1", "t2"}
    assert g.columns == ()


def test_tier1_select_one_keeps_from_table(two_table_schema: Schema) -> None:
    """``SELECT 1 FROM t1`` — no column refs anywhere; t1 survives via the
    no-references rule."""
    g = extract_tier1(_example("SELECT 1 FROM t1"), two_table_schema)
    assert g.tables == ("t1",)
    assert g.columns == ()


def test_tier1_qualified_star_keeps_that_table(
    two_table_schema: Schema,
) -> None:
    """``SELECT t2.* FROM t1 JOIN t2 ON t1.b = t2.x`` — ``t2.*`` keeps t2,
    JOIN-ON-only t1 is dropped."""
    ex = _example("SELECT t2.* FROM t1 JOIN t2 ON t1.b = t2.x")
    g = extract_tier1(ex, two_table_schema)
    assert g.tables == ("t2",)


def test_tier1_subset_of_tier2(two_table_schema: Schema) -> None:
    """Tier-1 must be a strict subset of Tier-2 for every query."""
    queries = [
        "SELECT a FROM t1",
        "SELECT a FROM t1 JOIN t2 ON t1.b = t2.x",
        "SELECT t1.b FROM t1 JOIN t2 ON t1.b = t2.x",
        "SELECT a FROM t1 WHERE a > 0 GROUP BY a HAVING COUNT(*) > 1 ORDER BY a",
        "SELECT a FROM t1 WHERE a IN (SELECT x FROM t2)",
    ]
    for q in queries:
        g1 = extract_tier1(_example(q), two_table_schema)
        g2 = extract_tier2(_example(q), two_table_schema)
        assert set(g1.tables).issubset(set(g2.tables)), q
        assert set(g1.columns).issubset(set(g2.columns)), q


def test_tier1_union_keeps_branch_columns(two_table_schema: Schema) -> None:
    """UNION's branches each contribute SELECT-list columns to Tier-1."""
    g = extract_tier1(
        _example("SELECT a FROM t1 UNION SELECT x FROM t2"),
        two_table_schema,
    )
    assert set(g.tables) == {"t1", "t2"}
    assert set(g.columns) == {("t1", "a"), ("t2", "x")}


def test_tier1_extract_all_integration_dev(
    real_schemas: dict[str, Schema],
) -> None:
    """Tier-1 over Spider dev must succeed at the same rate as Tier-2 and
    each entry must be a subset of the corresponding Tier-2 entry."""
    dev = load_spider_questions("dev")
    t1 = extract_tier1_all(dev, real_schemas)
    t2 = extract_tier2_all(dev, real_schemas)
    assert set(t1.keys()) == set(t2.keys())
    for qid, g1 in t1.items():
        g2 = t2[qid]
        assert set(g1.tables).issubset(set(g2.tables)), (
            f"Tier-1 has tables not in Tier-2 at qid {qid}: "
            f"{set(g1.tables) - set(g2.tables)}"
        )
        assert set(g1.columns).issubset(set(g2.columns)), (
            f"Tier-1 has columns not in Tier-2 at qid {qid}: "
            f"{set(g1.columns) - set(g2.columns)}"
        )


def test_integration_dev_spot_check_matches_taniguchi_for_simple_cases(
    real_schemas: dict[str, Schema],
) -> None:
    """Sanity: for the 3 paper Appendix-E questions, Tier-2 must include the
    Taniguchi-annotated tables and columns (Tier-2 is a superset of Tier-1
    mentioned items)."""
    dev = load_spider_questions("dev")
    by_q = {ex.question: ex for ex in dev}

    cases = [
        (
            "Count the number of templates.",
            {"Templates"},
            set(),
        ),
        (
            "Which airline has abbreviation 'UAL'?",
            {"airlines"},
            {("airlines", "Airline"), ("airlines", "Abbreviation")},
        ),
        (
            "How many orchestras does each record company manage?",
            {"orchestra"},
            {("orchestra", "Record_Company")},
        ),
    ]
    for q, exp_tables, exp_columns in cases:
        ex = by_q[q]
        g = extract_tier2(ex, real_schemas[ex.db_id])
        assert exp_tables.issubset(set(g.tables)), (
            f"missing tables for {q!r}: expected ⊇ {exp_tables}, got {g.tables}"
        )
        assert exp_columns.issubset(set(g.columns)), (
            f"missing columns for {q!r}: expected ⊇ {exp_columns}, got {g.columns}"
        )
