"""Tests for src/schema_linking/taniguchi_loader.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from schema_linking.data_loader import SpiderExample, load_spider_questions
from schema_linking.schema_parser import Schema, load_schemas
from schema_linking.taniguchi_loader import (
    GoldLinks,
    TaniguchiAnnotation,
    load_taniguchi_annotations,
    save_gold_links,
    to_gold_links,
)


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def annotations() -> dict[int, TaniguchiAnnotation]:
    return load_taniguchi_annotations()


@pytest.fixture(scope="module")
def spider_dev() -> tuple[SpiderExample, ...]:
    return load_spider_questions("dev")


@pytest.fixture(scope="module")
def schemas() -> dict[str, Schema]:
    return load_schemas()


@pytest.fixture(scope="module")
def gold(
    annotations: dict[int, TaniguchiAnnotation],
    spider_dev: tuple[SpiderExample, ...],
    schemas: dict[str, Schema],
) -> dict[int, GoldLinks]:
    return to_gold_links(annotations, spider_dev, schemas)


# ---------- load_taniguchi_annotations ----------


def test_total_annotations_cover_spider_dev(
    annotations: dict[int, TaniguchiAnnotation],
) -> None:
    """dev.jsonl + test.jsonl jointly cover Spider dev (1034 examples)."""
    assert len(annotations) == 1034


def test_annotation_records_are_typed(
    annotations: dict[int, TaniguchiAnnotation],
) -> None:
    for ann in annotations.values():
        assert isinstance(ann, TaniguchiAnnotation)
        assert isinstance(ann.id, int)
        assert ann.text
        for lbl in ann.labels:
            assert len(lbl) == 3
            start, end, text = lbl
            assert isinstance(start, int) and isinstance(end, int)
            assert isinstance(text, str) and text


def test_labels_resolved_to_table_or_table_dot_column(
    annotations: dict[int, TaniguchiAnnotation],
) -> None:
    for ann in annotations.values():
        for _start, _end, label in ann.labels:
            # Either a bare table name (no dot) or "table.column" (exactly one dot)
            assert label.count(".") in (0, 1)


def test_ids_are_contiguous(annotations: dict[int, TaniguchiAnnotation]) -> None:
    """The id-offset fallback relies on contiguous Taniguchi ids."""
    ids = sorted(annotations.keys())
    assert ids == list(range(ids[0], ids[-1] + 1))
    assert len(ids) == 1034


# ---------- to_gold_links ----------


def test_every_question_maps_to_one_spider_qid(
    annotations: dict[int, TaniguchiAnnotation],
    spider_dev: tuple[SpiderExample, ...],
    schemas: dict[str, Schema],
) -> None:
    """All 1034 Taniguchi annotations resolve to a Spider qid (text-first,
    id-offset fallback for the ~46 cases where Taniguchi normalised the
    question text)."""
    gold = to_gold_links(annotations, spider_dev, schemas)
    assert len(gold) == 1034
    spider_qids = {ex.question_id for ex in spider_dev}
    assert set(gold.keys()) == spider_qids


def test_unresolvable_label_rate_under_one_percent(
    annotations: dict[int, TaniguchiAnnotation],
    spider_dev: tuple[SpiderExample, ...],
    schemas: dict[str, Schema],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Conversion must succeed at the default 1% ceiling."""
    with caplog.at_level(logging.INFO, logger="schema_linking.taniguchi_loader"):
        gold = to_gold_links(annotations, spider_dev, schemas)
    summary = [r for r in caplog.records if "to_gold_links:" in r.getMessage()]
    assert summary, "expected a summary log line from to_gold_links"
    assert "unresolvable=0" in summary[0].getMessage() or any(
        "unresolvable=" in r.getMessage() for r in caplog.records
    )
    assert gold


def test_unresolvable_threshold_raises(
    spider_dev: tuple[SpiderExample, ...],
    schemas: dict[str, Schema],
) -> None:
    """A label that doesn't validate counts against the ceiling."""
    ex = spider_dev[0]
    bad = {
        1: TaniguchiAnnotation(
            id=1,
            text=ex.question,
            sql="",
            labels=(
                (0, 1, "NoSuchTable"),
                (0, 1, "AlsoMissing"),
            ),
        )
    }
    with pytest.raises(ValueError, match=r"unresolvable label fraction"):
        to_gold_links(bad, spider_dev, schemas, max_unresolvable_ratio=0.01)


def test_tables_and_columns_are_sorted_and_deduped(
    gold: dict[int, GoldLinks],
) -> None:
    for g in gold.values():
        assert list(g.tables) == sorted(set(g.tables))
        assert list(g.columns) == sorted(set(g.columns))


def test_column_implies_table(gold: dict[int, GoldLinks]) -> None:
    """Every (t, c) in columns must also surface in tables."""
    for g in gold.values():
        column_tables = {t for t, _c in g.columns}
        assert column_tables.issubset(set(g.tables))


# ---------- spot checks (paper Appendix E) ----------


SPOT_CHECKS: list[tuple[str, str, list[str], list[tuple[str, str]]]] = [
    (
        "Count the number of templates.",
        "cre_Doc_Template_Mgt",
        ["Templates"],
        [],
    ),
    (
        "Which airline has abbreviation 'UAL'?",
        "flight_2",
        ["airlines"],
        [("airlines", "Airline"), ("airlines", "Abbreviation")],
    ),
    (
        "How many orchestras does each record company manage?",
        "orchestra",
        ["orchestra"],
        [("orchestra", "Record_Company")],
    ),
]


@pytest.mark.parametrize("question,db_id,exp_tables,exp_columns", SPOT_CHECKS)
def test_spot_check_matches_paper(
    gold: dict[int, GoldLinks],
    spider_dev: tuple[SpiderExample, ...],
    question: str,
    db_id: str,
    exp_tables: list[str],
    exp_columns: list[tuple[str, str]],
) -> None:
    qid = next(ex.question_id for ex in spider_dev if ex.question == question)
    g = gold[qid]
    assert g.db_id == db_id
    assert set(g.tables) == set(exp_tables)
    assert set(g.columns) == set(exp_columns)


# ---------- save_gold_links ----------


def test_save_round_trip_preserves_content(tmp_path: Path) -> None:
    sample = {
        0: GoldLinks(
            db_id="concert_singer",
            tables=("singer",),
            columns=(),
        ),
        318: GoldLinks(
            db_id="cre_Doc_Template_Mgt",
            tables=("Templates",),
            columns=(),
        ),
    }
    out = tmp_path / "sub" / "gold.json"
    save_gold_links(sample, out)
    assert out.is_file()

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {"0", "318"}
    assert loaded["0"] == {
        "db_id": "concert_singer",
        "tables": ["singer"],
        "columns": [],
    }
    assert loaded["318"] == {
        "db_id": "cre_Doc_Template_Mgt",
        "tables": ["Templates"],
        "columns": [],
    }


# ---------- error path ----------


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_taniguchi_annotations([tmp_path / "nope.jsonl"])


def test_bad_meta_key_raises(tmp_path: Path) -> None:
    bad = {
        "id": 1,
        "text": "q",
        "meta": {"000_sql": "SELECT 1"},  # no label keys defined
        "labels": [[0, 1, "001_table_99"]],
    }
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not present in meta"):
        load_taniguchi_annotations([p])
