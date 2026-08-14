"""Tests for src/schema_linking/data_loader.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schema_linking.data_loader import SpiderExample, load_spider_questions
from schema_linking.utils.config import Config, DataConfig, OutputsConfig


def test_train_count() -> None:
    assert len(load_spider_questions("train")) == 7000


def test_dev_count() -> None:
    assert len(load_spider_questions("dev")) == 1034


def test_returns_tuple_of_spider_examples() -> None:
    examples = load_spider_questions("dev")
    assert isinstance(examples, tuple)
    assert all(isinstance(e, SpiderExample) for e in examples)


@pytest.mark.parametrize("split", ["train", "dev"])
def test_records_have_truthy_db_id_question_query(split: str) -> None:
    for ex in load_spider_questions(split):  # type: ignore[arg-type]
        assert ex.db_id
        assert ex.question
        assert ex.query


def test_train_dev_db_ids_disjoint() -> None:
    """Spider's cross-domain guarantee: no DB appears in both splits."""
    train_dbs = {e.db_id for e in load_spider_questions("train")}
    dev_dbs = {e.db_id for e in load_spider_questions("dev")}
    assert train_dbs.isdisjoint(dev_dbs)


def test_total_unique_db_ids_across_train_and_dev() -> None:
    """Spider train+dev cover 160 DBs (140 + 20).

    Note: ``tables.json`` contains 166 schemas — the extra 6 are for the
    hidden test set, which we never load. The 166 invariant lives in
    test_schema_parser.test_schema_count.
    """
    train_dbs = {e.db_id for e in load_spider_questions("train")}
    dev_dbs = {e.db_id for e in load_spider_questions("dev")}
    assert len(train_dbs | dev_dbs) == 160


def test_split_field_set_on_every_record() -> None:
    for ex in load_spider_questions("train"):
        assert ex.split == "train"
    for ex in load_spider_questions("dev"):
        assert ex.split == "dev"


def test_question_id_is_sequential_index() -> None:
    examples = load_spider_questions("dev")
    assert tuple(e.question_id for e in examples) == tuple(range(len(examples)))


def test_invalid_split_raises_value_error() -> None:
    with pytest.raises(ValueError, match="split must be"):
        load_spider_questions("test")  # type: ignore[arg-type]


def test_empty_db_id_raises_value_error_with_index(tmp_path: Path) -> None:
    """Synthetic corrupt fixture: one record with empty db_id at index 1."""
    spider_dir = tmp_path / "spider"
    spider_dir.mkdir()
    bad = [
        {"db_id": "ok", "question": "q1", "query": "Q1", "sql": {}},
        {"db_id": "",   "question": "q2", "query": "Q2", "sql": {}},
    ]
    (spider_dir / "dev.json").write_text(json.dumps(bad))
    cfg = Config(
        data=DataConfig(
            spider_dir=spider_dir,
            processed_dir=tmp_path,
            taniguchi_splits_dir=tmp_path,
        ),
        outputs=OutputsConfig(
            predictions_dir=tmp_path,
            results_dir=tmp_path,
            logs_dir=tmp_path,
        ),
    )
    with pytest.raises(ValueError, match=r"empty db_id at dev index 1"):
        load_spider_questions("dev", config=cfg)


def test_empty_question_raises_value_error(tmp_path: Path) -> None:
    spider_dir = tmp_path / "spider"
    spider_dir.mkdir()
    bad = [{"db_id": "ok", "question": "", "query": "Q", "sql": {}}]
    (spider_dir / "dev.json").write_text(json.dumps(bad))
    cfg = Config(
        data=DataConfig(
            spider_dir=spider_dir,
            processed_dir=tmp_path,
            taniguchi_splits_dir=tmp_path,
        ),
        outputs=OutputsConfig(
            predictions_dir=tmp_path,
            results_dir=tmp_path,
            logs_dir=tmp_path,
        ),
    )
    with pytest.raises(ValueError, match=r"empty question at dev index 0"):
        load_spider_questions("dev", config=cfg)


def test_empty_query_raises_value_error(tmp_path: Path) -> None:
    spider_dir = tmp_path / "spider"
    spider_dir.mkdir()
    bad = [{"db_id": "ok", "question": "q", "query": "", "sql": {}}]
    (spider_dir / "dev.json").write_text(json.dumps(bad))
    cfg = Config(
        data=DataConfig(
            spider_dir=spider_dir,
            processed_dir=tmp_path,
            taniguchi_splits_dir=tmp_path,
        ),
        outputs=OutputsConfig(
            predictions_dir=tmp_path,
            results_dir=tmp_path,
            logs_dir=tmp_path,
        ),
    )
    with pytest.raises(ValueError, match=r"empty query at dev index 0"):
        load_spider_questions("dev", config=cfg)


def test_explicit_config_overrides_default(tmp_path: Path) -> None:
    """Passing a config should bypass repo-root config.yaml entirely."""
    spider_dir = tmp_path / "spider"
    spider_dir.mkdir()
    one = [{"db_id": "x", "question": "q", "query": "Q", "sql": {}}]
    (spider_dir / "train_spider.json").write_text(json.dumps(one))
    cfg = Config(
        data=DataConfig(
            spider_dir=spider_dir,
            processed_dir=tmp_path,
            taniguchi_splits_dir=tmp_path,
        ),
        outputs=OutputsConfig(
            predictions_dir=tmp_path,
            results_dir=tmp_path,
            logs_dir=tmp_path,
        ),
    )
    examples = load_spider_questions("train", config=cfg)
    assert len(examples) == 1
    assert examples[0].db_id == "x"
    assert examples[0].split == "train"
