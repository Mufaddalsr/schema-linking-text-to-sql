"""Tests for ``schema_linking.utils.tuning`` (lexical and embedding tuning).

Self-contained: builds tiny synthetic schemas and datasets. Does NOT touch
real Spider data or download any model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import schema_linking.utils.tuning as tuning_module
from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Column, Schema, Table
from schema_linking.utils.tuning import tune_embedding, tune_fuzzy_threshold


def _table(name: str, cols: list[tuple[str, str]]) -> Table:
    columns = [
        Column(
            name=oname.replace("_", " "),
            original_name=oname,
            type=ctype,
            table_name=name,
            is_primary_key=False,
        )
        for oname, ctype in cols
    ]
    return Table(name=name, original_name=name, columns=columns)


@pytest.fixture
def schema() -> Schema:
    return Schema(
        db_id="test_db",
        tables=[
            _table("users", [("id", "number"), ("name", "text")]),
            _table("orders", [("id", "number"), ("amount", "number")]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def synthetic_examples_and_gold(
    schema: Schema,
) -> tuple[list[SpiderExample], dict[int, dict]]:
    """50 rotating examples with plausibly-correct gold."""
    plans = [
        ("how many users are there", ["users"], [["users", "id"]]),
        ("show all orders", ["orders"], []),
        ("list user names", ["users"], [["users", "name"]]),
        ("amount per order", ["orders"], [["orders", "amount"]]),
        ("find users with the largest order amount", ["users", "orders"], [["orders", "amount"]]),
    ]
    examples: list[SpiderExample] = []
    gold: dict[int, dict] = {}
    for i in range(50):
        question, tbls, cols = plans[i % len(plans)]
        examples.append(
            SpiderExample(
                question_id=i,
                db_id="test_db",
                question=question,
                query="SELECT 1",
                sql={},
                split="train",
            )
        )
        gold[i] = {"db_id": "test_db", "tables": list(tbls), "columns": [list(c) for c in cols]}
    return examples, gold


class TestTuneFuzzyThreshold:
    def test_best_is_one_of_the_candidates(
        self,
        schema: Schema,
        synthetic_examples_and_gold: tuple[list[SpiderExample], dict[int, dict]],
    ) -> None:
        examples, gold = synthetic_examples_and_gold
        candidates = [70, 80, 90]
        best, _ = tune_fuzzy_threshold(
            examples=examples,
            gold=gold,
            schemas={"test_db": schema},
            candidates=candidates,
        )
        assert best in candidates

    def test_sweep_table_has_one_row_per_candidate(
        self,
        schema: Schema,
        synthetic_examples_and_gold: tuple[list[SpiderExample], dict[int, dict]],
    ) -> None:
        examples, gold = synthetic_examples_and_gold
        candidates = [70, 75, 80, 85, 90, 95]
        _, sweep = tune_fuzzy_threshold(
            examples=examples,
            gold=gold,
            schemas={"test_db": schema},
            candidates=candidates,
        )
        assert isinstance(sweep, pd.DataFrame)
        assert len(sweep) == len(candidates)
        assert set(sweep["fuzzy_threshold"]) == set(candidates)

    def test_smoke_50_examples_no_crash(
        self,
        schema: Schema,
        synthetic_examples_and_gold: tuple[list[SpiderExample], dict[int, dict]],
    ) -> None:
        examples, gold = synthetic_examples_and_gold
        best, sweep = tune_fuzzy_threshold(
            examples=examples,
            gold=gold,
            schemas={"test_db": schema},
        )
        # default candidates list length is 6
        assert len(sweep) == 6
        assert isinstance(best, int)

    def test_column_element_type_picks_by_column_f1(
        self,
        schema: Schema,
        synthetic_examples_and_gold: tuple[list[SpiderExample], dict[int, dict]],
    ) -> None:
        examples, gold = synthetic_examples_and_gold
        candidates = [70, 80, 90]
        best, sweep = tune_fuzzy_threshold(
            examples=examples,
            gold=gold,
            schemas={"test_db": schema},
            candidates=candidates,
            element_type="column",
        )
        # The winning threshold must have the highest column_f1 in the sweep
        winner_col_f1 = sweep.loc[sweep["fuzzy_threshold"] == best, "column_f1"].iloc[0]
        assert winner_col_f1 == sweep["column_f1"].max()

    def test_sweep_columns_include_pr_f1_for_both_levels(
        self,
        schema: Schema,
        synthetic_examples_and_gold: tuple[list[SpiderExample], dict[int, dict]],
    ) -> None:
        examples, gold = synthetic_examples_and_gold
        _, sweep = tune_fuzzy_threshold(
            examples=examples,
            gold=gold,
            schemas={"test_db": schema},
            candidates=[80],
        )
        expected = {
            "fuzzy_threshold",
            "table_precision",
            "table_recall",
            "table_f1",
            "column_precision",
            "column_recall",
            "column_f1",
        }
        assert expected.issubset(sweep.columns)


class _FakeEmbeddingEncoder:
    """Stands in for ``SchemaEncoder``: fixed schema index, scripted question vectors."""

    def __init__(self, schema_index: dict, question_vectors: dict[str, np.ndarray]) -> None:
        self._schema_index = schema_index
        self._question_vectors = question_vectors

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.array([self._question_vectors[t] for t in texts])

    def encode_schema(self, schemas: dict[str, Schema]) -> dict:
        return self._schema_index


def _embedding_schema() -> Schema:
    """3 tables, 1 column each — matches the one-hot fake index below."""
    tables = []
    for name in ("A", "B", "C"):
        col = Column(
            name=f"{name} col", original_name=f"{name}Col", type="text",
            table_name=name, is_primary_key=False,
        )
        tables.append(Table(name=name, original_name=name, columns=[col]))
    return Schema(db_id="test_db", tables=tables, foreign_keys=[])


def _embedding_schema_index() -> dict:
    onehots = np.eye(3, dtype=np.float64)
    return {
        "test_db": {
            "table_names": ("A", "B", "C"),
            "table_vectors": onehots.copy(),
            "column_names": (("A", "ACol"), ("B", "BCol"), ("C", "CCol")),
            "column_vectors": onehots.copy(),
        }
    }


def _embedding_examples_and_gold() -> tuple[list[SpiderExample], dict[int, dict]]:
    """50 examples, each gold-linked to table A / column A.ACol."""
    examples: list[SpiderExample] = []
    gold: dict[int, dict] = {}
    for i in range(50):
        examples.append(
            SpiderExample(
                question_id=i,
                db_id="test_db",
                question=f"question {i}",
                query="SELECT 1",
                sql={},
                split="train",
            )
        )
        gold[i] = {"db_id": "test_db", "tables": ["A"], "columns": [["A", "ACol"]]}
    return examples, gold


def _patch_minimal_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2 x 1 x 1 x 2 = 4-cell grid, small enough for a fast smoke test."""
    monkeypatch.setattr(tuning_module, "TABLE_TOP_K_GRID", (1, 2))
    monkeypatch.setattr(tuning_module, "TABLE_THRESHOLD_GRID", (0.5,))
    monkeypatch.setattr(tuning_module, "COLUMN_TOP_K_GRID", (3,))
    monkeypatch.setattr(tuning_module, "COLUMN_THRESHOLD_GRID", (0.3, 0.5))


class TestTuneEmbedding:
    def test_smoke_minimal_grid_returns_winner_in_grid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_minimal_grid(monkeypatch)
        examples, gold = _embedding_examples_and_gold()
        question_vectors = {
            ex.question: np.array([1.0, 0.0, 0.0]) for ex in examples
        }
        encoder = _FakeEmbeddingEncoder(_embedding_schema_index(), question_vectors)

        best, sweep = tune_embedding(
            examples=examples,
            gold=gold,
            schemas={"test_db": _embedding_schema()},
            encoder=encoder,
            gold_tier="all_sql_used",
        )

        assert isinstance(sweep, pd.DataFrame)
        assert len(sweep) == 4  # 2 x 1 x 1 x 2
        assert set(best.keys()) == {
            "table_top_k",
            "table_threshold",
            "column_top_k",
            "column_threshold",
        }
        assert best["table_top_k"] in (1, 2)
        assert best["table_threshold"] == 0.5
        assert best["column_top_k"] == 3
        assert best["column_threshold"] in (0.3, 0.5)

        expected_cols = {
            "table_top_k",
            "table_threshold",
            "column_top_k",
            "column_threshold",
            "table_f1",
            "column_f1",
            "mean_f1",
            "mean_recall",
        }
        assert expected_cols.issubset(sweep.columns)

    def test_reproducible_same_input_same_winner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_minimal_grid(monkeypatch)
        examples, gold = _embedding_examples_and_gold()
        schemas = {"test_db": _embedding_schema()}
        question_vectors = {
            ex.question: np.array([1.0, 0.0, 0.0]) for ex in examples
        }

        best1, sweep1 = tune_embedding(
            examples=examples,
            gold=gold,
            schemas=schemas,
            encoder=_FakeEmbeddingEncoder(_embedding_schema_index(), question_vectors),
            gold_tier="all_sql_used",
        )
        best2, sweep2 = tune_embedding(
            examples=examples,
            gold=gold,
            schemas=schemas,
            encoder=_FakeEmbeddingEncoder(_embedding_schema_index(), question_vectors),
            gold_tier="all_sql_used",
        )

        assert best1 == best2
        pd.testing.assert_frame_equal(sweep1, sweep2)
