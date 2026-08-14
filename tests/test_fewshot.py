"""Tests for schema_linking.utils.fewshot's backward (Method D) reformatting.

pick_fewshot_examples/save_fewshot_examples (forward, Method C) are already
exercised via tests/test_prompts.py.
"""

from __future__ import annotations

import pytest

from schema_linking.data_loader import SpiderExample
from schema_linking.utils.fewshot import to_backward_fewshot_examples, to_graph_fewshot_examples


def test_to_backward_fewshot_examples_looks_up_gold_sql_by_question_id() -> None:
    train_examples = [
        SpiderExample(
            question_id=10, db_id="concert_singer", question="How many singers?",
            query="SELECT COUNT(*) FROM singer", sql={}, split="train",
        ),
        SpiderExample(
            question_id=20, db_id="pets_1", question="List pet names.",
            query="SELECT name FROM pets", sql={}, split="train",
        ),
    ]
    fewshot_examples = [
        {"question_id": 10, "db_id": "concert_singer", "question": "How many singers?", "pattern": "simple", "tables": ["singer"], "columns": []},
        {"question_id": 20, "db_id": "pets_1", "question": "List pet names.", "pattern": "multi_table", "tables": ["pets"], "columns": [["pets", "name"]]},
    ]

    out = to_backward_fewshot_examples(fewshot_examples, train_examples)

    assert out == [
        {"question_id": 10, "db_id": "concert_singer", "question": "How many singers?", "pattern": "simple", "gold_sql": "SELECT COUNT(*) FROM singer"},
        {"question_id": 20, "db_id": "pets_1", "question": "List pet names.", "pattern": "multi_table", "gold_sql": "SELECT name FROM pets"},
    ]


def test_to_backward_fewshot_examples_raises_on_missing_question_id() -> None:
    with pytest.raises(ValueError, match="99"):
        to_backward_fewshot_examples(
            [{"question_id": 99, "db_id": "x", "question": "q", "pattern": "simple", "tables": [], "columns": []}],
            [],
        )


def test_to_graph_fewshot_examples_renames_tables_to_core_tables() -> None:
    fewshot_examples = [
        {"question_id": 10, "db_id": "concert_singer", "question": "How many singers?", "pattern": "simple", "tables": ["singer"], "columns": []},
        {"question_id": 20, "db_id": "pets_1", "question": "List pet names.", "pattern": "multi_table", "tables": ["pets"], "columns": [["pets", "name"]]},
    ]

    out = to_graph_fewshot_examples(fewshot_examples)

    assert out == [
        {"question_id": 10, "db_id": "concert_singer", "question": "How many singers?", "pattern": "simple", "core_tables": ["singer"], "columns": []},
        {"question_id": 20, "db_id": "pets_1", "question": "List pet names.", "pattern": "multi_table", "core_tables": ["pets"], "columns": [["pets", "name"]]},
    ]


def test_to_graph_fewshot_examples_raises_on_column_outside_core_tables() -> None:
    with pytest.raises(ValueError, match="not in core_tables"):
        to_graph_fewshot_examples(
            [
                {
                    "question_id": 1, "db_id": "x", "question": "q", "pattern": "multi_table",
                    "tables": ["a"], "columns": [["b", "col"]],
                }
            ]
        )


def test_to_graph_fewshot_examples_raises_on_too_many_core_tables() -> None:
    with pytest.raises(ValueError, match="1-3 core tables"):
        to_graph_fewshot_examples(
            [
                {
                    "question_id": 1, "db_id": "x", "question": "q", "pattern": "multi_table",
                    "tables": ["a", "b", "c", "d"], "columns": [],
                }
            ]
        )
