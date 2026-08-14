"""Tests for ``schema_linking.lexical_linker.LexicalLinker``."""

from __future__ import annotations

import pytest

from schema_linking.base import Linker, Prediction
from schema_linking.data_loader import SpiderExample
from schema_linking.lexical_linker import LexicalLinker
from schema_linking.schema_parser import Column, Schema, Table


def _table(name: str, cols: list[tuple[str, str]]) -> Table:
    """Build a Table where ``cols`` is a list of ``(original_name, type)``."""
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
    """Synthetic 4-table schema covering all test scenarios."""
    return Schema(
        db_id="test_db",
        tables=[
            _table("singer", [("name", "text"), ("age", "number")]),
            _table("concert", [("year", "number"), ("stadium_id", "number")]),
            _table(
                "stadium",
                [
                    ("name", "text"),
                    ("capacity", "number"),
                    ("country", "text"),
                ],
            ),
            _table(
                "student",
                [("name", "text"), ("first_name", "text"), ("age", "number")],
            ),
        ],
        foreign_keys=[],
    )


def _example(question: str, qid: int = 0, db_id: str = "test_db") -> SpiderExample:
    return SpiderExample(
        question_id=qid,
        db_id=db_id,
        question=question,
        query="SELECT 1",
        sql={},
        split="dev",
    )


def _strategies(pred: Prediction) -> set[str]:
    """All strategies that fired across table_scores + column_scores."""
    strategies: set[str] = set()
    if pred.table_scores is not None:
        strategies.update(ts.strategy for ts in pred.table_scores)
    if pred.column_scores is not None:
        strategies.update(cs.strategy for cs in pred.column_scores)
    return strategies


class TestLexicalLinkerBasic:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(LexicalLinker(), Linker)

    def test_predicts_singer_table(self, schema: Schema) -> None:
        linker = LexicalLinker()
        ex = _example("How many singers are there?")
        pred = linker.predict_one(ex, schema)
        assert "singer" in pred.tables

    def test_predicts_stadium_table_and_name_column(self, schema: Schema) -> None:
        linker = LexicalLinker()
        ex = _example("What is the name of the stadium?")
        pred = linker.predict_one(ex, schema)
        assert "stadium" in pred.tables
        assert ("stadium", "name") in pred.columns

    def test_predicts_capacity_column(self, schema: Schema) -> None:
        linker = LexicalLinker()
        ex = _example("Show me the capacity")
        pred = linker.predict_one(ex, schema)
        assert ("stadium", "capacity") in pred.columns

    def test_predicts_concert_table(self, schema: Schema) -> None:
        linker = LexicalLinker()
        ex = _example("show concerts")
        pred = linker.predict_one(ex, schema)
        assert "concert" in pred.tables

    def test_predicts_student_name(self, schema: Schema) -> None:
        linker = LexicalLinker()
        ex = _example("list all student names")
        pred = linker.predict_one(ex, schema)
        assert ("student", "name") in pred.columns

    def test_predicts_underscore_column(self, schema: Schema) -> None:
        linker = LexicalLinker()
        ex = _example("what is the first name")
        pred = linker.predict_one(ex, schema)
        assert ("student", "first_name") in pred.columns


class TestLexicalLinkerNegative:
    def test_empty_match_returns_empty_prediction(self, schema: Schema) -> None:
        # Pin threshold so this test stays stable across config changes
        # (tuning may raise the configured threshold above 80).
        linker = LexicalLinker(fuzzy_threshold=80)
        ex = _example("abc xyz qwerty")
        pred = linker.predict_one(ex, schema)
        assert pred.tables == ()
        assert pred.columns == ()


class TestLexicalLinkerDiagnostics:
    def test_table_scores_cover_every_predicted_table(
        self, schema: Schema
    ) -> None:
        linker = LexicalLinker()
        ex = _example("What is the name of the stadium?")
        pred = linker.predict_one(ex, schema)
        assert pred.table_scores is not None
        scored_tables = {ts.table for ts in pred.table_scores}
        assert scored_tables == set(pred.tables)

    def test_substring_and_fuzzy_strategies_both_fire(
        self, schema: Schema
    ) -> None:
        # Pin threshold to 80 so the fuzzy case ("capasity" → "capacity",
        # partial_ratio ≈ 87) stays a hit regardless of the tuned value
        # written to config.yaml.
        linker = LexicalLinker(fuzzy_threshold=80)
        substring_pred = linker.predict_one(
            _example("What is the name of the stadium?"), schema
        )
        fuzzy_pred = linker.predict_one(
            _example("show the capasity of the stadium"), schema
        )
        combined = _strategies(substring_pred) | _strategies(fuzzy_pred)
        assert "substring" in combined
        assert "fuzzy" in combined


class TestLexicalLinkerPredictAll:
    def test_predict_all_returns_dict_keyed_by_qid(self, schema: Schema) -> None:
        linker = LexicalLinker()
        questions = [
            "How many singers are there?",
            "What is the name of the stadium?",
            "Show me the capacity",
            "show concerts",
            "list all student names",
            "what is the first name",
            "How old is the oldest singer?",
            "Which country has the most stadiums?",
            "How many concerts were there?",
            "List all ages",
        ]
        examples = [_example(q, qid=i) for i, q in enumerate(questions)]
        schemas = {"test_db": schema}
        result = linker.predict_all(examples, schemas)
        assert set(result.keys()) == set(range(10))
        for pred in result.values():
            assert isinstance(pred, Prediction)
