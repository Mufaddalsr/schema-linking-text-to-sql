"""Tests for ``schema_linking.base``: Prediction shape and Linker protocol."""

from __future__ import annotations

import json

import pytest

from schema_linking.base import (
    ColumnScore,
    Linker,
    Prediction,
    TableScore,
    from_predictions_to_dict,
    to_json,
)
from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema


def _make_prediction(
    *,
    db_id: str = "concert_singer",
    tables: tuple[str, ...] = ("singer", "concert"),
    columns: tuple[tuple[str, str], ...] = (
        ("singer", "name"),
        ("concert", "year"),
    ),
) -> Prediction:
    return Prediction(db_id=db_id, tables=tables, columns=columns)


class TestPredictionFrozen:
    def test_prediction_is_hashable(self) -> None:
        p1 = _make_prediction()
        p2 = _make_prediction()
        assert hash(p1) == hash(p2)
        assert {p1, p2} == {p1}

    def test_prediction_assignment_raises(self) -> None:
        p = _make_prediction()
        with pytest.raises((AttributeError, Exception)):
            p.db_id = "other"  # type: ignore[misc]

    def test_diagnostic_fields_default_none(self) -> None:
        p = _make_prediction()
        assert p.table_scores is None
        assert p.column_scores is None
        assert p.token_cost is None

    def test_prediction_with_diagnostics_still_hashable(self) -> None:
        p = Prediction(
            db_id="db",
            tables=("t",),
            columns=(("t", "c"),),
            table_scores=(TableScore(table="t", strategy="substring", score=100.0),),
            column_scores=(
                ColumnScore(table="t", column="c", strategy="fuzzy", score=87.0),
            ),
            token_cost=42,
        )
        assert hash(p) == hash(p)


class TestToJson:
    def test_to_json_canonical_shape(self) -> None:
        p = _make_prediction()
        d = to_json(p)
        assert d == {
            "db_id": "concert_singer",
            "tables": ["singer", "concert"],
            "columns": [["singer", "name"], ["concert", "year"]],
        }

    def test_to_json_is_json_serializable(self) -> None:
        p = _make_prediction()
        d = to_json(p)
        encoded = json.dumps(d)
        assert json.loads(encoded) == d

    def test_to_json_round_trip_through_prediction(self) -> None:
        p = _make_prediction()
        d = to_json(p)
        reconstructed = Prediction(
            db_id=d["db_id"],
            tables=tuple(d["tables"]),
            columns=tuple(tuple(c) for c in d["columns"]),
        )
        assert reconstructed == p

    def test_to_json_strips_diagnostic_fields(self) -> None:
        p = Prediction(
            db_id="db",
            tables=("t",),
            columns=(("t", "c"),),
            table_scores=(TableScore(table="t", strategy="substring", score=100.0),),
            column_scores=(
                ColumnScore(table="t", column="c", strategy="token", score=100.0),
            ),
            token_cost=99,
        )
        d = to_json(p)
        assert set(d.keys()) == {"db_id", "tables", "columns"}

    def test_to_json_empty_prediction(self) -> None:
        p = Prediction(db_id="db", tables=(), columns=())
        assert to_json(p) == {"db_id": "db", "tables": [], "columns": []}


class TestFromPredictionsToDict:
    def test_keys_are_qids_values_are_canonical(self) -> None:
        preds: dict[int, Prediction] = {
            0: _make_prediction(db_id="dbA", tables=("a",), columns=(("a", "x"),)),
            7: _make_prediction(db_id="dbB", tables=("b",), columns=(("b", "y"),)),
        }
        result = from_predictions_to_dict(preds)
        assert result == {
            0: {"db_id": "dbA", "tables": ["a"], "columns": [["a", "x"]]},
            7: {"db_id": "dbB", "tables": ["b"], "columns": [["b", "y"]]},
        }

    def test_empty_predictions(self) -> None:
        assert from_predictions_to_dict({}) == {}

    def test_diagnostics_not_in_file_ready_dict(self) -> None:
        preds = {
            0: Prediction(
                db_id="db",
                tables=("t",),
                columns=(("t", "c"),),
                table_scores=(
                    TableScore(table="t", strategy="substring", score=100.0),
                ),
                token_cost=10,
            )
        }
        result = from_predictions_to_dict(preds)
        assert set(result[0].keys()) == {"db_id", "tables", "columns"}


class _EmptyLinker:
    """Trivial linker that always predicts empty for any input."""

    def predict_one(
        self, example: SpiderExample, schema: Schema
    ) -> Prediction:
        return Prediction(db_id=example.db_id, tables=(), columns=())

    def predict_all(
        self,
        examples,
        schemas,
    ) -> dict[int, Prediction]:
        return {
            ex.question_id: self.predict_one(ex, schemas[ex.db_id])
            for ex in examples
        }


class TestLinkerProtocol:
    def test_empty_linker_satisfies_protocol(self) -> None:
        linker = _EmptyLinker()
        assert isinstance(linker, Linker)

    def test_non_linker_object_does_not_satisfy_protocol(self) -> None:
        class _Bogus:
            def something_else(self) -> None:
                return None

        assert not isinstance(_Bogus(), Linker)
