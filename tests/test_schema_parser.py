"""Tests for src/schema_linking/schema_parser.py."""

from __future__ import annotations

import pytest

from schema_linking.schema_parser import (
    Column,
    FKPair,
    Schema,
    Table,
    load_schemas,
)


@pytest.fixture(scope="module")
def schemas() -> dict[str, Schema]:
    return load_schemas()


def test_schema_count(schemas: dict[str, Schema]) -> None:
    assert len(schemas) == 166


def test_returns_schema_instances(schemas: dict[str, Schema]) -> None:
    for db_id, schema in schemas.items():
        assert isinstance(schema, Schema)
        assert schema.db_id == db_id


def test_at_least_one_schema_has_foreign_keys(schemas: dict[str, Schema]) -> None:
    assert any(s.foreign_keys for s in schemas.values())
    for schema in schemas.values():
        for fk in schema.foreign_keys:
            assert isinstance(fk, FKPair)


def test_star_column_excluded(schemas: dict[str, Schema]) -> None:
    for schema in schemas.values():
        for table in schema.tables:
            assert table.original_name != "*"
            for col in table.columns:
                assert col.original_name != "*", (
                    f"`*` column leaked into {schema.db_id}.{table.original_name}"
                )
                assert col.table_name, "column has empty table_name"


def test_column_table_name_matches_parent(schemas: dict[str, Schema]) -> None:
    for schema in schemas.values():
        for table in schema.tables:
            for col in table.columns:
                assert col.table_name == table.original_name


def test_concert_singer_structure(schemas: dict[str, Schema]) -> None:
    cs = schemas["concert_singer"]
    table_names = [t.original_name for t in cs.tables]
    assert table_names == ["stadium", "singer", "concert", "singer_in_concert"]
    assert len(cs.foreign_keys) == 3


def test_concert_singer_primary_keys_flagged(schemas: dict[str, Schema]) -> None:
    cs = schemas["concert_singer"]
    singer = next(t for t in cs.tables if t.original_name == "singer")

    singer_id = next(c for c in singer.columns if c.original_name == "Singer_ID")
    assert singer_id.is_primary_key is True

    name_col = next(c for c in singer.columns if c.original_name == "Name")
    assert name_col.is_primary_key is False

    stadium = next(t for t in cs.tables if t.original_name == "stadium")
    stadium_id = next(c for c in stadium.columns if c.original_name == "Stadium_ID")
    assert stadium_id.is_primary_key is True


def test_column_types_are_known(schemas: dict[str, Schema]) -> None:
    allowed = {"text", "number", "time", "boolean", "others"}
    for schema in schemas.values():
        for table in schema.tables:
            for col in table.columns:
                assert col.type in allowed, (
                    f"unexpected type {col.type!r} in "
                    f"{schema.db_id}.{table.original_name}.{col.original_name}"
                )


def test_dataclasses_are_frozen() -> None:
    col = Column("a", "a", "text", "t", False)
    with pytest.raises(AttributeError):
        col.name = "b"  # type: ignore[misc]

    fk = FKPair(1, 2)
    with pytest.raises(AttributeError):
        fk.from_col_idx = 3  # type: ignore[misc]
