"""Miniature schemas shared by every error-analysis unit test.

Small enough to hold in your head, and constructed so that each rule in the
cascade has a case it can fire on:

* ``singer`` and ``concert`` are FK-joined  -> SIBLING, JOIN-ONLY
* both carry a ``Name`` column              -> NAME-COLLISION, AMBIG-LOST
* ``flight_1`` is a separate database       -> WRONG-DB

Spider's global FK indices count the synthetic ``*`` column at position 0,
so column index 1 is ``singer.Singer_ID`` and index 6 is
``concert.Singer_ID``. See ``schema_parser`` module docstring.
"""

from __future__ import annotations

import pytest

from schema_linking.schema_parser import Column, FKPair, Schema, Table


def _col(name: str, original: str, table: str, *, pk: bool = False) -> Column:
    return Column(
        name=name,
        original_name=original,
        type="text",
        table_name=table,
        is_primary_key=pk,
    )


@pytest.fixture
def mini_schema() -> Schema:
    """Two FK-joined tables in db ``concert_singer``."""
    singer = Table(
        name="singer",
        original_name="singer",
        columns=[
            _col("singer id", "Singer_ID", "singer", pk=True),
            _col("name", "Name", "singer"),
            _col("country", "Country", "singer"),
        ],
    )
    concert = Table(
        name="concert",
        original_name="concert",
        columns=[
            _col("concert id", "Concert_ID", "concert", pk=True),
            _col("name", "Name", "concert"),
            _col("singer id", "Singer_ID", "concert"),
        ],
    )
    return Schema(
        db_id="concert_singer",
        tables=[singer, concert],
        foreign_keys=[FKPair(from_col_idx=6, to_col_idx=1)],
    )


@pytest.fixture
def other_schema() -> Schema:
    """A second database, for the WRONG-DB hallucination rule."""
    flights = Table(
        name="flights",
        original_name="flights",
        columns=[
            _col("flight number", "FlightNo", "flights", pk=True),
            _col("origin", "Origin", "flights"),
        ],
    )
    return Schema(db_id="flight_1", tables=[flights], foreign_keys=[])


@pytest.fixture
def all_schemas(mini_schema: Schema, other_schema: Schema) -> dict[str, Schema]:
    """Both fixture databases, keyed by ``db_id``."""
    return {mini_schema.db_id: mini_schema, other_schema.db_id: other_schema}
