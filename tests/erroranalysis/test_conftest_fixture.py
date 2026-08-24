"""The miniature fixture must be a valid Schema with a usable FK."""

from __future__ import annotations

from schema_linking.utils.graph import build_schema_graph


def test_mini_schema_shape(mini_schema):
    assert mini_schema.db_id == "concert_singer"
    assert [t.original_name for t in mini_schema.tables] == ["singer", "concert"]
    singer = mini_schema.tables[0]
    assert [c.original_name for c in singer.columns] == [
        "Singer_ID",
        "Name",
        "Country",
    ]
    concert = mini_schema.tables[1]
    assert [c.original_name for c in concert.columns] == [
        "Concert_ID",
        "Name",
        "Singer_ID",
    ]


def test_fk_is_usable_by_the_graph_util(mini_schema):
    graph = build_schema_graph(mini_schema)
    assert graph.has_edge("singer", "concert")


def test_name_collision_exists_by_construction(mini_schema):
    """Both tables have a ``Name`` column — the NAME-COLLISION rule needs it."""
    names = {
        (t.original_name, c.original_name)
        for t in mini_schema.tables
        for c in t.columns
    }
    assert ("singer", "Name") in names
    assert ("concert", "Name") in names


def test_other_schema_is_a_different_db(other_schema):
    assert other_schema.db_id == "flight_1"
    assert [t.original_name for t in other_schema.tables] == ["flights"]
