"""Tests for src/schema_linking/utils/graph.py."""

from __future__ import annotations

import logging

import networkx as nx
import pytest

from schema_linking.schema_parser import Column, FKPair, Schema, Table, load_schemas
from schema_linking.utils.graph import (
    build_schema_graph,
    resolve_endpoint_table,
    shortest_path_tables,
    steiner_subgraph_tables,
)


def _synthetic_abc_schema() -> Schema:
    """Tables a, b, c with a single FK a -> b. c is isolated."""
    a = Table(
        name="a",
        original_name="a",
        columns=[Column(name="a_id", original_name="a_id", type="number", table_name="a", is_primary_key=True)],
    )
    b = Table(
        name="b",
        original_name="b",
        columns=[Column(name="b_id", original_name="b_id", type="number", table_name="b", is_primary_key=True)],
    )
    c = Table(
        name="c",
        original_name="c",
        columns=[Column(name="c_id", original_name="c_id", type="number", table_name="c", is_primary_key=True)],
    )
    # global indices: 0 = star, 1 = a.a_id, 2 = b.b_id, 3 = c.c_id
    return Schema(db_id="synthetic_abc", tables=[a, b, c], foreign_keys=[FKPair(from_col_idx=1, to_col_idx=2)])


@pytest.fixture(scope="module")
def concert_singer_schema() -> Schema:
    return load_schemas()["concert_singer"]


def test_build_schema_graph_synthetic() -> None:
    graph = build_schema_graph(_synthetic_abc_schema())

    assert set(graph.nodes) == {"a", "b", "c"}
    assert graph.number_of_edges() == 1
    assert graph.has_edge("a", "b")
    assert graph["a"]["b"]["fk_columns"] == (("a_id", "b_id"),)
    assert graph.degree("c") == 0


def test_build_schema_graph_concert_singer(concert_singer_schema: Schema) -> None:
    graph = build_schema_graph(concert_singer_schema)

    assert set(graph.nodes) == {"stadium", "singer", "concert", "singer_in_concert"}
    assert graph.degree("singer_in_concert") >= 2


def test_shortest_path_tables_linear() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

    assert shortest_path_tables(graph, "a", "d") == ("a", "b", "c", "d")


def test_shortest_path_tables_disconnected() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b")
    graph.add_node("z")

    assert shortest_path_tables(graph, "a", "z") is None


def test_steiner_subgraph_tables_three_terminals() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("hub", "a"), ("hub", "b"), ("hub", "c")])

    result = steiner_subgraph_tables(graph, ["a", "b", "c"])

    assert result == ("a", "b", "c", "hub")


def test_steiner_subgraph_tables_single_terminal() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("hub", "a"), ("hub", "b")])

    assert steiner_subgraph_tables(graph, ["a"]) == ("a",)


def test_steiner_subgraph_tables_disconnected_terminal(caplog: pytest.LogCaptureFixture) -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b")
    graph.add_node("z")

    with caplog.at_level(logging.WARNING, logger="schema_linking.utils.graph"):
        result = steiner_subgraph_tables(graph, ["a", "b", "z"])

    assert result == ("a", "b", "z")
    assert any("disconnected" in record.message for record in caplog.records)


def test_resolve_endpoint_table(concert_singer_schema: Schema) -> None:
    assert resolve_endpoint_table("SINGER", concert_singer_schema) == "singer"
    assert resolve_endpoint_table("fake", concert_singer_schema) is None
