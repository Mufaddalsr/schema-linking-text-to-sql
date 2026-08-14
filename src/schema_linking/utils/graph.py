"""Build and query a schema's foreign-key graph.

Pure, stateless helpers for the graph linker (Method F). No LLM calls, no
network I/O — everything here operates on an already-loaded :class:`Schema`.

Column-index resolution
------------------------
:class:`~schema_linking.schema_parser.FKPair` stores Spider's raw global
column indices (positions in the original ``column_names`` list, which
includes the synthetic ``*`` column at index 0). :class:`Schema` does not
retain that raw list, but Spider's ``tables.json`` groups columns by table in
table order, so the global index can be reconstructed by walking
``schema.tables`` in order, offsetting by 1 for the ``*`` column at index 0.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx

from schema_linking.schema_parser import Schema

logger = logging.getLogger(__name__)

_STAR_COLUMN_COUNT = 1


def _column_index_map(schema: Schema) -> dict[int, tuple[str, str]]:
    """Map Spider global column index -> ``(table_name, column_name)``."""
    index_map: dict[int, tuple[str, str]] = {}
    col_idx = _STAR_COLUMN_COUNT
    for table in schema.tables:
        for column in table.columns:
            index_map[col_idx] = (table.original_name, column.original_name)
            col_idx += 1
    return index_map


def build_schema_graph(schema: Schema) -> nx.Graph:
    """Build an undirected foreign-key graph over a schema's tables.

    Parameters
    ----------
    schema
        Parsed Spider schema.

    Returns
    -------
    nx.Graph
        One node per table (``original_name`` case), with a ``"columns"``
        attribute holding a tuple of that table's column names. One edge per
        unique unordered pair of tables that share at least one foreign key,
        with a ``"fk_columns"`` attribute holding a tuple of
        ``(from_col, to_col)`` name pairs for every FK between that pair.
        Tables with no foreign keys are present as isolated nodes.
    """
    graph: nx.Graph = nx.Graph()
    for table in schema.tables:
        graph.add_node(
            table.original_name,
            columns=tuple(c.original_name for c in table.columns),
        )

    index_map = _column_index_map(schema)
    for fk in schema.foreign_keys:
        from_table, from_col = index_map[fk.from_col_idx]
        to_table, to_col = index_map[fk.to_col_idx]
        if graph.has_edge(from_table, to_table):
            graph[from_table][to_table]["fk_columns"] += ((from_col, to_col),)
        else:
            graph.add_edge(from_table, to_table, fk_columns=((from_col, to_col),))

    return graph


def shortest_path_tables(
    graph: nx.Graph,
    source: str,
    target: str,
) -> tuple[str, ...] | None:
    """Shortest path between two tables, as an ordered tuple of table names.

    Parameters
    ----------
    graph
        Schema graph from :func:`build_schema_graph`.
    source
        Source table name (must match a graph node id exactly).
    target
        Target table name (must match a graph node id exactly).

    Returns
    -------
    tuple[str, ...] | None
        Table names along the shortest path, ``source`` to ``target``
        inclusive, or ``None`` if the two tables are disconnected.
    """
    try:
        path = nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        return None
    return tuple(path)


def steiner_subgraph_tables(
    graph: nx.Graph,
    terminals: Iterable[str],
) -> tuple[str, ...]:
    """Greedy Steiner-tree approximation over a set of terminal tables.

    Starts from the first terminal and repeatedly connects the nearest
    remaining terminal to the growing tree via its shortest path.

    Parameters
    ----------
    graph
        Schema graph from :func:`build_schema_graph`.
    terminals
        Table names that must be connected (3+ typical; 1 or 2 handled as
        base cases).

    Returns
    -------
    tuple[str, ...]
        Sorted table names in the resulting subgraph. A terminal
        disconnected from the rest is still included, alone, with a warning
        logged.
    """
    terminal_list = list(dict.fromkeys(terminals))
    if not terminal_list:
        return ()
    if len(terminal_list) == 1:
        return (terminal_list[0],)

    tree_nodes: set[str] = {terminal_list[0]}
    for terminal in terminal_list[1:]:
        if terminal in tree_nodes:
            continue

        best_path: list[str] | None = None
        for node in tree_nodes:
            try:
                path = nx.shortest_path(graph, node, terminal)
            except nx.NetworkXNoPath:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path

        if best_path is None:
            logger.warning(
                "terminal %r disconnected from other terminals — including it alone",
                terminal,
            )
            tree_nodes.add(terminal)
        else:
            tree_nodes.update(best_path)

    return tuple(sorted(tree_nodes))


def resolve_endpoint_table(name: str, schema: Schema) -> str | None:
    """Case-insensitively resolve an LLM-produced table name against a schema.

    Parameters
    ----------
    name
        Raw table name as produced by an LLM (may have stray whitespace or
        differing case).
    schema
        Schema to resolve against.

    Returns
    -------
    str | None
        The canonical ``original_name`` from the schema, or ``None`` if no
        exact (case-insensitive) match exists. No fuzzy matching — a
        hallucinated table name returns ``None``.
    """
    target = name.strip().lower()
    for table in schema.tables:
        if table.original_name.lower() == target:
            return table.original_name
    return None
