"""Parse Spider ``tables.json`` into typed :class:`Schema` records.

The ``*`` column convention
---------------------------
Spider represents the SQL ``*`` selector as a synthetic column at index 0 of
``column_names`` / ``column_names_original`` with ``table_idx == -1``. This
parser **excludes** that synthetic column from the resulting :class:`Schema`.

Rationale: ``*`` is a syntactic SQL construct, not a schema element that a
linker should ever predict. Including it as a synthetic table would force
every downstream consumer (linkers, evaluator, error analysis) to
special-case-skip it. The gold-link extractor handles ``*``-bearing
constructs (``COUNT(*)``, ``SELECT *``) at the SQL-parserlayer where they belong.

Foreign-key indices
-------------------
:class:`FKPair` stores Spider's **raw** global column indices, i.e. the
positions in the original ``column_names`` list which *includes* the ``*``
column at position 0. These indices are therefore **not** directly indexable
into the flattened ``Schema`` columns. A resolver mapping
``col_idx -> (table_name, column_name)`` will be added when a downstream
consumer (e.g. the graph linker) needs it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_TABLES_PATH: Path = (
    Path(__file__).resolve().parents[2] / "data" / "spider" / "tables.json"
)
TABLES_FILE: str = "tables.json"

_STAR_TABLE_IDX: int = -1


@dataclass(frozen=True, slots=True)
class Column:
    """A single column within a table.

    Attributes
    ----------
    name
        Human-readable column name (Spider's ``column_names`` entry), e.g.
        ``"concert ID"``.
    original_name
        DB-canonical column name (Spider's ``column_names_original`` entry),
        e.g. ``"concert_ID"``. This is the identifier that appears in SQL.
    type
        Spider column type — one of ``"text"``, ``"number"``, ``"time"``,
        ``"boolean"``, ``"others"``.
    table_name
        ``original_name`` of the parent :class:`Table`.
    is_primary_key
        ``True`` iff this column's global index appears in Spider's
        ``primary_keys`` list for its database.
    """

    name: str
    original_name: str
    type: str
    table_name: str
    is_primary_key: bool


@dataclass(frozen=True, slots=True)
class Table:
    """A single table in a Spider database.

    Attributes
    ----------
    name
        Human-readable table name (``table_names`` entry).
    original_name
        DB-canonical table name (``table_names_original`` entry) — the form
        used in SQL.
    columns
        Columns belonging to this table, in Spider source order.
    """

    name: str
    original_name: str
    columns: list[Column]


@dataclass(frozen=True, slots=True)
class FKPair:
    """A single foreign-key relation.

    Attributes
    ----------
    from_col_idx
        Spider global column index of the referencing column.
    to_col_idx
        Spider global column index of the referenced column.

    Notes
    -----
    Indices refer to positions in Spider's ``column_names`` list, which
    includes the synthetic ``*`` column at position 0. They are **not**
    indices into ``Schema.tables[t].columns`` — see module docstring.
    """

    from_col_idx: int
    to_col_idx: int


@dataclass(frozen=True, slots=True)
class Schema:
    """The schema of a single Spider database.

    Attributes
    ----------
    db_id
        Spider database identifier, e.g. ``"concert_singer"``.
    tables
        Tables in Spider source order. Excludes any synthetic ``*`` table.
    foreign_keys
        Foreign-key pairs as raw Spider indices. May be empty.
    """

    db_id: str
    tables: list[Table]
    foreign_keys: list[FKPair]


def load_schemas(path: Path = _DEFAULT_TABLES_PATH) -> dict[str, Schema]:
    """Load all Spider schemas from ``tables.json``.

    Parameters
    ----------
    path
        Path to Spider's ``tables.json``. Defaults to ``data/spider/tables.json``.

    Returns
    -------
    dict[str, Schema]
        Mapping from ``db_id`` to :class:`Schema`. Spider's dev set
        references 166 distinct databases.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    return {entry["db_id"]: _parse_schema(entry) for entry in raw}


def _parse_schema(entry: dict) -> Schema:
    """Build one :class:`Schema` from a single ``tables.json`` entry."""
    table_names: list[str] = entry["table_names"]
    table_names_original: list[str] = entry["table_names_original"]
    column_names: list[list] = entry["column_names"]
    column_names_original: list[list] = entry["column_names_original"]
    column_types: list[str] = entry["column_types"]
    primary_keys: set[int] = set(entry["primary_keys"])
    foreign_keys_raw: list[list[int]] = entry["foreign_keys"]

    columns_by_table: dict[int, list[Column]] = {
        i: [] for i in range(len(table_names_original))
    }

    for col_idx, ((tbl_idx, hname), (_, oname), ctype) in enumerate(
        zip(column_names, column_names_original, column_types, strict=True)
    ):
        if tbl_idx == _STAR_TABLE_IDX:
            continue  # exclude the synthetic `*` column — see module docstring
        columns_by_table[tbl_idx].append(
            Column(
                name=hname,
                original_name=oname,
                type=ctype,
                table_name=table_names_original[tbl_idx],
                is_primary_key=col_idx in primary_keys,
            )
        )

    tables = [
        Table(
            name=table_names[i],
            original_name=table_names_original[i],
            columns=columns_by_table[i],
        )
        for i in range(len(table_names_original))
    ]

    foreign_keys = [FKPair(from_col_idx=a, to_col_idx=b) for a, b in foreign_keys_raw]

    return Schema(db_id=entry["db_id"], tables=tables, foreign_keys=foreign_keys)
