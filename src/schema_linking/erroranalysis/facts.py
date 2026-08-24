"""Precomputed per-case facts consumed by the rule cascade.

Every rule in :mod:`schema_linking.erroranalysis.rules` is a pure predicate
over a :class:`CaseFacts`. Anything a rule needs to know is computed once
here, so rules stay cheap, order-independent, and trivially testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from schema_linking.erroranalysis.taxonomy import Element
from schema_linking.schema_parser import Schema
from schema_linking.utils.graph import build_schema_graph


class SemanticScorer(Protocol):
    """Supplies question-to-element cosine similarity.

    Implemented for real by :class:`EmbeddingSemanticScorer` (Task 5) and by
    a stub in unit tests, so the test suite never loads the encoder.
    """

    def score(
        self, question: str, elements: Sequence[Element]
    ) -> dict[Element, float]:
        """Cosine similarity in ``[0, 1]`` for each element against the question."""
        ...


@dataclass(frozen=True, slots=True)
class SchemaIndex:
    """Canonical lookup structures for one database.

    Attributes
    ----------
    db_id
        Spider database identifier.
    tables
        Canonical table names.
    columns
        Every column as a canonical :class:`Element`.
    columns_by_table
        Canonical table name to that table's column elements.
    columns_by_name
        Canonical *column* name to every element bearing it, across tables.
        Drives the ``NAME-COLLISION`` and ``AMBIG-LOST`` rules.
    fk_adjacent
        Canonical table name to the set of tables reachable by one
        foreign-key edge. Symmetric. Drives the ``SIBLING`` rule.
    """

    db_id: str
    tables: frozenset[str]
    columns: frozenset[Element]
    columns_by_table: Mapping[str, frozenset[Element]]
    columns_by_name: Mapping[str, frozenset[Element]]
    fk_adjacent: Mapping[str, frozenset[str]]

    @classmethod
    def build(cls, schema: Schema) -> "SchemaIndex":
        """Index one :class:`Schema`.

        Foreign-key adjacency is taken from
        :func:`schema_linking.utils.graph.build_schema_graph` rather than
        re-derived, so the error analysis and the graph linker agree on what
        "adjacent" means.
        """
        by_table: dict[str, set[Element]] = {}
        by_name: dict[str, set[Element]] = {}
        for table in schema.tables:
            t = Element.table_el(table.original_name).table
            by_table.setdefault(t, set())
            for column in table.columns:
                el = Element.column_el(table.original_name, column.original_name)
                by_table[t].add(el)
                by_name.setdefault(el.column, set()).add(el)

        graph = build_schema_graph(schema)
        adjacency = {
            t: frozenset(
                Element.table_el(n).table for n in graph.neighbors(t)
            )
            if t in graph
            else frozenset()
            for t in by_table
        }

        return cls(
            db_id=schema.db_id,
            tables=frozenset(by_table),
            columns=frozenset(e for els in by_table.values() for e in els),
            columns_by_table={k: frozenset(v) for k, v in by_table.items()},
            columns_by_name={k: frozenset(v) for k, v in by_name.items()},
            fk_adjacent=adjacency,
        )


def elements_from_record(raw: Mapping[str, Any]) -> frozenset[Element]:
    """Canonicalise a raw gold/predicted record into a set of elements.

    Parameters
    ----------
    raw
        A mapping shaped ``{"tables": [str, ...], "columns": [[table, col], ...]}``,
        as produced by the gold-link extractor and by each linker's saved
        predictions. Table and column names may be in any case.

    Returns
    -------
    frozenset[Element]
        One :class:`Element` per table and per column entry, canonicalised
        via :meth:`Element.table_el` / :meth:`Element.column_el`.
    """
    tables = (Element.table_el(t) for t in raw.get("tables", ()))
    columns = (Element.column_el(t, c) for t, c in raw.get("columns", ()))
    return frozenset((*tables, *columns))


@dataclass(frozen=True, slots=True)
class CaseFacts:
    """Everything the cascade needs about one (question, method) pair.

    Attributes
    ----------
    question_id, db_id, question, gold_sql
        Identity and raw inputs. ``gold_sql`` backs the ``GOLD-DEFECT`` and
        ``IMPLICIT-AGG`` rules.
    index
        The database's :class:`SchemaIndex`.
    gold_tier1, gold_tier2, predicted
        Canonical element sets.
    lexical_scores
        Element to rapidfuzz ``partial_ratio`` against the question, ``0-100``.
        Empty until Task 5.
    semantic_scores
        Element to cosine similarity against the question, ``0-1``. Empty
        until Task 5.
    hardness
        Spider hardness label from ``utils.difficulty.eval_hardness``.
    n_tables, n_columns
        Schema size, carried onto every error row for RQ4.
    """

    question_id: int
    db_id: str
    question: str
    gold_sql: str
    index: SchemaIndex
    gold_tier1: frozenset[Element]
    gold_tier2: frozenset[Element]
    predicted: frozenset[Element]
    hardness: str
    n_tables: int
    n_columns: int
    lexical_scores: Mapping[Element, int] = field(default_factory=dict)
    semantic_scores: Mapping[Element, float] = field(default_factory=dict)

    def gold_for(self, tier: str) -> frozenset[Element]:
        """Gold set for ``"tier1"`` or ``"tier2"``."""
        if tier == "tier1":
            return self.gold_tier1
        if tier == "tier2":
            return self.gold_tier2
        raise ValueError(f"unknown tier: {tier!r}")

    @staticmethod
    def other_tier(tier: str) -> str:
        """The tier that ``tier`` is not."""
        if tier == "tier1":
            return "tier2"
        if tier == "tier2":
            return "tier1"
        raise ValueError(f"unknown tier: {tier!r}")


def build_case_facts(
    *,
    question_id: int,
    question: str,
    gold_sql: str,
    schema: Schema,
    gold_tier1_raw: Mapping[str, Any],
    gold_tier2_raw: Mapping[str, Any],
    predicted_raw: Mapping[str, Any],
    hardness: str,
    index: SchemaIndex | None = None,
    lexical_scores: Mapping[Element, int] | None = None,
    semantic_scores: Mapping[Element, float] | None = None,
) -> CaseFacts:
    """Assemble a :class:`CaseFacts`.

    Parameters
    ----------
    index
        Prebuilt :class:`SchemaIndex`. Pass one when looping over a split so
        the index is built once per database rather than once per question.
    lexical_scores, semantic_scores
        Precomputed scores. ``None`` leaves them empty (Task 4 behaviour).
    """
    idx = index if index is not None else SchemaIndex.build(schema)
    return CaseFacts(
        question_id=question_id,
        db_id=schema.db_id,
        question=question,
        gold_sql=gold_sql,
        index=idx,
        gold_tier1=elements_from_record(gold_tier1_raw),
        gold_tier2=elements_from_record(gold_tier2_raw),
        predicted=elements_from_record(predicted_raw),
        hardness=hardness,
        n_tables=len(idx.tables),
        n_columns=len(idx.columns),
        lexical_scores=dict(lexical_scores or {}),
        semantic_scores=dict(semantic_scores or {}),
    )
