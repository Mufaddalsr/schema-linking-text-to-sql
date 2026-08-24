"""Set logic and schema indexing for CaseFacts."""

from __future__ import annotations

import pytest

from schema_linking.erroranalysis.facts import (
    CaseFacts,
    SchemaIndex,
    build_case_facts,
    elements_from_record,
)
from schema_linking.erroranalysis.taxonomy import Element
from schema_linking.schema_parser import Column, FKPair, Schema, Table


def _col(name: str, original: str, table: str, *, pk: bool = False) -> Column:
    return Column(
        name=name,
        original_name=original,
        type="text",
        table_name=table,
        is_primary_key=pk,
    )


def test_schema_index_collects_canonical_tables(mini_schema):
    idx = SchemaIndex.build(mini_schema)
    assert idx.tables == {"singer", "concert"}


def test_schema_index_collects_canonical_columns(mini_schema):
    idx = SchemaIndex.build(mini_schema)
    assert Element.column_el("singer", "Singer_ID") in idx.columns
    assert Element.column_el("concert", "Name") in idx.columns
    assert len(idx.columns) == 6


def test_columns_by_table_groups_correctly(mini_schema):
    idx = SchemaIndex.build(mini_schema)
    assert idx.columns_by_table["singer"] == {
        Element.column_el("singer", "Singer_ID"),
        Element.column_el("singer", "Name"),
        Element.column_el("singer", "Country"),
    }


def test_columns_by_name_finds_the_collision(mini_schema):
    """Both tables have a Name column; the index must surface both."""
    idx = SchemaIndex.build(mini_schema)
    assert idx.columns_by_name["name"] == {
        Element.column_el("singer", "Name"),
        Element.column_el("concert", "Name"),
    }


def test_fk_adjacency_is_symmetric(mini_schema):
    idx = SchemaIndex.build(mini_schema)
    assert idx.fk_adjacent["singer"] == {"concert"}
    assert idx.fk_adjacent["concert"] == {"singer"}


def test_fk_adjacency_of_isolated_table_is_empty(other_schema):
    idx = SchemaIndex.build(other_schema)
    assert idx.fk_adjacent["flights"] == frozenset()


def test_fk_adjacency_is_symmetric_for_mixed_case_table_names():
    """Regression: table original_name may not be lowercase (e.g. "Singer").

    build_schema_graph keys its nodes by the raw, uncanonicalised
    original_name. SchemaIndex.build must canonicalise on the way out, not
    look the raw name up under its canonical form.
    """
    singer = Table(
        name="singer",
        original_name="Singer",
        columns=[
            _col("singer id", "Singer_ID", "Singer", pk=True),
            _col("name", "Name", "Singer"),
        ],
    )
    concert = Table(
        name="concert",
        original_name="Concert",
        columns=[
            _col("concert id", "Concert_ID", "Concert", pk=True),
            _col("singer id", "Singer_ID", "Concert"),
        ],
    )
    schema = Schema(
        db_id="mixed_case_db",
        tables=[singer, concert],
        foreign_keys=[FKPair(from_col_idx=4, to_col_idx=1)],
    )

    idx = SchemaIndex.build(schema)

    assert idx.fk_adjacent["singer"] == {"concert"}
    assert idx.fk_adjacent["concert"] == {"singer"}


def test_elements_from_record_canonicalises_mixed_case():
    record = {
        "tables": ["Singer", "CONCERT"],
        "columns": [["Singer", "Singer_ID"], ["concert", "NAME"]],
    }
    assert elements_from_record(record) == {
        Element.table_el("singer"),
        Element.table_el("concert"),
        Element.column_el("singer", "Singer_ID"),
        Element.column_el("concert", "Name"),
    }


def _facts(mini_schema, **kw):
    defaults = dict(
        question_id=0,
        question="How many singers are there?",
        gold_sql="SELECT count(*) FROM singer",
        schema=mini_schema,
        gold_tier1_raw={"tables": ["singer"], "columns": []},
        gold_tier2_raw={"tables": ["singer"], "columns": []},
        predicted_raw={"tables": ["singer"], "columns": [["singer", "Name"]]},
        hardness="easy",
    )
    defaults.update(kw)
    return build_case_facts(**defaults)


def test_build_case_facts_canonicalises_gold(mini_schema):
    facts = _facts(
        mini_schema,
        gold_tier1_raw={"tables": ["Singer"], "columns": [["Singer", "Singer_ID"]]},
    )
    assert facts.gold_tier1 == {
        Element.table_el("singer"),
        Element.column_el("singer", "Singer_ID"),
    }


def test_build_case_facts_canonicalises_predictions(mini_schema):
    facts = _facts(
        mini_schema,
        predicted_raw={"tables": ["SINGER"], "columns": [["SINGER", "NAME"]]},
    )
    assert facts.predicted == {
        Element.table_el("singer"),
        Element.column_el("singer", "Name"),
    }


def test_build_case_facts_records_schema_size(mini_schema):
    facts = _facts(mini_schema)
    assert facts.n_tables == 2
    assert facts.n_columns == 6


def test_scores_are_empty_until_task_5(mini_schema):
    facts = _facts(mini_schema)
    assert facts.lexical_scores == {}
    assert facts.semantic_scores == {}


def test_gold_for_tier_dispatches(mini_schema):
    facts = _facts(
        mini_schema,
        gold_tier1_raw={"tables": ["singer"], "columns": []},
        gold_tier2_raw={"tables": ["singer", "concert"], "columns": []},
    )
    assert facts.gold_for("tier1") == {Element.table_el("singer")}
    assert facts.gold_for("tier2") == {
        Element.table_el("singer"),
        Element.table_el("concert"),
    }


def test_other_tier_gold_is_the_complement(mini_schema):
    facts = _facts(mini_schema)
    assert facts.other_tier("tier1") == "tier2"
    assert facts.other_tier("tier2") == "tier1"


def test_gold_for_raises_on_unknown_tier(mini_schema):
    facts = _facts(mini_schema)
    with pytest.raises(ValueError):
        facts.gold_for("tier3")


def test_other_tier_raises_on_unknown_tier():
    with pytest.raises(ValueError):
        CaseFacts.other_tier("tier3")


def test_elements_from_record_of_empty_record_is_empty_set():
    assert elements_from_record({"tables": [], "columns": []}) == frozenset()
