"""Tests for ``schema_linking.utils.prompts`` and ``.fewshot``.

No real Anthropic calls: the one test that exercises a full assembled
prompt through a client interface uses ``MockLLMClient``.
"""

from __future__ import annotations

import json
from pathlib import Path

from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Column, FKPair, Schema, Table
from schema_linking.utils.fewshot import pick_fewshot_examples
from schema_linking.utils.llm_client import MockLLMClient, MockTurn, _last_user_message_key
from schema_linking.utils.prompts import (
    BACKWARD_V1,
    FORWARD_V1,
    GRAPH_ENDPOINT_V1,
    estimate_tokens,
    render_backward_fewshot_block,
    render_backward_user_message,
    render_graph_endpoint_fewshot_block,
    render_graph_endpoint_user_message,
    render_schema_block,
    render_user_message,
)


def _schema_with_fk() -> Schema:
    singer_id = Column(
        name="singer id", original_name="singer_id", type="number",
        table_name="singer", is_primary_key=True,
    )
    name = Column(
        name="name", original_name="name", type="text",
        table_name="singer", is_primary_key=False,
    )
    singer = Table(name="singer", original_name="singer", columns=[singer_id, name])

    concert_id = Column(
        name="concert id", original_name="concert_id", type="number",
        table_name="concert", is_primary_key=True,
    )
    singer_ref = Column(
        name="singer id", original_name="singer_id", type="number",
        table_name="concert", is_primary_key=False,
    )
    concert = Table(name="concert", original_name="concert", columns=[concert_id, singer_ref])

    # Global indices: 0 = `*` (excluded), 1-2 = singer.*, 3-4 = concert.*.
    fk = FKPair(from_col_idx=4, to_col_idx=1)  # concert.singer_id -> singer.singer_id
    return Schema(db_id="concert_singer", tables=[singer, concert], foreign_keys=[fk])


def test_render_schema_block_produces_sql_ish_text() -> None:
    block = render_schema_block(_schema_with_fk())

    assert "CREATE TABLE singer (" in block
    assert "CREATE TABLE concert (" in block
    assert "singer_id INT PRIMARY KEY" in block
    assert "FOREIGN KEY (singer_id) REFERENCES singer(singer_id)" in block


def _simple_example() -> tuple[SpiderExample, dict]:
    example = SpiderExample(
        question_id=0, db_id="db_simple", question="How many singers are there?",
        query="SELECT COUNT(*) FROM singer", sql={
            "select": [False, [[3, [0, [0, 0, False], None]]]],
            "from": {"table_units": [["table_unit", 0]], "conds": []},
            "where": [],
        }, split="train",
    )
    gold = {"db_id": "db_simple", "tables": ["singer"], "columns": []}
    return example, gold


def _multi_table_example() -> tuple[SpiderExample, dict]:
    example = SpiderExample(
        question_id=1, db_id="db_multi", question="List singer names and concert names.",
        query="SELECT T1.name, T2.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.id",
        sql={
            "select": [False, [[0, [0, 0, False]]]],
            "from": {"table_units": [["table_unit", 0], ["table_unit", 1]], "conds": []},
            "where": [],
        }, split="train",
    )
    gold = {
        "db_id": "db_multi", "tables": ["singer", "concert"],
        "columns": [["singer", "name"], ["concert", "name"]],
    }
    return example, gold


def _schemas_for_fewshot_tests() -> dict[str, Schema]:
    singer_cols = [
        Column(name="id", original_name="id", type="number", table_name="singer", is_primary_key=True),
        Column(name="name", original_name="name", type="text", table_name="singer", is_primary_key=False),
    ]
    concert_cols = [
        Column(name="id", original_name="id", type="number", table_name="concert", is_primary_key=True),
        Column(name="name", original_name="name", type="text", table_name="concert", is_primary_key=False),
    ]
    return {
        "db_simple": Schema(
            db_id="db_simple",
            tables=[Table(name="singer", original_name="singer", columns=singer_cols)],
            foreign_keys=[],
        ),
        "db_multi": Schema(
            db_id="db_multi",
            tables=[
                Table(name="singer", original_name="singer", columns=singer_cols),
                Table(name="concert", original_name="concert", columns=concert_cols),
            ],
            foreign_keys=[],
        ),
    }


def test_pick_fewshot_examples_returns_n_with_expected_patterns() -> None:
    simple_ex, simple_gold = _simple_example()
    multi_ex, multi_gold = _multi_table_example()
    train_examples = [simple_ex, multi_ex]
    gold = {0: simple_gold, 1: multi_gold}
    schemas = _schemas_for_fewshot_tests()

    picked = pick_fewshot_examples(train_examples, schemas, gold, n=2, seed=42)

    assert len(picked) == 2
    assert {ex["pattern"] for ex in picked} == {"simple", "multi_table"}


def test_pick_fewshot_examples_uses_different_db_ids() -> None:
    simple_ex, simple_gold = _simple_example()
    multi_ex, multi_gold = _multi_table_example()
    train_examples = [simple_ex, multi_ex]
    gold = {0: simple_gold, 1: multi_gold}
    schemas = _schemas_for_fewshot_tests()

    picked = pick_fewshot_examples(train_examples, schemas, gold, n=2, seed=42)

    assert picked[0]["db_id"] != picked[1]["db_id"]


def test_full_prompt_assembly_under_4000_tokens(tmp_path: Path) -> None:
    schema = _schema_with_fk()
    schema_block = render_schema_block(schema)
    fewshot_examples = [
        {"question": "How many singers are there?", "tables": ["singer"], "columns": []},
        {
            "question": "List singer and concert names.",
            "tables": ["singer", "concert"],
            "columns": [["singer", "name"], ["concert", "name"]],
        },
    ]
    fewshot_schema_blocks = [schema_block, schema_block]

    user_message = render_user_message(
        FORWARD_V1, schema_block, "Show all singers who performed in a concert.",
        fewshot_examples=fewshot_examples, fewshot_schema_blocks=fewshot_schema_blocks,
    )
    full_prompt = FORWARD_V1.system + "\n\n" + user_message

    assert estimate_tokens(full_prompt) < 4000

    # End-to-end wiring sanity check, via MockLLMClient (no real API call).
    messages = [{"role": "user", "content": user_message}]
    key = _last_user_message_key(messages)
    canned = json.dumps({"tables": ["singer"], "columns": [["singer", "name"]]})
    client = MockLLMClient(
        model="claude-haiku-4-5-20251001", temperature=0.0, max_tokens=512,
        log_path=tmp_path / "llm_calls.jsonl",
        responses={key: MockTurn(text=canned)}, cost_cap_usd=None,
    )
    response = client.call(system=FORWARD_V1.system, messages=messages)
    parsed = json.loads(response.text)
    assert set(parsed.keys()) == {"tables", "columns"}


def test_backward_v1_version_tag() -> None:
    assert BACKWARD_V1.version == "backward_v1"


def test_backward_v1_has_no_output_schema() -> None:
    assert BACKWARD_V1.output_schema is None


def test_backward_fewshot_block_uses_sql_not_output() -> None:
    block = render_backward_fewshot_block(
        1,
        {"question": "How many singers are there?", "gold_sql": "SELECT COUNT(*) FROM singer"},
        "CREATE TABLE singer (\n  singer_id INT PRIMARY KEY\n);",
    )
    assert "SQL: SELECT COUNT(*) FROM singer" in block
    assert "Output:" not in block


def test_backward_full_prompt_assembly_under_4000_tokens() -> None:
    schema = _schema_with_fk()
    schema_block = render_schema_block(schema)
    fewshot_examples = [
        {"question": "How many singers are there?", "gold_sql": "SELECT COUNT(*) FROM singer"},
        {
            "question": "List singer and concert names.",
            "gold_sql": "SELECT T1.name, T2.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id",
        },
    ]
    fewshot_schema_blocks = [schema_block, schema_block]

    user_message = render_backward_user_message(
        BACKWARD_V1, schema_block, "Show all singers who performed in a concert.",
        fewshot_examples=fewshot_examples, fewshot_schema_blocks=fewshot_schema_blocks,
    )
    full_prompt = BACKWARD_V1.system + "\n\n" + user_message

    assert estimate_tokens(full_prompt) < 4000
    assert user_message.rstrip().endswith("SQL:")


def test_graph_endpoint_v1_version_tag() -> None:
    assert GRAPH_ENDPOINT_V1.version == "graph_endpoint_v1"


def test_graph_endpoint_fewshot_block_uses_core_tables_not_tables() -> None:
    block = render_graph_endpoint_fewshot_block(
        1,
        {"question": "How many singers are there?", "core_tables": ["singer"], "columns": []},
        "CREATE TABLE singer (\n  singer_id INT PRIMARY KEY\n);",
    )
    assert '"core_tables"' in block
    assert '"tables"' not in block


def test_graph_endpoint_full_prompt_assembly_under_4500_tokens() -> None:
    schema = _schema_with_fk()
    schema_block = render_schema_block(schema)
    fewshot_examples = [
        {"question": "How many singers are there?", "core_tables": ["singer"], "columns": []},
        {
            "question": "List singer and concert names.",
            "core_tables": ["singer", "concert"],
            "columns": [["singer", "name"], ["concert", "name"]],
        },
    ]
    fewshot_schema_blocks = [schema_block, schema_block]

    user_message = render_graph_endpoint_user_message(
        GRAPH_ENDPOINT_V1, schema_block, "Show all singers who performed in a concert.",
        fewshot_examples=fewshot_examples, fewshot_schema_blocks=fewshot_schema_blocks,
    )
    full_prompt = GRAPH_ENDPOINT_V1.system + "\n\n" + user_message

    assert estimate_tokens(full_prompt) < 4500
