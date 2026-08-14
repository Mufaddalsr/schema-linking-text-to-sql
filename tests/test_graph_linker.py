"""Tests for ``schema_linking.graph_linker.GraphLinker``.

All tests use ``MockLLMClient`` and a synthetic "concert_singer_lite" schema
— no real Anthropic API calls, no dependency on the real Spider dataset.

Schema shape (linear FK chain, by design — makes shortest-path/Steiner
results hand-verifiable):

    singer -- singer_in_concert -- concert -- stadium
"""

from __future__ import annotations

import json
from pathlib import Path

from schema_linking.data_loader import SpiderExample
from schema_linking.graph_linker import GraphLinker
from schema_linking.schema_parser import Column, FKPair, Schema, Table
from schema_linking.utils.llm_client import MockLLMClient, MockTurn, _last_user_message_key
from schema_linking.utils.prompts import GRAPH_ENDPOINT_V1, render_graph_endpoint_user_message, render_schema_block

MODEL = "claude-haiku-4-5-20251001"


def _lite_schema(db_id: str = "concert_singer_lite") -> Schema:
    singer_cols = [
        Column(name="id", original_name="singer_id", type="number", table_name="singer", is_primary_key=True),
        Column(name="name", original_name="name", type="text", table_name="singer", is_primary_key=False),
    ]
    concert_cols = [
        Column(name="id", original_name="concert_id", type="number", table_name="concert", is_primary_key=True),
        Column(name="stadium id", original_name="stadium_id", type="number", table_name="concert", is_primary_key=False),
        Column(name="year", original_name="year", type="number", table_name="concert", is_primary_key=False),
    ]
    stadium_cols = [
        Column(name="id", original_name="stadium_id", type="number", table_name="stadium", is_primary_key=True),
        Column(name="location", original_name="location", type="text", table_name="stadium", is_primary_key=False),
    ]
    sic_cols = [
        Column(name="concert id", original_name="concert_id", type="number", table_name="singer_in_concert", is_primary_key=False),
        Column(name="singer id", original_name="singer_id", type="number", table_name="singer_in_concert", is_primary_key=False),
    ]
    tables = [
        Table(name="singer", original_name="singer", columns=singer_cols),
        Table(name="concert", original_name="concert", columns=concert_cols),
        Table(name="stadium", original_name="stadium", columns=stadium_cols),
        Table(name="singer_in_concert", original_name="singer_in_concert", columns=sic_cols),
    ]
    # global indices: 0=*, 1-2=singer, 3-5=concert, 6-7=stadium, 8-9=singer_in_concert
    foreign_keys = [
        FKPair(from_col_idx=9, to_col_idx=1),  # singer_in_concert.singer_id -> singer.singer_id
        FKPair(from_col_idx=8, to_col_idx=3),  # singer_in_concert.concert_id -> concert.concert_id
        FKPair(from_col_idx=4, to_col_idx=6),  # concert.stadium_id -> stadium.stadium_id
    ]
    return Schema(db_id=db_id, tables=tables, foreign_keys=foreign_keys)


def _example(
    qid: int = 0, db_id: str = "concert_singer_lite", question: str = "Some question?"
) -> SpiderExample:
    return SpiderExample(question_id=qid, db_id=db_id, question=question, query="SELECT 1", sql={}, split="dev")


def _key(example: SpiderExample, schema: Schema) -> str:
    """Compute the same key GraphLinker will use, for programming MockLLMClient."""
    schema_block = render_schema_block(schema)
    user_message = render_graph_endpoint_user_message(GRAPH_ENDPOINT_V1, schema_block, example.question)
    return _last_user_message_key([{"role": "user", "content": user_message}])


def _linker(
    client: MockLLMClient, schemas: dict[str, Schema], trace_path: Path | None = None
) -> GraphLinker:
    return GraphLinker(
        llm_client=client, prompt=GRAPH_ENDPOINT_V1, few_shot=[], schemas=schemas, trace_path=trace_path,
    )


def test_two_endpoints_shortest_path(tmp_path: Path) -> None:
    schema = _lite_schema()
    example = _example(question="Where did singers perform?")
    key = _key(example, schema)
    answer = json.dumps(
        {"core_tables": ["singer", "stadium"], "columns": [["singer", "name"], ["stadium", "location"]]}
    )
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, {schema.db_id: schema})

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer", "singer_in_concert", "concert", "stadium")
    assert prediction.columns == (("singer", "name"), ("stadium", "location"))
    assert prediction.extra["failure"] is None


def test_single_endpoint(tmp_path: Path) -> None:
    schema = _lite_schema()
    example = _example(question="List singer names.")
    key = _key(example, schema)
    answer = json.dumps({"core_tables": ["singer"], "columns": [["singer", "name"]]})
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, {schema.db_id: schema})

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)
    assert prediction.columns == (("singer", "name"),)


def test_three_endpoints_steiner(tmp_path: Path) -> None:
    schema = _lite_schema()
    example = _example(question="Singers, concerts, and stadiums?")
    key = _key(example, schema)
    answer = json.dumps({"core_tables": ["singer", "concert", "stadium"], "columns": []})
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, {schema.db_id: schema})

    prediction = linker.predict_one(example, schema)

    # Hand-verified: linear chain singer-SIC-concert-stadium, so connecting
    # singer, concert, stadium necessarily pulls in singer_in_concert too.
    assert prediction.tables == ("concert", "singer", "singer_in_concert", "stadium")


def test_hallucinated_endpoint_dropped(tmp_path: Path) -> None:
    schema = _lite_schema()
    example = _example(question="List singer names.")
    key = _key(example, schema)
    answer = json.dumps({"core_tables": ["singer", "fake_table"], "columns": []})
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, {schema.db_id: schema})

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)
    assert prediction.extra["llm_endpoints_resolved"] == ["singer"]
    assert prediction.extra["n_endpoints_hallucinated"] == 1


def test_all_endpoints_hallucinated(tmp_path: Path) -> None:
    schema = _lite_schema()
    example = _example(question="Nonsense question.")
    key = _key(example, schema)
    answer = json.dumps({"core_tables": ["fake1", "fake2"], "columns": []})
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, {schema.db_id: schema})

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ()
    assert prediction.columns == ()
    assert prediction.extra["failure"] == "no_valid_endpoints"


def test_column_dropped_off_predicted_path(tmp_path: Path) -> None:
    schema = _lite_schema()
    example = _example(question="List singer names.")
    key = _key(example, schema)
    answer = json.dumps(
        {"core_tables": ["singer"], "columns": [["singer", "name"], ["concert", "year"]]}
    )
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, {schema.db_id: schema})

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)
    assert prediction.columns == (("singer", "name"),)
    assert prediction.extra["columns_dropped_off_path"] == 1


def test_parse_failure_returns_empty_and_logs(tmp_path: Path, caplog) -> None:
    schema = _lite_schema()
    example = _example(question="Garbage response question.")
    key = _key(example, schema)
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="garbage not json")},
    )
    linker = _linker(client, {schema.db_id: schema})

    with caplog.at_level("WARNING", logger="schema_linking.graph_linker"):
        prediction = linker.predict_one(example, schema)

    assert prediction.tables == ()
    assert prediction.columns == ()
    assert prediction.extra["failure"] == "parse"
    assert any("failed to parse" in record.message for record in caplog.records)


def test_predict_all_writes_trace_file(tmp_path: Path) -> None:
    schema = _lite_schema()
    examples = [
        _example(qid=1, question="Q1?"),
        _example(qid=2, question="Q2?"),
        _example(qid=3, question="Q3?"),
    ]
    responses = {}
    for ex in examples:
        answer = json.dumps({"core_tables": ["singer"], "columns": [["singer", "name"]]})
        responses[_key(ex, schema)] = MockTurn(text=answer)

    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses=responses,
    )
    trace_path = tmp_path / "graph_dev_traces.jsonl"
    linker = _linker(client, {schema.db_id: schema}, trace_path=trace_path)

    linker.predict_all(examples, {schema.db_id: schema})

    lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line, ex in zip(lines, examples):
        record = json.loads(line)
        assert record["qid"] == ex.question_id
        assert record["db_id"] == ex.db_id
        assert record["question"] == ex.question
        assert record["final_tables"] == ["singer"]
        assert record["final_columns"] == [["singer", "name"]]
        assert record["graph_result"] == ["singer"]
        assert record["endpoints_resolved"] == ["singer"]
        assert record["failure"] is None
        assert "llm_raw" in record


def test_predict_all_batches_by_db_id(tmp_path: Path) -> None:
    schema_a = _lite_schema(db_id="db_a")
    schema_b = _lite_schema(db_id="db_b")
    ex1 = _example(qid=1, db_id="db_a", question="Same question")
    ex2 = _example(qid=2, db_id="db_b", question="Same question")
    ex3 = _example(qid=3, db_id="db_a", question="Same question")

    answer = json.dumps({"core_tables": ["singer"], "columns": []})
    key_a = _key(ex1, schema_a)
    key_b = _key(ex2, schema_b)

    class SpyMockLLMClient(MockLLMClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.call_order: list[tuple[str, int]] = []

        def call(self, system, messages, cacheable_prefix=None, metadata=None):
            self.call_order.append((metadata["db_id"], metadata["qid"]))
            return super().call(system, messages, cacheable_prefix=cacheable_prefix, metadata=metadata)

    client = SpyMockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key_a: MockTurn(text=answer), key_b: MockTurn(text=answer)},
    )
    linker = _linker(client, {"db_a": schema_a, "db_b": schema_b}, trace_path=tmp_path / "traces.jsonl")

    linker.predict_all([ex1, ex2, ex3], {"db_a": schema_a, "db_b": schema_b})

    assert client.call_order == [("db_a", 1), ("db_a", 3), ("db_b", 2)]


def test_extra_metadata_merged_into_call_metadata(tmp_path: Path) -> None:
    schema = _lite_schema()
    example = _example(question="List singer names.")
    key = _key(example, schema)
    answer = json.dumps({"core_tables": ["singer"], "columns": []})

    captured: dict = {}

    class SpyMockLLMClient(MockLLMClient):
        def call(self, system, messages, cacheable_prefix=None, metadata=None):
            captured.update(metadata)
            return super().call(system, messages, cacheable_prefix=cacheable_prefix, metadata=metadata)

    client = SpyMockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = GraphLinker(
        llm_client=client, prompt=GRAPH_ENDPOINT_V1, few_shot=[], schemas={schema.db_id: schema},
        extra_metadata={"phase": "graph_dev_run", "prompt_version": GRAPH_ENDPOINT_V1.version},
    )

    linker.predict_one(example, schema)

    assert captured["phase"] == "graph_dev_run"
    assert captured["prompt_version"] == GRAPH_ENDPOINT_V1.version
    assert captured["method"] == "graph"
    assert captured["qid"] == example.question_id
