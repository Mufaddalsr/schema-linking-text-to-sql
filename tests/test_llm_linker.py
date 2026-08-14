"""Tests for ``schema_linking.llm_linker.LLMForwardLinker``.

All tests use ``MockLLMClient`` — no real Anthropic API calls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from schema_linking.data_loader import SpiderExample
from schema_linking.llm_linker import BidirectionalLinker, LLMBackwardLinker, LLMForwardLinker
from schema_linking.schema_parser import Column, Schema, Table
from schema_linking.utils.llm_client import MockLLMClient, MockTurn, _last_user_message_key
from schema_linking.utils.prompts import (
    BACKWARD_V1,
    FORWARD_V1,
    render_backward_user_message,
    render_schema_block,
    render_user_message,
)

MODEL = "claude-haiku-4-5-20251001"


def _schema(db_id: str = "concert_singer") -> Schema:
    cols = [
        Column(name="id", original_name="singer_id", type="number", table_name="singer", is_primary_key=True),
        Column(name="name", original_name="name", type="text", table_name="singer", is_primary_key=False),
    ]
    return Schema(db_id=db_id, tables=[Table(name="singer", original_name="singer", columns=cols)], foreign_keys=[])


def _example(
    qid: int = 0, db_id: str = "concert_singer", question: str = "How many singers are there?"
) -> SpiderExample:
    return SpiderExample(question_id=qid, db_id=db_id, question=question, query="SELECT 1", sql={}, split="dev")


def _user_message_key(example: SpiderExample, schema: Schema) -> str:
    """Compute the same key LLMForwardLinker will use, for programming MockLLMClient."""
    schema_block = render_schema_block(schema)
    user_message = render_user_message(FORWARD_V1, schema_block, example.question)
    return _last_user_message_key([{"role": "user", "content": user_message}])


def _linker(
    client: MockLLMClient, k_samples: int = 3, aggregation: str = "union", tmp_path: Path | None = None
) -> LLMForwardLinker:
    return LLMForwardLinker(
        llm_client=client, prompt=FORWARD_V1, few_shot=[], k_samples=k_samples,
        aggregation=aggregation,
        parse_failure_log_path=(tmp_path / "llm_parse_failures.jsonl") if tmp_path else None,
    )


def test_basic_identical_samples(tmp_path: Path) -> None:
    example, schema = _example(), _schema()
    key = _user_message_key(example, schema)
    answer = json.dumps({"tables": ["singer"], "columns": [["singer", "name"]]})
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, k_samples=3, tmp_path=tmp_path)

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)
    assert prediction.columns == (("singer", "name"),)
    assert prediction.extra["n_samples_parsed"] == 3
    assert prediction.extra["n_samples_valid"] == 3


def test_union_aggregation_takes_any_element(tmp_path: Path) -> None:
    example, schema = _example(), _schema()
    key = _user_message_key(example, schema)
    texts = [
        json.dumps({"tables": ["singer"], "columns": [["singer", "name"]]}),
        json.dumps({"tables": ["singer"], "columns": []}),
        json.dumps({"tables": [], "columns": []}),
    ]
    client = MockLLMClient(
        model=MODEL, temperature=1.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(texts=texts)},
    )
    linker = _linker(client, k_samples=3, aggregation="union", tmp_path=tmp_path)

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)  # 2/3 samples
    assert prediction.columns == (("singer", "name"),)  # 1/3 samples, union still takes it


def test_majority_aggregation_requires_ceil_half(tmp_path: Path) -> None:
    example, schema = _example(), _schema()
    key = _user_message_key(example, schema)
    texts = [
        json.dumps({"tables": ["singer"], "columns": [["singer", "name"]]}),
        json.dumps({"tables": ["singer"], "columns": []}),
        json.dumps({"tables": [], "columns": []}),
    ]
    client = MockLLMClient(
        model=MODEL, temperature=1.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(texts=texts)},
    )
    linker = _linker(client, k_samples=3, aggregation="majority", tmp_path=tmp_path)

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)  # 2/3 >= ceil(3/2)=2
    assert prediction.columns == ()  # 1/3 < 2


def test_one_parse_failure_still_aggregates_other_samples(tmp_path: Path) -> None:
    example, schema = _example(), _schema()
    key = _user_message_key(example, schema)
    valid = json.dumps({"tables": ["singer"], "columns": [["singer", "name"]]})
    texts = [valid, "not json at all", valid]
    client = MockLLMClient(
        model=MODEL, temperature=1.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(texts=texts)},
    )
    linker = _linker(client, k_samples=3, aggregation="union", tmp_path=tmp_path)

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)
    assert prediction.extra["n_samples_parsed"] == 2
    assert prediction.extra["n_samples_valid"] == 2


def test_all_samples_fail_to_parse_returns_empty_and_logs(tmp_path: Path) -> None:
    example, schema = _example(), _schema()
    key = _user_message_key(example, schema)
    texts = ["garbage one", "garbage two", "garbage three"]
    log_path = tmp_path / "llm_parse_failures.jsonl"
    client = MockLLMClient(
        model=MODEL, temperature=1.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(texts=texts)},
    )
    linker = LLMForwardLinker(
        llm_client=client, prompt=FORWARD_V1, few_shot=[], k_samples=3,
        parse_failure_log_path=log_path,
    )

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ()
    assert prediction.columns == ()
    assert prediction.extra["n_samples_parsed"] == 0
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["reason"] == "parse_error" for line in lines)


def test_hallucinated_table_passes_through_unfiltered(tmp_path: Path) -> None:
    example, schema = _example(), _schema()
    key = _user_message_key(example, schema)
    answer = json.dumps({"tables": ["fake_table"], "columns": []})
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer)},
    )
    linker = _linker(client, k_samples=1, tmp_path=tmp_path)

    prediction = linker.predict_one(example, schema)

    assert "fake_table" in prediction.tables


def test_total_cost_usd_sums_across_samples(tmp_path: Path) -> None:
    example, schema = _example(), _schema()
    key = _user_message_key(example, schema)
    answer = json.dumps({"tables": ["singer"], "columns": []})
    client = MockLLMClient(
        model=MODEL, temperature=0.0, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text=answer, input_tokens=1000, output_tokens=100)},
    )
    linker = _linker(client, k_samples=3, tmp_path=tmp_path)

    prediction = linker.predict_one(example, schema)

    # Haiku 4.5 pricing: $1.00/MTok in, $5.00/MTok out (see llm_client.py).
    per_call_cost = (1000 * 1.00 + 100 * 5.00) / 1_000_000
    assert prediction.extra["total_cost_usd"] == pytest.approx(3 * per_call_cost)
    assert prediction.extra["total_input_tokens"] == 3000
    assert prediction.extra["total_output_tokens"] == 300


def test_predict_all_batches_calls_by_db_id(tmp_path: Path) -> None:
    schema_a = _schema(db_id="db_a")
    schema_b = _schema(db_id="db_b")
    ex1 = _example(qid=1, db_id="db_a")
    ex2 = _example(qid=2, db_id="db_b")
    ex3 = _example(qid=3, db_id="db_a")

    answer = json.dumps({"tables": [], "columns": []})
    key_a = _user_message_key(ex1, schema_a)  # ex1 and ex3 render the same prompt (same db, same question)
    key_b = _user_message_key(ex2, schema_b)

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
    linker = _linker(client, k_samples=1, tmp_path=tmp_path)

    linker.predict_all([ex1, ex2, ex3], {"db_a": schema_a, "db_b": schema_b})

    assert client.call_order == [("db_a", 1), ("db_a", 3), ("db_b", 2)]


# ============================================================
# LLMBackwardLinker (Method D)
# ============================================================


def _backward_schema(db_id: str = "concert_singer") -> Schema:
    singer_cols = [
        Column(name="id", original_name="singer_id", type="number", table_name="singer", is_primary_key=True),
        Column(name="name", original_name="name", type="text", table_name="singer", is_primary_key=False),
        Column(name="age", original_name="age", type="number", table_name="singer", is_primary_key=False),
    ]
    concert_cols = [
        Column(name="id", original_name="concert_id", type="number", table_name="concert", is_primary_key=True),
        Column(name="name", original_name="name", type="text", table_name="concert", is_primary_key=False),
        Column(name="singer id", original_name="singer_id", type="number", table_name="concert", is_primary_key=False),
    ]
    return Schema(
        db_id=db_id,
        tables=[
            Table(name="singer", original_name="singer", columns=singer_cols),
            Table(name="concert", original_name="concert", columns=concert_cols),
        ],
        foreign_keys=[],
    )


def _backward_user_message_key(example: SpiderExample, schema: Schema) -> str:
    schema_block = render_schema_block(schema)
    user_message = render_backward_user_message(BACKWARD_V1, schema_block, example.question)
    return _last_user_message_key([{"role": "user", "content": user_message}])


def _backward_linker(
    client: MockLLMClient, tmp_path: Path, temperature_override: float | None = 0.0
) -> LLMBackwardLinker:
    return LLMBackwardLinker(
        llm_client=client, prompt=BACKWARD_V1, few_shot=[],
        temperature_override=temperature_override,
        parse_failure_log_path=tmp_path / "llm_sql_parse_failures.jsonl",
        sql_output_path=tmp_path / "llm_backward_dev_sql.jsonl",
    )


def test_backward_basic_where_clause(tmp_path: Path) -> None:
    example, schema = _example(question="Which singers are over 30?"), _backward_schema()
    key = _backward_user_message_key(example, schema)
    client = MockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="SELECT name FROM singer WHERE age > 30")},
    )
    linker = _backward_linker(client, tmp_path)

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("singer",)
    assert prediction.columns == (("singer", "age"), ("singer", "name"))


def test_backward_join(tmp_path: Path) -> None:
    example, schema = _example(question="List concert names by singer."), _backward_schema()
    key = _backward_user_message_key(example, schema)
    client = MockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(
            text="SELECT c.name FROM singer s JOIN concert c ON s.singer_id = c.singer_id"
        )},
    )
    linker = _backward_linker(client, tmp_path)

    prediction = linker.predict_one(example, schema)

    assert set(prediction.tables) == {"singer", "concert"}
    assert ("concert", "name") in prediction.columns
    assert ("singer", "singer_id") in prediction.columns
    assert ("concert", "singer_id") in prediction.columns


def test_backward_hallucinated_table_passes_through(tmp_path: Path) -> None:
    example, schema = _example(question="Show me the widgets."), _backward_schema()
    key = _backward_user_message_key(example, schema)
    client = MockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="SELECT * FROM fake_table")},
    )
    linker = _backward_linker(client, tmp_path)

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("fake_table",)
    assert any(i["kind"] == "unknown_table" for i in prediction.extra["parse_issues"])


def test_backward_hallucinated_column_passes_through(tmp_path: Path) -> None:
    example, schema = _example(question="Show me the fake column."), _backward_schema()
    key = _backward_user_message_key(example, schema)
    client = MockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="SELECT fake_col FROM singer")},
    )
    linker = _backward_linker(client, tmp_path)

    prediction = linker.predict_one(example, schema)

    assert ("singer", "fake_col") in prediction.columns
    assert any(i["kind"] == "unknown_column" for i in prediction.extra["parse_issues"])


def test_backward_parse_failure_returns_empty_and_logs(tmp_path: Path) -> None:
    example, schema = _example(question="Broken query test."), _backward_schema()
    key = _backward_user_message_key(example, schema)
    client = MockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="SELECT FROM broken WHERE")},
    )
    log_path = tmp_path / "llm_sql_parse_failures.jsonl"
    linker = LLMBackwardLinker(
        llm_client=client, prompt=BACKWARD_V1, few_shot=[],
        parse_failure_log_path=log_path,
        sql_output_path=tmp_path / "llm_backward_dev_sql.jsonl",
    )

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ()
    assert prediction.columns == ()
    assert any(i["kind"] == "parse_error" for i in prediction.extra["parse_issues"])
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["reason"] == "parse_error"


def test_backward_strips_markdown_fence(tmp_path: Path) -> None:
    schema = Schema(
        db_id="concert_singer",
        tables=[Table(name="t", original_name="t", columns=[
            Column(name="a", original_name="a", type="number", table_name="t", is_primary_key=False),
        ])],
        foreign_keys=[],
    )
    example = _example(question="Show all t.")
    key = _backward_user_message_key(example, schema)
    client = MockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="```sql\nSELECT * FROM t\n```")},
    )
    linker = _backward_linker(client, tmp_path)

    prediction = linker.predict_one(example, schema)

    assert prediction.tables == ("t",)
    assert prediction.extra["raw_sql"] == "SELECT * FROM t"


def test_backward_temperature_override_spies_actual_call_temperature(tmp_path: Path) -> None:
    example, schema = _example(), _backward_schema()
    key = _backward_user_message_key(example, schema)

    class SpyMockLLMClient(MockLLMClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.temperatures_at_call: list[float] = []

        def call(self, system, messages, cacheable_prefix=None, metadata=None):
            self.temperatures_at_call.append(self.temperature)
            return super().call(system, messages, cacheable_prefix=cacheable_prefix, metadata=metadata)

    client = SpyMockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="SELECT name FROM singer")},
    )
    linker = _backward_linker(client, tmp_path, temperature_override=0.0)

    linker.predict_one(example, schema)

    assert client.temperatures_at_call == [0.0]
    assert client.temperature == 0.3  # restored after the call


def test_backward_predict_all_batches_by_db_id_and_writes_sql_dump(tmp_path: Path) -> None:
    schema_a = _backward_schema(db_id="db_a")
    schema_b = _backward_schema(db_id="db_b")
    ex1 = _example(qid=1, db_id="db_a", question="Q1")
    ex2 = _example(qid=2, db_id="db_b", question="Q2")
    ex3 = _example(qid=3, db_id="db_a", question="Q3")

    key1 = _backward_user_message_key(ex1, schema_a)
    key2 = _backward_user_message_key(ex2, schema_b)
    key3 = _backward_user_message_key(ex3, schema_a)

    class SpyMockLLMClient(MockLLMClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.call_order: list[tuple[str, int]] = []

        def call(self, system, messages, cacheable_prefix=None, metadata=None):
            self.call_order.append((metadata["db_id"], metadata["qid"]))
            return super().call(system, messages, cacheable_prefix=cacheable_prefix, metadata=metadata)

    client = SpyMockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={
            key1: MockTurn(text="SELECT name FROM singer"),
            key2: MockTurn(text="SELECT name FROM singer"),
            key3: MockTurn(text="SELECT name FROM singer"),
        },
    )
    sql_output_path = tmp_path / "llm_backward_dev_sql.jsonl"
    linker = LLMBackwardLinker(
        llm_client=client, prompt=BACKWARD_V1, few_shot=[],
        parse_failure_log_path=tmp_path / "llm_sql_parse_failures.jsonl",
        sql_output_path=sql_output_path,
    )

    linker.predict_all([ex1, ex2, ex3], {"db_a": schema_a, "db_b": schema_b})

    assert client.call_order == [("db_a", 1), ("db_a", 3), ("db_b", 2)]

    lines = [json.loads(line) for line in sql_output_path.read_text().strip().splitlines()]
    assert len(lines) == 3
    assert {line["question_id"] for line in lines} == {1, 2, 3}
    assert all(
        set(line.keys()) == {"question_id", "db_id", "question", "raw_sql", "parse_issues"}
        for line in lines
    )


def test_backward_predict_all_isolates_one_failing_example_and_continues(
    tmp_path: Path,
) -> None:
    """One example's underlying call raising an unexpected (non-retryable)
    exception must not lose the other examples' real results — this is
    exactly what happened for real: 912/1034 paid calls succeeded, then one
    uncaught exception crashed the whole batch and lost all of them, since
    predict_all only writes output after the full loop completes."""
    schema_a = _backward_schema(db_id="db_a")
    ex1 = _example(qid=1, db_id="db_a", question="Q1")
    ex2 = _example(qid=2, db_id="db_a", question="Q2 (this one blows up)")
    ex3 = _example(qid=3, db_id="db_a", question="Q3")

    key1 = _backward_user_message_key(ex1, schema_a)
    key2 = _backward_user_message_key(ex2, schema_a)
    key3 = _backward_user_message_key(ex3, schema_a)

    log_path = tmp_path / "llm_sql_parse_failures.jsonl"
    client = MockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={
            key1: MockTurn(text="SELECT name FROM singer"),
            key2: MockTurn(raises=[RuntimeError("simulated unexpected failure")]),
            key3: MockTurn(text="SELECT name FROM singer"),
        },
    )
    sql_output_path = tmp_path / "llm_backward_dev_sql.jsonl"
    linker = LLMBackwardLinker(
        llm_client=client, prompt=BACKWARD_V1, few_shot=[],
        parse_failure_log_path=log_path,
        sql_output_path=sql_output_path,
    )

    predictions = linker.predict_all([ex1, ex2, ex3], {"db_a": schema_a})

    assert set(predictions.keys()) == {1, 2, 3}
    assert predictions[1].tables == ("singer",)
    assert predictions[3].tables == ("singer",)
    assert predictions[2].tables == ()
    assert predictions[2].columns == ()
    assert any(i["kind"] == "other" for i in predictions[2].extra["parse_issues"])

    lines = [json.loads(line) for line in sql_output_path.read_text().strip().splitlines()]
    assert len(lines) == 3
    assert {line["question_id"] for line in lines} == {1, 2, 3}

    log_lines = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert any(
        entry["question_id"] == 2 and entry["reason"] == "unexpected_error"
        for entry in log_lines
    )


def test_backward_extra_metadata_merged_into_call_metadata(tmp_path: Path) -> None:
    example, schema = _example(), _backward_schema()
    key = _backward_user_message_key(example, schema)

    class SpyMockLLMClient(MockLLMClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.last_metadata: dict | None = None

        def call(self, system, messages, cacheable_prefix=None, metadata=None):
            self.last_metadata = metadata
            return super().call(system, messages, cacheable_prefix=cacheable_prefix, metadata=metadata)

    client = SpyMockLLMClient(
        model=MODEL, temperature=0.3, max_tokens=512, log_path=tmp_path / "calls.jsonl",
        responses={key: MockTurn(text="SELECT name FROM singer")},
    )
    linker = LLMBackwardLinker(
        llm_client=client, prompt=BACKWARD_V1, few_shot=[],
        parse_failure_log_path=tmp_path / "llm_sql_parse_failures.jsonl",
        sql_output_path=tmp_path / "llm_backward_dev_sql.jsonl",
        extra_metadata={"phase": "backward_prompt_iteration"},
    )

    linker.predict_one(example, schema)

    assert client.last_metadata["phase"] == "backward_prompt_iteration"
    assert client.last_metadata["method"] == "llm_backward"


# ============================================================
# BidirectionalLinker (Method E)
# ============================================================


def _write_predictions(path: Path, entries: dict[str, dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_bidirectional_basic_union(tmp_path: Path) -> None:
    forward_path = tmp_path / "forward.json"
    backward_path = tmp_path / "backward.json"
    _write_predictions(forward_path, {"0": {"db_id": "d", "tables": ["a"], "columns": []}})
    _write_predictions(backward_path, {"0": {"db_id": "d", "tables": ["b"], "columns": []}})
    linker = BidirectionalLinker(forward_path, backward_path)

    prediction = linker.predict_one(_example(qid=0, db_id="d"), _schema(db_id="d"))

    assert set(prediction.tables) == {"a", "b"}


def test_bidirectional_overlap_reports_both_count(tmp_path: Path) -> None:
    forward_path = tmp_path / "forward.json"
    backward_path = tmp_path / "backward.json"
    _write_predictions(forward_path, {"0": {"db_id": "d", "tables": ["a", "b"], "columns": []}})
    _write_predictions(backward_path, {"0": {"db_id": "d", "tables": ["b", "c"], "columns": []}})
    linker = BidirectionalLinker(forward_path, backward_path)

    prediction = linker.predict_one(_example(qid=0, db_id="d"), _schema(db_id="d"))

    assert set(prediction.tables) == {"a", "b", "c"}
    assert prediction.extra["n_tables_both"] == 1
    assert prediction.extra["n_tables_forward_only"] == 1
    assert prediction.extra["n_tables_backward_only"] == 1


def test_bidirectional_missing_forward_falls_back_to_backward(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    forward_path = tmp_path / "forward.json"
    backward_path = tmp_path / "backward.json"
    _write_predictions(forward_path, {})
    _write_predictions(backward_path, {"0": {"db_id": "d", "tables": ["b"], "columns": []}})
    linker = BidirectionalLinker(forward_path, backward_path)

    with caplog.at_level(logging.WARNING, logger="schema_linking.llm_linker"):
        prediction = linker.predict_one(_example(qid=0, db_id="d"), _schema(db_id="d"))

    assert prediction.tables == ("b",)
    assert any("forward" in r.getMessage() for r in caplog.records)


def test_bidirectional_missing_both_returns_empty_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    forward_path = tmp_path / "forward.json"
    backward_path = tmp_path / "backward.json"
    _write_predictions(forward_path, {})
    _write_predictions(backward_path, {})
    linker = BidirectionalLinker(forward_path, backward_path)

    with caplog.at_level(logging.WARNING, logger="schema_linking.llm_linker"):
        prediction = linker.predict_one(_example(qid=0, db_id="d"), _schema(db_id="d"))

    assert prediction.tables == ()
    assert prediction.columns == ()
    assert len(caplog.records) == 1


def test_bidirectional_extra_breakdown_for_columns(tmp_path: Path) -> None:
    forward_path = tmp_path / "forward.json"
    backward_path = tmp_path / "backward.json"
    _write_predictions(forward_path, {
        "0": {"db_id": "d", "tables": ["a"], "columns": [["a", "x"], ["a", "y"]]},
    })
    _write_predictions(backward_path, {
        "0": {"db_id": "d", "tables": ["a"], "columns": [["a", "y"], ["a", "z"]]},
    })
    linker = BidirectionalLinker(forward_path, backward_path)

    prediction = linker.predict_one(_example(qid=0, db_id="d"), _schema(db_id="d"))

    assert set(prediction.columns) == {("a", "x"), ("a", "y"), ("a", "z")}
    assert prediction.extra["n_columns_forward_only"] == 1  # x
    assert prediction.extra["n_columns_backward_only"] == 1  # z
    assert prediction.extra["n_columns_both"] == 1  # y
    assert prediction.extra["source"] == "bidirectional"
