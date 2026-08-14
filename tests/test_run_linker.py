"""Tests for ``schema_linking.run_linker``.

Synthetic only — no Spider data touched here, no real Anthropic API calls
(the LLM forward tests use ``MockLLMClient``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schema_linking.data_loader import SpiderExample
from schema_linking.graph_linker import GraphLinker
from schema_linking.lexical_linker import LexicalLinker
from schema_linking.llm_linker import BidirectionalLinker, LLMBackwardLinker, LLMForwardLinker
from schema_linking.run_linker import (
    estimate_graph_cost,
    estimate_llm_backward_cost,
    estimate_llm_forward_cost,
    run_bidirectional,
    run_graph,
    run_lexical,
    run_llm_backward,
    run_llm_forward,
    write_llm_cost_report,
)
from schema_linking.schema_parser import Column, Schema, Table
from schema_linking.utils.llm_client import MockLLMClient, MockTurn, _last_user_message_key
from schema_linking.utils.prompts import (
    BACKWARD_V1,
    FORWARD_V1,
    GRAPH_ENDPOINT_V1,
    render_backward_user_message,
    render_graph_endpoint_user_message,
    render_schema_block,
    render_user_message,
)


def _table(name: str, cols: list[tuple[str, str]]) -> Table:
    columns = [
        Column(
            name=oname.replace("_", " "),
            original_name=oname,
            type=ctype,
            table_name=name,
            is_primary_key=False,
        )
        for oname, ctype in cols
    ]
    return Table(name=name, original_name=name, columns=columns)


@pytest.fixture
def schema() -> Schema:
    return Schema(
        db_id="test_db",
        tables=[
            _table("singer", [("name", "text"), ("age", "number")]),
            _table("stadium", [("name", "text"), ("capacity", "number")]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def synthetic_examples() -> list[SpiderExample]:
    questions = [
        "How many singers are there?",
        "What is the stadium capacity?",
        "List singer names",
        "Show me the stadium name",
        "How old is each singer?",
    ]
    return [
        SpiderExample(
            question_id=i,
            db_id="test_db",
            question=q,
            query="SELECT 1",
            sql={},
            split="dev",
        )
        for i, q in enumerate(questions)
    ]


def test_run_lexical_writes_valid_json(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    out = tmp_path / "lexical_synth.json"
    run_lexical(
        linker=LexicalLinker(fuzzy_threshold=80),
        examples=synthetic_examples,
        schemas={"test_db": schema},
        output_path=out,
    )

    assert out.is_file()
    with out.open(encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 5
    assert set(data.keys()) == {"0", "1", "2", "3", "4"}
    for entry in data.values():
        assert set(entry.keys()) == {"db_id", "tables", "columns"}
        assert entry["db_id"] == "test_db"
        assert isinstance(entry["tables"], list)
        assert isinstance(entry["columns"], list)


def test_run_lexical_creates_parent_dirs(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    out = tmp_path / "nested" / "deeper" / "lexical.json"
    run_lexical(
        linker=LexicalLinker(fuzzy_threshold=80),
        examples=synthetic_examples,
        schemas={"test_db": schema},
        output_path=out,
    )
    assert out.is_file()


def _mock_llm_forward_linker(
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    tmp_path: Path,
    k_samples: int = 2,
    tables: list[str] | None = None,
    columns: list[list[str]] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> LLMForwardLinker:
    """Build an ``LLMForwardLinker`` over ``MockLLMClient`` that answers every
    example in ``examples`` with the same canned ``{tables, columns}``."""
    answer = json.dumps({"tables": tables or ["singer"], "columns": columns or []})
    responses = {}
    for ex in examples:
        schema_block = render_schema_block(schemas[ex.db_id])
        user_message = render_user_message(FORWARD_V1, schema_block, ex.question)
        key = _last_user_message_key([{"role": "user", "content": user_message}])
        responses[key] = MockTurn(text=answer, input_tokens=input_tokens, output_tokens=output_tokens)

    client = MockLLMClient(
        model="claude-haiku-4-5-20251001",
        temperature=0.3,
        max_tokens=512,
        log_path=tmp_path / "llm_calls.jsonl",
        responses=responses,
        cost_cap_usd=None,
    )
    return LLMForwardLinker(
        llm_client=client,
        prompt=FORWARD_V1,
        few_shot=[],
        k_samples=k_samples,
        aggregation="union",
        parse_failure_log_path=tmp_path / "parse_failures.jsonl",
    )


def test_run_llm_forward_writes_predictions_and_samples(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    linker = _mock_llm_forward_linker(
        synthetic_examples, {"test_db": schema}, tmp_path,
        k_samples=2, tables=["singer"], columns=[["singer", "name"]],
    )
    output_path = tmp_path / "llm_forward.json"
    samples_path = tmp_path / "llm_forward_samples.jsonl"

    run_llm_forward(linker, synthetic_examples, {"test_db": schema}, output_path, samples_path)

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(predictions) == 5
    for entry in predictions.values():
        assert entry["db_id"] == "test_db"
        assert entry["tables"] == ["singer"]
        assert entry["columns"] == [["singer", "name"]]

    lines = samples_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        record = json.loads(line)
        assert set(record.keys()) == {
            "question_id", "db_id", "sample_predictions", "n_samples_parsed",
            "n_samples_valid", "total_input_tokens", "total_output_tokens", "total_cost_usd",
        }
        assert record["n_samples_parsed"] == 2
        assert record["n_samples_valid"] == 2
        assert record["total_input_tokens"] == 200
        assert record["total_output_tokens"] == 40
        assert len(record["sample_predictions"]) == 2


def test_estimate_llm_forward_cost_projects_linearly(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    dry_run_examples = synthetic_examples[:2]
    linker = _mock_llm_forward_linker(
        dry_run_examples, {"test_db": schema}, tmp_path,
        k_samples=2, input_tokens=100, output_tokens=20,
    )

    report = estimate_llm_forward_cost(
        linker, dry_run_examples, {"test_db": schema}, full_dev_size=100
    )

    # Haiku 4.5 pricing: $1.00/MTok in, $5.00/MTok out (see llm_client.py).
    # Per call: (100*1 + 20*5) / 1e6 = 0.0002; per query (k=2): 0.0004.
    per_query_cost = pytest.approx(0.0004)
    assert report["n_dry_run"] == 2
    assert report["dry_run_cost_usd"] == pytest.approx(0.0008)
    assert report["avg_cost_per_query_usd"] == per_query_cost
    assert report["full_dev_size"] == 100
    assert report["projected_total_cost_usd"] == pytest.approx(0.04)


def _mock_llm_backward_linker(
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    tmp_path: Path,
    sql_text: str = "SELECT name FROM singer",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> LLMBackwardLinker:
    """Build an ``LLMBackwardLinker`` over ``MockLLMClient`` that answers
    every example with the same canned SQL string."""
    responses = {}
    for ex in examples:
        schema_block = render_schema_block(schemas[ex.db_id])
        user_message = render_backward_user_message(BACKWARD_V1, schema_block, ex.question)
        key = _last_user_message_key([{"role": "user", "content": user_message}])
        responses[key] = MockTurn(text=sql_text, input_tokens=input_tokens, output_tokens=output_tokens)

    client = MockLLMClient(
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
        max_tokens=512,
        log_path=tmp_path / "llm_calls_backward.jsonl",
        responses=responses,
        cost_cap_usd=None,
    )
    return LLMBackwardLinker(
        llm_client=client,
        prompt=BACKWARD_V1,
        few_shot=[],
        parse_failure_log_path=tmp_path / "parse_failures_backward.jsonl",
        sql_output_path=tmp_path / "llm_backward_sql_dump.jsonl",
    )


def test_run_llm_backward_writes_predictions_and_sql_dump(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    linker = _mock_llm_backward_linker(synthetic_examples, {"test_db": schema}, tmp_path)
    output_path = tmp_path / "llm_backward.json"

    run_llm_backward(linker, synthetic_examples, {"test_db": schema}, output_path)

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(predictions) == 5
    for entry in predictions.values():
        assert entry["db_id"] == "test_db"
        assert entry["tables"] == ["singer"]

    # LLMBackwardLinker.predict_all writes its own sql_output_path — run_llm_backward
    # must not duplicate it, just leave it where the linker put it.
    sql_lines = linker.sql_output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(sql_lines) == 5
    for line in sql_lines:
        record = json.loads(line)
        assert record["raw_sql"] == "SELECT name FROM singer"


def test_estimate_llm_backward_cost_projects_linearly(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    dry_run_examples = synthetic_examples[:2]
    linker = _mock_llm_backward_linker(
        dry_run_examples, {"test_db": schema}, tmp_path,
        input_tokens=100, output_tokens=20,
    )

    report = estimate_llm_backward_cost(
        linker, dry_run_examples, {"test_db": schema}, full_dev_size=100
    )

    # Haiku 4.5 pricing: $1.00/MTok in, $5.00/MTok out.
    # Per call: (100*1 + 20*5) / 1e6 = 0.0002; k=1 so per-query == per-call.
    assert report["n_dry_run"] == 2
    assert report["dry_run_cost_usd"] == pytest.approx(0.0004)
    assert report["avg_cost_per_query_usd"] == pytest.approx(0.0002)
    assert report["full_dev_size"] == 100
    assert report["projected_total_cost_usd"] == pytest.approx(0.02)


def _mock_graph_linker(
    examples: list[SpiderExample],
    schemas: dict[str, Schema],
    tmp_path: Path,
    tables: list[str] | None = None,
    columns: list[list[str]] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> GraphLinker:
    """Build a ``GraphLinker`` over ``MockLLMClient`` that answers every
    example with the same canned ``{core_tables, columns}``."""
    answer = json.dumps({"core_tables": tables or ["singer"], "columns": columns or []})
    responses = {}
    for ex in examples:
        schema_block = render_schema_block(schemas[ex.db_id])
        user_message = render_graph_endpoint_user_message(GRAPH_ENDPOINT_V1, schema_block, ex.question)
        key = _last_user_message_key([{"role": "user", "content": user_message}])
        responses[key] = MockTurn(text=answer, input_tokens=input_tokens, output_tokens=output_tokens)

    client = MockLLMClient(
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
        max_tokens=512,
        log_path=tmp_path / "llm_calls_graph.jsonl",
        responses=responses,
        cost_cap_usd=None,
    )
    return GraphLinker(
        llm_client=client,
        prompt=GRAPH_ENDPOINT_V1,
        few_shot=[],
        schemas=schemas,
        trace_path=tmp_path / "graph_traces.jsonl",
    )


def test_run_graph_writes_predictions_and_traces(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    linker = _mock_graph_linker(
        synthetic_examples, {"test_db": schema}, tmp_path,
        tables=["singer"], columns=[["singer", "name"]],
    )
    output_path = tmp_path / "graph.json"

    run_graph(linker, synthetic_examples, {"test_db": schema}, output_path)

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(predictions) == 5
    for entry in predictions.values():
        assert entry["db_id"] == "test_db"
        assert entry["tables"] == ["singer"]
        assert entry["columns"] == [["singer", "name"]]

    # GraphLinker.predict_all writes its own trace_path — run_graph must not
    # duplicate it, just leave it where the linker put it.
    trace_lines = linker.trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == 5
    for line in trace_lines:
        record = json.loads(line)
        assert record["final_tables"] == ["singer"]
        assert record["failure"] is None


def test_estimate_graph_cost_projects_linearly(
    schema: Schema,
    synthetic_examples: list[SpiderExample],
    tmp_path: Path,
) -> None:
    dry_run_examples = synthetic_examples[:2]
    linker = _mock_graph_linker(
        dry_run_examples, {"test_db": schema}, tmp_path,
        input_tokens=100, output_tokens=20,
    )

    report = estimate_graph_cost(
        linker, dry_run_examples, {"test_db": schema}, full_dev_size=100
    )

    # Haiku 4.5 pricing: $1.00/MTok in, $5.00/MTok out.
    # Per call: (100*1 + 20*5) / 1e6 = 0.0002; k=1 so per-query == per-call.
    assert report["n_dry_run"] == 2
    assert report["dry_run_cost_usd"] == pytest.approx(0.0004)
    assert report["avg_cost_per_query_usd"] == pytest.approx(0.0002)
    assert report["full_dev_size"] == 100
    assert report["projected_total_cost_usd"] == pytest.approx(0.02)


def test_run_bidirectional_unions_forward_and_backward(
    schema: Schema, tmp_path: Path
) -> None:
    forward_path = tmp_path / "forward.json"
    backward_path = tmp_path / "backward.json"
    forward_path.write_text(json.dumps({
        "0": {"db_id": "test_db", "tables": ["singer"], "columns": []},
    }))
    backward_path.write_text(json.dumps({
        "0": {"db_id": "test_db", "tables": ["stadium"], "columns": []},
    }))
    linker = BidirectionalLinker(forward_path, backward_path)
    example = SpiderExample(
        question_id=0, db_id="test_db", question="q", query="SELECT 1", sql={}, split="dev",
    )
    output_path = tmp_path / "bidirectional.json"

    run_bidirectional(linker, [example], {"test_db": schema}, output_path)

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(predictions["0"]["tables"]) == {"singer", "stadium"}


def test_write_llm_cost_report_aggregates_across_phases(tmp_path: Path) -> None:
    iteration_path = tmp_path / "iteration.jsonl"
    dev_run_path = tmp_path / "dev_run.jsonl"

    iteration_lines = [
        {"cost_usd": 0.01, "input_tokens": 1000, "output_tokens": 100,
         "cache_read_input_tokens": 500, "cache_creation_input_tokens": 0,
         "metadata": {"phase": "iteration"}},
        {"cost_usd": 0.02, "input_tokens": 2000, "output_tokens": 200,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1000,
         "metadata": {"phase": "iteration"}},
    ]
    iteration_path.write_text("\n".join(json.dumps(e) for e in iteration_lines) + "\n")

    dev_run_lines = [
        {"cost_usd": 5.0, "input_tokens": 10000, "output_tokens": 1000,
         "cache_read_input_tokens": 8000, "cache_creation_input_tokens": 0,
         "metadata": {"phase": "dev_run"}},
    ]
    dev_run_path.write_text("\n".join(json.dumps(e) for e in dev_run_lines) + "\n")

    output_path = tmp_path / "llm_cost_report.csv"
    result = write_llm_cost_report(
        log_paths_by_phase={"iteration": iteration_path, "dev_run": dev_run_path},
        output_path=output_path,
    )

    assert output_path.is_file()
    by_phase = result.set_index("phase")

    assert by_phase.loc["iteration", "calls"] == 2
    assert by_phase.loc["iteration", "cost_usd"] == pytest.approx(0.03)
    assert by_phase.loc["iteration", "avg_input_tokens"] == pytest.approx(1500)
    assert by_phase.loc["iteration", "avg_output_tokens"] == pytest.approx(150)
    assert by_phase.loc["iteration", "cache_hit_rate"] == pytest.approx(500 / 4500)

    assert by_phase.loc["dev_run", "calls"] == 1
    assert by_phase.loc["dev_run", "cost_usd"] == pytest.approx(5.0)
    assert by_phase.loc["dev_run", "cache_hit_rate"] == pytest.approx(8000 / 18000)

    assert by_phase.loc["total", "calls"] == 3
    assert by_phase.loc["total", "cost_usd"] == pytest.approx(5.03)
    assert by_phase.loc["total", "avg_input_tokens"] == pytest.approx(13000 / 3)
    assert by_phase.loc["total", "cache_hit_rate"] == pytest.approx(8500 / 22500)


def test_write_llm_cost_report_two_phases_share_one_file(tmp_path: Path) -> None:
    """The exact situation Week 7 created: two phases' entries interleaved
    in one physical log file, distinguished only by metadata.phase."""
    shared_path = tmp_path / "shared.jsonl"
    lines = [
        {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 10,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
         "metadata": {"phase": "prompt_iteration"}},
        {"cost_usd": 0.05, "input_tokens": 200, "output_tokens": 20,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
         "metadata": {"phase": "backward_prompt_iteration"}},
    ]
    shared_path.write_text("\n".join(json.dumps(e) for e in lines) + "\n")

    result = write_llm_cost_report(
        log_paths_by_phase={
            "prompt_iteration": shared_path,
            "backward_prompt_iteration": shared_path,
        },
        output_path=tmp_path / "report.csv",
    )
    by_phase = result.set_index("phase")
    assert by_phase.loc["prompt_iteration", "calls"] == 1
    assert by_phase.loc["prompt_iteration", "cost_usd"] == pytest.approx(0.01)
    assert by_phase.loc["backward_prompt_iteration", "calls"] == 1
    assert by_phase.loc["backward_prompt_iteration", "cost_usd"] == pytest.approx(0.05)
    assert by_phase.loc["total", "calls"] == 2


def test_write_llm_cost_report_skips_missing_log_file(tmp_path: Path) -> None:
    dev_run_path = tmp_path / "dev_run.jsonl"
    dev_run_path.write_text(
        json.dumps({"cost_usd": 1.0, "input_tokens": 100, "output_tokens": 10,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                    "metadata": {"phase": "dev_run"}}) + "\n"
    )

    result = write_llm_cost_report(
        log_paths_by_phase={
            "iteration": tmp_path / "does_not_exist.jsonl",
            "dev_run": dev_run_path,
        },
        output_path=tmp_path / "report.csv",
    )

    assert set(result["phase"]) == {"dev_run", "total"}
