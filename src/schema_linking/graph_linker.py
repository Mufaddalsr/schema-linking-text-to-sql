"""Graph schema linker (Method G — SchemaGraphSQL-style).

One deterministic LLM call per question identifies the 1-3 "core" (endpoint)
tables and their projection/filter columns (``GRAPH_ENDPOINT_V1``, see
``schema_linking.utils.prompts``). The tables actually predicted are then
computed by a pure graph algorithm over the schema's foreign-key graph —
shortest path for 2 endpoints, a greedy Steiner approximation for 3 (see
``schema_linking.utils.graph``) — never by the LLM guessing join-bridge
tables itself.

Hallucination policy
---------------------
Unlike Method C (LLM forward), hallucinated table/column names ARE filtered
out here: a core table that doesn't resolve against the real schema can't
be fed into the graph algorithm at all (there's no node for it), and a
column whose table falls off the predicted path is architecturally
meaningless for this method. Both are logged, and counted in
``Prediction.extra`` (``n_endpoints_hallucinated``,
``columns_dropped_off_path``) for error analysis.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from schema_linking.base import Prediction
from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema
from schema_linking.utils.graph import (
    build_schema_graph,
    resolve_endpoint_table,
    shortest_path_tables,
    steiner_subgraph_tables,
)
from schema_linking.utils.llm_client import LLMClient
from schema_linking.utils.prompts import (
    PromptTemplate,
    render_graph_endpoint_user_message,
    render_schema_block,
)

logger = logging.getLogger(__name__)

_DEFAULT_TRACE_PATH: Path = (
    Path(__file__).resolve().parents[2] / "outputs" / "predictions" / "graph_dev_traces.jsonl"
)

_CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _parse_response_text(text: str) -> Any | None:
    """Parse ``text`` as JSON, tolerating a ```json fenced block.

    Returns ``None`` (never raises) on failure.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = _strip_code_fence(text)
    if stripped == text:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _matches_array_spec(value: Any, spec: dict[str, Any]) -> bool:
    if spec.get("type") != "array":
        return True
    if not isinstance(value, list):
        return False
    min_items = spec.get("minItems")
    max_items = spec.get("maxItems")
    if min_items is not None and len(value) < min_items:
        return False
    if max_items is not None and len(value) > max_items:
        return False
    item_spec = spec.get("items", {})
    if item_spec.get("type") == "string":
        return all(isinstance(v, str) for v in value)
    if item_spec.get("type") == "array":
        inner_min = item_spec.get("minItems")
        inner_max = item_spec.get("maxItems")
        inner = item_spec.get("items", {})
        for pair in value:
            if not isinstance(pair, list):
                return False
            if inner_min is not None and len(pair) < inner_min:
                return False
            if inner_max is not None and len(pair) > inner_max:
                return False
            if inner.get("type") == "string" and not all(isinstance(x, str) for x in pair):
                return False
        return True
    return True


def _validate_output_schema(parsed: Any, output_schema: dict[str, Any]) -> bool:
    """Structural check driven by ``output_schema`` (mirrors
    ``llm_linker._validate_output_schema`` but also enforces the outer
    array's own ``minItems``/``maxItems`` — relevant here since
    ``GRAPH_ENDPOINT_V1`` caps ``core_tables`` at 3)."""
    if not isinstance(parsed, dict):
        return False
    required = output_schema.get("required", [])
    if any(key not in parsed for key in required):
        return False
    for key, spec in output_schema.get("properties", {}).items():
        if key in parsed and not _matches_array_spec(parsed[key], spec):
            return False
    return True


def _resolve_column_name(table_original_name: str, column_name: str, schema: Schema) -> str | None:
    """Case-insensitive column lookup within an already-resolved table."""
    target = column_name.strip().lower()
    for table in schema.tables:
        if table.original_name != table_original_name:
            continue
        for column in table.columns:
            if column.original_name.lower() == target:
                return column.original_name
        return None
    return None


class GraphLinker:
    """Graph schema linker (Method G — SchemaGraphSQL-style).

    Parameters
    ----------
    llm_client
        Configured :class:`~schema_linking.utils.llm_client.LLMClient` (or
        :class:`~schema_linking.utils.llm_client.MockLLMClient` in tests).
    prompt
        The :class:`~schema_linking.utils.prompts.PromptTemplate` to render
        (``GRAPH_ENDPOINT_V1``).
    few_shot
        Fixed few-shot examples, each a dict with ``question``,
        ``core_tables``, ``columns``, and a pre-rendered ``schema_block``
        (that example's own db's schema — same convention as
        :class:`~schema_linking.llm_linker.LLMForwardLinker`'s ``few_shot``).
        Pass ``[]`` for zero-shot.
    schemas
        Every Spider schema this linker will be asked to predict against,
        keyed by ``db_id``. Used to precompute and cache each schema's
        foreign-key graph and rendered schema block at construction time —
        neither changes after load.
    k_samples
        Locked at ``1`` — no self-consistency sampling for Method G. Present
        for interface parity with the other LLM linkers; any other value
        raises ``ValueError``.
    temperature_override
        Temperature used for the single deterministic call. Locked at
        ``0.0`` by default.
    trace_path
        Where :meth:`predict_all` writes the mandatory per-query trace
        JSONL (Week 9 error-analysis material). Defaults to
        ``outputs/predictions/graph_dev_traces.jsonl`` under the repo root;
        tests pass a ``tmp_path``.
    extra_metadata
        Extra tags merged into every call's cost-log ``metadata`` (e.g.
        ``{"phase": "graph_dev_run"}``), ahead of the fixed
        ``method``/``qid``/``db_id`` keys — same convention as
        :class:`~schema_linking.llm_linker.LLMForwardLinker`, needed so
        :func:`~schema_linking.run_linker.write_llm_cost_report` can filter
        this linker's calls out of a shared log by phase.
    """

    name: str = "graph"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt: PromptTemplate,
        few_shot: list[dict[str, Any]],
        schemas: dict[str, Schema],
        k_samples: int = 1,
        temperature_override: float = 0.0,
        trace_path: Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if k_samples != 1:
            raise ValueError(
                "GraphLinker.k_samples is locked at 1 (no self-consistency sampling "
                f"for Method G), got {k_samples}"
            )
        self.llm_client = llm_client
        self.prompt = prompt
        self.few_shot = few_shot
        self.k_samples = k_samples
        self.temperature_override = temperature_override
        self.trace_path = trace_path or _DEFAULT_TRACE_PATH
        self.extra_metadata = extra_metadata or {}

        self._graphs = {db_id: build_schema_graph(schema) for db_id, schema in schemas.items()}
        self._schema_blocks = {
            db_id: render_schema_block(schema) for db_id, schema in schemas.items()
        }

    def predict_one(self, example: SpiderExample, schema: Schema) -> Prediction:
        """Link one example: render the prompt, call once at
        ``temperature_override``, resolve endpoints, run the graph
        algorithm, filter columns.

        Returns
        -------
        Prediction
            ``tables`` is the graph result (shortest path / Steiner
            subgraph / single endpoint); ``columns`` is filtered to only
            those whose resolved table is in ``tables``. ``extra`` carries
            ``llm_endpoints_raw``, ``llm_endpoints_resolved``,
            ``n_endpoints_hallucinated``, ``graph_path_or_subgraph``,
            ``columns_dropped_off_path``, ``input_tokens``,
            ``output_tokens``, ``cost_usd``, ``failure`` (``None``,
            ``"parse"``, or ``"no_valid_endpoints"``).
        """
        prediction, _raw_text = self._predict_with_raw_text(example, schema)
        return prediction

    def _predict_with_raw_text(
        self, example: SpiderExample, schema: Schema
    ) -> tuple[Prediction, str]:
        """Same as :meth:`predict_one`, but also returns the raw LLM
        response text — needed by :meth:`predict_all` for the trace file,
        not part of the public ``Prediction`` contract."""
        graph = self._graphs[example.db_id]
        schema_block = self._schema_blocks[example.db_id]
        fewshot_schema_blocks = (
            [ex["schema_block"] for ex in self.few_shot] if self.few_shot else None
        )
        user_message = render_graph_endpoint_user_message(
            self.prompt,
            schema_block,
            example.question,
            fewshot_examples=self.few_shot or None,
            fewshot_schema_blocks=fewshot_schema_blocks,
        )
        messages = [{"role": "user", "content": user_message}]

        original_temperature = self.llm_client.temperature
        self.llm_client.temperature = self.temperature_override
        try:
            response = self.llm_client.call(
                system=self.prompt.system,
                messages=messages,
                cacheable_prefix=schema_block,
                metadata={
                    **self.extra_metadata,
                    "method": "graph",
                    "qid": example.question_id,
                    "db_id": example.db_id,
                },
            )
        finally:
            self.llm_client.temperature = original_temperature

        parsed = _parse_response_text(response.text)
        if parsed is None or not _validate_output_schema(parsed, self.prompt.output_schema):
            logger.warning(
                "graph linker qid %s (db %r): failed to parse/validate LLM response: %r",
                example.question_id, example.db_id, response.text,
            )
            return (
                Prediction(
                    db_id=example.db_id,
                    tables=(),
                    columns=(),
                    extra={
                        "llm_endpoints_raw": [],
                        "llm_endpoints_resolved": [],
                        "n_endpoints_hallucinated": 0,
                        "graph_path_or_subgraph": (),
                        "columns_dropped_off_path": 0,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "failure": "parse",
                    },
                ),
                response.text,
            )

        raw_endpoints = list(parsed["core_tables"])
        raw_columns = [list(pair) for pair in parsed["columns"]]

        resolved_endpoints: list[str] = []
        n_hallucinated = 0
        for name in raw_endpoints:
            resolved = resolve_endpoint_table(name, schema)
            if resolved is None:
                n_hallucinated += 1
                logger.warning(
                    "graph linker qid %s (db %r): hallucinated core table %r",
                    example.question_id, example.db_id, name,
                )
                continue
            if resolved not in resolved_endpoints:
                resolved_endpoints.append(resolved)

        if not resolved_endpoints:
            logger.warning(
                "graph linker qid %s (db %r): all core tables hallucinated: %r",
                example.question_id, example.db_id, raw_endpoints,
            )
            return (
                Prediction(
                    db_id=example.db_id,
                    tables=(),
                    columns=(),
                    extra={
                        "llm_endpoints_raw": raw_endpoints,
                        "llm_endpoints_resolved": [],
                        "n_endpoints_hallucinated": n_hallucinated,
                        "graph_path_or_subgraph": (),
                        "columns_dropped_off_path": 0,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "failure": "no_valid_endpoints",
                    },
                ),
                response.text,
            )

        if len(resolved_endpoints) == 1:
            predicted_tables: tuple[str, ...] = (resolved_endpoints[0],)
        elif len(resolved_endpoints) == 2:
            a, b = resolved_endpoints
            path = shortest_path_tables(graph, a, b)
            predicted_tables = path if path is not None else (a, b)
        else:
            predicted_tables = steiner_subgraph_tables(graph, resolved_endpoints)

        predicted_table_set = set(predicted_tables)
        resolved_columns: list[tuple[str, str]] = []
        columns_dropped = 0
        for table_name, column_name in raw_columns:
            resolved_table = resolve_endpoint_table(table_name, schema)
            if resolved_table is None:
                columns_dropped += 1
                logger.warning(
                    "graph linker qid %s (db %r): column references hallucinated table %r",
                    example.question_id, example.db_id, table_name,
                )
                continue
            resolved_column = _resolve_column_name(resolved_table, column_name, schema)
            if resolved_column is None:
                columns_dropped += 1
                logger.warning(
                    "graph linker qid %s (db %r): hallucinated column %s.%s",
                    example.question_id, example.db_id, resolved_table, column_name,
                )
                continue
            if resolved_table not in predicted_table_set:
                columns_dropped += 1
                logger.warning(
                    "graph linker qid %s (db %r): dropping column %s.%s — table off predicted path",
                    example.question_id, example.db_id, resolved_table, resolved_column,
                )
                continue
            pair = (resolved_table, resolved_column)
            if pair not in resolved_columns:
                resolved_columns.append(pair)

        prediction = Prediction(
            db_id=example.db_id,
            tables=tuple(predicted_tables),
            columns=tuple(resolved_columns),
            extra={
                "llm_endpoints_raw": raw_endpoints,
                "llm_endpoints_resolved": resolved_endpoints,
                "n_endpoints_hallucinated": n_hallucinated,
                "graph_path_or_subgraph": tuple(predicted_tables),
                "columns_dropped_off_path": columns_dropped,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "failure": None,
            },
        )
        return prediction, response.text

    def predict_all(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, Prediction]:
        """Link a batch, grouped by ``db_id`` to maximise prompt-cache hits
        (same pattern as :class:`~schema_linking.llm_linker.LLMForwardLinker`).

        Side effect
        -----------
        Overwrites :attr:`trace_path` with one JSONL line per query —
        ``{qid, db_id, question, llm_raw, endpoints_resolved, graph_result,
        final_tables, final_columns, failure}`` — the material for Week 9
        error analysis. Never skipped.
        """
        groups: dict[str, list[SpiderExample]] = {}
        for ex in examples:
            groups.setdefault(ex.db_id, []).append(ex)

        predictions: dict[int, Prediction] = {}
        trace_lines: list[dict[str, Any]] = []
        for db_id, group in groups.items():
            schema = schemas[db_id]
            for ex in group:
                prediction, raw_text = self._predict_with_raw_text(ex, schema)
                predictions[ex.question_id] = prediction
                trace_lines.append(
                    {
                        "qid": ex.question_id,
                        "db_id": ex.db_id,
                        "question": ex.question,
                        "llm_raw": raw_text,
                        "endpoints_resolved": list(prediction.extra["llm_endpoints_resolved"]),
                        "graph_result": list(prediction.extra["graph_path_or_subgraph"]),
                        "final_tables": list(prediction.tables),
                        "final_columns": [list(c) for c in prediction.columns],
                        "failure": prediction.extra["failure"],
                    }
                )

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("w", encoding="utf-8") as f:
            for line in trace_lines:
                f.write(json.dumps(line) + "\n")

        return predictions
