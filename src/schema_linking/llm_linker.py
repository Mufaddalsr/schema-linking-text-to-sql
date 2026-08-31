"""LLM forward schema linker (Method C).

Prompts an LLM with the full rendered schema and question, samples
``k_samples`` times at the configured temperature (self-consistency), and
aggregates the parsed per-sample predictions by vote threshold. See
:mod:`schema_linking.utils.prompts` and :mod:`schema_linking.utils.fewshot`
for the prompt and few-shot design.

Hallucination policy
---------------------
Predicted tables/columns are never filtered against the real schema here —
a sample naming a nonexistent table is passed straight through to the
returned :class:`~schema_linking.base.Prediction`. The Week 3 evaluator
computes ``hallucination_rate`` from exactly this raw, unfiltered signal;
silently dropping hallucinations in this module would make that metric
meaningless.

Aggregation denominator
------------------------
Vote thresholds (``union``/``majority``/``intersection``) are always taken
over the fixed ``k_samples``, never over how many samples happened to parse
successfully. A sample that fails to parse or fails output-schema
validation contributes an empty vote (i.e. it can only ever hurt
``majority``/``intersection`` recall, never inflate them) — it does not
shrink the required-vote bar.
"""

from __future__ import annotations

import json
import logging
import math
import re
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from schema_linking.base import Prediction
from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema
from schema_linking.utils.llm_client import LLMClient
from schema_linking.utils.prompts import (
    PromptTemplate,
    render_backward_user_message,
    render_schema_block,
    render_user_message,
)
from schema_linking.utils.sql_parsing import ParseIssue, extract_schema_references

logger = logging.getLogger(__name__)

_DEFAULT_PARSE_FAILURE_LOG_PATH: Path = (
    Path(__file__).resolve().parents[2] / "outputs" / "logs" / "llm_parse_failures.jsonl"
)
_DEFAULT_BACKWARD_PARSE_FAILURE_LOG_PATH: Path = (
    Path(__file__).resolve().parents[2] / "outputs" / "logs" / "llm_sql_parse_failures.jsonl"
)
_DEFAULT_BACKWARD_SQL_OUTPUT_PATH: Path = (
    Path(__file__).resolve().parents[2] / "outputs" / "predictions" / "llm_backward_dev_sql.jsonl"
)

_CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n?(.*?)\n?```$", re.DOTALL)

Aggregation = Literal["union", "majority", "intersection"]
_VALID_AGGREGATIONS: tuple[Aggregation, ...] = ("union", "majority", "intersection")


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _parse_response_text(text: str) -> Any | None:
    """Parse ``text`` as JSON, tolerating a ```json fenced block.

    Returns ``None`` (never raises) on failure — the caller treats that as
    an empty-prediction sample and logs it.
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
    item_spec = spec.get("items", {})
    if item_spec.get("type") == "string":
        return all(isinstance(v, str) for v in value)
    if item_spec.get("type") == "array":
        min_items = item_spec.get("minItems")
        max_items = item_spec.get("maxItems")
        inner = item_spec.get("items", {})
        for pair in value:
            if not isinstance(pair, list):
                return False
            if min_items is not None and len(pair) < min_items:
                return False
            if max_items is not None and len(pair) > max_items:
                return False
            if inner.get("type") == "string" and not all(isinstance(x, str) for x in pair):
                return False
        return True
    return True


def _validate_output_schema(parsed: Any, output_schema: dict[str, Any]) -> bool:
    """Minimal structural check driven by ``output_schema`` (not a full JSON
    Schema validator — this project's prompt output shape is simple and
    fixed: an object with array-of-string / array-of-pair properties)."""
    if not isinstance(parsed, dict):
        return False
    required = output_schema.get("required", [])
    if any(key not in parsed for key in required):
        return False
    for key, spec in output_schema.get("properties", {}).items():
        if key in parsed and not _matches_array_spec(parsed[key], spec):
            return False
    return True


def _log_failure(
    log_path: Path,
    question_id: int,
    db_id: str,
    sample_index: int,
    reason: str,
    raw_text: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": question_id,
                    "db_id": db_id,
                    "sample_index": sample_index,
                    "reason": reason,
                    "raw_text": raw_text,
                }
            )
            + "\n"
        )


def _issue_to_dict(issue: ParseIssue) -> dict[str, str]:
    return {"kind": issue.kind, "detail": issue.detail}


def _aggregate(
    sample_predictions: list[dict[str, list]], k_samples: int, aggregation: Aggregation
) -> tuple[list[str], list[list[str]]]:
    if aggregation == "union":
        threshold = 1
    elif aggregation == "majority":
        threshold = math.ceil(k_samples / 2)
    elif aggregation == "intersection":
        threshold = k_samples
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")

    table_votes: dict[str, int] = {}
    column_votes: dict[tuple[str, str], int] = {}
    for sample in sample_predictions:
        for table in dict.fromkeys(sample["tables"]):
            table_votes[table] = table_votes.get(table, 0) + 1
        for pair in dict.fromkeys(tuple(p) for p in sample["columns"]):
            column_votes[pair] = column_votes.get(pair, 0) + 1

    tables = [t for t, count in table_votes.items() if count >= threshold]
    columns = [list(pair) for pair, count in column_votes.items() if count >= threshold]
    return tables, columns


class LLMForwardLinker:
    """LLM forward schema linker (Method C) — self-consistency over ``k_samples``.

    Parameters
    ----------
    llm_client
        Configured :class:`~schema_linking.utils.llm_client.LLMClient` (or
        :class:`~schema_linking.utils.llm_client.MockLLMClient` in tests).
    prompt
        The :class:`~schema_linking.utils.prompts.PromptTemplate` to render
        (e.g. ``FORWARD_V1``).
    few_shot
        Fixed few-shot examples, each a dict with ``question``, ``tables``,
        ``columns``, and a pre-rendered ``schema_block`` (that example's
        *own* db's schema, rendered once via
        :func:`~schema_linking.utils.prompts.render_schema_block` — this
        class never sees the few-shot examples' schemas itself). Pass ``[]``
        for zero-shot.
    k_samples
        Number of self-consistency samples per question.
    aggregation
        Vote-threshold rule applied over the fixed ``k_samples`` denominator
        — see the module docstring.
    parse_failure_log_path
        Where unparseable/invalid samples are logged. Defaults to
        ``outputs/logs/llm_parse_failures.jsonl`` under the repo root; tests
        pass a ``tmp_path`` to avoid writing into the real log.
    extra_metadata
        Extra tags merged into every call's cost-log ``metadata`` (e.g.
        ``{"phase": "prompt_iteration"}``), ahead of the fixed
        ``method``/``qid``/``db_id``/``sample_index`` keys — for
        traceability when the same log file accumulates calls from more
        than one run purpose.
    """

    name: str = "llm_forward"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt: PromptTemplate,
        few_shot: list[dict[str, Any]],
        k_samples: int = 3,
        aggregation: Aggregation = "union",
        parse_failure_log_path: Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, got {aggregation!r}"
            )
        self.llm_client = llm_client
        self.prompt = prompt
        self.few_shot = few_shot
        self.k_samples = k_samples
        self.aggregation: Aggregation = aggregation
        self.parse_failure_log_path = parse_failure_log_path or _DEFAULT_PARSE_FAILURE_LOG_PATH
        self.extra_metadata = extra_metadata or {}

    def predict_one(self, example: SpiderExample, schema: Schema) -> Prediction:
        """Link one example: render the prompt, sample ``k_samples`` times, aggregate.

        Returns
        -------
        Prediction
            ``tables``/``columns`` are the vote-aggregated raw predictions
            (never filtered against ``schema`` — see module docstring).
            ``extra`` carries per-sample diagnostics: ``n_samples_parsed``,
            ``n_samples_valid``, ``sample_predictions`` (one ``{tables,
            columns}`` dict per sample, empty for failed samples),
            ``total_input_tokens``, ``total_output_tokens``,
            ``total_cost_usd``.
        """
        schema_block = render_schema_block(schema)
        fewshot_schema_blocks = (
            [ex["schema_block"] for ex in self.few_shot] if self.few_shot else None
        )
        user_message = render_user_message(
            self.prompt,
            schema_block,
            example.question,
            fewshot_examples=self.few_shot or None,
            fewshot_schema_blocks=fewshot_schema_blocks,
        )
        messages = [{"role": "user", "content": user_message}]

        sample_predictions: list[dict[str, list]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = 0.0
        n_samples_parsed = 0
        n_samples_valid = 0

        for i in range(self.k_samples):
            response = self.llm_client.call(
                system=self.prompt.system,
                messages=messages,
                cacheable_prefix=schema_block,
                metadata={
                    **self.extra_metadata,
                    "method": "llm_forward",
                    "qid": example.question_id,
                    "db_id": example.db_id,
                    "sample_index": i,
                },
            )
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_cost_usd += response.cost_usd

            parsed = _parse_response_text(response.text)
            if parsed is None:
                _log_failure(
                    self.parse_failure_log_path, example.question_id, example.db_id,
                    i, "parse_error", response.text,
                )
                sample_predictions.append({"tables": [], "columns": []})
                continue
            n_samples_parsed += 1

            if not _validate_output_schema(parsed, self.prompt.output_schema):
                _log_failure(
                    self.parse_failure_log_path, example.question_id, example.db_id,
                    i, "schema_invalid", response.text,
                )
                sample_predictions.append({"tables": [], "columns": []})
                continue
            n_samples_valid += 1
            sample_predictions.append(
                {"tables": list(parsed["tables"]), "columns": [list(p) for p in parsed["columns"]]}
            )

        tables, columns = _aggregate(sample_predictions, self.k_samples, self.aggregation)

        return Prediction(
            db_id=example.db_id,
            tables=tuple(tables),
            columns=tuple(tuple(pair) for pair in columns),
            extra={
                "n_samples_parsed": n_samples_parsed,
                "n_samples_valid": n_samples_valid,
                "sample_predictions": sample_predictions,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_usd": total_cost_usd,
            },
        )

    def predict_all(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, Prediction]:
        """Link a batch, grouped by ``db_id`` to maximise prompt-cache hits.

        Examples are grouped by ``db_id`` in first-seen order (a plain dict
        preserves insertion order) before any calls are made, so every
        ``llm_client.call()`` for a given database happens back-to-back —
        consecutive calls reuse the same ``cacheable_prefix`` (the schema
        block), which is what makes Anthropic's ephemeral prompt cache hit.
        """
        groups: dict[str, list[SpiderExample]] = {}
        for ex in examples:
            groups.setdefault(ex.db_id, []).append(ex)

        predictions: dict[int, Prediction] = {}
        for db_id, group in groups.items():
            schema = schemas[db_id]
            for ex in group:
                predictions[ex.question_id] = self.predict_one(ex, schema)
        return predictions


class LLMBackwardLinker:
    """LLM backward schema linker (Method D) — question -> SQL -> schema references.

    Prompts an LLM to generate a single SQL query for the question (no
    self-consistency — one deterministic call per question), then resolves
    the generated SQL's table/column references against the real schema via
    :func:`schema_linking.utils.sql_parsing.extract_schema_references` with
    ``strict=False`` — a hallucinated table/column is a valid Method D
    prediction (see that module's docstring), not something to filter out.

    Parameters
    ----------
    llm_client
        Configured :class:`~schema_linking.utils.llm_client.LLMClient` (or
        :class:`~schema_linking.utils.llm_client.MockLLMClient` in tests).
    prompt
        The :class:`~schema_linking.utils.prompts.PromptTemplate` to render
        (``BACKWARD_V1``).
    few_shot
        Fixed few-shot examples, each a dict with ``question``, ``gold_sql``,
        and a pre-rendered ``schema_block`` (that example's own db's schema
        — same convention as :class:`LLMForwardLinker`'s ``few_shot``). Pass
        ``[]`` for zero-shot.
    k_samples
        Locked at ``1``. Method D takes a single deterministic call, no
        self-consistency sampling. Present for
        interface parity with :class:`LLMForwardLinker`; any other value
        raises ``ValueError``.
    temperature_override
        Locked at ``0.0``. When not ``None``,
        ``llm_client.temperature`` is temporarily swapped to this value for
        the duration of the call (and restored afterwards) — so a shared
        client configured for Method C's self-consistency temperature can
        still be reused for Method D's deterministic calls. Pass ``None``
        to use the client's own configured temperature unchanged.
    parse_failure_log_path
        Where SQL that fails to parse at all is logged. Defaults to
        ``outputs/logs/llm_sql_parse_failures.jsonl`` under the repo root;
        tests pass a ``tmp_path``.
    sql_output_path
        Where :meth:`predict_all` writes the per-query raw-SQL JSONL dump
        (material for Week 9 error analysis). Never skipped. Defaults to
        ``outputs/predictions/llm_backward_dev_sql.jsonl`` under the repo
        root; tests pass a ``tmp_path``.
    extra_metadata
        Extra tags merged into every call's cost-log ``metadata`` (e.g.
        ``{"phase": "backward_prompt_iteration"}``), ahead of the fixed
        ``method``/``qid``/``db_id`` keys — for traceability when the same
        log file accumulates calls from more than one run purpose.
    """

    name: str = "llm_backward"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt: PromptTemplate,
        few_shot: list[dict[str, Any]],
        k_samples: int = 1,
        temperature_override: float | None = 0.0,
        parse_failure_log_path: Path | None = None,
        sql_output_path: Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if k_samples != 1:
            raise ValueError(
                "LLMBackwardLinker.k_samples is locked at 1 (no self-consistency "
                f"sampling for Method D), got {k_samples}"
            )
        self.llm_client = llm_client
        self.prompt = prompt
        self.few_shot = few_shot
        self.k_samples = k_samples
        self.temperature_override = temperature_override
        self.parse_failure_log_path = (
            parse_failure_log_path or _DEFAULT_BACKWARD_PARSE_FAILURE_LOG_PATH
        )
        self.sql_output_path = sql_output_path or _DEFAULT_BACKWARD_SQL_OUTPUT_PATH
        self.extra_metadata = extra_metadata or {}

    def predict_one(self, example: SpiderExample, schema: Schema) -> Prediction:
        """Link one example: render the prompt, call once, resolve the SQL.

        Returns
        -------
        Prediction
            ``tables``/``columns`` from :func:`extract_schema_references`
            (``strict=False`` — never filtered against ``schema``). Empty
            for both if the generated SQL fails to parse at all. ``extra``
            carries ``raw_sql`` (fence-stripped), ``parse_issues`` (list of
            ``{"kind": ..., "detail": ...}`` dicts), ``input_tokens``,
            ``output_tokens``, ``cost_usd``.
        """
        schema_block = render_schema_block(schema)
        fewshot_schema_blocks = (
            [ex["schema_block"] for ex in self.few_shot] if self.few_shot else None
        )
        user_message = render_backward_user_message(
            self.prompt,
            schema_block,
            example.question,
            fewshot_examples=self.few_shot or None,
            fewshot_schema_blocks=fewshot_schema_blocks,
        )
        messages = [{"role": "user", "content": user_message}]

        original_temperature = self.llm_client.temperature
        if self.temperature_override is not None:
            self.llm_client.temperature = self.temperature_override
        try:
            response = self.llm_client.call(
                system=self.prompt.system,
                messages=messages,
                cacheable_prefix=schema_block,
                metadata={
                    **self.extra_metadata,
                    "method": "llm_backward",
                    "qid": example.question_id,
                    "db_id": example.db_id,
                },
            )
        finally:
            self.llm_client.temperature = original_temperature

        raw_sql = _strip_code_fence(response.text.strip())
        refs, issues = extract_schema_references(raw_sql, schema, strict=False)
        issue_dicts = [_issue_to_dict(i) for i in issues]

        if any(i.kind == "parse_error" for i in issues):
            _log_failure(
                self.parse_failure_log_path, example.question_id, example.db_id,
                0, "parse_error", raw_sql,
            )
            tables: tuple[str, ...] = ()
            columns: tuple[tuple[str, str], ...] = ()
        else:
            tables, columns = refs.tables, refs.columns

        return Prediction(
            db_id=example.db_id,
            tables=tables,
            columns=columns,
            extra={
                "raw_sql": raw_sql,
                "parse_issues": issue_dicts,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )

    def predict_all(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, Prediction]:
        """Link a batch, grouped by ``db_id`` to maximise prompt-cache hits
        (same pattern as :meth:`LLMForwardLinker.predict_all`).

        Per-example isolation
        ----------------------
        Any exception raised while linking one example (an LLM call that
        exhausts its retries, an unanticipated SQL construct the shared
        walker doesn't handle, ...) is caught, logged to
        :attr:`parse_failure_log_path` with ``reason="unexpected_error"``
        (full traceback included), and recorded as an empty
        :class:`~schema_linking.base.Prediction` for that query — it does
        **not** abort the batch. A batch here is a real, paid, slow run
        over up to 1034 dev examples; one bad query must never destroy the
        other ~1000 already-completed results. Mirrors the same
        per-example isolation :func:`schema_linking.gold_link_extractor.
        _extract_all` already applies to a batch of ``ExtractionError``\\ s.

        Side effect
        -----------
        Overwrites :attr:`sql_output_path` with one JSONL line per query —
        ``{question_id, db_id, question, raw_sql, parse_issues}`` — the
        material for Week 9 error analysis. Never skipped.
        """
        groups: dict[str, list[SpiderExample]] = {}
        for ex in examples:
            groups.setdefault(ex.db_id, []).append(ex)

        predictions: dict[int, Prediction] = {}
        dump_lines: list[dict[str, Any]] = []
        for db_id, group in groups.items():
            schema = schemas[db_id]
            for ex in group:
                try:
                    pred = self.predict_one(ex, schema)
                except Exception as exc:  # noqa: BLE001 - isolate one bad query from the whole paid batch, see docstring
                    _log_failure(
                        self.parse_failure_log_path, ex.question_id, ex.db_id,
                        0, "unexpected_error",
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                    pred = Prediction(
                        db_id=ex.db_id,
                        tables=(),
                        columns=(),
                        extra={
                            "raw_sql": "",
                            "parse_issues": [
                                {
                                    "kind": "other",
                                    "detail": f"unexpected_error: {type(exc).__name__}: {exc}",
                                }
                            ],
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cost_usd": 0.0,
                        },
                    )
                predictions[ex.question_id] = pred
                dump_lines.append(
                    {
                        "question_id": ex.question_id,
                        "db_id": ex.db_id,
                        "question": ex.question,
                        "raw_sql": pred.extra["raw_sql"],
                        "parse_issues": pred.extra["parse_issues"],
                    }
                )

        self.sql_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.sql_output_path.open("w", encoding="utf-8") as f:
            for line in dump_lines:
                f.write(json.dumps(line) + "\n")

        return predictions


class BidirectionalLinker:
    """Bidirectional schema linker (Method E) — union of Methods C and D.

    Makes no LLM calls of its own: it unions the already-saved canonical
    predictions from :class:`LLMForwardLinker` and :class:`LLMBackwardLinker`
    (the same ``{qid: {db_id, tables, columns}}`` JSON shape written by
    ``run_linker.py``, keyed by stringified ``question_id``).

    Parameters
    ----------
    forward_predictions_path
        Path to Method C's canonical predictions JSON.
    backward_predictions_path
        Path to Method D's canonical predictions JSON.
    """

    name: str = "llm_bidirectional"

    def __init__(self, forward_predictions_path: Path, backward_predictions_path: Path) -> None:
        self.forward_predictions: dict[str, Any] = json.loads(
            Path(forward_predictions_path).read_text(encoding="utf-8")
        )
        self.backward_predictions: dict[str, Any] = json.loads(
            Path(backward_predictions_path).read_text(encoding="utf-8")
        )

    def predict_one(self, example: SpiderExample, schema: Schema) -> Prediction:
        """Union the forward and backward predictions for one question.

        Falls back to a single source (with a logged warning) if the other
        is missing for this ``question_id``; returns an empty
        :class:`Prediction` (with a logged warning) if both are missing.
        """
        qid_key = str(example.question_id)
        fwd = self.forward_predictions.get(qid_key)
        bwd = self.backward_predictions.get(qid_key)

        if fwd is None and bwd is None:
            logger.warning(
                "no forward or backward prediction for qid %s (db %r) — "
                "returning empty prediction", qid_key, example.db_id,
            )
        elif fwd is None:
            logger.warning(
                "no forward prediction for qid %s (db %r) — using backward-only",
                qid_key, example.db_id,
            )
        elif bwd is None:
            logger.warning(
                "no backward prediction for qid %s (db %r) — using forward-only",
                qid_key, example.db_id,
            )

        fwd_tables = set(fwd["tables"]) if fwd else set()
        bwd_tables = set(bwd["tables"]) if bwd else set()
        fwd_columns = {tuple(c) for c in fwd["columns"]} if fwd else set()
        bwd_columns = {tuple(c) for c in bwd["columns"]} if bwd else set()

        return Prediction(
            db_id=example.db_id,
            tables=tuple(sorted(fwd_tables | bwd_tables)),
            columns=tuple(sorted(fwd_columns | bwd_columns)),
            extra={
                "source": "bidirectional",
                "n_tables_forward_only": len(fwd_tables - bwd_tables),
                "n_tables_backward_only": len(bwd_tables - fwd_tables),
                "n_tables_both": len(fwd_tables & bwd_tables),
                "n_columns_forward_only": len(fwd_columns - bwd_columns),
                "n_columns_backward_only": len(bwd_columns - fwd_columns),
                "n_columns_both": len(fwd_columns & bwd_columns),
            },
        )

    def predict_all(
        self,
        examples: Sequence[SpiderExample],
        schemas: Mapping[str, Schema],
    ) -> dict[int, Prediction]:
        """Straightforward loop — no batching benefit since there are no LLM calls."""
        return {
            ex.question_id: self.predict_one(ex, schemas[ex.db_id])
            for ex in examples
        }
