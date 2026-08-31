"""Versioned LLM prompt templates for schema linking (Method C — LLM forward).

Rendering conventions
----------------------
``render_schema_block`` renders a :class:`~schema_linking.schema_parser.Schema`
as CREATE TABLE statements using DB-canonical (``original_name``) identifiers
— the form that appears in real SQL and in Spider's gold-link JSON — with
inline FOREIGN KEY constraints. The synthetic ``*`` column is never rendered;
it's already excluded from ``Schema`` (see ``schema_parser`` module
docstring).

Rule 3 and tiering
-------------------
The system prompt's rule against join-bridge-only columns targets **Tier 1**
gold. Tier 2 (Tier 1 + JOIN tables/columns)
is derived from Tier-1-style predictions at gold-comparison time by the
evaluator, not by asking the model to overpredict joins it can't distinguish
from genuinely-selected columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from schema_linking.schema_parser import FKPair, Schema, Table

_SQL_TYPE_MAP: dict[str, str] = {
    "text": "TEXT",
    "number": "INT",
    "time": "DATE",
    "boolean": "BOOLEAN",
    "others": "TEXT",
}


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A versioned prompt template.

    Attributes
    ----------
    version
        Short version tag, e.g. ``"forward_v1"``. Logged alongside LLM calls
        (see ``LLMClient`` call metadata) so results are traceable to the
        exact template that produced them.
    system
        System prompt text.
    user_template
        ``str.format``-style template with ``{schema_block}`` and
        ``{question}`` placeholders, used when there are no few-shot
        examples. See :func:`render_user_message` for the few-shot path.
    output_schema
        JSON Schema (as a plain dict) describing the expected model output.
        Stored for the calling linker to validate against; not used here.
        ``None`` for templates whose output isn't JSON (e.g.
        :data:`BACKWARD_V1`, which emits a raw SQL string).
    """

    version: str
    system: str
    user_template: str
    output_schema: dict[str, Any] | None = None


_FORWARD_V1_SYSTEM = """You extract the tables and columns from a database schema that are relevant to answering a natural-language question.

Rules:
1. Use ONLY tables and columns that appear in the provided schema.
2. Return every table you need. Return every column you need.
3. Do NOT return columns whose ONLY purpose is joining tables together — return only columns that would appear in SELECT, WHERE, GROUP BY, HAVING, or ORDER BY of the target SQL.
4. Output ONLY a JSON object with keys "tables" (list of strings) and "columns" (list of [table_name, column_name] pairs). No prose, no explanation, no markdown fences."""

_FORWARD_V1_USER_TEMPLATE = """Schema:
{schema_block}

Question: {question}

Return the relevant tables and columns as JSON."""

_FORWARD_V1_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tables": {"type": "array", "items": {"type": "string"}},
        "columns": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
        },
    },
    "required": ["tables", "columns"],
}

FORWARD_V1 = PromptTemplate(
    version="forward_v1",
    system=_FORWARD_V1_SYSTEM,
    user_template=_FORWARD_V1_USER_TEMPLATE,
    output_schema=_FORWARD_V1_OUTPUT_SCHEMA,
)

# forward_v2 — see notebooks/06_prompt_iteration.ipynb
# ("Prompt iteration: forward_v1 vs forward_v2") for the failure analysis that
# motivated this change. Only rule 3 differs from v1: it generalises from
# "columns whose only purpose is joining tables together" to also cover whole
# TABLES, and adds the "topically related but not actually referenced by the
# SQL" case — 4/5 of the worst forward_v1 failures on the 20-example
# iteration set were the model adding a table/column that was schema-real
# and topically plausible but never referenced in the gold SQL at all (not
# even via JOIN). Rules 1, 2, 4 and the user template/output schema are
# unchanged from v1.
_FORWARD_V2_SYSTEM = """You extract the tables and columns from a database schema that are relevant to answering a natural-language question.

Rules:
1. Use ONLY tables and columns that appear in the provided schema.
2. Return every table you need. Return every column you need.
3. A table or column must correspond to something the target SQL actually references (in SELECT, WHERE, GROUP BY, HAVING, ORDER BY, or as the FROM/JOIN table holding one of those referenced columns). Do NOT include a table only because it is topically related to the question, and do NOT include a column whose only purpose is joining tables together.
4. Output ONLY a JSON object with keys "tables" (list of strings) and "columns" (list of [table_name, column_name] pairs). No prose, no explanation, no markdown fences."""

FORWARD_V2 = PromptTemplate(
    version="forward_v2",
    system=_FORWARD_V2_SYSTEM,
    user_template=_FORWARD_V1_USER_TEMPLATE,
    output_schema=_FORWARD_V1_OUTPUT_SCHEMA,
)

_BACKWARD_V1_SYSTEM = """You are a SQL expert. Given a database schema and a natural-language question, generate a single SQL query that would answer the question.

Rules:
1. Use only tables and columns that appear in the provided schema.
2. Return only the SQL query. No explanation, no markdown fences, no prose.
3. Do not add comments inside the SQL.
4. If the question is ambiguous, choose the most literal interpretation."""

_BACKWARD_V1_USER_TEMPLATE = """Schema:
{schema_block}

Question: {question}

SQL:"""

BACKWARD_V1 = PromptTemplate(
    version="backward_v1",
    system=_BACKWARD_V1_SYSTEM,
    user_template=_BACKWARD_V1_USER_TEMPLATE,
    output_schema=None,
)

_GRAPH_ENDPOINT_V1_SYSTEM = """You identify the 1 to 3 core tables and the specific columns needed from a database schema to answer a natural-language question.

Rules:
1. Return between 1 and 3 core tables. Prefer fewer.
2. Do NOT include tables that only exist to join others — those will be added automatically by a graph algorithm.
3. For columns, return only those that would appear in SELECT, WHERE, GROUP BY, HAVING, or ORDER BY of the target SQL. Not join columns.
4. Every table and column must be from the provided schema.
5. Output ONLY a JSON object with keys "core_tables" (list of 1-3 strings) and "columns" (list of [table_name, column_name] pairs)."""

_GRAPH_ENDPOINT_V1_USER_TEMPLATE = """Schema:
{schema_block}

Question: {question}

Return the core tables and needed columns as JSON."""

_GRAPH_ENDPOINT_V1_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "core_tables": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "columns": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
        },
    },
    "required": ["core_tables", "columns"],
}

GRAPH_ENDPOINT_V1 = PromptTemplate(
    version="graph_endpoint_v1",
    system=_GRAPH_ENDPOINT_V1_SYSTEM,
    user_template=_GRAPH_ENDPOINT_V1_USER_TEMPLATE,
    output_schema=_GRAPH_ENDPOINT_V1_OUTPUT_SCHEMA,
)

# graph_endpoint_v2 — see notebooks/08a_graph_prompt_iteration.ipynb for the
# failure analysis that motivated this change. Rules 1 and 2 differ from v1;
# rules 3, 4, 5 and the user template/output schema are unchanged. Motivated
# by the "Wrong endpoint" pattern dominating v1's 20-train-example worst
# cases (6/11 imperfect examples): the model repeatedly named either (a) a
# topically-plausible-but-unreferenced second table as a core table (same
# failure forward_v1/v2 already documented for Method C — see
# notebooks/06_prompt_iteration.ipynb), or (b) a genuine join-bridge table as
# a core table, directly contradicting v1's existing rule 2. Rule 1 gets an
# explicit "must be needed by the SQL" criterion (generalising forward_v2's
# fix from columns to tables); rule 2 gets a concrete, testable definition of
# "join-only" instead of relying on the model to infer it.
_GRAPH_ENDPOINT_V2_SYSTEM = """You identify the 1 to 3 core tables and the specific columns needed from a database schema to answer a natural-language question.

Rules:
1. Return between 1 and 3 core tables — tables the target SQL actually needs to compute the answer. Do NOT include a table just because it is topically related to the question if no column of it would ever appear in the SQL. Prefer fewer.
2. Do NOT include tables that only exist to join others — those will be added automatically by a graph algorithm. A table is join-only if none of its own columns would appear in SELECT, WHERE, GROUP BY, HAVING, or ORDER BY, and it is used only via its foreign keys to connect two other tables. If you are unsure whether a table contributes an output column, leave it out — the graph algorithm will add it automatically if it lies on the path connecting your core tables.
3. For columns, return only those that would appear in SELECT, WHERE, GROUP BY, HAVING, or ORDER BY of the target SQL. Not join columns.
4. Every table and column must be from the provided schema.
5. Output ONLY a JSON object with keys "core_tables" (list of 1-3 strings) and "columns" (list of [table_name, column_name] pairs)."""

GRAPH_ENDPOINT_V2 = PromptTemplate(
    version="graph_endpoint_v2",
    system=_GRAPH_ENDPOINT_V2_SYSTEM,
    user_template=_GRAPH_ENDPOINT_V1_USER_TEMPLATE,
    output_schema=_GRAPH_ENDPOINT_V1_OUTPUT_SCHEMA,
)

PROMPTS: dict[str, PromptTemplate] = {
    FORWARD_V1.version: FORWARD_V1,
    FORWARD_V2.version: FORWARD_V2,
    BACKWARD_V1.version: BACKWARD_V1,
    GRAPH_ENDPOINT_V1.version: GRAPH_ENDPOINT_V1,
    GRAPH_ENDPOINT_V2.version: GRAPH_ENDPOINT_V2,
}


def get_prompt_template(version: str) -> PromptTemplate:
    """Look up a template by version tag.

    Raises
    ------
    KeyError
        If ``version`` is not registered in :data:`PROMPTS`.
    """
    try:
        return PROMPTS[version]
    except KeyError as exc:
        raise KeyError(
            f"no registered prompt template {version!r}; have {sorted(PROMPTS)}"
        ) from exc


def _global_column_index_map(schema: Schema) -> dict[int, tuple[str, str]]:
    """Map Spider's raw global column index back to ``(table, column)``
    original names, for resolving :class:`FKPair` indices.

    Relies on Spider's ``tables.json`` listing columns in per-table
    contiguous blocks in ascending table order — the same assumption
    ``schema_parser`` relies on when building ``Table.columns`` in source
    order. Index 0 (the synthetic ``*`` column) is never in this map, since
    ``Schema`` already excludes it.
    """
    mapping: dict[int, tuple[str, str]] = {}
    idx = 1
    for table in schema.tables:
        for column in table.columns:
            mapping[idx] = (table.original_name, column.original_name)
            idx += 1
    return mapping


def _foreign_key_lines(
    table: Table, foreign_keys: list[FKPair], fk_map: dict[int, tuple[str, str]]
) -> list[str]:
    lines: list[str] = []
    for fk in foreign_keys:
        from_ref = fk_map.get(fk.from_col_idx)
        to_ref = fk_map.get(fk.to_col_idx)
        if from_ref is None or to_ref is None or from_ref[0] != table.original_name:
            continue
        _, from_col = from_ref
        to_table, to_col = to_ref
        lines.append(f"  FOREIGN KEY ({from_col}) REFERENCES {to_table}({to_col})")
    return lines


def render_schema_block(schema: Schema) -> str:
    """Render ``schema`` as CREATE TABLE statements with inline FK constraints.

    Uses DB-canonical (``original_name``) identifiers throughout. Column
    types are mapped through :data:`_SQL_TYPE_MAP` (Spider's ``text`` /
    ``number`` / ``time`` / ``boolean`` / ``others`` -> SQL-ish types);
    unrecognised types fall back to ``TEXT``.
    """
    fk_map = _global_column_index_map(schema)
    tables_sql: list[str] = []
    for table in schema.tables:
        col_lines = [
            f"  {column.original_name} {_SQL_TYPE_MAP.get(column.type, 'TEXT')}"
            + (" PRIMARY KEY" if column.is_primary_key else "")
            for column in table.columns
        ]
        col_lines.extend(_foreign_key_lines(table, schema.foreign_keys, fk_map))
        body = ",\n".join(col_lines)
        tables_sql.append(f"CREATE TABLE {table.original_name} (\n{body}\n);")
    return "\n\n".join(tables_sql)


def render_fewshot_block(index: int, example: dict[str, Any], schema_block: str) -> str:
    """Render one few-shot example as ``Example {index}: ...``."""
    answer_json = json.dumps({"tables": example["tables"], "columns": example["columns"]})
    return (
        f"Example {index}:\n"
        f"Schema: {schema_block}\n"
        f"Question: {example['question']}\n"
        f"Output: {answer_json}"
    )


def render_backward_fewshot_block(index: int, example: dict[str, Any], schema_block: str) -> str:
    """Render one backward-style (Method D) few-shot example as ``Example {index}: ...``."""
    return (
        f"Example {index}:\n"
        f"Schema: {schema_block}\n"
        f"Question: {example['question']}\n"
        f"SQL: {example['gold_sql']}"
    )


def render_graph_endpoint_fewshot_block(index: int, example: dict[str, Any], schema_block: str) -> str:
    """Render one graph-endpoint-style (Method F) few-shot example as ``Example {index}: ...``."""
    answer_json = json.dumps({"core_tables": example["core_tables"], "columns": example["columns"]})
    return (
        f"Example {index}:\n"
        f"Schema: {schema_block}\n"
        f"Question: {example['question']}\n"
        f"Output: {answer_json}"
    )


def render_user_message(
    template: PromptTemplate,
    schema_block: str,
    question: str,
    fewshot_examples: list[dict[str, Any]] | None = None,
    fewshot_schema_blocks: list[str] | None = None,
) -> str:
    """Render the final user message, optionally prefixed with few-shot examples.

    Parameters
    ----------
    fewshot_schema_blocks
        Rendered :func:`render_schema_block` output for each fewshot
        example's *own* ``db_id`` schema (not the target question's schema).
        Required (and must be the same length) when ``fewshot_examples`` is
        given.

    Raises
    ------
    ValueError
        If ``fewshot_examples`` is given without a matching
        ``fewshot_schema_blocks``.
    """
    if not fewshot_examples:
        return template.user_template.format(schema_block=schema_block, question=question)

    if fewshot_schema_blocks is None or len(fewshot_schema_blocks) != len(fewshot_examples):
        raise ValueError("fewshot_schema_blocks must have one entry per fewshot_examples")

    blocks = [
        render_fewshot_block(i, ex, block)
        for i, (ex, block) in enumerate(zip(fewshot_examples, fewshot_schema_blocks), start=1)
    ]
    real_task = (
        "Now the real task:\n"
        f"Schema: {schema_block}\n"
        f"Question: {question}\n\n"
        "Return the relevant tables and columns as JSON."
    )
    return "\n\n".join([*blocks, real_task])


def render_backward_user_message(
    template: PromptTemplate,
    schema_block: str,
    question: str,
    fewshot_examples: list[dict[str, Any]] | None = None,
    fewshot_schema_blocks: list[str] | None = None,
) -> str:
    """Render the final backward (Method D) user message, optionally prefixed
    with few-shot examples. See :func:`render_user_message` for the forward
    (JSON-output) counterpart — kept separate because the two templates'
    real-task suffixes differ (JSON instruction vs. bare ``SQL:``).

    Raises
    ------
    ValueError
        If ``fewshot_examples`` is given without a matching
        ``fewshot_schema_blocks``.
    """
    if not fewshot_examples:
        return template.user_template.format(schema_block=schema_block, question=question)

    if fewshot_schema_blocks is None or len(fewshot_schema_blocks) != len(fewshot_examples):
        raise ValueError("fewshot_schema_blocks must have one entry per fewshot_examples")

    blocks = [
        render_backward_fewshot_block(i, ex, block)
        for i, (ex, block) in enumerate(zip(fewshot_examples, fewshot_schema_blocks), start=1)
    ]
    real_task = (
        "Now the real task:\n"
        f"Schema: {schema_block}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )
    return "\n\n".join([*blocks, real_task])


def render_graph_endpoint_user_message(
    template: PromptTemplate,
    schema_block: str,
    question: str,
    fewshot_examples: list[dict[str, Any]] | None = None,
    fewshot_schema_blocks: list[str] | None = None,
) -> str:
    """Render the final graph-endpoint (Method F) user message, optionally
    prefixed with few-shot examples. See :func:`render_user_message` for the
    forward counterpart — kept separate because the few-shot answer JSON key
    differs (``core_tables`` vs ``tables``).

    Raises
    ------
    ValueError
        If ``fewshot_examples`` is given without a matching
        ``fewshot_schema_blocks``.
    """
    if not fewshot_examples:
        return template.user_template.format(schema_block=schema_block, question=question)

    if fewshot_schema_blocks is None or len(fewshot_schema_blocks) != len(fewshot_examples):
        raise ValueError("fewshot_schema_blocks must have one entry per fewshot_examples")

    blocks = [
        render_graph_endpoint_fewshot_block(i, ex, block)
        for i, (ex, block) in enumerate(zip(fewshot_examples, fewshot_schema_blocks), start=1)
    ]
    real_task = (
        "Now the real task:\n"
        f"Schema: {schema_block}\n"
        f"Question: {question}\n\n"
        "Return the core tables and needed columns as JSON."
    )
    return "\n\n".join([*blocks, real_task])


def estimate_tokens(text: str) -> int:
    """Rough token-count estimate (~4 chars/token for English prose).

    For prompt-size sanity checks only, not billing — real counts come from
    ``LLMResponse.input_tokens`` after an actual API call (see
    ``schema_linking.utils.llm_client``).
    """
    return max(1, len(text) // 4)
