"""Load Taniguchi et al.'s human-annotated schema-linking labels.

The dataset
-----------
Taniguchi et al. distribute their annotations as two JSON Lines files at
``<taniguchi_splits_dir>/dev.jsonl`` and ``<taniguchi_splits_dir>/test.jsonl``
(517 examples each, jointly covering all 1034 examples of Spider's dev set).
Their sibling ``.txt`` files contain BIO-tagged token sequences derived from
the same data and are ignored here.

Line schema
~~~~~~~~~~~
Each line is a JSON object::

    {
      "id": 3616,
      "text": "How many singers do we have?",
      "meta": {
        "000_sql": "<gold SQL>",
        "001_table_0": "stadium", "001_table_1": "singer", ...,
        "002_col_1": "stadium.Stadium_ID", "002_col_9": "singer.Name", ...
      },
      "labels": [[9, 16, "001_table_1"]]
    }

Each ``labels`` entry is a ``(start, end, meta_key)`` triple referencing
into the ``meta`` dict. ``meta`` values are Spider's *original* (DB-canonical)
names — the same form found in ``tables.json``'s ``table_names_original``
and ``column_names_original`` fields.

Spider-question matching
------------------------
:func:`to_gold_links` joins each Taniguchi annotation to a Spider dev
example primarily by question text. 988 of 1034 Taniguchi texts match
Spider's ``question`` field exactly; the remaining 46 differ by typo
fixes, punctuation, or whitespace that Taniguchi normalised. For those,
the loader falls back to Taniguchi's ``id`` field: empirically, Taniguchi
ids are contiguous from 3616 to 4649 with ``id - 3616`` equal to the
Spider dev row index, and the per-annotation ``meta`` tables always match
the mapped Spider db's schema. Each fallback use is logged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from schema_linking.data_loader import SpiderExample
from schema_linking.schema_parser import Schema
from schema_linking.utils.config import Config, load_config

logger = logging.getLogger(__name__)

_DEFAULT_SPLIT_FILES: tuple[str, ...] = ("dev.jsonl", "test.jsonl")
_TANIGUCHI_ID_OFFSET: int = 3616  # Taniguchi id 3616 → Spider dev row 0


@dataclass(frozen=True, slots=True)
class TaniguchiAnnotation:
    """A single human-annotated example from Taniguchi et al.

    Attributes
    ----------
    id
        Taniguchi's intrinsic annotation id (contiguous 3616..4649 on
        Spider dev).
    text
        Question text as it appears in the Taniguchi file. Differs from
        Spider's ``question`` for ~46/1034 examples where Taniguchi fixed
        typos or normalised whitespace.
    sql
        Gold SQL from the ``meta["000_sql"]`` field — kept for reference;
        not used by the converter.
    labels
        Tuple of ``(start, end, label)`` triples. Character offsets are
        relative to :attr:`text`. ``label`` is the *resolved* form, i.e.
        either ``"TableName"`` (table label) or ``"TableName.ColumnName"``
        (column label), using Spider's original casing.
    """

    id: int
    text: str
    sql: str
    labels: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True, slots=True)
class GoldLinks:
    """Schema-linking gold set for a single Spider dev question.

    Attributes
    ----------
    db_id
        Spider database identifier.
    tables
        Distinct table names (Spider ``*_original`` form), lexically sorted.
        Includes both explicitly-labelled tables and tables implied by any
        labelled column.
    columns
        Distinct ``(table, column)`` pairs (Spider ``*_original`` form),
        lexically sorted.
    """

    db_id: str
    tables: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]


def load_taniguchi_annotations(
    paths: Iterable[Path] | None = None,
    *,
    config: Config | None = None,
) -> dict[int, TaniguchiAnnotation]:
    """Load Taniguchi annotations from one or more JSONL files.

    Parameters
    ----------
    paths
        Iterable of JSONL file paths. If ``None``, loads ``dev.jsonl`` and
        ``test.jsonl`` from ``config.data.taniguchi_splits_dir``.
    config
        Optional :class:`Config`. Used only when ``paths`` is ``None``.

    Returns
    -------
    dict[int, TaniguchiAnnotation]
        Mapping from Taniguchi ``id`` to annotation record. Labels are
        resolved through each line's ``meta`` dict at load time.

    Raises
    ------
    FileNotFoundError
        If any input path is missing.
    ValueError
        If duplicate Taniguchi ids are seen across input files, or a
        label references a ``meta`` key that is not present.
    """
    resolved_paths = _resolve_paths(paths, config)

    out: dict[int, TaniguchiAnnotation] = {}
    for path in resolved_paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                ann = _parse_annotation(raw, source=f"{path.name}:{line_no}")
                if ann.id in out:
                    raise ValueError(
                        f"duplicate Taniguchi id {ann.id} at " f"{path.name}:{line_no}"
                    )
                out[ann.id] = ann
    return out


def to_gold_links(
    annotations: dict[int, TaniguchiAnnotation],
    spider_examples: tuple[SpiderExample, ...],
    schemas: dict[str, Schema],
    *,
    max_unresolvable_ratio: float = 0.01,
) -> dict[int, GoldLinks]:
    """Convert Taniguchi annotations into Spider-keyed gold-link sets.

    Matches each Taniguchi annotation to a Spider dev example primarily by
    question text. For annotations whose text doesn't exactly match any
    Spider question, falls back to ``Spider qid = Taniguchi id -
    _TANIGUCHI_ID_OFFSET`` (verified empirically; see module docstring).

    Each label is validated against the target db's schema. Tables are
    matched on :attr:`schema_parser.Table.original_name`; columns on
    ``(table.original_name, column.original_name)`` — Taniguchi's ``meta``
    uses Spider's original casing, so equality is case-sensitive.

    Parameters
    ----------
    annotations
        Output of :func:`load_taniguchi_annotations`.
    spider_examples
        Spider dev examples in source order (from
        :func:`schema_linking.data_loader.load_spider_questions`).
    schemas
        Output of :func:`schema_linking.schema_parser.load_schemas`.
    max_unresolvable_ratio
        Hard ceiling on the fraction of labels that fail schema validation.
        If exceeded, raises :class:`ValueError` — that indicates the
        matching or validation logic is broken, not a few bad annotations.

    Returns
    -------
    dict[int, GoldLinks]
        Keyed by Spider's :attr:`SpiderExample.question_id` (the
        zero-based dev row index).

    Raises
    ------
    ValueError
        If the unresolvable-label fraction exceeds ``max_unresolvable_ratio``.
    """
    spider_by_question: dict[str, SpiderExample] = {
        ex.question: ex for ex in spider_examples
    }

    total_labels = 0
    bad_labels = 0
    unmatched_questions = 0
    fallback_used = 0

    by_qid: dict[int, tuple[set[str], set[tuple[str, str]], str]] = {}

    for ann in annotations.values():
        ex = spider_by_question.get(ann.text)
        if ex is None:
            ex = _fallback_by_id(ann, spider_examples)
            if ex is None:
                unmatched_questions += 1
                logger.warning(
                    "no Spider dev question matched Taniguchi id %d; "
                    "text=%r — skipping",
                    ann.id,
                    ann.text,
                )
                continue
            fallback_used += 1
            logger.info(
                "Taniguchi id %d: text mismatch — used id-offset fallback "
                "to Spider qid %d; T=%r vs S=%r",
                ann.id,
                ex.question_id,
                ann.text,
                ex.question,
            )

        schema = schemas.get(ex.db_id)
        if schema is None:
            raise ValueError(
                f"no schema for db_id {ex.db_id!r} (Spider qid {ex.question_id})"
            )

        tables_index = {t.original_name: t for t in schema.tables}

        qid_tables, qid_cols, _ = by_qid.setdefault(
            ex.question_id, (set(), set(), ex.db_id)
        )

        for _start, _end, label in ann.labels:
            total_labels += 1
            if "." in label:
                table_name, column_name = label.split(".", 1)
                table = tables_index.get(table_name)
                if table is None or not _table_has_column(table, column_name):
                    bad_labels += 1
                    logger.warning(
                        "label %r at Taniguchi id %d (Spider qid %d, db %r) "
                        "does not resolve in schema",
                        label,
                        ann.id,
                        ex.question_id,
                        ex.db_id,
                    )
                    continue
                qid_tables.add(table_name)
                qid_cols.add((table_name, column_name))
            else:
                if label not in tables_index:
                    bad_labels += 1
                    logger.warning(
                        "table label %r at Taniguchi id %d (Spider qid %d, "
                        "db %r) does not resolve in schema",
                        label,
                        ann.id,
                        ex.question_id,
                        ex.db_id,
                    )
                    continue
                qid_tables.add(label)

    if total_labels == 0:
        ratio = 0.0
    else:
        ratio = bad_labels / total_labels
    if ratio > max_unresolvable_ratio:
        raise ValueError(
            f"unresolvable label fraction {ratio:.3%} exceeds threshold "
            f"{max_unresolvable_ratio:.3%} "
            f"({bad_labels}/{total_labels} labels failed schema validation)"
        )

    logger.info(
        "to_gold_links: %d annotations → %d Spider qids; "
        "labels total=%d, unresolvable=%d (%.3f%%); "
        "id-offset fallbacks=%d; unmatched questions=%d",
        len(annotations),
        len(by_qid),
        total_labels,
        bad_labels,
        ratio * 100.0,
        fallback_used,
        unmatched_questions,
    )

    return {
        qid: GoldLinks(
            db_id=db_id,
            tables=tuple(sorted(tables)),
            columns=tuple(sorted(cols)),
        )
        for qid, (tables, cols, db_id) in by_qid.items()
    }


def save_gold_links(gold: dict[int, GoldLinks], path: Path) -> None:
    """Write a gold-links dict to JSON at ``path``.

    The on-disk shape is a JSON object keyed by stringified
    :attr:`SpiderExample.question_id`::

        {
          "0":   {"db_id": "concert_singer", "tables": ["singer"], "columns": []},
          "318": {"db_id": "cre_Doc_Template_Mgt",
                  "tables": ["Templates"], "columns": []},
          ...
        }

    Parent directories are created if needed.
    """
    serialisable = {
        str(qid): {
            "db_id": g.db_id,
            "tables": list(g.tables),
            "columns": [list(c) for c in g.columns],
        }
        for qid, g in sorted(gold.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, sort_keys=False)
        f.write("\n")


def _resolve_paths(paths: Iterable[Path] | None, config: Config | None) -> list[Path]:
    if paths is not None:
        resolved = [Path(p) for p in paths]
    else:
        cfg = config if config is not None else load_config()
        resolved = [
            cfg.data.taniguchi_splits_dir / name for name in _DEFAULT_SPLIT_FILES
        ]
    for p in resolved:
        if not p.is_file():
            raise FileNotFoundError(f"Taniguchi annotation file not found: {p}")
    return resolved


def _parse_annotation(raw: dict, *, source: str) -> TaniguchiAnnotation:
    meta: dict[str, str] = raw["meta"]
    labels: list[tuple[int, int, str]] = []
    for entry in raw["labels"]:
        start, end, key = entry
        if key not in meta:
            raise ValueError(f"label key {key!r} at {source} is not present in meta")
        labels.append((int(start), int(end), meta[key]))
    return TaniguchiAnnotation(
        id=int(raw["id"]),
        text=raw["text"],
        sql=meta.get("000_sql", ""),
        labels=tuple(labels),
    )


def _fallback_by_id(
    ann: TaniguchiAnnotation, spider_examples: tuple[SpiderExample, ...]
) -> SpiderExample | None:
    qid = ann.id - _TANIGUCHI_ID_OFFSET
    if 0 <= qid < len(spider_examples):
        return spider_examples[qid]
    return None


def _table_has_column(table, column_name: str) -> bool:
    return any(c.original_name == column_name for c in table.columns)
