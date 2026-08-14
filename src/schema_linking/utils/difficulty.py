"""Spider query difficulty — vendored from the official ``evaluation.py``.

Source: https://github.com/taoyds/spider/blob/master/evaluation.py
License: MIT (Spider — Tao Yu, Rui Zhang, et al., Yale LILY Lab)

The implementation below is a verbatim port of Spider's hardness logic
with only cosmetic changes:

* type hints on the public functions
* short docstrings on each helper
* tolerant ``.get()`` lookups so the helpers don't crash on synthetic
  ``sql`` dicts that omit optional keys (Spider's own evaluation.py
  assumes all keys are present)

The bucketing thresholds, the structure of ``count_component1`` /
``count_component2`` / ``count_others``, and Spider's ``count_agg``
applied to ``sql['where'][::2]`` and ``sql['having'][::2]`` (which
operates on ``cond_unit[0]`` — i.e. the ``not_op`` flag — and is thus
in effect a NOT-count rather than an aggregation count) are reproduced
exactly as published. Reproducing the quirk is required to match
Spider's published per-bucket counts on the dev split.

Input is Spider's pre-parsed ``sql`` dict (the value of
``SpiderExample.sql``). Output of :func:`eval_hardness` is one of
``"easy"`` / ``"medium"`` / ``"hard"`` / ``"extra"``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from schema_linking.data_loader import SpiderExample

__all__ = [
    "Hardness",
    "count_component1",
    "count_component2",
    "count_others",
    "eval_hardness",
    "difficulty_for_examples",
]

Hardness = Literal["easy", "medium", "hard", "extra"]

# Spider's WHERE_OPS table: 'like' is at index 9.
_LIKE_OP_INDEX = 9
# Spider's AGG_OPS table: 'none' is at index 0.
_AGG_NONE_INDEX = 0


def _has_agg(unit: Any) -> bool:
    """Reproduces Spider's ``has_agg`` — checks ``unit[0] != 0``."""
    return bool(unit) and unit[0] != _AGG_NONE_INDEX


def _count_agg(units: Iterable[Any]) -> int:
    """Reproduces Spider's ``count_agg`` — number of units with non-zero ``unit[0]``."""
    return sum(1 for u in units if _has_agg(u))


def count_component1(sql: dict[str, Any]) -> int:
    """Component-1 count: structural clauses + JOIN count + OR + LIKE.

    Adds 1 for each of ``WHERE`` / ``GROUP BY`` / ``ORDER BY`` /
    ``LIMIT`` present, plus ``n-1`` for ``n`` tables in ``FROM``, plus
    one for every ``or`` in ``FROM`` conds / ``WHERE`` / ``HAVING``,
    plus one for every ``LIKE`` op in those same lists.
    """
    count = 0
    if sql.get("where"):
        count += 1
    if sql.get("groupBy"):
        count += 1
    if sql.get("limit") is not None:
        count += 1
    if sql.get("orderBy"):
        count += 1

    table_units = (sql.get("from") or {}).get("table_units") or []
    if len(table_units) > 0:
        count += len(table_units) - 1

    ao = (
        ((sql.get("from") or {}).get("conds") or [])[1::2]
        + (sql.get("where") or [])[1::2]
        + (sql.get("having") or [])[1::2]
    )
    count += sum(1 for token in ao if token == "or")

    cond_units = (
        ((sql.get("from") or {}).get("conds") or [])[::2]
        + (sql.get("where") or [])[::2]
        + (sql.get("having") or [])[::2]
    )
    count += sum(
        1
        for cu in cond_units
        if isinstance(cu, list) and len(cu) >= 2 and cu[1] == _LIKE_OP_INDEX
    )
    return count


def count_component2(sql: dict[str, Any]) -> int:
    """Component-2 count: nested SQLs.

    Counts dict-valued ``val1`` / ``val2`` slots in any cond_unit (from
    ``FROM`` conds, ``WHERE``, or ``HAVING``) plus a count of 1 for each
    of ``INTERSECT`` / ``UNION`` / ``EXCEPT`` that is present.
    """
    nested: list[Any] = []
    for cu in (
        ((sql.get("from") or {}).get("conds") or [])[::2]
        + (sql.get("where") or [])[::2]
        + (sql.get("having") or [])[::2]
    ):
        if isinstance(cu, list):
            if len(cu) >= 4 and isinstance(cu[3], dict):
                nested.append(cu[3])
            if len(cu) >= 5 and isinstance(cu[4], dict):
                nested.append(cu[4])
    if sql.get("intersect") is not None:
        nested.append(sql["intersect"])
    if sql.get("except") is not None:
        nested.append(sql["except"])
    if sql.get("union") is not None:
        nested.append(sql["union"])
    return len(nested)


def count_others(sql: dict[str, Any]) -> int:
    """Complexity flags.

    Sets one of four 1-bits when the corresponding count exceeds its
    threshold: total ``count_agg`` > 1, ``SELECT`` items > 1, ``WHERE``
    conditions > 1, ``GROUP BY`` columns > 1.
    """
    count = 0

    select_items = (sql.get("select") or [None, []])[1] or []
    agg_count = _count_agg(select_items)
    agg_count += _count_agg((sql.get("where") or [])[::2])
    agg_count += _count_agg(sql.get("groupBy") or [])
    order_by = sql.get("orderBy") or []
    if len(order_by) > 0:
        agg_count += _count_agg(
            [u[1] for u in order_by[1] if u[1]]
            + [u[2] for u in order_by[1] if u[2]]
        )
    agg_count += _count_agg(sql.get("having") or [])
    if agg_count > 1:
        count += 1

    if len(select_items) > 1:
        count += 1
    if len(sql.get("where") or []) > 1:
        count += 1
    if len(sql.get("groupBy") or []) > 1:
        count += 1
    return count


def eval_hardness(sql: dict[str, Any]) -> Hardness:
    """Spider's easy / medium / hard / extra binning for one query.

    Parameters
    ----------
    sql
        Spider's pre-parsed ``sql`` dict — the same one carried by
        :attr:`SpiderExample.sql`.

    Returns
    -------
    Hardness
        One of ``"easy"`` / ``"medium"`` / ``"hard"`` / ``"extra"``.
    """
    c1 = count_component1(sql)
    c2 = count_component2(sql)
    co = count_others(sql)

    if c1 <= 1 and co == 0 and c2 == 0:
        return "easy"
    if (co <= 2 and c1 <= 1 and c2 == 0) or (
        c1 <= 2 and co < 2 and c2 == 0
    ):
        return "medium"
    if (
        (co > 2 and c1 <= 2 and c2 == 0)
        or (2 < c1 <= 3 and co <= 2 and c2 == 0)
        or (c1 <= 1 and co == 0 and c2 <= 1)
    ):
        return "hard"
    return "extra"


def difficulty_for_examples(
    examples: Iterable[SpiderExample],
) -> dict[int, Hardness]:
    """Map each example's ``question_id`` to its Spider hardness label.

    Parameters
    ----------
    examples
        Spider examples (from :func:`schema_linking.data_loader.load_spider_questions`).

    Returns
    -------
    dict[int, Hardness]
        ``{question_id: "easy"|"medium"|"hard"|"extra"}``. Keys are
        preserved from the iterable's order (``dict`` is ordered).
    """
    return {ex.question_id: eval_hardness(ex.sql) for ex in examples}
