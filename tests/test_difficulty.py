"""Tests for src/schema_linking/utils/difficulty.py.

Four hand-crafted ``sql`` dicts — one per Spider hardness bucket — plus
an integration test on Spider dev that asserts the per-bucket counts
land within ±50 of the published numbers (easy~248, medium~446,
hard~174, extra~166).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from schema_linking.data_loader import load_spider_questions
from schema_linking.utils.difficulty import (
    count_component1,
    count_component2,
    count_others,
    difficulty_for_examples,
    eval_hardness,
)


# ---------- synthetic sql dicts ----------


def _base_sql() -> dict[str, Any]:
    """Minimal ``sql`` dict skeleton with every optional key present."""
    return {
        "select": [False, []],
        "from": {"table_units": [["table_unit", 0]], "conds": []},
        "where": [],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }


def _col_unit(col_id: int = 0) -> list[Any]:
    """Spider ``col_unit`` = ``[agg_id, col_id, distinct]`` (agg_id 0 = none)."""
    return [0, col_id, False]


def _val_unit(col_id: int = 0) -> list[Any]:
    """Spider ``val_unit`` = ``[unit_op, col_unit1, col_unit2]``."""
    return [0, _col_unit(col_id), None]


def _select_item(col_id: int = 0) -> tuple[int, list[Any]]:
    """One ``SELECT`` projection: ``(agg_id, val_unit)`` with no aggregation."""
    return (0, _val_unit(col_id))


def _cond_unit(op_id: int = 2, col_id: int = 0, val: Any = 5) -> list[Any]:
    """``cond_unit`` = ``[not_op, op_id, val_unit, val1, val2]`` (op 2 = ``=``)."""
    return [False, op_id, _val_unit(col_id), val, None]


# ---------- 1. Synthetic per-bucket assertions ----------


def test_easy_single_column_single_table() -> None:
    """``SELECT col FROM t`` → c1=0, c2=0, others=0 → easy."""
    sql = _base_sql()
    sql["select"] = [False, [_select_item(0)]]
    assert count_component1(sql) == 0
    assert count_component2(sql) == 0
    assert count_others(sql) == 0
    assert eval_hardness(sql) == "easy"


def test_medium_where_and_groupby() -> None:
    """``SELECT col FROM t WHERE x = 5 GROUP BY y`` → c1=2 → medium."""
    sql = _base_sql()
    sql["select"] = [False, [_select_item(0)]]
    sql["where"] = [_cond_unit(op_id=2, col_id=1, val=5)]
    sql["groupBy"] = [_col_unit(col_id=2)]
    assert count_component1(sql) == 2  # where + groupBy
    assert count_component2(sql) == 0
    assert count_others(sql) == 0
    assert eval_hardness(sql) == "medium"


def test_hard_where_groupby_orderby() -> None:
    """``SELECT col FROM t WHERE x=5 GROUP BY y ORDER BY z`` → c1=3 → hard."""
    sql = _base_sql()
    sql["select"] = [False, [_select_item(0)]]
    sql["where"] = [_cond_unit(op_id=2, col_id=1, val=5)]
    sql["groupBy"] = [_col_unit(col_id=2)]
    sql["orderBy"] = ["asc", [_val_unit(col_id=3)]]
    assert count_component1(sql) == 3  # where + groupBy + orderBy
    assert count_component2(sql) == 0
    assert count_others(sql) == 0
    assert eval_hardness(sql) == "hard"


def test_extra_two_set_ops() -> None:
    """Bare query with both ``INTERSECT`` and ``UNION`` non-None → c2=2 → extra."""
    sql = _base_sql()
    sql["select"] = [False, [_select_item(0)]]
    nested = _base_sql()
    nested["select"] = [False, [_select_item(0)]]
    sql["intersect"] = nested
    sql["union"] = dict(nested)
    assert count_component1(sql) == 0
    assert count_component2(sql) == 2
    assert eval_hardness(sql) == "extra"


# ---------- 2. Bucket-boundary corner cases ----------


def test_select_with_aggregate_is_others_zero() -> None:
    """One aggregation in SELECT is fine (others triggers at agg_count > 1)."""
    sql = _base_sql()
    sql["select"] = [False, [(3, _val_unit(0))]]  # COUNT(col)
    assert count_others(sql) == 0
    assert eval_hardness(sql) == "easy"


def test_join_adds_n_minus_one_to_component1() -> None:
    sql = _base_sql()
    sql["select"] = [False, [_select_item(0)]]
    sql["from"]["table_units"] = [
        ["table_unit", 0],
        ["table_unit", 1],
        ["table_unit", 2],
    ]
    assert count_component1(sql) == 2  # 3 tables → 2 extra joins


# ---------- 3. difficulty_for_examples shape ----------


def test_difficulty_for_examples_keys_are_question_ids() -> None:
    dev = load_spider_questions("dev")
    out = difficulty_for_examples(dev)
    assert len(out) == len(dev)
    assert set(out.keys()) == {ex.question_id for ex in dev}
    assert set(out.values()) <= {"easy", "medium", "hard", "extra"}


# ---------- 4. Integration: distribution on Spider dev ----------

# Published Spider 1.0 dev split counts (Yu et al. 2018).
_EXPECTED = {"easy": 248, "medium": 446, "hard": 174, "extra": 166}
_TOLERANCE = 50


@pytest.mark.parametrize("bucket", list(_EXPECTED))
def test_dev_distribution_within_tolerance(bucket: str) -> None:
    dev = load_spider_questions("dev")
    counts = Counter(eval_hardness(ex.sql) for ex in dev)
    n = counts[bucket]
    expected = _EXPECTED[bucket]
    assert abs(n - expected) <= _TOLERANCE, (
        f"{bucket}: got {n}, expected {expected}±{_TOLERANCE}"
    )


def test_dev_distribution_sums_to_1034() -> None:
    dev = load_spider_questions("dev")
    counts = Counter(eval_hardness(ex.sql) for ex in dev)
    assert sum(counts.values()) == len(dev) == 1034
