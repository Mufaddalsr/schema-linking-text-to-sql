"""Chapter tables, each a pure function of the classified census.

Every rate here has an explicit denominator. ``MISS`` causes are reported
per gold element and ``SPUR`` / ``HALL`` causes per predicted element,
because a method that predicts 20 elements and one that predicts 8 are not
comparable on raw counts. v1 reported counts out of a 50-case sample with no
base at all, which is why its cross-tab could not be read across methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from schema_linking.erroranalysis.facts import elements_from_record
from schema_linking.erroranalysis.taxonomy import Shape

if TYPE_CHECKING:
    from schema_linking.erroranalysis.loading import Corpus

_SELF_EVIDENCING_GATE_RULES: frozenset[str] = frozenset(
    {"gold_element_not_in_schema", "tier1_gold_absent_from_sql"}
)
"""Gate rules that exclude automatically. ``missed_by_most_methods`` does not
— it needs manual confirmation (design §7.3)."""


def shape_by_method(census: pd.DataFrame) -> pd.DataFrame:
    """Axis-1 profile: MISS / SPUR / HALL counts per method and tier."""
    table = (
        census.pivot_table(
            index=["method", "tier"],
            columns="shape_code",
            values="element",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for shape in Shape:
        if str(shape) not in table.columns:
            table[str(shape)] = 0
    ordered = ["method", "tier", *(str(s) for s in Shape)]
    table = table[ordered]
    return table.assign(
        total=table[[str(s) for s in Shape]].sum(axis=1)
    )


def cause_by_method(
    census: pd.DataFrame,
    bases: pd.DataFrame,
    tier: str = "tier1",
) -> pd.DataFrame:
    """The RQ4 cross-tab, with denominators.

    Parameters
    ----------
    census
        Classified census.
    bases
        ``method``, ``tier``, ``n_gold``, ``n_predicted`` — the denominators.
        Produced by :func:`compute_bases`.
    tier
        Which tier to report.

    Returns
    -------
    pandas.DataFrame
        ``method``, ``cause``, ``shape_code``, ``n``, ``base``,
        ``base_kind``, ``rate``.
    """
    subset = census[census.tier == tier]
    counts = (
        subset.groupby(["method", "shape_code", "cause"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    tier_bases = bases[bases.tier == tier]
    merged = counts.merge(tier_bases, on="method", how="left")
    is_miss = merged.shape_code == str(Shape.MISS)
    merged = merged.assign(
        base=merged.n_gold.where(is_miss, merged.n_predicted),
        base_kind=pd.Series("predicted_elements", index=merged.index).where(
            ~is_miss, "gold_elements"
        ),
    )
    merged = merged.assign(rate=merged.n / merged.base.replace(0, pd.NA))
    return merged[
        ["method", "cause", "shape_code", "n", "base", "base_kind", "rate"]
    ].sort_values(["method", "n"], ascending=[True, False], ignore_index=True)


def compute_bases(
    corpus: "Corpus",
    tier_gold: dict[str, dict[int, dict[str, Any]]],
    methods: Sequence[str],
) -> pd.DataFrame:
    """Denominators per method and tier.

    ``n_gold`` is the total gold elements across all questions for that tier;
    ``n_predicted`` is the total predicted elements for that method. Both are
    corpus-level sums, matching the corpus-level error counts they divide.

    Uses :func:`schema_linking.erroranalysis.facts.elements_from_record`,
    the project's single canonicalisation path.
    """
    rows = []
    for tier, gold_by_qid in tier_gold.items():
        n_gold = sum(
            len(elements_from_record(rec)) for rec in gold_by_qid.values()
        )
        for method in methods:
            n_pred = sum(
                len(elements_from_record(rec))
                for rec in corpus.predictions[method].values()
            )
            rows.append(
                {
                    "method": method,
                    "tier": tier,
                    "n_gold": n_gold,
                    "n_predicted": n_pred,
                }
            )
    return pd.DataFrame(rows)


def _within_group_shares(
    census: pd.DataFrame, group: str, tier: str
) -> pd.DataFrame:
    """Cause distribution within each level of ``group``.

    Shares are within-group, so a hardness band with many errors does not
    dominate. This is what "how do error categories shift with X" means.
    """
    subset = census[census.tier == tier]
    counts = (
        subset.groupby([group, "cause"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    totals = counts.groupby(group, as_index=False)["n"].sum().rename(
        columns={"n": "group_total"}
    )
    merged = counts.merge(totals, on=group)
    return merged.assign(share=merged.n / merged.group_total).sort_values(
        [group, "n"], ascending=[True, False], ignore_index=True
    )


def cause_by_hardness(census: pd.DataFrame, tier: str = "tier1") -> pd.DataFrame:
    """RQ4: how the cause distribution shifts with Spider hardness."""
    return _within_group_shares(census, "hardness", tier)


def cause_by_schema_size(census: pd.DataFrame, tier: str = "tier1") -> pd.DataFrame:
    """RQ4: how the cause distribution shifts with schema size."""
    return _within_group_shares(census, "schema_size_bin", tier)


def gold_defects(census: pd.DataFrame) -> pd.DataFrame:
    """Every gate hit, one row per defective gold element.

    De-duplicated across methods: a gold element that does not exist in the
    schema is one annotation defect, not six method errors.
    ``needs_confirmation`` marks the ``missed_by_most_methods`` rows you must
    check by hand before excluding them.
    """
    defects = census[census.cause == "GOLD-DEFECT"]
    if defects.empty:
        return pd.DataFrame(
            columns=[
                "question_id",
                "db_id",
                "tier",
                "level",
                "element",
                "rule_name",
                "evidence",
                "n_methods_affected",
                "needs_confirmation",
            ]
        )
    grouped = (
        defects.groupby(
            ["question_id", "db_id", "tier", "level", "element", "rule_name"],
            as_index=False,
        )
        .agg(evidence=("evidence", "first"), n_methods_affected=("method", "nunique"))
    )
    needs_confirmation = pd.array(
        [
            rule_name not in _SELF_EVIDENCING_GATE_RULES
            for rule_name in grouped.rule_name
        ],
        dtype=object,
    )
    return grouped.assign(needs_confirmation=needs_confirmation).sort_values(
        ["needs_confirmation", "question_id", "element"],
        ascending=[False, True, True],
        ignore_index=True,
    )
