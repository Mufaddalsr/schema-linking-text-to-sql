"""Blind human validation of the automatic codes.

The cascade is exhaustive, so there is no residual to adjudicate. What needs
establishing instead is whether the rules agree with a human reading the same
evidence. This module draws a cause-stratified sample, writes a sheet that
withholds the machine's code, and computes agreement once the sheet comes
back.

Withholding the code is not a formality: a coder shown "SIBLING" will find
reasons to agree with it, and the resulting kappa would measure nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from schema_linking.erroranalysis.taxonomy import Cause

if TYPE_CHECKING:
    from schema_linking.erroranalysis.loading import Corpus

VALIDATION_SHEET_COLUMNS: tuple[str, ...] = (
    "case_id",
    "question",
    "db_id",
    "gold_sql",
    "tier",
    "shape_code",
    "element",
    "gold_elements",
    "predicted_elements",
    "human_cause",
    "notes",
)


def _case_id(row: pd.Series) -> str:
    """Stable 12-hex-character id for one census row.

    Derived from the identifying tuple rather than the row index, so a
    re-run that reorders the census still produces the same ids.
    """
    key = f"{row.method}|{row.tier}|{row.question_id}|{row.element}|{row.shape_code}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def draw_validation_sample(
    census: pd.DataFrame,
    n_per_cause: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw a cause-stratified sample for blind human coding.

    With the default ``n_per_cause=15``, the twelve causes that actually
    fire on this corpus (``IMPLICIT-AGG``, ``MALFORMED`` and ``WRONG-DB``
    have zero rows) yield up to 12 * 15 = 180 cases to hand-code — fewer
    if any cause has fewer than ``n_per_cause`` rows, since a rare cause is
    taken whole rather than padded.

    Parameters
    ----------
    census
        A classified census frame.
    n_per_cause
        Target per cause. A cause with fewer rows is taken whole — never
        padded, never resampled.
    seed
        Passed to ``DataFrame.sample`` for reproducibility.

    Returns
    -------
    pandas.DataFrame
        The sampled rows plus a ``case_id`` column, sorted by ``case_id`` so
        the sheet order carries no information about the assigned cause.
    """
    parts = [
        group.sample(n=min(n_per_cause, len(group)), random_state=seed)
        for _, group in census.groupby("cause", sort=True)
    ]
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.assign(case_id=sample.apply(_case_id, axis=1))
    return sample.sort_values("case_id", ignore_index=True)


def write_validation_sheet(
    sample: pd.DataFrame,
    corpus: "Corpus",
    path: Path,
) -> Path:
    """Write the blind coding sheet.

    The ``cause``, ``rule_name`` and ``evidence`` columns are deliberately
    omitted. ``human_cause`` and ``notes`` are left empty for you to fill.
    """
    examples = corpus.example_by_qid()
    rows = []
    for row in sample.itertuples(index=False):
        example = examples[int(row.question_id)]
        gold = corpus.gold_tier1 if row.tier == "tier1" else corpus.gold_tier2
        gold_rec = gold[int(row.question_id)]
        pred_rec = corpus.predictions[row.method][int(row.question_id)]
        rows.append(
            {
                "case_id": row.case_id,
                "question": example.question,
                "db_id": row.db_id,
                "gold_sql": example.query,
                "tier": row.tier,
                "shape_code": row.shape_code,
                "element": row.element,
                "gold_elements": _render(gold_rec),
                "predicted_elements": _render(pred_rec),
                "human_cause": "",
                "notes": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(VALIDATION_SHEET_COLUMNS)).to_csv(
        path, index=False
    )
    return path


def _render(record: dict) -> str:
    """Compact ``tables | table.column`` rendering for the sheet."""
    tables = ", ".join(sorted(record.get("tables", ())))
    columns = ", ".join(sorted(f"{t}.{c}" for t, c in record.get("columns", ())))
    return f"tables: {tables} || columns: {columns}"


def read_validation_sheet(path: Path) -> pd.DataFrame:
    """Read a completed sheet back.

    Raises
    ------
    ValueError
        If any ``human_cause`` is not a member of :class:`Cause`. Typos in a
        hand-edited CSV must fail loud, not silently become a new category.
    """
    frame = pd.read_csv(path).fillna({"human_cause": ""})
    valid = {str(c) for c in Cause}
    coded = frame[frame.human_cause != ""]
    unknown = sorted(set(coded.human_cause) - valid)
    if unknown:
        raise ValueError(f"unknown cause(s) in {path}: {unknown}")
    return coded[["case_id", "human_cause"]].reset_index(drop=True)


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa between two label series over the same cases."""
    labels = sorted(set(a) | set(b))
    matrix = pd.crosstab(a, b).reindex(index=labels, columns=labels, fill_value=0)
    total = matrix.to_numpy().sum()
    observed = np.trace(matrix.to_numpy()) / total
    expected = float(
        (matrix.sum(axis=1).to_numpy() * matrix.sum(axis=0).to_numpy()).sum()
    ) / (total**2)
    if expected == 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def agreement(
    sample: pd.DataFrame, human: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Compare machine codes against human codes.

    Parameters
    ----------
    sample
        Output of :func:`draw_validation_sample` — carries the machine's
        ``cause`` and the ``case_id``.
    human
        Output of :func:`read_validation_sheet` — ``case_id`` and
        ``human_cause``.

    Returns
    -------
    tuple[pandas.DataFrame, float]
        Per-cause agreement (``cause``, ``n``, ``n_agreed``, ``agreement``)
        and Cohen's kappa over all sampled cases.

    Raises
    ------
    ValueError
        If any sampled case is missing a human code. A partial sheet would
        bias the kappa toward whichever cases were easiest to code.
    """
    merged = sample[["case_id", "cause"]].merge(human, on="case_id", how="left")
    missing = merged[merged.human_cause.isna()]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} sampled case(s) are uncoded: "
            f"{missing.case_id.head(5).tolist()}"
        )
    merged = merged.assign(agreed=(merged.cause == merged.human_cause).astype(int))
    per_cause = (
        merged.groupby("cause", as_index=False)
        .agg(n=("agreed", "size"), n_agreed=("agreed", "sum"))
    )
    per_cause = per_cause.assign(agreement=per_cause.n_agreed / per_cause.n)
    return per_cause, _cohen_kappa(merged.cause, merged.human_cause)
