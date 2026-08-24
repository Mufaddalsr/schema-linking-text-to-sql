"""The census must be the same error population as main_per_query.csv.

This is the guard against the error analysis drifting away from the metrics
the thesis reports. It compares per-query FN, FP, and hallucination counts,
per method, per tier, per level, for all 12,408 rows.

``main_per_query.csv`` is produced after
``schema_linking.evaluator.filter_hallucinated`` splits hallucinated
elements out of the prediction *before* computing precision/recall: a
predicted element that does not exist in the schema is never counted in
``{level}_fp`` — it is counted separately in ``{level}_hallucinated``. The
mapping from the census's ``shape_code`` axis onto the reported columns is
therefore three-way, not two-way:

    reported *_fn           == count of shape_code == "MISS"
    reported *_fp           == count of shape_code == "SPUR"   (HALL excluded)
    reported *_hallucinated == count of shape_code == "HALL"

Concrete check (question_id=26, method=llm_backward, tier=tier1): predicted
columns are ``{concert.Year, concert.concert_count}``, gold tier-1 columns
are ``{concert.Year}``. ``concert.concert_count`` does not exist in the
``concert_singer`` schema, so it is HALL, not SPUR: ``column_fp=0``,
``column_hallucinated=1`` in ``main_per_query.csv``, even though the raw
``|predicted - gold|`` is 1.
"""

from __future__ import annotations

import pandas as pd
import pytest

from schema_linking.erroranalysis.census import build_census
from schema_linking.erroranalysis.loading import load_corpus
from schema_linking.erroranalysis.scoring import NullSemanticScorer
from schema_linking.utils.config import load_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def census():
    cfg = load_config()
    return build_census(load_corpus("dev"), NullSemanticScorer(), cfg)


@pytest.fixture(scope="module")
def reported(census):
    cfg = load_config()
    return pd.read_csv(cfg.outputs.results_dir / "main_per_query.csv")


def _census_counts(census: pd.DataFrame) -> pd.DataFrame:
    """Per (qid, method, tier, level): counts of MISS, SPUR, and HALL rows.

    Three separate counters, matching the three-way mapping onto
    ``main_per_query.csv`` described in the module docstring.
    """
    census = census.assign(
        fn=(census.shape_code == "MISS").astype(int),
        fp=(census.shape_code == "SPUR").astype(int),
        hall=(census.shape_code == "HALL").astype(int),
    )
    return (
        census.groupby(["question_id", "method", "tier", "level"], as_index=False)[
            ["fn", "fp", "hall"]
        ]
        .sum()
    )


def _reported_counts(reported: pd.DataFrame) -> pd.DataFrame:
    """Reshape main_per_query.csv into the same long form.

    ``{level}_hallucinated`` is pulled in alongside ``{level}_fp`` /
    ``{level}_fn`` — it is what the census's HALL rows must match.
    """
    frames = []
    for level in ("table", "column"):
        frames.append(
            reported[
                [
                    "question_id",
                    "method",
                    "tier",
                    f"{level}_fp",
                    f"{level}_fn",
                    f"{level}_hallucinated",
                ]
            ]
            .rename(
                columns={
                    f"{level}_fp": "fp",
                    f"{level}_fn": "fn",
                    f"{level}_hallucinated": "hall",
                }
            )
            .assign(level=level)
        )
    return pd.concat(frames, ignore_index=True)[
        ["question_id", "method", "tier", "level", "fn", "fp", "hall"]
    ]


def test_total_error_count_matches(census, reported):
    rep = _reported_counts(reported)
    assert int(census.shape[0]) == int(rep.fn.sum() + rep.fp.sum() + rep.hall.sum())


def test_every_per_query_count_matches(census, reported):
    merged = _reported_counts(reported).merge(
        _census_counts(census),
        on=["question_id", "method", "tier", "level"],
        how="outer",
        suffixes=("_reported", "_census"),
    ).fillna(0)
    bad = merged[
        (merged.fn_reported != merged.fn_census)
        | (merged.fp_reported != merged.fp_census)
        | (merged.hall_reported != merged.hall_census)
    ]
    assert bad.empty, bad.head(20).to_string()


def test_grand_total_is_24130(census):
    """MISS 5744 + SPUR 18342 + HALL 44, computed directly off the predictions."""
    assert int(census.shape[0]) == 24130


def test_tier1_total_is_13160(census):
    assert int((census.tier == "tier1").sum()) == 13160


def test_tier2_total_is_10970(census):
    assert int((census.tier == "tier2").sum()) == 10970


def test_miss_total_is_5744(census):
    assert int((census.shape_code == "MISS").sum()) == 5744


def test_spur_total_is_18342(census):
    assert int((census.shape_code == "SPUR").sum()) == 18342


def test_hall_total_is_44(census):
    assert int((census.shape_code == "HALL").sum()) == 44


def test_hallucinations_only_come_from_llm_methods(census):
    """main_results.csv reports a nonzero hallucination rate only for D and E."""
    halluc = set(census[census.shape_code == "HALL"].method.unique())
    assert halluc <= {"llm_backward", "llm_bidirectional", "llm_forward"}
