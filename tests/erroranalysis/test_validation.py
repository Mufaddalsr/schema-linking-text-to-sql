"""Blind validation sampling and agreement."""

from __future__ import annotations

import pandas as pd
import pytest

from schema_linking.erroranalysis.census import CENSUS_COLUMNS
from schema_linking.erroranalysis.validation import (
    agreement,
    draw_validation_sample,
    read_validation_sheet,
)


def _census(n_per_cause=40):
    rows = []
    qid = 0
    for cause in ("UNFORCED", "PARAPHRASE", "SIBLING", "UNANCHORED"):
        for i in range(n_per_cause):
            rows.append(
                {
                    "question_id": qid,
                    "db_id": "concert_singer",
                    "method": ["lexical", "embedding", "graph"][i % 3],
                    "tier": "tier1",
                    "level": "column",
                    "element": f"singer.c{i}",
                    "shape_code": "MISS" if cause in ("UNFORCED", "PARAPHRASE") else "SPUR",
                    "cause": cause,
                    "rule_name": "r",
                    "evidence": "e",
                    "hardness": ["easy", "medium", "hard", "extra"][i % 4],
                    "n_tables": 2,
                    "n_columns": 6,
                    "schema_size_bin": "Q1_smallest",
                }
            )
            qid += 1
    return pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))


def test_sample_is_stratified_by_cause():
    sample = draw_validation_sample(_census(), n_per_cause=10, seed=42)
    assert sample.cause.value_counts().to_dict() == {
        "UNFORCED": 10,
        "PARAPHRASE": 10,
        "SIBLING": 10,
        "UNANCHORED": 10,
    }


def test_sample_is_reproducible():
    a = draw_validation_sample(_census(), n_per_cause=10, seed=42)
    b = draw_validation_sample(_census(), n_per_cause=10, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_a_different_sample():
    a = draw_validation_sample(_census(), n_per_cause=10, seed=42)
    b = draw_validation_sample(_census(), n_per_cause=10, seed=7)
    assert not a.element.tolist() == b.element.tolist()


def test_rare_cause_is_taken_whole_not_padded():
    census = pd.concat(
        [_census(3)[lambda d: d.cause == "UNFORCED"], _census(40)[lambda d: d.cause != "UNFORCED"]],
        ignore_index=True,
    )
    sample = draw_validation_sample(census, n_per_cause=10, seed=42)
    assert (sample.cause == "UNFORCED").sum() == 3


def test_sample_carries_a_stable_case_id():
    sample = draw_validation_sample(_census(), n_per_cause=5, seed=42)
    assert sample.case_id.is_unique
    assert sample.case_id.str.match(r"^[a-z0-9]{12}$").all()


def test_agreement_is_one_when_codes_match():
    sample = draw_validation_sample(_census(), n_per_cause=5, seed=42)
    human = sample[["case_id", "cause"]].rename(columns={"cause": "human_cause"})
    per_cause, kappa = agreement(sample, human)
    assert (per_cause.agreement == 1.0).all()
    assert kappa == pytest.approx(1.0)


def test_agreement_detects_a_systematic_disagreement():
    sample = draw_validation_sample(_census(), n_per_cause=5, seed=42)
    human = sample[["case_id", "cause"]].rename(columns={"cause": "human_cause"})
    human = human.assign(
        human_cause=human.human_cause.replace({"PARAPHRASE": "UNVERBALISED"})
    )
    per_cause, kappa = agreement(sample, human)
    row = per_cause[per_cause.cause == "PARAPHRASE"].iloc[0]
    assert row.agreement == 0.0
    assert kappa < 1.0


def test_agreement_raises_on_missing_human_codes():
    sample = draw_validation_sample(_census(), n_per_cause=5, seed=42)
    human = sample[["case_id", "cause"]].rename(columns={"cause": "human_cause"}).iloc[:3]
    with pytest.raises(ValueError, match="uncoded"):
        agreement(sample, human)


def test_read_validation_sheet_rejects_an_unknown_cause(tmp_path):
    path = tmp_path / "sheet.csv"
    pd.DataFrame(
        {"case_id": ["abc123def456"], "human_cause": ["NOT_A_REAL_CAUSE"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="NOT_A_REAL_CAUSE"):
        read_validation_sheet(path)
