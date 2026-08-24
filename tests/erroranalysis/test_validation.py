"""Blind validation sampling and agreement."""

from __future__ import annotations

import pandas as pd
import pytest

from schema_linking.data_loader import SpiderExample
from schema_linking.erroranalysis.census import CENSUS_COLUMNS
from schema_linking.erroranalysis.loading import Corpus
from schema_linking.erroranalysis.validation import (
    VALIDATION_SHEET_COLUMNS,
    agreement,
    draw_validation_sample,
    read_validation_sheet,
    write_validation_sheet,
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


def test_equal_size_causes_do_not_share_relative_positions():
    """Two equal-sized cause groups drawn with the same base seed must not
    select the same relative row positions within their group — otherwise
    every stratum over- or under-represents whichever method or
    question-range sits at that position (task-14 fix round 1, Important 2).
    """
    sample = draw_validation_sample(_census(40), n_per_cause=10, seed=42)

    def _positions(cause):
        rows = sample[sample.cause == cause]
        return sorted(int(e.rsplit("c", 1)[-1]) for e in rows.element)

    assert _positions("UNFORCED") != _positions("PARAPHRASE")


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


def _write_sheet_corpus() -> Corpus:
    """A minimal two-question corpus for exercising `write_validation_sheet`.

    Only the fields `write_validation_sheet` touches
    (`example_by_qid`, `gold_tier1`, `gold_tier2`, `predictions`) carry real
    content; `schemas`/`indices`/`hardness` are unused by the function under
    test and left empty.
    """
    examples = (
        SpiderExample(
            question_id=0,
            db_id="concert_singer",
            question="How many singers are there?",
            query="SELECT count(*) FROM singer",
            sql={},
            split="dev",
        ),
        SpiderExample(
            question_id=1,
            db_id="concert_singer",
            question="What is the name of the oldest singer?",
            query="SELECT name FROM singer ORDER BY age DESC LIMIT 1",
            sql={},
            split="dev",
        ),
    )
    gold_tier1 = {
        0: {
            "db_id": "concert_singer",
            "tables": ["singer"],
            "columns": [["singer", "singer_id"]],
        },
        1: {
            "db_id": "concert_singer",
            "tables": ["singer"],
            "columns": [["singer", "name"]],
        },
    }
    gold_tier2 = {
        0: {
            "db_id": "concert_singer",
            "tables": ["singer", "concert"],
            "columns": [["singer", "singer_id"], ["concert", "singer_id"]],
        },
        1: {
            "db_id": "concert_singer",
            "tables": ["singer"],
            "columns": [["singer", "name"], ["singer", "age"]],
        },
    }
    predictions = {
        "lexical": {
            0: {
                "db_id": "concert_singer",
                "tables": ["singer"],
                "columns": [["singer", "singer_id"]],
            },
            1: {
                "db_id": "concert_singer",
                "tables": ["singer"],
                "columns": [["singer", "name"]],
            },
        }
    }
    return Corpus(
        split="dev",
        examples=examples,
        schemas={},
        indices={},
        gold_tier1=gold_tier1,
        gold_tier2=gold_tier2,
        predictions=predictions,
        hardness={0: "easy", 1: "medium"},
    )


def _write_sheet_sample() -> pd.DataFrame:
    """One tier-1 and one tier-2 case with distinctive, non-colliding
    machine-code tokens, so the blindness test can search for them.
    """
    rows = [
        {
            "question_id": 0,
            "db_id": "concert_singer",
            "method": "lexical",
            "tier": "tier1",
            "level": "column",
            "element": "singer.singer_id",
            "shape_code": "MISS",
            "cause": "UNFORCED",
            "rule_name": "unforced_miss_rule",
            "evidence": "lexical_anchor_token_zzz",
            "hardness": "easy",
            "n_tables": 1,
            "n_columns": 3,
            "schema_size_bin": "Q1_smallest",
        },
        {
            "question_id": 1,
            "db_id": "concert_singer",
            "method": "lexical",
            "tier": "tier2",
            "level": "column",
            "element": "singer.age",
            "shape_code": "SPUR",
            "cause": "SIBLING",
            "rule_name": "sibling_rule",
            "evidence": "fk_adjacent_token_zzz",
            "hardness": "medium",
            "n_tables": 1,
            "n_columns": 3,
            "schema_size_bin": "Q1_smallest",
        },
    ]
    return pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))


def test_write_validation_sheet_has_exactly_the_blind_columns(tmp_path):
    sample = draw_validation_sample(_write_sheet_sample(), n_per_cause=1, seed=42)
    path = write_validation_sheet(sample, _write_sheet_corpus(), tmp_path / "sheet.csv")
    written = pd.read_csv(path, keep_default_na=False)
    assert list(written.columns) == list(VALIDATION_SHEET_COLUMNS)
    assert (written.human_cause == "").all()
    assert (written.notes == "").all()


def test_write_validation_sheet_withholds_the_machine_code(tmp_path):
    """The blindness contract: none of the machine's `cause`, `rule_name`
    or `evidence` values may appear anywhere in the written sheet. This
    would fail if someone later added a helpful "hint" column.
    """
    sample = draw_validation_sample(_write_sheet_sample(), n_per_cause=1, seed=42)
    path = write_validation_sheet(sample, _write_sheet_corpus(), tmp_path / "sheet.csv")
    text = path.read_text(encoding="utf-8")
    for leaked in (
        "UNFORCED",
        "SIBLING",
        "unforced_miss_rule",
        "sibling_rule",
        "lexical_anchor_token_zzz",
        "fk_adjacent_token_zzz",
    ):
        assert leaked not in text, leaked


def test_write_validation_sheet_picks_tier_specific_gold_and_matching_example(tmp_path):
    sample = draw_validation_sample(_write_sheet_sample(), n_per_cause=1, seed=42)
    corpus = _write_sheet_corpus()
    path = write_validation_sheet(sample, corpus, tmp_path / "sheet.csv")
    written = pd.read_csv(path, keep_default_na=False)

    tier1_row = written[written.tier == "tier1"].iloc[0]
    assert tier1_row.question == "How many singers are there?"
    assert tier1_row.gold_sql == "SELECT count(*) FROM singer"
    assert "singer_id" in tier1_row.gold_elements
    assert "concert" not in tier1_row.gold_elements  # tier-2-only table

    tier2_row = written[written.tier == "tier2"].iloc[0]
    assert tier2_row.question == "What is the name of the oldest singer?"
    assert tier2_row.gold_sql == "SELECT name FROM singer ORDER BY age DESC LIMIT 1"
    assert "age" in tier2_row.gold_elements  # only present in tier-2 gold
    assert "singer.name" in tier2_row.predicted_elements
