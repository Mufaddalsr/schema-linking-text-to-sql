"""Enumerate the error census: one row per erroneous link.

Axis 1 only. Cause assignment is layered on by
:mod:`schema_linking.erroranalysis.rules`, and is deliberately separate so
this enumeration can be proved equal to ``main_per_query.csv`` before any
judgement enters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import pandas as pd

from schema_linking.erroranalysis.facts import CaseFacts
from schema_linking.erroranalysis.taxonomy import Element, ErrorInstance, Shape

if TYPE_CHECKING:
    from schema_linking.erroranalysis.facts import SemanticScorer
    from schema_linking.erroranalysis.loading import Corpus
    from schema_linking.utils.config import Config

METHODS: tuple[str, ...] = (
    "lexical",
    "embedding",
    "llm_forward",
    "llm_backward",
    "llm_bidirectional",
    "graph",
)
"""The six methods in scope §3 order (A, B, C, D, E, G)."""

TIERS: tuple[str, ...] = ("tier1", "tier2")

CENSUS_COLUMNS: tuple[str, ...] = (
    "question_id",
    "db_id",
    "method",
    "tier",
    "level",
    "element",
    "shape_code",
    "cause",
    "rule_name",
    "evidence",
    "hardness",
    "n_tables",
    "n_columns",
    "schema_size_bin",
)
"""Column order of ``census.csv``.

``shape_code`` rather than ``shape`` because ``DataFrame.shape`` is taken —
``row.shape`` would silently return the frame's dimensions.
"""


def _exists(element: Element, facts: CaseFacts) -> bool:
    """Whether ``element`` is present in the case's database schema."""
    if element.level == "table":
        return element.table in facts.index.tables
    return element in facts.index.columns


def enumerate_errors(
    facts: CaseFacts,
    method: str,
    tier: str,
) -> list[ErrorInstance]:
    """Every erroneous link for one (question, method, tier).

    Parameters
    ----------
    facts
        The case bundle. ``facts.predicted`` is this method's prediction.
    method
        One of :data:`METHODS`.
    tier
        ``"tier1"`` or ``"tier2"``.

    Returns
    -------
    list[ErrorInstance]
        Misses first, then spurious and hallucinated predictions. Each
        element appears at most once: ``HALL`` pre-empts ``SPUR``.
    """
    gold = facts.gold_for(tier)
    errors = [
        ErrorInstance(
            question_id=facts.question_id,
            db_id=facts.db_id,
            method=method,
            tier=tier,
            element=el,
            shape=Shape.MISS,
        )
        for el in sorted(gold - facts.predicted, key=lambda e: (e.level, e.render()))
    ]
    for el in sorted(
        facts.predicted - gold, key=lambda e: (e.level, e.render())
    ):
        errors.append(
            ErrorInstance(
                question_id=facts.question_id,
                db_id=facts.db_id,
                method=method,
                tier=tier,
                element=el,
                shape=Shape.SPUR if _exists(el, facts) else Shape.HALL,
            )
        )
    return errors


def errors_to_frame(
    errors: Sequence[ErrorInstance],
    facts_by_qid: Mapping[int, CaseFacts],
) -> pd.DataFrame:
    """Render error instances as the long census frame.

    ``cause``, ``rule_name``, ``evidence`` and ``schema_size_bin`` are left
    empty; they are filled by :mod:`rules` and by
    :func:`add_schema_size_bin` respectively.
    """
    rows = []
    for err in errors:
        facts = facts_by_qid[err.question_id]
        rows.append(
            {
                "question_id": err.question_id,
                "db_id": err.db_id,
                "method": err.method,
                "tier": err.tier,
                "level": err.element.level,
                "element": err.element.render(),
                "shape_code": str(err.shape),
                "cause": "",
                "rule_name": "",
                "evidence": "",
                "hardness": facts.hardness,
                "n_tables": facts.n_tables,
                "n_columns": facts.n_columns,
                "schema_size_bin": "",
            }
        )
    return pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))


def _bin_labels(n_bins: int) -> list[str]:
    """Quartile-style labels for ``n_bins`` bins, smallest to largest.

    ``pd.qcut(..., duplicates="drop")`` can return fewer than four bins
    when quantile-edge collisions occur — most often because many
    databases share the same ``n_columns``. Labels are generated for
    however many bins actually survive, rather than assuming four, so the
    endpoints stay legible (the first always reads "smallest", the last
    always reads "largest") no matter how far the count collapses.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if n_bins == 1:
        return ["Q1_smallest"]
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    labels[0] += "_smallest"
    labels[-1] += "_largest"
    return labels


def add_schema_size_bin(frame: pd.DataFrame) -> pd.DataFrame:
    """Add quartile bins over ``n_columns``, computed across databases.

    Binning is over the set of distinct ``(db_id, n_columns)`` pairs, not
    over error rows — otherwise a database with many errors would drag the
    quartile boundaries toward itself.

    Quantile-edge collisions can leave fewer than four bins — e.g. every
    database sharing the same column count, or only two or three distinct
    counts among many databases. Rather than raising, as many bins as the
    data supports are produced and labelled by :func:`_bin_labels`. When
    every database has the same ``n_columns`` there is exactly one bin.
    """
    per_db = frame[["db_id", "n_columns"]].drop_duplicates()
    if per_db["n_columns"].nunique() <= 1:
        per_db = per_db.assign(schema_size_bin="Q1_smallest")
    else:
        binned = pd.qcut(per_db["n_columns"], q=4, duplicates="drop")
        label_map = dict(
            zip(binned.cat.categories, _bin_labels(binned.cat.categories.size))
        )
        per_db = per_db.assign(schema_size_bin=binned.map(label_map).astype(str))
    return frame.drop(columns=["schema_size_bin"]).merge(
        per_db[["db_id", "schema_size_bin"]], on="db_id", how="left"
    )[list(CENSUS_COLUMNS)]


def build_facts(
    corpus: "Corpus",
    method: str,
    scorer: "SemanticScorer",
    cfg: "Config",
) -> dict[int, CaseFacts]:
    """Build a :class:`CaseFacts` per question for one method.

    Scores are computed over the union of gold and predicted elements that
    actually exist in the schema — scoring every element of every schema
    would be wasteful, and hallucinated predictions must be screened out
    first: :func:`~schema_linking.erroranalysis.scoring.lexical_scores`
    raises ``KeyError`` for an element absent from the schema.
    """
    from schema_linking.erroranalysis.facts import build_case_facts
    from schema_linking.erroranalysis.scoring import lexical_scores

    out: dict[int, CaseFacts] = {}
    for example in corpus.examples:
        qid = example.question_id
        schema = corpus.schemas[example.db_id]
        bare = build_case_facts(
            question_id=qid,
            question=example.question,
            gold_sql=example.query,
            schema=schema,
            gold_tier1_raw=corpus.gold_tier1[qid],
            gold_tier2_raw=corpus.gold_tier2[qid],
            predicted_raw=corpus.predictions[method][qid],
            hardness=corpus.hardness[qid],
            index=corpus.indices[example.db_id],
        )
        scorable = sorted(
            {
                el
                for el in (bare.gold_tier1 | bare.gold_tier2 | bare.predicted)
                if el.level == "table"
                and el.table in bare.index.tables
                or el.level == "column"
                and el in bare.index.columns
            },
            key=lambda e: (e.level, e.render()),
        )
        out[qid] = build_case_facts(
            question_id=qid,
            question=example.question,
            gold_sql=example.query,
            schema=schema,
            gold_tier1_raw=corpus.gold_tier1[qid],
            gold_tier2_raw=corpus.gold_tier2[qid],
            predicted_raw=corpus.predictions[method][qid],
            hardness=corpus.hardness[qid],
            index=corpus.indices[example.db_id],
            lexical_scores=lexical_scores(example.question, scorable, schema),
            semantic_scores=scorer.score(example.question, scorable),
        )
    return out


def build_census(
    corpus: "Corpus",
    scorer: "SemanticScorer",
    cfg: "Config",
    methods: Sequence[str] = METHODS,
) -> pd.DataFrame:
    """Enumerate the full census for a corpus.

    Parameters
    ----------
    corpus
        Everything loaded by :func:`schema_linking.erroranalysis.loading.load_corpus`.
    scorer
        Semantic scorer, e.g. :class:`~schema_linking.erroranalysis.scoring.NullSemanticScorer`
        or :class:`~schema_linking.erroranalysis.scoring.EmbeddingSemanticScorer`.
    cfg
        Project configuration, forwarded to :func:`build_facts`.
    methods
        Methods to census. Defaults to all six.

    Returns
    -------
    pandas.DataFrame
        One row per error instance, with :data:`CENSUS_COLUMNS`.
        ``cause`` is empty — see :func:`rules.classify_census`.
    """
    frames = []
    for method in methods:
        facts_by_qid = build_facts(corpus, method, scorer, cfg)
        errors = [
            err
            for tier in TIERS
            for facts in facts_by_qid.values()
            for err in enumerate_errors(facts, method, tier)
        ]
        frames.append(errors_to_frame(errors, facts_by_qid))
    frame = pd.concat(frames, ignore_index=True)
    return add_schema_size_bin(frame)
