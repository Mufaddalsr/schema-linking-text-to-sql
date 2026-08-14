# Schema Linking Thesis — `code/`

Implementation of the six schema-linking methods compared in the thesis
"Investigations into Schema Linking in Text-to-SQL". Scope and the
twelve-week plan live in [`docs/scope.md`](docs/scope.md). Implementation
decisions and gotchas live in [`docs/decisions.md`](docs/decisions.md).

## Layout

```
src/schema_linking/       library code (data loading, parsing, gold extraction, …)
tests/                    pytest suite — runs in <5s end-to-end
notebooks/                exploratory analyses (production logic lives in src/)
data/spider/              raw Spider release (train_spider.json, dev.json, tables.json)
data/spider_schema_linking/  Taniguchi et al.'s human Tier-1 annotations
data/processed/           extracted gold-link JSON (see below)
outputs/                  predictions, results CSVs, run logs
config.yaml               paths configurable from one place
```

## Data

### Raw inputs (read-only, never modified)

- `data/spider/train_spider.json` — 7000 examples (cross-domain Spider train).
- `data/spider/dev.json` — 1034 examples (Spider dev — the headline-reporting split).
- `data/spider/tables.json` — 166 database schemas (tables, columns, types, primary keys, foreign keys).
- `data/spider_schema_linking/data/schema-linking/splits/{dev,test}.jsonl` —
  Taniguchi et al.'s human Tier-1 character-span annotations. Two
  517-example halves of Spider dev (jointly 1034); their `.txt` siblings
  are BIO-tagged token sequences derived from the same data and are
  unused here. See `taniguchi_loader.py` and `docs/decisions.md` for the
  span-to-`(table, column)` resolution.

### Processed gold-link files

Four canonical files under `data/processed/`. The on-disk shape is the
same for all four: a JSON object keyed by stringified Spider question
id, each value
`{"db_id": str, "tables": list[str], "columns": list[[str, str]]}`.

| File                                | Split | Tier                | Source                | Use                                            |
| ----------------------------------- | ----- | ------------------- | --------------------- | ---------------------------------------------- |
| `gold_links_dev_mentioned.json`     | dev   | 1 (Mentioned)       | **Taniguchi** (human) | Headline Tier-1 evaluation on dev              |
| `gold_links_dev_all_sql.json`       | dev   | 2 (All-SQL-used)    | sqlglot               | Headline Tier-2 evaluation on dev              |
| `gold_links_train_mentioned.json`   | train | 1 (Mentioned)       | sqlglot               | Few-shot selection, threshold tuning (no human Tier-1 exists for train) |
| `gold_links_train_all_sql.json`     | train | 2 (All-SQL-used)    | sqlglot               | Few-shot selection, threshold tuning           |

`gold_link_extractor.extract_tier1` / `extract_tier2` produce the
sqlglot files from `(SpiderExample, Schema)` pairs.
`taniguchi_loader.to_gold_links` produces the human dev Tier-1 file by
joining Taniguchi spans to Spider dev questions (text-first; falls back
to Taniguchi's contiguous `id - 3616 = qid` for the 46/1034 cases where
Taniguchi normalised typos / whitespace).

Train has no Taniguchi annotations and never contributes to reported
numbers — per [`docs/scope.md`](docs/scope.md), train is for threshold
tuning, few-shot selection, and sanity-only. sqlglot is the only Tier-1
source on train and that's fine.

### Cross-validation of the two Tier-1 sources

[`notebooks/02_gold_extraction_sanity.ipynb`](notebooks/02_gold_extraction_sanity.ipynb)
cross-validates sqlglot Tier-1 against Taniguchi on dev. Numbers go to:

- `outputs/results/tier1_validation.csv` — micro P / R / F1 / SRR, tables and columns.
- `outputs/results/tier1_validation_by_hardness.csv` — same, broken down by Spider easy / medium / hard / extra.

About 40 % of dev queries show some Tier-1 disagreement (almost all
join-bridge tables that Taniguchi marks as "mentioned" but strict-Tier-1
drops). The divergence-analysis subsection in the methodology chapter
will work from these CSVs.

## Running tests

```bash
.venv/bin/python -m pytest
```

Should be ≤5s on a laptop and finish green.

## Regenerating the processed files

Imports plus a few one-liners (no dedicated script — these are run
rarely and the call sites are stable):

```python
from pathlib import Path
from schema_linking.data_loader import load_spider_questions
from schema_linking.schema_parser import load_schemas
from schema_linking.taniguchi_loader import load_taniguchi_annotations, to_gold_links
from schema_linking.gold_link_extractor import (
    extract_tier1_all, extract_tier2_all, save_gold_links,
)

dev = load_spider_questions("dev")
train = load_spider_questions("train")
schemas = load_schemas()

# dev Tier-1 — Taniguchi
save_gold_links(
    to_gold_links(load_taniguchi_annotations(), dev, schemas),
    Path("data/processed/gold_links_dev_mentioned.json"),
)
# dev Tier-2 — sqlglot
save_gold_links(extract_tier2_all(dev, schemas),
                Path("data/processed/gold_links_dev_all_sql.json"))
# train Tier-1 / Tier-2 — sqlglot only
save_gold_links(extract_tier1_all(train, schemas),
                Path("data/processed/gold_links_train_mentioned.json"))
save_gold_links(extract_tier2_all(train, schemas),
                Path("data/processed/gold_links_train_all_sql.json"))
```
