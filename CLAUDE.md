# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RCAJ-X Mini: a lightweight cross-attention PyTorch model that grades student answers against per-question rubric criteria and produces grounded, evidence-linked explanations. Designed to stay small/bounded enough for eventual ZK-circuit compilation (EZKL) — this repo is the model/data hardening pass, not the ZK integration (see "Hardening plan" below).

## Setup & commands

```bash
source venv/bin/activate          # venv already exists in repo root
pip install -r requirements.txt
```

Pipeline order (each stage reads the previous stage's output from disk — there is no single entrypoint):

```bash
python scripts/ingest_data.py     # datainj/*.json -> data/train, data/test, data/raw/rubrics.json, data/dataset_manifest.json
python src/preprocessing.py       # data/{train,test} -> data/{train,test}_embedded.pt (chunking, spellcheck, BGE embeddings)
python src/train.py               # data/train_embedded.pt -> checkpoints/rcaj_x_best.pt + results/ablation_results.csv
python src/benchmark.py           # checkpoints + data/test_embedded.pt -> results/benchmark_results.csv + results/benchmark_report.md
python src/compare.py             # results/benchmark_results.csv vs data/friend_model_predictions.csv -> results/comparison_report.md
python src/explain.py             # sample explanations -> results/explanations_sample.json (also self-checks evidence grounding & confidence consistency)
streamlit run app.py              # interactive grading console at http://localhost:8501
```

Tests:
```bash
pytest tests/                     # all tests
pytest tests/test_model.py::test_output_shapes   # single test
```

`data/*_embedded.pt`, `checkpoints/`, and `results/` are gitignored — they're regenerated locally by the pipeline above, not committed.

## Architecture

**Core model (`src/model.py`)**: `RCAJ_X` = `MultiHeadCrossAttention` + `ScoringHead`.
- Input: `R` (n_criteria × 384, rubric criterion embeddings) as queries, `A` (n_chunks × 384, answer sentence-chunk embeddings) as keys/values.
- Cross-attention produces per-criterion context vectors plus per-head attention weights `(h, n_c, n_a)`.
- `ScoringHead` takes `[context, per-head spread, negation_flag]` per criterion and outputs a raw (currently **unbounded**) per-criterion score via an MLP.
- Everything is per-example (no batch dimension) — one forward pass grades one answer against one rubric's full criteria list.

**Data flow contract** — this shape is threaded through every script and is not obvious from any single file:
- Raw reviewed data lives in `datainj/` (not shown above, upstream of this repo's `data/`) and is copied/normalized into `data/{train,test}/*.json` + `data/raw/rubrics.json` + `data/dataset_manifest.json` by `ingest_data.py`.
- `preprocessing.py` chunks each answer into sentences (spaCy), normalizes glossary terms via Levenshtein distance to rubric vocabulary, embeds both criteria and chunks with `BAAI/bge-small-en-v1.5`, and computes a `negation_flags` tensor — **against `chunks[0]` only**, not the model's actual top-attended chunk, because the flag must be computed before the model runs (see the inline comment in `preprocessing.py:preprocess_dataset` for the full reasoning — this is a known approximation, not a bug to silently "fix").
- Output is one `.pt` file per split (`data/train_embedded.pt`, `data/test_embedded.pt`), a list of dicts each carrying `R`, `A`, `chunks`, `negation_flags`, `human_scores`, and identifiers (`answer_id`, `question_id`, `variant_type` for test, `derived_from_train_id` linking test variants back to their source train example).
- `train.py` grid-searches `{n_heads, d_k, lr, weight_decay}`, keeps the best val-loss state dict, and saves it with its config into `checkpoints/rcaj_x_best.pt` — `benchmark.py`, `explain.py`, and `app.py` all reconstruct `RCAJ_X` from that saved config, so model hyperparameters are not hardcoded anywhere downstream of training.
- `benchmark.py` evaluates per test example, and also computes a **paired delta**: predicted score minus the predicted score of the train example that test row was derived from (`derived_from_train_id`) — this isolates how much a specific stress-test perturbation (paraphrase, negation flip, typo, etc.) moved the model's score, independent of the model's baseline accuracy on that question.
- Stress-test variant types (`variant_type` field) are the evaluation axis throughout `benchmark.py`, `report.py`, and `compare.py`: `paraphrase`, `scattered_evidence`, `diffuse_padded`, `partial_credit_shift`, `negation_flipped`, `confidently_wrong`, `typo_injected`, `genuinely_ambiguous`. `report.py` has hardcoded expected-direction heuristics per variant type (e.g. `negation_flipped` should show a strong negative score delta) used to flag likely regressions.
- `explain.py` re-runs the model per example to attach grounded explanations: top-2 attended chunks per criterion, an ambiguity flag from attention spread (`mean_spread < 0.4`), and a negation-mismatch flag — then self-checks that evidence chunks actually appear in the raw answer (`check_evidence_grounding`).
- `compare.py` benchmarks against `data/friend_model_predictions.csv` (auto-generates a mock if absent) to produce a comparative report against a baseline/teammate approach.

## Known issue (see `rcaj-x-hardening-plan/`)

`ScoringHead`'s final layer (`src/model.py`) has **no output bounding** — nothing constrains a predicted score to `[0, max_marks]`, and this is the primary suspected cause of scores that exceed max marks or don't track the model's own stated explanation. `rcaj-x-hardening-plan/01_score_bounding_and_consistency_fix.md` is the fix-first task; the other numbered files in that directory (`02`–`04`) cover data strategy, rubric-locking/fine-tuning, and live-input sanity checks, in that read order per `00_OVERVIEW.md`. ZK/EZKL circuit integration is explicitly out of scope for this repo/plan.
