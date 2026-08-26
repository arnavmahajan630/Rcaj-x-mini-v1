# RCAJ-X Mini v1 — Hardening & Data Upgrade Plan
## Agent Overview

## Context
`Rcaj-x-mini-v1` (https://github.com/arnavmahajan630/Rcaj-x-mini-v1) is built and runs. Confirmed structure matches spec: `src/model.py` (MultiHeadCrossAttention + ScoringHead), `src/preprocessing.py`, `src/train.py`, `src/explain.py`, `src/benchmark.py`, `src/compare.py`, `app.py` (Streamlit), `datainj/` (raw reviewed data), `data/` (embedded tensors), `results/`, `checkpoints/`.

Known issue reported by the project owner: trained on LLM-synthetically-generated data, results are broadly reasonable but **scores sometimes exceed the criterion's max_marks, or don't track the model's own stated reasoning** — i.e. the explanation correctly identifies a partial/flawed answer, but the numeric score doesn't reflect that judgment consistently. This was reproduced structurally in a prior debugging session: the `ScoringHead`'s final `nn.Linear(hidden, 1)` layer has **no output bounding**, so nothing constrains predictions to `[0, max_marks]` — this is the most likely primary cause and the first fix in this plan.

## Hardware Constraint (governs every "how much" decision below)
**Target machine: i5-13400H + RTX 3050 (laptop, likely 4GB VRAM).** This is real but modest — enough for full fine-tuning of BGE-small (33M params) in minutes, not enough for large-batch training of anything bigger, and not a machine to run multi-hour sweeps on. Every recommendation in this plan is scoped to **demo timeframe** (a focused work session, not unlimited time) — see `03_rubric_locking_and_finetuning_plan.md` for the specific budget breakdown. Do not scale any recommendation here up to the "Complete Build Plan" scope from earlier planning (full ASAP-scale datasets, multi-day hyperparameter sweeps, multi-party certification) — that's out of scope for this hardening pass.

## The Four Things This Plan Addresses (mapped to the project owner's explicit asks)
1. **Better data, 5 subjects, high volume and quality, short AND long paragraph answers** → `02_data_strategy_5_subjects.md`
2. **Locked vs. manually-entered rubrics — which performs better** → `03_rubric_locking_and_finetuning_plan.md` (Part A)
3. **Fine-tuning / negation hardening, feasible on this specific GPU, in demo time** → `03_rubric_locking_and_finetuning_plan.md` (Part B)
4. **Sane scores for arbitrary live evaluator input, not just curated benchmark answers** → `04_eval_input_sanity_and_definition_of_done.md`

Plus the standing bug: **`01_score_bounding_and_consistency_fix.md`** — fix this first, before anything else, since it likely explains a real chunk of the "results are fine but sometimes wrong" symptom, and every other fix in this plan is easier to evaluate once scores are actually bounded and trustworthy.

## Read Order
1. `00_OVERVIEW.md` (this file)
2. `01_score_bounding_and_consistency_fix.md` — fix first, it's cheap and unblocks honest evaluation of everything else
3. `02_data_strategy_5_subjects.md`
4. `03_rubric_locking_and_finetuning_plan.md`
5. `04_eval_input_sanity_and_definition_of_done.md` — also the final self-check

## What This Plan Does NOT Cover
ZK/EZKL integration into CertiProof — that's the separate `certiproof-integration-plan/` file set, which assumes this hardening pass is complete first. Do not start on EZKL circuit work from this repo until the model itself is fixed, retrained, and benchmarked well per this plan.
