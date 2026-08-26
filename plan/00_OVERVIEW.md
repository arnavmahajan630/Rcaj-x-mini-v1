# RCAJ-X Model Evaluation Build — Agent Overview

## Mission
This is **not** a hackathon-deadline build. It is a genuine comparative research evaluation. A teammate is independently building a different NLP-based grading approach. The purpose of this build is to construct RCAJ-X at a smaller scale, benchmark it heavily and honestly, generate human-readable score explanations, and produce results that can be directly compared against the teammate's approach using the same test set and metrics — so that whichever architecture performs better is the one carried forward into the ZK-proving build and the final pitch. Treat every metric and report produced here as a real decision input, not a demo talking point — do not round favorably, do not omit weak results.

Scope: build the RCAJ-X model, its explainability/reasoning layer, a lightweight manual-testing UI, and the comparison protocol against the teammate's model. No ZK proving, no EZKL, no ONNX circuit work in this scope — that decision comes after the comparison in `09_comparative_evaluation_vs_alternative_approach.md` is complete. See `06_definition_of_done.md` for what "done" means and the end of this file for the full exclusion list.

## Read Order
Read and execute in this order. Do not skip ahead — each file assumes the previous one's checkpoint passed.
1. `00_OVERVIEW.md` (this file)
2. `01_dataset_and_benchmark_protocol.md`
3. `02_preprocessing_pipeline.md`
4. `03_model_architecture.md`
5. `04_training_pipeline.md`
6. `05_benchmarking_and_reporting.md`
7. `07_explainability_and_reasoning.md`
8. `08_streamlit_test_app.md`
9. `09_comparative_evaluation_vs_alternative_approach.md`
10. `10_EXCLUSIONS.md` — read alongside `06`, documents every deliberate scope reduction and its expected quality impact
11. `06_definition_of_done.md` — self-verify against this before declaring the task complete

## The Core Benchmark Design Principle — Read This Carefully

This is the single most important constraint in the whole plan, specified by the project owner directly, and every other file is written to serve it:

> Train the model on a fixed set of **(rubric, question)** pairs using one style/distribution of answers ("X-type answers"). Benchmark it on the **same rubrics and same questions**, but with **unseen answer variations** ("Y-type answers" — a deliberate variation of X-type: different phrasing, scattered evidence, typos, negation flips, padding, etc.) that the model never saw during training.

This means, critically:
- **Do not split train/test by rubric or by question.** Every rubric/question used in training must *also* appear in the test set — only the specific answer instances differ.
- The test set is not a random held-out slice of the same distribution. It is **deliberately out-of-distribution on the answer side, in-distribution on the rubric/question side.** This directly tests whether the model generalizes to how a real, unpredictable student might phrase a correct or incorrect answer to a question it has already been calibrated on — which is the realistic deployment scenario (a certified exam has fixed questions; it does not have fixed student answers).
- Build the data generation step (`01_dataset_and_benchmark_protocol.md`) to produce train and test answers as **explicitly paired variations of each other**, not independently sampled — so every test example has a traceable "which X-type answer was this Y-type variation derived from" link, enabling direct before/after comparisons per pair.

## Project Folder Structure to Create

```
rcaj-x/
├── data/
│   ├── raw/                     # source rubric/question definitions
│   ├── train/                   # X-type answers, paired to rubrics
│   ├── test/                    # Y-type answer variations, same rubrics/questions
│   ├── dataset_manifest.json    # traceability: test_id -> train_id it was derived from, variant_type
│   └── friend_model_predictions.csv   # teammate's model output on the SAME test set (see 09)
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── benchmark.py
│   ├── report.py
│   ├── explain.py               # score-reasoning / evidence generation (07)
│   └── compare.py                # head-to-head comparison vs. teammate's model (09)
├── checkpoints/
│   └── rcaj_x_best.pt
├── results/
│   ├── benchmark_results.csv
│   ├── benchmark_report.md
│   ├── ablation_results.csv
│   ├── explanations_sample.json
│   └── comparison_report.md
├── app.py                        # Streamlit manual-testing UI (08)
└── requirements.txt
```

## Environment
```
python 3.11
torch>=2.2
sentence-transformers
scikit-learn
pandas
numpy
spacy (+ en_core_web_sm)
streamlit
```
Install and confirm all imports succeed before proceeding to `01_dataset_and_benchmark_protocol.md`.

## Non-Negotiable Constraints for the Agent
- Do not implement ZK/EZKL/ONNX export in this scope. That decision is downstream of `09_comparative_evaluation_vs_alternative_approach.md`, not part of this build.
- Do not implement encoder fine-tuning in this scope — use frozen, off-the-shelf `BAAI/bge-small-en-v1.5`. This is a deliberate, named exclusion, accepted for this round of evaluation. `06_definition_of_done.md` and the accompanying notes quantify the expected quality delta this leaves on the table, so the decision can be revisited with eyes open once results are in.
- Do not silently deviate from the train/test protocol described in `01_dataset_and_benchmark_protocol.md`. If the data generation approach seems to conflict with any convenience shortcut (e.g., random splitting), the protocol file wins.
- Every benchmark number reported in `05_benchmarking_and_reporting.md` and `09_comparative_evaluation_vs_alternative_approach.md` must be computed on the Y-type (unseen answer variation) test set — never report a number computed on training data as if it were a benchmark result.
- **Explainability is a required deliverable, not optional polish.** Every score the model produces must be paired with a human-readable reason — see `07_explainability_and_reasoning.md`. This is one of the two things this evaluation is explicitly testing (the other being raw accuracy/robustness), since it is likely to be a deciding factor against the teammate's approach if raw accuracy ends up close.
