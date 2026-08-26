# 06 — Definition of Done

Self-verify against every item below before declaring the task complete. If any item fails, go back to the relevant plan file and fix it — do not report completion with known-failing items.

## Data
- [ ] `data/raw/rubrics.json` exists with 8–12 questions, 2–4 criteria each, spanning ≥3 subjects.
- [ ] `data/train/` contains 150–300 human-scored X-type answers.
- [ ] `data/test/` contains 150–300 human-scored Y-type answers, covering all 8 required `variant_type` categories.
- [ ] Every `question_id` in `data/test/` also appears in `data/train/`.
- [ ] Every test example has a valid `derived_from_train_id` resolving to a real train example, recorded in `data/dataset_manifest.json`.
- [ ] No test answer is a verbatim duplicate of its paired train source.

## Preprocessing
- [ ] `data/train_embedded.pt` and `data/test_embedded.pt` both exist and load without error.
- [ ] Spelling normalization verified on at least 2 `typo_injected` examples.
- [ ] Negation flags computed for all examples in both splits.

## Model
- [ ] All three unit tests in `03_model_architecture.md` pass.
- [ ] `RCAJ_X` is parameterized by `n_heads`, `d_k`, `d_v` — confirmed by successfully instantiating with at least 2 different configs.

## Training
- [ ] `results/ablation_results.csv` contains all 12 grid configurations with logged validation loss.
- [ ] `checkpoints/rcaj_x_best.pt` exists, loads, and its validation loss beats the "predict mean training score" baseline (both numbers reported).
- [ ] Training and validation splits are drawn only from `data/train/` — the Y-type test set was never used for model selection or early stopping.

## Benchmarking
- [ ] `results/benchmark_results.csv` has one row per Y-type test example.
- [ ] `results/benchmark_report.md` includes: overall accuracy, per-`variant_type` accuracy table, per-`variant_type` precision/recall/FP/FN, paired-delta table with actual-vs-expected direction for every category, at least 3 concrete demo-usable example rows, and an explicit "weakest categories" section.
- [ ] Every one of the 8 required `variant_type` categories appears in the final report — none silently dropped or merged.
- [ ] Any category where actual results contradicted the expected direction (see `05_benchmarking_and_reporting.md` Step 5 table) is explicitly called out in the report, not smoothed over.

## Explainability
- [ ] `src/explain.py` generates a grounded, template-based reason for every score, referencing actual attention weights, spread, and negation flags — not generic filler text.
- [ ] `results/explanations_sample.json` covers all `variant_type` categories.
- [ ] `check_evidence_grounding` and `check_confidence_consistency` both pass 100% over the sample set.
- [ ] Manual review confirms explanations for `negation_flipped`, `genuinely_ambiguous`, `scattered_evidence`, and a `fully_correct`-derived example all read as sensible to a human.

## Streamlit Test UI
- [ ] `streamlit run app.py` launches without error.
- [ ] Selecting a rubric, entering an answer, and grading produces a per-criterion score, confidence badge, reason text, and evidence chunks.
- [ ] Preset test examples can be loaded by `variant_type` and graded live.

## Comparative Evaluation vs. Teammate's Approach
- [ ] `data/friend_model_predictions.csv` collected on the identical `data/test/` set as RCAJ-X.
- [ ] `results/comparison_report.md` contains the full decision table and a written, non-hedging verdict.
- [ ] Every `variant_type` category appears in the per-category comparison for both models.
- [ ] The report explicitly states whether the teammate's approach has any grounded, evidence-linked explanation mechanism, and whether either approach is meaningfully more/less ZK-provable at reasonable circuit cost.

## Final Sanity Pass
- [ ] Re-run the full pipeline once end-to-end (`data_generation.py` → `preprocessing.py` → `train.py` → `benchmark.py` → `report.py`) from a clean checkpoint directory to confirm reproducibility.
- [ ] Confirm no step in the pipeline reads from or writes to `data/test/` before `05_benchmarking_and_reporting.md` — this is the single most important process integrity check for this project, since the whole benchmark's credibility depends on the test set being genuinely unseen during training.

## Explicitly Out of Scope for This Build — Do Not Attempt
See `10_EXCLUSIONS.md`, which documents every deliberate scope reduction for this evaluation round along with the expected quality impact of each one if added later. If asked to extend into any of those areas, treat it as a new, separate task requiring a new plan, not an implicit extension of this one.
