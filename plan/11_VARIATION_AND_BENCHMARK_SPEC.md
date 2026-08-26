# 11 — Master Variation & Benchmark Specification
## Single-File Consolidated Reference: Every Required Answer Variation, Every Benchmark Check, One Worked Example

This file exists so the agent (or anyone reviewing the build) doesn't have to cross-reference `01`, `02`, `04`, `05`, `07`, `08`, `09` to find every variation type and every test that's required somewhere in the plan. Everything is repeated here in one place, consolidated. If anything here conflicts with an earlier file, the earlier file's detailed instructions win — this is a summary/index, not a replacement.

---

## Section A — The 8 Required Answer Variation Types (Full Spec)

Every variation type below must appear in `data/test/` (Y-type), each as a traceable transformation of a `data/train/` (X-type) source answer, per `01_dataset_and_benchmark_protocol.md`. This table is the authoritative definition — use it directly when generating data.

| # | `variant_type` | Transformation Rule | Generation Guidance | Expected Score Direction (vs. X-type source) | Expected Spread Behavior |
|---|---|---|---|---|---|
| 1 | `paraphrase` | Same meaning, fully different wording/sentence structure | Rewrite the source answer using different vocabulary and sentence order while preserving all factual content | ~No change | Low (concentrated) |
| 2 | `scattered_evidence` | Same correct content, split across 2+ non-adjacent sentences with unrelated filler between them | Take a single-sentence correct answer, break its content into two separate sentences, insert 1–2 filler/unrelated sentences between them | ~No change (should still get credit) | Low-to-moderate; attention should combine both chunks |
| 3 | `diffuse_padded` | Same correct content, buried inside a long, mostly-irrelevant answer | Take a correct answer, surround it with 3–5 sentences of plausible-sounding but off-topic padding | ~No change (should not be penalized for length, should not be inflated either) | Low on the truly relevant chunk; test that padding doesn't drag spread up artificially |
| 4 | `partial_credit_shift` | A full-credit answer degraded to address only part of a compound criterion | For any compound criterion ("X and Y"), remove or omit the second half | Reduced, proportional to omitted content | Low (confident partial match) |
| 5 | `negation_flipped` | Explicit or implicit negation inserted, flipping correctness while remaining fluent | Insert "not"/"never"/"fails to" etc., or flip a comparative ("increases"→"decreases") | Strongly reduced | Low (confident but wrong — this is the key signature to check) |
| 6 | `confidently_wrong` | Fluent, on-topic, factually altered — no negation word used | Change a key fact/number/term to something plausible-sounding but incorrect | Strongly reduced | Low (confident but wrong, same signature as negation — validates the "safe to auto-reject" case) |
| 7 | `typo_injected` | 1–2 character typos injected into the criterion-critical term(s) only | Take a correct answer, misspell the key technical term (transpose/drop/substitute 1-2 characters) | ~No change (small negative at most — graceful degradation) | Low, should remain low after spelling normalization |
| 8 | `genuinely_ambiguous` | Deliberately vague phrasing a reasonable human grader could read as partial credit either way | Write an answer that gestures at the right idea without committing to specifics | Variable/moderate | **High** (this is the one category where high spread is the correct, desired outcome) |

**Minimum required volume per type:** at least 5–8 examples per `variant_type` per question, per `01_dataset_and_benchmark_protocol.md` — aim for 150–300 total Y-type examples across all types and all questions.

---

## Section B — Master Benchmark & Test Checklist (Consolidated Across All Files)

Every check below must pass before the build is considered complete. Source file is listed for traceability back to full instructions.

| # | Source File | Check | What It Verifies | Pass Criteria |
|---|---|---|---|---|
| 1 | `01` | Rubric/question overlap | Every `question_id` in test also exists in train | 100% overlap, no orphaned test question_ids |
| 2 | `01` | Manifest traceability | Every test example has a valid `derived_from_train_id` | 100% resolve to a real train file |
| 3 | `01` | No duplication | No test answer is a verbatim copy of its paired train source | 0 exact-text duplicates |
| 4 | `01` | Human-scored ground truth | Scores are human-assigned for both train and test, never LLM-assigned | Manual confirmation |
| 5 | `02` | Embedding shape | `R.shape == (n_criteria, 384)`, `A.shape == (n_chunks, 384)` | Confirmed on ≥3 sample examples |
| 6 | `02` | Spelling normalization | Typos on glossary terms are corrected pre-embedding | Verified on ≥2 `typo_injected` examples |
| 7 | `02` | Levenshtein tie-break | Ambiguous (tied-distance) words are left un-normalized, logged | Confirmed no arbitrary tie-break occurs |
| 8 | `03` | Softmax normalization | `attn_weights.sum(dim=-1)` is all-ones per criterion per head | Unit test passes |
| 9 | `03` | Output shapes | `per_criterion_scores.shape == (n_criteria,)`, `final_score` is scalar | Unit test passes |
| 10 | `03` | Head-count configurability | Model instantiates correctly for `n_heads ∈ {2, 4, 6}` | Unit test passes |
| 11 | `04` | Hyperparameter sweep completeness | All grid configs (12 total: 3 heads × 2 d_k × 2 lr) logged | `ablation_results.csv` has 12 rows |
| 12 | `04` | Beats baseline | Best validation loss beats "predict mean training score" baseline | Both numbers reported |
| 13 | `04` | Train/val isolation | Y-type test set never used for model selection or early stopping | Code review confirms no test-set reads in `train.py` |
| 14 | `05` | Per-category accuracy | MAE, % within 1 mark computed and reported for **every** `variant_type` | All 8 types appear in `benchmark_report.md` |
| 15 | `05` | Precision/recall/FP/FN | Computed on binary "criterion met" framing, overall and per-category | Reported in `benchmark_report.md` |
| 16 | `05` | Paired-delta analysis | Score delta between each Y-type answer and its X-type source, per `variant_type` | Actual vs. expected direction (Section A table above) stated for every category |
| 17 | `05` | Mismatch transparency | Any category where actual results contradict expected direction is explicitly flagged | No silent omission |
| 18 | `07` | Evidence grounding | Every `evidence_chunk` in a generated explanation is verbatim present in the source answer's chunks | `check_evidence_grounding` passes 100% on sample set |
| 19 | `07` | Confidence consistency | The `confidence` label matches the underlying spread value | `check_confidence_consistency` passes 100% |
| 20 | `07` | Explanation coverage | Explanations generated for examples spanning all `variant_type` categories | `explanations_sample.json` covers all 8 types |
| 21 | `07` | Manual sensibility check | Explanation text reads as sensible to a human for `negation_flipped`, `genuinely_ambiguous`, `scattered_evidence`, and one `fully_correct`-derived example | Manual review confirms |
| 22 | `08` | UI functional check | `streamlit run app.py` launches, grading produces score + reason + evidence | Manual run confirms |
| 23 | `08` | Preset example walkthrough | A `negation_flipped` and a `scattered_evidence` preset both grade correctly through the UI | Manual run confirms |
| 24 | `09` | Shared test set | Teammate's predictions collected on the **identical** `data/test/` set, no substitutions | File-level confirmation |
| 25 | `09` | Identical metrics | Same metric functions applied to both models, no divergent logic | Code review confirms reuse from `05` |
| 26 | `09` | Full category coverage in comparison | Every `variant_type` appears in the head-to-head comparison table | All 8 types present in `comparison_report.md` |
| 27 | `09` | Explainability comparison stated | Report explicitly states whether teammate's approach has a grounded explanation mechanism | Stated, not skipped |
| 28 | `09` | No code coupling | `src/compare.py` contains only CSV read/merge/metric logic, no network calls or teammate-codebase imports | Code review confirms |
| 29 | Final (`06`) | End-to-end reproducibility | Full pipeline re-runs cleanly from a fresh checkpoint directory | Manual re-run confirms |
| 30 | Final (`06`) | Test-set isolation | No pipeline step reads `data/test/` before the benchmarking step | Code review confirms |

---

## Section C — One Fully Worked Example (Template to Replicate Across All Rubrics)

Use this as the concrete template when generating data for every question in `data/raw/rubrics.json` — this is what "8 variants of one X-type answer" should actually look like end to end.

**Question:** *"Explain why plants appear green."*
**Criterion:** `c1` — "Correctly attributes the green color to chlorophyll reflecting/not absorbing green light" (max_marks: 2)

**X-type source answer** (`plants_train_004`):
> "Chlorophyll in plant cells absorbs red and blue light for photosynthesis but reflects green light, which is why plants appear green to us." — **human score: 2/2**

| `variant_type` | Y-type answer text | Expected score | Expected spread |
|---|---|---|---|
| `paraphrase` | "The green pigment chlorophyll doesn't absorb green wavelengths — it bounces them back, and that reflected light is what makes leaves look green." | 2/2 | Low |
| `scattered_evidence` | "Plants have many pigments involved in growth. Chlorophyll is one of them and it's essential for photosynthesis. It happens to reflect green light rather than absorbing it, which is the reason we see plants as green." | 2/2 | Low-moderate |
| `diffuse_padded` | "Plants are fascinating organisms that have evolved over millions of years. They need sunlight, water, and nutrients to survive. Different species have different leaf shapes and sizes. Chlorophyll, found in their cells, reflects green light instead of absorbing it, which is why they look green. Photosynthesis also produces oxygen as a byproduct, which is important for animal life." | 2/2 | Low on relevant chunk |
| `partial_credit_shift` | "Plants are green because of chlorophyll." (mechanism/reason omitted) | ~1/2 | Low |
| `negation_flipped` | "Chlorophyll absorbs green light, which is why plants appear green." (factually inverted via negation of the correct relationship) | 0/2 | Low (confident but wrong) |
| `confidently_wrong` | "Plants appear green because of a pigment called xanthophyll, which is the dominant pigment in most leaf cells." (fluent, on-topic, wrong fact, no negation word) | 0/2 | Low (confident but wrong) |
| `typo_injected` | "Chlorphyll in plant cells absorbs red and blue light but reflects gren light, which is why plants appear green." | ~2/2 (after normalization) | Low |
| `genuinely_ambiguous` | "It has something to do with the pigments in the leaves interacting with sunlight in a certain way." | ~0.5–1/2 | **High** |

Replicate this 8-row pattern for every criterion of every question in `data/raw/rubrics.json`, scoring each row yourself per `01_dataset_and_benchmark_protocol.md`'s hard constraint (human-scored, never LLM-scored).

---

## Quick-Use Summary

- **8 variant types, minimum 5–8 examples each per question** → data generation target.
- **30 checks across the whole build** → the master checklist in Section B; nothing is "done" until all 30 are confirmed, matching `06_definition_of_done.md`.
- **One worked example** → the template in Section C to replicate consistently across every rubric so variation quality doesn't drift between questions.
