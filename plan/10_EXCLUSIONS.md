# 10 — Exclusions: What This Build Deliberately Leaves Out of the Full RCAJ-X Architecture

This file exists because these exclusions were previously only stated in conversation with the project owner, not written into any plan file — the agent building this system needs them documented here directly, not assumed from context it doesn't have access to. Every item below is a **deliberate, accepted scope reduction for this evaluation round**, not an oversight. If a future task asks to close one of these gaps, treat it as new work requiring its own plan, not an implicit extension of files `00`–`09`.

## 1. No ZK / EZKL / ONNX proving
This build is model-only. No circuit compilation, no proof generation, no ONNX export for proving purposes. That work only begins after the comparison in `09_comparative_evaluation_vs_alternative_approach.md` produces a decision on which architecture to carry forward.

## 2. No encoder fine-tuning
Using frozen, off-the-shelf `BAAI/bge-small-en-v1.5` as-is. The full architecture calls for contrastive fine-tuning on `(criterion, correct_chunk, hard_negative_chunk)` triplets, including deliberately-constructed negation-flipped hard negatives. This is skipped here.

**Expected impact if added later:** meaningful but not transformative. Fine-tuning on your own rubric-answer distribution should sharpen separation specifically within your subject vocabulary and improve handling of domain-specific paraphrase — expect the largest gains in the `paraphrase` and `confidently_wrong` benchmark categories. It will not fix cross-domain generalization or reasoning-heavy correctness on its own.

## 3. Simplified negation checker
`negation_mismatch_flag` (in `02_preprocessing_pipeline.md`) is a flat keyword-list check, not the dependency-parse-based rule set (checking `neg` dependency relations attached to the shared predicate) described in the full architecture.

**Expected impact if added later:** narrow and category-specific — should show up almost entirely in the `negation_flipped` benchmark category, and specifically on **implicit** negation (no explicit negation word — e.g. "lacks," "fails to," a flipped comparative like "decreases" vs "increases"). If your `negation_flipped` test examples lean toward explicit negation words, the keyword version already catches most of them and the gap is small. If they lean implicit, expect a real, visible improvement confined to that one category.

## 4. No entropy vs. spread comparison
Only the `spread = max(w) - mean(w)` ambiguity proxy is implemented. The full architecture calls for comparing this against true attention entropy (`-Σw·log(w)`) and picking whichever separates ambiguous from confident cases more cleanly, with entropy only usable in a proved model if EZKL's `log` op is confirmed supported.

**Expected impact if added later:** likely small for this evaluation round specifically, since ZK-provability isn't in scope yet — entropy might separate categories marginally better, but spread is the safe, provable-either-way default and this is not expected to be a major accuracy driver either way.

## 5. No confidence-routing threshold calibration or review-queue workflow
`07_explainability_and_reasoning.md` produces a `confidence` label (`high_confidence` / `review_recommended`) per criterion using a fixed default `spread_threshold`, but there is no calibration step tuning this threshold against a validation set, and no actual teacher-review-queue system built around it.

**Expected impact if added later:** primarily affects how well-calibrated the *rate* of flagged answers is (e.g. "15% flagged" vs. "40% flagged") — not the underlying accuracy of the score itself. Worth doing once the comparison in `09` is complete and RCAJ-X (if chosen) moves toward a real deployment story.

## 6. Small, handcrafted/synthetic dataset, not full real-dataset scale
Data comes from the handcrafted, deliberately-stratified generation process in `01_dataset_and_benchmark_protocol.md` (150–300 train, 150–300 test), not the full ASAP-SAS/Mohler/CBSE-ICSE-scale datasets described in the complete architecture plan.

**Expected impact if added later:** more data generally helps, but the handcrafted set is deliberately *stratified* by failure mode in a way a large, unstratified real dataset isn't — for this specific evaluation's purpose (does the architecture handle ambiguity/negation/scattered-evidence correctly), the current dataset design is arguably more informative per example than raw scale would be. Real-dataset scale matters more once accuracy needs to generalize across a much wider variety of real student phrasing than a hand-built set can cover.

## 7. No certification-integrity layer
No reproducible-build pipeline, no multi-party sign-off, no transparency log, no golden-set regression testing. This entire layer exists to solve the "was the certified model the legitimate one" problem, which only matters once there's a certified, proved model to protect — not relevant until after `09`'s decision and the ZK-integration phase.

## 8. No stack integration
Nothing is wired into the existing Gateway / audit-chain / frontend from the original CertiProof system. This build is a standalone model, explanation layer, and Streamlit test console — not a deployed service.

## 9. No code-level integration with the teammate's model
Per explicit instruction: RCAJ-X and the teammate's NLP approach are built completely independently, by different people, with no shared codebase. The only exchange point is a manually-produced CSV of predictions on the shared test set (`09_comparative_evaluation_vs_alternative_approach.md`). Do not build an adapter, HTTP client, or any runtime dependency on the teammate's system.

## What IS in scope, for clarity (not an exclusion — stated here to avoid ambiguity)
Explainability/score-reasoning (`07`) and the Streamlit manual-testing console (`08`) are **in scope** for this build, despite being deferred in an earlier version of this plan — they were added back in because grounded score explanation is one of the two things this evaluation is explicitly testing (alongside raw accuracy/robustness), and is likely to matter directly in the comparison against the teammate's approach.
