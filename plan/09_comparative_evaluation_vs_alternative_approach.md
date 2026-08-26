# 09 — Comparative Evaluation vs. the Teammate's NLP Approach

## Objective
Compare RCAJ-X against the teammate's **independently and separately built** NLP grading approach, using the same benchmark (`data/test/`, the Y-type unseen-answer-variation set) and the same metrics, to jointly decide which architecture proceeds to ZK integration and the final pitch. This is a **manual, collaborative comparison, not a code-level integration.** RCAJ-X and the teammate's model are built entirely separately, by different people, with no shared codebase, no adapter, and no runtime dependency between them. The only shared artifact is the test set and its ground-truth scores, and the only exchange point is a results file. Treat the fairness of this comparison as more important than making RCAJ-X look good.

## Step 1 — Establish a Fair, Shared Evaluation Contract
Before running anything, confirm with the teammate:
- Both approaches are evaluated on the **identical** `data/test/` set — same files, same `human_scores` ground truth, no per-model test set substitutions. Share `data/test/` and `data/raw/rubrics.json` with the teammate directly (as files), since their model needs the same rubric/question inputs to be scored against the same criteria.
- Both approaches' outputs land in the same simple output schema, so results can be merged into one dataframe for comparison:
```json
{"answer_id": "q1_test_017", "question_id": "q1", "variant_type": "scattered_evidence",
 "pred_scores": {"q1_c1": 1.6, "q1_c2": 0.2}, "model": "rcaj_x"}
```
- If the teammate's approach cannot produce **per-criterion** scores (e.g. it only produces one holistic score), note this explicitly as a structural difference in the comparison report rather than forcing an artificial per-criterion breakdown onto it — an overall-score-only comparison is still valid, just recorded as a known asymmetry.

## Step 2 — Collect Teammate's Predictions (Manual Exchange Only)
The teammate runs their own model, independently, on their own system, against the shared `data/test/` set, and exports their predictions to `data/friend_model_predictions.csv` in the schema above. **Do not build an HTTP adapter, a shared API, or any code path in `src/compare.py` that calls into the teammate's model directly** — the two systems are built and run independently; the CSV file is the entire integration surface. `src/compare.py` only needs to **read** this CSV and merge it with RCAJ-X's own results — nothing more.

## Step 3 — Compute Identical Metrics for Both Models
Reuse the exact metric functions from `05_benchmarking_and_reporting.md` — do not write separate, slightly-different metric logic for the teammate's model, since any inconsistency here invalidates the comparison. For both models, compute:
- Overall MAE, % within 1 mark, Cohen's κ / quadratic weighted κ.
- Per-`variant_type` MAE and % within 1 mark (this is the core comparison — a model that's accurate on `fully_correct`-derived answers but falls apart on `negation_flipped` or `genuinely_ambiguous` should show that clearly here).
- Precision/recall/false-positive/false-negative on the binary "criterion met" framing, overall and per `variant_type`.
- **Explainability**: does the teammate's approach produce any grounded, evidence-linked explanation for its score? If yes, apply the same `check_evidence_grounding`-style test from `07_explainability_and_reasoning.md` to it. If no, record this as a straightforward structural gap in the comparison table — do not attempt to manufacture an explanation on the teammate's behalf.
- **Latency**: rough per-example inference time for both, gathered independently on each person's own machine and reported as-is (not normalized to identical hardware) — relevant if either approach is meaningfully slower, since that has real implications for later ZK-proving cost if RCAJ-X is chosen, or deployment cost either way. Note the hardware difference explicitly in the report rather than presenting it as an apples-to-apples number.

## Step 4 — Build the Decision Table
Produce `results/comparison_report.md` with a summary table structured as:

| Dimension | RCAJ-X | Teammate's Approach | Notes |
|---|---|---|---|
| Overall MAE | ... | ... | |
| % within 1 mark | ... | ... | |
| Cohen's κ | ... | ... | |
| Precision / Recall (criterion-met) | ... | ... | |
| Accuracy on `negation_flipped` | ... | ... | |
| Accuracy on `genuinely_ambiguous` | ... | ... | |
| Accuracy on `scattered_evidence` | ... | ... | |
| Accuracy on `typo_injected` | ... | ... | |
| Per-criterion score granularity | Yes | ... | |
| Grounded, evidence-linked explanation | Yes (attention-based) | ... | |
| Inference latency (per answer) | ... | ... | different hardware, not directly comparable |
| Provable in ZK at reasonable circuit cost | Yes (bounded nonlinear ops) | ... (assess based on architecture — a full LLM or deep transformer approach would not be) | |

Do not pre-weight these into a single composite score algorithmically — list them plainly and let the presenter and teammate jointly weigh what matters most for the pitch (e.g. if the ZK-provability row rules out the teammate's approach regardless of raw accuracy, that should be a visible, separate consideration, not silently folded into one number).

## Step 5 — Write an Honest Verdict Section
End the report with a short, direct verdict: which approach performed better overall, where each one specifically won or lost, and an explicit recommendation for which to carry forward — including any conditions (e.g. "RCAJ-X wins on ambiguity handling and explainability, but only if the `genuinely_ambiguous` gap versus the teammate's approach closes after encoder fine-tuning — see `10_EXCLUSIONS.md` for expected impact"). If results are close or mixed, say so plainly rather than forcing a clean winner.

## Checkpoint
- `data/friend_model_predictions.csv` received from the teammate (manually exported, not fetched via code) on the identical `data/test/` set.
- `results/comparison_report.md` contains the full decision table (Step 4) and a written verdict (Step 5).
- Every `variant_type` category present in `data/test/` appears in the per-category comparison — no category silently dropped from either side.
- The explainability comparison explicitly states whether the teammate's approach has a grounded, evidence-linked reasoning mechanism or not, rather than skipping the question.
- `src/compare.py` contains only CSV-reading and metric/merge logic — no network calls, no imports from or references to the teammate's codebase.
