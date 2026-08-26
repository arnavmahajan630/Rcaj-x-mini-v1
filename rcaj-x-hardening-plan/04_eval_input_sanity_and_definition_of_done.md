# 04 — Sane Scores for Live Evaluator Input & Definition of Done

## The Requirement
The project owner's core demo target: if an evaluator (judge, teacher, or the owner themself) types an answer live — not one of the curated benchmark examples — the score they get back should be **sane**, not a wild or nonsensical number, even for input the model has never specifically seen the shape of. This is a distinct requirement from benchmark accuracy: a model can score 90% on a held-out test set and still occasionally produce something absurd on genuinely novel live input, because live input can land further outside the training distribution than any curated test set does.

## Why This Needs Explicit Guardrails, Not Just "Train It Well"
No amount of training data eliminates the possibility of an out-of-distribution input producing an unreliable score — this is a structural property of neural regression, not a data quantity problem. The fix is architectural: detect when input is unusual and respond differently, rather than trusting every prediction equally.

## Guardrail 1 — Output Bounding (Already Covered)
Per `01_score_bounding_and_consistency_fix.md` — this is the floor-level fix and must be in place before any of the following make sense to add.

## Guardrail 2 — Input Sanity Pre-Checks (New)
Before running the model at all, in `app.py` (Streamlit) and any future API endpoint:
```python
def input_sanity_check(answer_text, question_text):
    issues = []
    word_count = len(answer_text.split())
    if word_count < 3:
        issues.append("Answer is extremely short (<3 words) — score reliability is low.")
    if word_count > 500:
        issues.append("Answer is unusually long (>500 words) — outside typical training range.")
    # crude language/gibberish check: ratio of dictionary-recognizable tokens
    # (reuse the existing spell-check glossary/dictionary from preprocessing.py for this,
    # don't add a new dependency)
    return issues
```
Surface these as a visible warning banner in the UI alongside the score — not a hard block, since the evaluator should still get *a* score, but they should know it's in shakier territory before trusting it.

## Guardrail 3 — Confidence Routing Is Already the Right Mechanism, Just Underused
The existing `spread`-based confidence flag (`high_confidence` / `review_recommended`) is exactly the mechanism designed for this — but per the project owner's testing, it may not currently be surfaced prominently enough for **live** input specifically (it was designed and validated against the curated benchmark set, not stress-tested against live typed input). Concretely:
- In `app.py`, make the confidence badge visually prominent (not a small caption) whenever the input triggered any `input_sanity_check` issue — stack the signals so a short, garbled, or unusual answer is very hard to mistake for a clean high-confidence score.
- Add a third confidence tier if useful: `input_unusual` — distinct from `review_recommended` (which reflects the model's own attention-spread uncertainty about *content*), to separately flag when the *input itself* was outside normal range regardless of what the model's internal signals say. These are different failure modes and conflating them loses information: a short-but-clear answer might have low spread (model is "confident") while still deserving an input-length warning.

## Guardrail 4 — A Standing Live-Input Stress Test (Do This Once, Keep It)
Add `tests/test_live_input_robustness.py` — not curated benchmark data, deliberately adversarial/unusual inputs an evaluator might plausibly type while testing the system live:
- Single-word answers.
- Answers that just restate the question back.
- Answers in a mildly different register (very informal slang, or overly formal/verbose).
- An answer to a *different* question entirely (topic mismatch) — confirm this doesn't accidentally score high due to generic fluency.
- Empty or whitespace-only input.

For each, assert: score stays within bounds (guaranteed by `01`), and at least one guardrail (sanity check or confidence flag) fires — the goal isn't a "correct" score on nonsense input, it's a score that's bounded *and* honestly flagged as low-confidence, never a wild number presented with false confidence.

## Checkpoint (Also Serves as Overall Definition of Done for This Whole Hardening Pass)
- [ ] `01`: 0 out-of-bounds predictions confirmed by explicit assertion; consistency-flag check implemented and reviewed.
- [ ] `02`: 5 subjects represented with real (not exclusively synthetic) data, short/medium/long length buckets genuinely populated per subject, ≥1000 total examples, data quality report reviewed.
- [ ] `03` Part A: dev and locked rubric modes both implemented; locked mode caches `R` per rubric hash.
- [ ] `03` Part B: encoder fine-tuned on this hardware within a documented, small time budget; before/after ablation shows measured improvement on negation/confidently-wrong/paraphrase categories.
- [ ] `04` (this file): input sanity checks implemented and surfaced in the UI; confidence routing visibly distinguishes model-uncertainty from input-unusualness; live-input stress test suite passes (bounded + flagged, not necessarily "correct").
- [ ] Full benchmark suite (`src/benchmark.py`) re-run after all changes above, with a fresh `results/benchmark_report.md` and `results/consistency_flags.csv` — do not carry forward pre-fix numbers as if they still apply.
- [ ] The project owner has personally tested at least 15–20 live, off-the-cuff answers (not from any curated set) through the Streamlit UI and confirms scores feel sane, consistent with the reasoning shown, and appropriately flagged when uncertain.

Only once every box above is checked should this repo be considered ready for the CertiProof integration work in the separate `certiproof-integration-plan/` file set.
