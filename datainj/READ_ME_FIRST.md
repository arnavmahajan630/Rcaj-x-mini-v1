# READ ME FIRST — Status of This BIG Draft Dataset

## What this is
`rubrics.json`, `train_answers_DRAFT.json`, `test_answers_DRAFT.json`, `manifest_DRAFT.json` in this folder are the **full combined dataset**: your original 6 rubrics + 20 newly generated rubrics across biology, chemistry, physics, computer science, mathematics, civics, history, geography, economics, and environmental science.

**Totals: 26 rubrics, 104 X-type training answers, 208 Y-type test answers, 312 examples overall — exactly 26 per required `variant_type` category.**

## Still not ground truth — same rule as before
Every score is under `"ai_suggested_scores"`, every record is `"human_reviewed": false`. Nothing changes about the review requirement in `01_dataset_and_benchmark_protocol.md` — this batch is larger, not exempt. Do not let any pipeline step treat these as final until reviewed.

## How this batch was built (so you can spot-check the method, not just the output)
Rather than hand-typing 312 answers individually, this was built from **20 hand-authored core snippets per new rubric** (the correct answer, an incorrect answer, a partial fragment, a vague/ambiguous version, a paraphrase, a negated version, and a key technical term) plus **4 hand-authored core snippets** per original rubric. A script then mechanically derived the remaining structural variants from those:
- `scattered_evidence` — the hand-written correct answer's two sentences, split apart with a filler sentence inserted between them (filler drawn from a fixed per-subject pool).
- `diffuse_padded` — the correct answer surrounded by 2–3 filler sentences from the same pool.
- `typo_injected` — a deterministic 2-character swap applied to the answer's key technical term.
- `longer_explanation` (X-type) — a generic framing sentence prepended to the correct answer.
- `partial_credit_shift` — a further-truncated version of the hand-written partial fragment.

The **content-bearing** variants (`paraphrase`, `negation_flipped`, `confidently_wrong`/incorrect, `genuinely_ambiguous`) were all hand-authored per rubric, not mechanically generated — those are the ones that most need your review attention, since a script can't reliably judge whether a paraphrase preserves meaning or a negation is subtly wrong.

## Review Priority (given the volume, review smart, not just linearly)
With 312 examples, reviewing everything with equal scrutiny will take a while. Suggested triage:
1. **High priority — review carefully:** all `negation_flipped`, `confidently_wrong`, and `genuinely_ambiguous` examples (78 total) — these carry the most subjective scoring judgment and are the categories your benchmark's paired-delta analysis (`05` Step 5) leans on most heavily.
2. **Medium priority — spot-check a sample:** `paraphrase` and `partial_credit_shift` (52 total) — check a handful per subject rather than every single one, since these follow a consistent, checkable pattern.
3. **Low priority — mechanically verifiable, sanity-check rather than deep-review:** `scattered_evidence`, `diffuse_padded`, `typo_injected` (78 total) — these were built by deterministic transformation of already-reviewed correct answers, so a quick glance to confirm the transformation didn't produce something nonsensical is enough.
4. **Training set (104 examples)** — same priority split applies: `x_type_clearly_incorrect` and `x_type_partial_correct` deserve real attention; `x_type_full_correct` and `x_type_longer_explanation` are lower-risk.

## Same conversion step as before
Once reviewed, rename `ai_suggested_scores` → `human_scores` and flip `human_reviewed` to `true` per example, then follow `12_HOW_TO_RUN.md` to feed this into the pipeline.

## If You Want Even More Volume
The generation script (`rubric_content.py` + `generate.py`, used to build the 20-rubric batch) is reusable — adding more rubrics is a matter of appending more hand-authored core snippets in the same 7-field format (correct/incorrect/partial/vague/paraphrase/negated/key_term) and re-running the script. Ask if you'd like another batch in a different set of subjects.
