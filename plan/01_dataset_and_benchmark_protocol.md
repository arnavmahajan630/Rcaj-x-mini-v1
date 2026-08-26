# 01 — Dataset Generation and the Train/Test Generalization Protocol

## Objective
Produce two answer sets over the **same fixed set of rubrics and questions**:
- `data/train/` — X-type answers (used to train the model)
- `data/test/` — Y-type answers (deliberate variations of X-type answers, unseen during training, used only for benchmarking)

with a manifest linking every test example back to the train example it was derived from, plus a `variant_type` tag describing how it was varied.

## Step 1 — Define the Fixed Rubric/Question Set (`data/raw/rubrics.json`)

Implement `generate_rubrics()` in `src/data_generation.py` producing 8–12 rubrics, each with 2–4 criteria, spanning at least 3 distinct subject areas (e.g. general science, CS/programming fundamentals, civics/general knowledge — adjust to whatever the presenter knows well enough to self-verify scores). Schema:

```json
{
  "question_id": "q1",
  "question_text": "Explain osmotic pressure and give one real-world application.",
  "subject": "biology",
  "criteria": [
    {"criterion_id": "q1_c1", "text": "Correctly defines osmotic pressure", "max_marks": 2},
    {"criterion_id": "q1_c2", "text": "Gives one valid real-world application", "max_marks": 2}
  ]
}
```

This file is **fixed for the entire project** once created. Both train and test answers below are generated against these exact same questions/criteria — never add or remove rubrics after this step.

## Step 2 — Generate X-Type (Training) Answers (`data/train/`)

For every `(question, criterion)` pair, generate 4–6 X-type answer instances covering a controlled, "training-representative" distribution:
- Fully correct, direct, single-clause statement
- Fully correct, slightly longer explanation
- Partially correct (addresses one of two criteria if compound)
- Clearly incorrect, unrelated content

Schema per example (`data/train/{question_id}_{answer_id}.json`):
```json
{
  "answer_id": "q1_train_003",
  "question_id": "q1",
  "answer_text": "...",
  "human_scores": {"q1_c1": 2, "q1_c2": 0},
  "style": "x_type"
}
```

Score every example yourself (or have the presenter score them) — do not use an LLM to assign scores, only to help draft candidate text if needed. This matches the credibility constraint from the project's data-sourcing guidance: synthetic answer *generation* is fine, synthetic *scoring* is not.

**Target volume:** 150–300 total X-type training examples across all rubrics.

## Step 3 — Generate Y-Type (Test) Answers, as Explicit Variations of X-Type Answers (`data/test/`)

This is the step that implements the project owner's core benchmark design. For a defined subset of X-type training answers (aim for covering every question/criterion at least once), generate a **paired variation** — same underlying correctness intent, different surface form. Implement `generate_variation(train_answer, variant_type)` producing one Y-type answer per `(train_answer, variant_type)` pair, for each of the following required variant types:

| `variant_type` | Transformation applied to the paired X-type answer |
|---|---|
| `paraphrase` | Same meaning, different wording/sentence structure entirely |
| `scattered_evidence` | Same correct content, but split across 2 non-adjacent sentences with unrelated filler between them |
| `diffuse_padded` | Same correct content, buried inside a long, mostly-irrelevant answer |
| `partial_credit_shift` | Originally-full-credit answer degraded to address only part of a compound criterion |
| `negation_flipped` | Explicit or implicit negation inserted, flipping correctness while staying fluent |
| `confidently_wrong` | Fluent, on-topic, but factually altered (no negation word) |
| `typo_injected` | 1–2 character typos injected specifically into the criterion-critical term(s) |
| `genuinely_ambiguous` | Deliberately vague phrasing a reasonable human grader could read as partial credit either way |

Schema per example (`data/test/{question_id}_{answer_id}.json`):
```json
{
  "answer_id": "q1_test_017",
  "question_id": "q1",
  "derived_from_train_id": "q1_train_003",
  "variant_type": "scattered_evidence",
  "answer_text": "...",
  "human_scores": {"q1_c1": 2, "q1_c2": 0}
}
```

Score every Y-type example yourself as well, using the same criteria definitions — this is your ground truth for benchmarking, and it must be independently and correctly assigned, not inferred from the paired training answer's score (a variation can legitimately deserve a different score than its source, e.g. `negation_flipped` should score lower).

**Target volume:** at least 8 variant types × 5–8 examples per question (where applicable) — aim for 150–300 total Y-type test examples, roughly matching train volume so per-category benchmark numbers are statistically meaningful.

## Step 4 — Build the Manifest (`data/dataset_manifest.json`)

```json
[
  {"test_id": "q1_test_017", "derived_from_train_id": "q1_train_003", "question_id": "q1", "variant_type": "scattered_evidence"}
]
```
This traceability is required by `05_benchmarking_and_reporting.md` to compute **paired score deltas** (e.g. "how much did the score drop from the X-type source to its negation_flipped variation") — a plain aggregate accuracy number is not sufficient, the paired comparison is the actual point of this benchmark design.

## Step 1.5 — LLM-Assisted Drafting (Optional Tooling, Human-in-the-Loop by Design)

Manually writing 300–600 answer instances from scratch is unnecessarily slow. Use an LLM to **draft candidate answer text only** — never to assign scores (see the hard constraint at the bottom of this file, unchanged). Recommended workflow, in order of preference:

**Preferred: manual chat-based drafting, no code.** Do not build a live API integration into `data_generation.py` for this — it adds credential/cost/rate-limit handling for a one-time task that doesn't need to be programmatic. Instead, for each rubric, run a fixed prompt through Claude Pro chat directly:

> *"For this rubric criterion: [criterion text]. Generate 6 draft answer variants: fully correct, longer explanation, partially correct, clearly incorrect. Return as JSON: [{answer_text, intended_style}]."*

and for the Y-type variations:

> *"Here is a correct student answer: [X-type answer text]. Generate one variation of type '[variant_type]' — [insert the transformation description from the variant_type table above]. Return only the transformed answer text."*

Paste the output into `data/train/` or `data/test/` files, then score every example yourself before it's used anywhere downstream. This keeps human scoring as the actual bottleneck (as intended) while removing the drafting bottleneck.

**If a scripted version is genuinely preferred**, implement `draft_via_llm(prompt: str) -> str` in `src/data_generation.py` as a thin, optional wrapper around the Anthropic API (`claude-sonnet-5` for quality on harder variant types like `genuinely_ambiguous`; `claude-haiku-4-5-20251001` is a reasonable cheaper/faster substitute for simpler ones like `paraphrase` or `typo_injected`), gated behind an environment variable (`ANTHROPIC_API_KEY`). If the key isn't set, the function should raise a clear error rather than silently falling back to placeholder text, so a missing-drafting-step failure is never mistaken for a data-generation bug. Either way, the scoring step immediately after drafting remains fully manual — this is unchanged and non-negotiable.

**If neither is available right now**: populate `data/train/` and `data/test/` with a small number (10–20) of dummy/placeholder examples so the rest of the pipeline (`02` through `05`) can be built and tested end-to-end against realistic-shaped data, clearly marked with `"placeholder": true` in each file, and swap in the real, human-scored dataset before any benchmark numbers in `05` or `09` are treated as final. Do not report placeholder-derived numbers as real results.

## Hard Constraints (Do Not Violate)
- **No rubric or question appears only in train or only in test.** Every `question_id` in `data/test/` must also exist in `data/train/`.
- **No test answer is a verbatim duplicate of a train answer.** Every Y-type answer must be a genuine transformation, not a copy.
- **Every test example must trace to exactly one train example** via `derived_from_train_id`, and that link must be recorded in the manifest.
- **Scores are human-assigned, not LLM-assigned**, for both train and test sets.

## Checkpoint
- `data/raw/rubrics.json` exists, fixed, 8–12 questions.
- `data/train/` contains 150–300 scored X-type answers across all questions.
- `data/test/` contains 150–300 scored Y-type answers, each traceable via the manifest, covering all 8 required `variant_type` categories.
- Run a validation script confirming: every `question_id` in test also appears in train; every `derived_from_train_id` resolves to a real train file; no duplicate `answer_text` between a test example and its paired train source.
