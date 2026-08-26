# 02 — Data Strategy: 5 Subjects, High Volume, Short & Long Paragraph Answers

## Goal
Move off pure LLM-synthetic data as the sole source, target 5 subjects with real depth, and explicitly cover **both short-paragraph and long-paragraph answer styles** — the project owner flagged this as a distinct requirement, since real student answers vary enormously in length and a model trained mostly on one length regime will misjudge the other.

## Step 1 — Pick 5 Subjects with Real Public Data Available
Recommend subjects where a real, labeled short-answer dataset exists, so "pull huge data then convert to our needs" is actually executable rather than aspirational:
1. **Biology / general science** — ASAP-SAS (Kaggle, 10 items, real rubrics, 17k+ responses) is still the strongest available public source for exactly this task shape.
2. **Computer science fundamentals** — Mohler dataset (short-answer CS grading, human-scored) is smaller but directly on-topic and well-suited to short-paragraph answers specifically.
3. **Civics / social studies** — fewer ready-made labeled datasets; source via public exam board marking schemes (e.g. CBSE/ICSE civics past papers) for real rubrics, paired with your own answer generation (see Step 3) rather than expecting a pre-labeled public set.
4. **Physics or chemistry (pick one)** — similarly likely to need marking-scheme-sourced rubrics + generated answers rather than a ready public dataset.
5. **English comprehension / reading response** — deliberately included because this is where **long-paragraph** answers naturally occur (analysis, interpretation questions), balancing the short-answer bias of ASAP-SAS/Mohler.

## Step 2 — Ingestion & Conversion Pipeline (for the real datasets)
Extend `scripts/ingest_data.py`:
1. Download ASAP-SAS and Mohler raw files.
2. Parse into the existing `(question_id, criteria, answer_text, human_scores)` schema already used by `datainj/`.
3. **Explicitly tag each ingested answer with a `length_bucket`**: `short` (< 40 words), `medium` (40–100 words), `long` (100+ words) — computed automatically from word count, not guessed. This tag is what makes the short/long paragraph requirement checkable later rather than assumed.
4. Run the existing `src/preprocessing.py` chunking/normalization pipeline on ingested data exactly as it runs on existing data — no special-casing needed there.

## Step 3 — Fill the Gap: Long-Paragraph Coverage
Public short-answer datasets skew short by construction. To get genuine long-paragraph coverage without waiting on scarce public data:
1. For a subset of ASAP-SAS/Mohler questions, **manually (or LLM-draft-assisted, human-scored per the existing project rule) write extended versions** of existing correct answers — same content, expanded with justification, examples, and elaboration, 100–200+ words. This directly stress-tests whether RCAJ-X's cross-attention still correctly locates the relevant evidence chunks inside a genuinely long answer, which is a stronger test of the architecture's core claim than the earlier `diffuse_padded` synthetic padding was (padding was irrelevant filler; a real long-paragraph answer is dense with *relevant but verbose* content, a different failure mode).
2. English comprehension subject data (Step 1, subject 5) naturally supplies this — prioritize sourcing real long-form student responses there if any labeled set is available (e.g. a past-paper marking scheme with sample answers), since real long verbose answers are qualitatively different from artificially-extended short ones.

## Step 4 — Volume & Balance Targets
- **Per subject**: aim for at least 150–250 labeled examples once combined (real + supplementary), covering all 8 `variant_type` categories from the existing benchmark spec.
- **Length balance, per subject**: no subject should be more than ~70% short-answer — actively check this with a `length_bucket` distribution report before training, not after.
- **Total target**: 5 subjects × ~200 = ~1000 examples, meaningfully larger than the current LLM-synthetic-only set, and with a real (not purely synthetic) core.

## Step 5 — Data Quality Report (Required Before Retraining)
Add `scripts/data_quality_report.py`, run before every retrain:
- Count and % by subject, `variant_type`, and `length_bucket`.
- Flag any subject/variant_type/length_bucket combination with fewer than 5 examples — these cells will be under-tested in benchmarking, and it's better to know that explicitly than discover it silently in a benchmark report.

## Checkpoint
- 5 subjects represented, each with real (not exclusively LLM-synthetic) source data where publicly available.
- `length_bucket` distribution report shows genuine short/medium/long spread per subject, not a short-answer monoculture.
- Total dataset volume ≥ ~1000 examples, human-scored per the existing project rule (no LLM-assigned ground truth).
- `data_quality_report.py` run and reviewed, with any under-populated cells explicitly acknowledged (not silently left thin).
