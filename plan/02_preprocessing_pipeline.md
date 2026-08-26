# 02 — Preprocessing Pipeline

## Objective
Implement `src/preprocessing.py` to turn raw rubric/answer text (from both `data/train/` and `data/test/`) into chunked, embedded tensors ready for the model, with a lightweight spelling-normalization step to support the `typo_injected` benchmark category.

## Step 1 — Chunking
```python
import spacy
nlp = spacy.load("en_core_web_sm")

def chunk_answer(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
```
Use spaCy sentence segmentation, not a regex split — more robust across the variety of punctuation your `diffuse_padded` and `scattered_evidence` examples will contain.

## Step 2 — Spelling Normalization (lightweight version for this scope)
Implement `normalize_spelling(text: str, glossary: dict[str, str]) -> str`:
- Build `glossary` from the exact-match set of criterion-critical terms across all rubrics in `data/raw/rubrics.json` (extract nouns/technical terms from criterion text).
- For each word in the input text, compute Levenshtein distance to every glossary term.
- **Only normalize if there is a single, unique closest glossary term within distance ≤ 2.** If a word is equidistant to two or more glossary terms (a genuine tie), **leave it un-normalized** — do not default to "first found" or any arbitrary tie-break. Silently picking between two different technical terms risks conflating them, which is a worse outcome than leaving a typo uncorrected.
- Log every ambiguous (tied) case encountered to `results/normalization_ambiguous_cases.log` (word, candidate terms, distances) for manual review — this is useful signal on whether the glossary itself needs refinement (e.g. two terms that are too similar to each other).
- Use `python-Levenshtein` or a simple DP edit-distance function — no need for a full SymSpell dependency-frequency setup at this scope.
- Apply this **only** as a preprocessing step before embedding — never mutate the stored `answer_text` in the dataset files; normalization output is intermediate, not persisted as ground truth.

This is intentionally the simplified version of the full spelling-robustness design — sufficient to test whether the `typo_injected` benchmark category degrades gracefully, without building the full SymSpell + language-tool pipeline described in the complete architecture.

## Step 3 — Embedding
```python
from sentence_transformers import SentenceTransformer
import torch

encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')

def embed_example(criteria_texts: list[str], answer_text: str, glossary: dict) -> dict:
    chunks = chunk_answer(answer_text)
    normalized_chunks = [normalize_spelling(c, glossary) for c in chunks]
    R = encoder.encode(criteria_texts, convert_to_tensor=True)
    A = encoder.encode(normalized_chunks, convert_to_tensor=True)
    return {"R": R, "A": A, "chunks": chunks, "normalized_chunks": normalized_chunks}
```
Note: keep both raw `chunks` and `normalized_chunks` — the benchmark step needs raw chunks for the verification-report-style output, but embeddings should be computed on normalized text.

## Step 4 — Negation Flag (simplified rule-based version)
```python
NEGATION_WORDS = {"not", "no", "never", "cannot", "isn't", "doesn't", "won't", "n't", "lacks", "fails"}

def negation_mismatch_flag(criterion_text: str, top_chunk_text: str) -> float:
    c_neg = any(w in criterion_text.lower().split() for w in NEGATION_WORDS)
    a_neg = any(w in top_chunk_text.lower().split() for w in NEGATION_WORDS)
    return float(c_neg != a_neg)
```
This is the simplified keyword-based negation check (not the full dependency-parse version from the complete architecture) — sufficient for this scope, explicitly named as a scope-reduction in the exclusion list.

## Step 5 — Batch Process and Cache
Write `preprocess_dataset(split: str)` that runs the full pipeline (chunk → normalize → embed) over every file in `data/train/` or `data/test/`, and caches results to `data/{split}_embedded.pt` (a serialized list of dicts including the original metadata: `answer_id`, `question_id`, `human_scores`, `variant_type` where applicable). Do not re-run embedding inside the training or benchmarking loop — always read from this cache.

## Checkpoint
- `preprocess_dataset("train")` and `preprocess_dataset("test")` both run end to end without error.
- Spot-check: embed 3 examples, confirm `R.shape == (n_criteria, 384)` and `A.shape == (n_chunks, 384)`.
- Spot-check the normalization step on at least 2 `typo_injected` test examples — confirm the injected typo is corrected back to the glossary term.
- Cached files exist at `data/train_embedded.pt` and `data/test_embedded.pt`.
