# 08 — Streamlit Manual-Testing UI

## Objective
A minimal Streamlit app (`app.py`) for manually exercising the model — pick a rubric, type or paste an answer, see the score, per-criterion breakdown, evidence, and confidence, generated live. This is a **testing tool for the presenter and teammate**, not a polished product UI — keep it simple, prioritize function over styling.

## Required Features (Minimum)

1. **Rubric selector**: dropdown populated from `data/raw/rubrics.json`, showing the question text and its criteria once selected.
2. **Answer input**: free-text box for pasting/typing a student answer.
3. **Grade button**: runs the full pipeline (chunk → normalize → embed → model forward → explain) on the input and displays results.
4. **Results display**, per criterion:
   - Score / max marks
   - Confidence badge (`high_confidence` / `review_recommended`)
   - Reason text (from `07_explainability_and_reasoning.md`)
   - Evidence chunks, visually highlighted within the original answer text if feasible (e.g. bold or colored span) — this is the single most convincing thing to show live, since it makes the attention mechanism tangible rather than abstract.
5. **Overall score** (weighted sum across criteria).
6. **Preset test examples dropdown**: let the user quickly load one of the existing `data/test/` Y-type examples by `variant_type`, so the presenter/teammate can walk through "here's a negation-flipped example, here's a scattered-evidence example" live without retyping answers.

## Implementation Sketch

```python
import streamlit as st
import json
import torch
from src.model import RCAJ_X
from src.preprocessing import embed_example, negation_mismatch_flag
from src.explain import generate_explanation

st.set_page_config(page_title="RCAJ-X Grader — Test Console", layout="wide")

@st.cache_resource
def load_model():
    checkpoint = torch.load("checkpoints/rcaj_x_best.pt")
    model = RCAJ_X(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

@st.cache_data
def load_rubrics():
    return json.load(open("data/raw/rubrics.json"))

model = load_model()
rubrics = load_rubrics()

st.title("RCAJ-X — Grading Test Console")

question = st.selectbox("Question", rubrics, format_func=lambda q: q["question_text"])

preset = st.selectbox(
    "Load a preset test example (optional)",
    ["-- none --"] + [f"{ex['answer_id']} ({ex['variant_type']})" for ex in load_matching_test_examples(question["question_id"])]
)
default_text = get_preset_text(preset) if preset != "-- none --" else ""

answer_text = st.text_area("Student Answer", value=default_text, height=200)

if st.button("Grade Answer"):
    with st.spinner("Grading..."):
        R, A, chunks = embed_example([c["text"] for c in question["criteria"]], answer_text, glossary={})
        neg_flags = torch.tensor([
            negation_mismatch_flag(c["text"], chunks[0] if chunks else "") for c in question["criteria"]
        ])
        explanations = generate_explanation(model, R, A, chunks, question["criteria"], neg_flags)

    total_score = sum(e["score"] for e in explanations)
    total_max = sum(e["max_marks"] for e in explanations)
    st.metric("Overall Score", f"{total_score:.1f} / {total_max}")

    for exp in explanations:
        with st.container(border=True):
            badge = "🟢 High Confidence" if exp["confidence"] == "high_confidence" else "🟡 Flagged for Review"
            st.markdown(f"**{exp['criterion_text']}** — {exp['score']:.1f}/{exp['max_marks']}  {badge}")
            st.write(exp["reason_text"])
            st.caption("Evidence: " + " | ".join(exp["evidence_chunks"]))
```

Fill in `load_matching_test_examples` and `get_preset_text` as thin helper functions reading from `data/test/`.

## Optional (Only If Time Allows) — Side-by-Side Comparison Panel
If the teammate's model is exposed as an importable function or a simple HTTP endpoint by the time this is built, add a second column in the results display showing the teammate's model's score and (if it has one) its own explanation for the same input, side by side. This turns the Streamlit app into a live comparison tool, not just a solo test console — valuable for the joint decision this evaluation is meant to produce. If not available yet, skip this and rely on `09_comparative_evaluation_vs_alternative_approach.md`'s offline comparison instead.

## Checkpoint
- `streamlit run app.py` launches without error.
- Selecting a question, typing an answer, and clicking "Grade Answer" produces a per-criterion breakdown with visible reason text and evidence.
- Loading a preset `negation_flipped` or `scattered_evidence` example and grading it produces a sensible, correctly-flagged result — use this as your own first manual sanity check of the whole pipeline before trusting the batch benchmark numbers.
