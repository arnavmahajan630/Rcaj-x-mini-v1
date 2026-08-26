import streamlit as st
import json
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import RCAJ_X
from src.preprocessing import embed_answer, negation_mismatch_flag
from src.explain import generate_explanation
from src.rubric_cache import get_rubric_R
from src.guardrails import input_sanity_check

st.set_page_config(page_title="RCAJ-X Grader — Test Console", layout="wide")

@st.cache_resource
def load_model():
    checkpoint = torch.load("checkpoints/rcaj_x_best.pt", weights_only=False)
    model_config = {k: v for k, v in checkpoint["config"].items() if k in ["n_heads", "d_k", "d_v", "hidden"]}
    model = RCAJ_X(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

@st.cache_data
def load_rubrics():
    with open("data/raw/rubrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_test_examples():
    examples = []
    if os.path.exists("data/test"):
        for filename in os.listdir("data/test"):
            if filename.endswith(".json"):
                with open(os.path.join("data/test", filename), "r") as f:
                    examples.append(json.load(f))
    return examples

model = load_model()
rubrics = load_rubrics()
test_examples = load_test_examples()

st.title("RCAJ-X — Grading Test Console")

rubric_mode_label = st.sidebar.radio(
    "Rubric Mode",
    ["Dev (recompute each time)", "Locked (cached, hash-verified)"],
    help="Locked mode caches each rubric's criteria embedding on disk, keyed by a hash of its "
         "question_id + criteria text. If rubrics.json changes for a question_id after it's been "
         "cached, locked mode refuses to silently re-embed the new text — it errors instead.",
)
rubric_mode = "locked" if rubric_mode_label.startswith("Locked") else "dev"

question = st.selectbox("Question", rubrics, format_func=lambda q: f"[{q['question_id'].upper()}] ({q['subject'].title()}) — {q['question_text']}")

# Display Rubric Criteria Breakdown
with st.expander("📋 Active Rubric & Criteria Breakdown", expanded=True):
    st.markdown(f"**Question ID:** `{question['question_id']}` | **Subject:** `{question['subject'].title()}`")
    st.markdown(f"**Question Prompt:** {question['question_text']}")
    st.markdown("**Evaluation Criteria:**")
    for idx, c in enumerate(question["criteria"], 1):
        st.markdown(f"- **{c['criterion_id']}** (*Max Marks: {c['max_marks']}*): {c['text']}")

def load_matching_test_examples(q_id):
    return [ex for ex in test_examples if ex["question_id"] == q_id]

matching_examples = load_matching_test_examples(question["question_id"])
preset_options = ["-- None (Type custom answer) --"] + [f"{ex['answer_id']} ({ex.get('variant_type', ex.get('style', 'preset'))})" for ex in matching_examples]

preset = st.selectbox("Select Preset Test Case", preset_options)

selected_example = None
if preset != "-- None (Type custom answer) --":
    ans_id = preset.split(" (")[0]
    for ex in matching_examples:
        if ex["answer_id"] == ans_id:
            selected_example = ex
            break

if selected_example:
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Selected Test Case:** `{selected_example['answer_id']}`")
            st.markdown(f"**Stress-Test Variant:** `{selected_example.get('variant_type', selected_example.get('style', 'preset'))}`")
        with col2:
            scores = selected_example.get("human_scores", selected_example.get("ai_suggested_scores", {}))
            score_str = ", ".join([f"{k}: {v}" for k, v in scores.items()])
            st.markdown(f"**Ground Truth Target Marks:** `{score_str}`")

default_text = selected_example["answer_text"] if selected_example else ""

answer_text = st.text_area("Student Answer", value=default_text, height=180)

if st.button("Grade Answer"):
    sanity_issues = input_sanity_check(answer_text)

    try:
        with st.spinner("Grading..."):
            R = get_rubric_R(question, mode=rubric_mode)
            # Minimal glossary for test console (ideally we load the full one)
            # For this UI, we can pass an empty glossary or rebuild it.
            # Building empty glossary to speed up.
            ans_dict = embed_answer(answer_text, glossary={})
            A = ans_dict["A"]
            chunks = ans_dict["chunks"]

            neg_flags = torch.tensor([
                negation_mismatch_flag(c["text"], chunks[0] if chunks else "") for c in question["criteria"]
            ], dtype=torch.float32)

            explanations = generate_explanation(model, R, A, chunks, question["criteria"], neg_flags)
    except ValueError as e:
        st.error(f"Rubric lock error: {e}")
        st.stop()

    if sanity_issues:
        st.warning("⚠️ **Input flagged as unusual — treat this score with caution:**\n\n" +
                    "\n".join(f"- {issue}" for issue in sanity_issues))

    total_score = sum(e["score"] for e in explanations)
    total_max = sum(e["max_marks"] for e in explanations)
    st.metric("Overall Score", f"{total_score:.1f} / {total_max}")

    for exp in explanations:
        with st.container(border=True):
            if sanity_issues:
                badge = "🔴 Input Unusual"
            elif exp["confidence"] == "high_confidence":
                badge = "🟢 High Confidence"
            else:
                badge = "🟡 Flagged for Review"
            st.markdown(f"**{exp['criterion_text']}** — {exp['score']:.1f}/{exp['max_marks']}  {badge}")
            st.write(exp["reason_text"])

            # Highlight evidence chunks in the original text
            # We can just list them for simplicity as requested by Step 4 of 08
            st.caption("Evidence: " + " | ".join(exp["evidence_chunks"]))
