import streamlit as st
import json
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import RCAJ_X
from src.preprocessing import embed_example, negation_mismatch_flag
from src.explain import generate_explanation

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

question = st.selectbox("Question", rubrics, format_func=lambda q: q["question_text"])

def load_matching_test_examples(q_id):
    return [ex for ex in test_examples if ex["question_id"] == q_id]

matching_examples = load_matching_test_examples(question["question_id"])
preset_options = ["-- none --"] + [f"{ex['answer_id']} ({ex['variant_type']})" for ex in matching_examples]

preset = st.selectbox("Load a preset test example (optional)", preset_options)

def get_preset_text(preset_label):
    if preset_label == "-- none --":
        return ""
    ans_id = preset_label.split(" (")[0]
    for ex in matching_examples:
        if ex["answer_id"] == ans_id:
            return ex["answer_text"]
    return ""

default_text = get_preset_text(preset)

answer_text = st.text_area("Student Answer", value=default_text, height=200)

if st.button("Grade Answer"):
    if not answer_text.strip():
        st.warning("Please enter an answer to grade.")
    else:
        with st.spinner("Grading..."):
            # Minimal glossary for test console (ideally we load the full one)
            # For this UI, we can pass an empty glossary or rebuild it.
            # Building empty glossary to speed up.
            R_dict = embed_example([c["text"] for c in question["criteria"]], answer_text, glossary={})
            R = R_dict["R"]
            A = R_dict["A"]
            chunks = R_dict["chunks"]
            
            neg_flags = torch.tensor([
                negation_mismatch_flag(c["text"], chunks[0] if chunks else "") for c in question["criteria"]
            ], dtype=torch.float32)
            
            explanations = generate_explanation(model, R, A, chunks, question["criteria"], neg_flags)

        total_score = sum(e["score"] for e in explanations)
        total_max = sum(e["max_marks"] for e in explanations)
        st.metric("Overall Score", f"{total_score:.1f} / {total_max}")

        for exp in explanations:
            with st.container(border=True):
                badge = "🟢 High Confidence" if exp["confidence"] == "high_confidence" else "🟡 Flagged for Review"
                st.markdown(f"**{exp['criterion_text']}** — {exp['score']:.1f}/{exp['max_marks']}  {badge}")
                st.write(exp["reason_text"])
                
                # Highlight evidence chunks in the original text
                # We can just list them for simplicity as requested by Step 4 of 08
                st.caption("Evidence: " + " | ".join(exp["evidence_chunks"]))
