import spacy
from sentence_transformers import SentenceTransformer
import torch
import json
import os
import Levenshtein

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If not loaded yet, download it.
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
NEGATION_WORDS = {"not", "no", "never", "cannot", "isn't", "doesn't", "won't", "n't", "lacks", "fails"}

def chunk_answer(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def normalize_spelling(text: str, glossary: dict) -> str:
    if not glossary:
        return text
    
    words = text.split()
    normalized_words = []
    
    os.makedirs("results", exist_ok=True)
    log_file = "results/normalization_ambiguous_cases.log"
    
    for word in words:
        clean_word = word.strip(".,!?()[]{}\"'").lower()
        if not clean_word:
            normalized_words.append(word)
            continue
            
        distances = []
        for term, proper_term in glossary.items():
            dist = Levenshtein.distance(clean_word, term)
            distances.append((dist, proper_term))
        
        distances.sort(key=lambda x: x[0])
        
        if distances[0][0] <= 2:
            # Check for ties
            if len(distances) > 1 and distances[0][0] == distances[1][0]:
                with open(log_file, "a") as f:
                    f.write(f"Ambiguous: '{word}', Candidates: {[d[1] for d in distances[:2]]}, Distance: {distances[0][0]}\n")
                normalized_words.append(word) # Un-normalized on tie
            else:
                # Need to match capitalization or just replace? The instructions say "corrected back to the glossary term"
                # Let's just replace with the glossary term but preserve the original word's capitalization if possible, 
                # or just use proper_term directly since it's a technical term.
                # To be simple and robust:
                replacement = distances[0][1]
                # Try to preserve punctuation around it
                prefix = word[:len(word) - len(word.lstrip(".,!?()[]{}\"'"))]
                suffix = word[len(word.rstrip(".,!?()[]{}\"'")):]
                normalized_words.append(f"{prefix}{replacement}{suffix}")
        else:
            normalized_words.append(word)
            
    return " ".join(normalized_words)

def build_glossary(rubrics):
    glossary = {}
    for r in rubrics:
        for c in r["criteria"]:
            doc = nlp(c["text"])
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
                    term = token.text.lower()
                    glossary[term] = token.text
    return glossary

def embed_example(criteria_texts: list[str], answer_text: str, glossary: dict) -> dict:
    chunks = chunk_answer(answer_text)
    if not chunks:
        chunks = [""] # Handle empty answers
    normalized_chunks = [normalize_spelling(c, glossary) for c in chunks]
    R = encoder.encode(criteria_texts, convert_to_tensor=True).cpu()
    A = encoder.encode(normalized_chunks, convert_to_tensor=True).cpu()
    return {"R": R, "A": A, "chunks": chunks, "normalized_chunks": normalized_chunks}

def negation_mismatch_flag(criterion_text: str, top_chunk_text: str) -> float:
    c_neg = any(w in criterion_text.lower().split() for w in NEGATION_WORDS)
    a_neg = any(w in top_chunk_text.lower().split() for w in NEGATION_WORDS)
    return float(c_neg != a_neg)

def preprocess_dataset(split: str):
    with open("data/raw/rubrics.json", "r") as f:
        rubrics = json.load(f)
        
    glossary = build_glossary(rubrics)
    
    rubric_map = {r["question_id"]: r for r in rubrics}
    
    examples = []
    files = [f for f in os.listdir(f"data/{split}") if f.endswith('.json')]
    for file in files:
        with open(f"data/{split}/{file}", "r") as f:
            data = json.load(f)
            
        question_id = data["question_id"]
        rubric = rubric_map[question_id]
        criteria_texts = [c["text"] for c in rubric["criteria"]]
        
        embedded = embed_example(criteria_texts, data["answer_text"], glossary)
        
        # Calculate initial negation flags assuming chunks[0] as top chunk, 
        # but the prompt says: "computed against whichever chunk that criterion's top attention weight points to".
        # Wait, the prompt says "computed upstream ... passed in as a tensor of shape (n_criteria,)".
        # But attention weights aren't known until the model runs!
        # Ah, "negation_flags must be computed upstream (in 02_preprocessing_pipeline.md's pipeline) and passed in 
        # as a tensor of shape (n_criteria,) — one flag per criterion, computed against whichever chunk that criterion's 
        # top attention weight (averaged across heads) points to."
        # This is a paradox: you need attention weights to know the top chunk, but you need negation flags for the model input.
        # Wait, the `ScoringHead` takes `negation_flags`, but `MultiHeadCrossAttention` doesn't!
        # So we can't compute it upstream in `preprocessing.py` fully, UNLESS we just approximate it, or we compute it dynamically.
        # Wait, in the plan 02: "negation_mismatch_flag(criterion_text: str, top_chunk_text: str)".
        # It seems the pipeline in 08 does:
        # `neg_flags = torch.tensor([negation_mismatch_flag(c["text"], chunks[0] if chunks else "") for c in question["criteria"]])`
        # In 02_preprocessing_pipeline.md it just mentions Step 4 - Negation Flag.
        # So let's compute it against all chunks, or just the whole answer? The instructions say `chunks[0]`.
        # I'll just follow the `app.py` logic: compute against `chunks[0]`.
        
        neg_flags = []
        top_chunk = embedded["chunks"][0] if embedded["chunks"] else ""
        for c in criteria_texts:
            neg_flags.append(negation_mismatch_flag(c, top_chunk))
            
        data["R"] = embedded["R"]
        data["A"] = embedded["A"]
        data["chunks"] = embedded["chunks"]
        data["normalized_chunks"] = embedded["normalized_chunks"]
        data["negation_flags"] = torch.tensor(neg_flags, dtype=torch.float32)
        
        examples.append(data)
        
    torch.save(examples, f"data/{split}_embedded.pt")

if __name__ == "__main__":
    preprocess_dataset("train")
    preprocess_dataset("test")
    print("Preprocessing complete.")
