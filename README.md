# RCAJ-X Mini v1

**Rubric-Criterion-Attention-Judgement with eXplanations (RCAJ-X)** is a lightweight, interpretable, cross-attention neural architecture designed for automated, multi-criterion answer grading and evidence grounding. 

RCAJ-X uses bounded nonlinear operators and a compact parameter budget, making it optimized for **Zero-Knowledge (ZK) circuit compilation** (e.g., via EZKL) while offering verifiable, grounded feedback.

---

## 🌟 Key Features

* **Multi-Criterion Cross-Attention**: Computes cross-attention between rubric criteria embeddings and sentence chunks of student answers.
* **Grounded Evidence Explanations**: Automatically extracts top-attending chunks to provide grounded, evidence-linked justifications for each criterion score.
* **Ambiguity & Negation Detection**: Measures attention spread across answer chunks to flag ambiguous answers, and detects negation mismatches between criteria and text.
* **ZK-Friendly Architecture**: Built with lightweight linear transformations and bounded operations suitable for ZK-SNARK circuit proving.
* **Stress-Test Benchmarking**: Evaluates performance across 8 distinct answer variant types (e.g., `paraphrase`, `diffuse_padded`, `negation_flipped`, `confidently_wrong`, `typo_injected`).
* **Interactive Streamlit Dashboard**: Provides a visual workspace for testing custom answers, viewing attention heatmaps, and running comparisons.

---

## 🏗️ Architecture Overview

```
               +-----------------------+
               |  Student Answer Text  |
               +-----------+-----------+
                           |
                           v  (Sentence Chunking & Levenshtein Normalization)
               +-----------+-----------+
               |   Sentence Chunks (A) |
               +-----------+-----------+
                           |
                           v  (BAAI/bge-small-en-v1.5 Embedder)
+------------------+  +----+------------------+
| Rubric Criteria  |  | Chunk Embeddings (A)  |
| Embeddings (R)   |  | (shape: n_chunks x d) |
+--------+---------+  +----+------------------+
         |                 |
         +--------+--------+
                  |
                  v
   +--------------+---------------+
   |   MultiHeadCrossAttention     |
   |   Q = R * W_q, K/V = A * W_k/v|
   +--------------+---------------+
                  |
                  v
   +--------------+---------------+
   |   ScoringHead (MLP)           |
   |   Inputs: Context + Spread +  |
   |           Negation Mismatch  |
   +--------------+---------------+
                  |
                  v
   +--------------+---------------+
   |  Per-Criterion Scores &      |
   |  Grounded Explanations       |
   +------------------------------+
```

---

## 📁 Repository Structure

```
Rcaj-x-mini-v1/
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Python Dependencies
├── scripts/
│   └── ingest_data.py          # Data ingestion & normalization utility
├── src/
│   ├── model.py                # MultiHeadCrossAttention & ScoringHead PyTorch module
│   ├── preprocessing.py        # spaCy chunking, Levenshtein spell-check & BGE embedding
│   ├── train.py                # Grid sweep ablation trainer & checkpoint saver
│   ├── explain.py              # Explanation generator & attention-spread analyzer
│   ├── benchmark.py            # Variant-wise evaluation & paired-delta report generator
│   └── compare.py              # Comparative evaluation vs baseline models
├── tests/
│   └── test_model.py           # Unit tests for model forward pass & output shapes
├── data/                       # Preprocessed datasets and embedded tensors (.pt)
├── datainj/                    # Raw human-reviewed dataset & rubric definitions
├── results/                    # Benchmark reports, comparison tables & logs
└── checkpoints/                # Saved model weights (rcaj_x_best.pt)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and set up a Python virtual environment:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Ingestion & Preprocessing

Ingest raw rubric data and convert text answers into sentence-level embeddings:

```bash
# Ingest raw dataset
python scripts/ingest_data.py

# Chunk, spell-check, and generate embeddings
python src/preprocessing.py
```

### 3. Model Training

Train the RCAJ-X cross-attention model across hyperparameter configurations:

```bash
python src/train.py
```
*The best performing model weights will be saved to `checkpoints/rcaj_x_best.pt`.*

---

## 📊 Benchmarking & Comparative Evaluation

To run stress-tests across all 8 variant categories and generate comparative reports:

```bash
# Run benchmark evaluations
python src/benchmark.py

# Generate comparative evaluation report
python src/compare.py
```

Reports will be updated in:
* `results/benchmark_report.md`
* `results/comparison_report.md`

---

## 🖥️ Running the Interactive Streamlit UI

To launch the web interface for real-time grading, score inspection, and explanation viewing:

```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧪 Running Unit Tests

To run pytest unit tests verifying model forward pass dimensions and loss function behavior:

```bash
pytest tests/
```

---

## 📄 License

This project is open-source under the MIT License.
