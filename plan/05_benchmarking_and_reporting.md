# 05 — Benchmarking and Reporting

## Objective
Implement `src/benchmark.py`: run the trained model (`checkpoints/rcaj_x_best.pt`) against `data/test_embedded.pt` (Y-type, unseen answer variations, same rubrics/questions as training) and produce `results/benchmark_results.csv` and `results/benchmark_report.md`. This file implements the specific generalization-testing protocol the project owner specified — do not substitute a generic random-split benchmark.

## Step 1 — Load Model and Test Set
```python
checkpoint = torch.load("checkpoints/rcaj_x_best.pt")
model = RCAJ_X(**checkpoint["config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()

test_examples = torch.load("data/test_embedded.pt")   # Y-type, never seen in training
manifest = json.load(open("data/dataset_manifest.json"))
```

## Step 2 — Run Inference Over the Full Test Set
```python
results = []
with torch.no_grad():
    for ex in test_examples:
        out = model(ex["R"], ex["A"], ex["negation_flags"])
        results.append({
            "answer_id": ex["answer_id"],
            "question_id": ex["question_id"],
            "variant_type": ex["variant_type"],
            "derived_from_train_id": ex["derived_from_train_id"],
            "human_scores": ex["human_scores"],
            "pred_scores": out["per_criterion_scores"].tolist(),
            "mean_spread": out["spread"].mean().item(),
        })
df = pd.DataFrame(results)
df.to_csv("results/benchmark_results.csv", index=False)
```

## Step 3 — Compute Core Accuracy Metrics (Overall and Per `variant_type`)
For every row, compute per-criterion absolute error, then aggregate:
```python
def per_criterion_errors(row):
    human = list(row["human_scores"].values())
    pred = row["pred_scores"]
    return [abs(h - p) for h, p in zip(human, pred)]

df["mae"] = df.apply(lambda r: sum(per_criterion_errors(r)) / len(r["pred_scores"]), axis=1)
df["within_1_mark"] = df.apply(lambda r: all(e <= 1.0 for e in per_criterion_errors(r)), axis=1)
```
Report, grouped by `variant_type`:
- Mean absolute error
- % of examples within 1 mark
- Mean attention spread

```python
summary = df.groupby("variant_type").agg(
    mean_mae=("mae", "mean"),
    pct_within_1=("within_1_mark", "mean"),
    mean_spread=("mean_spread", "mean"),
    n=("answer_id", "count"),
)
```

## Step 4 — Precision/Recall/False-Positive/False-Negative Per Criterion (Binary "met/not-met" framing)
Define a threshold (e.g. score ≥ 50% of max_marks → "criterion met"). Compute:
```python
from sklearn.metrics import precision_score, recall_score, confusion_matrix

y_true = [...]  # human "met" labels, flattened across all criteria in test set
y_pred = [...]  # model "met" labels, same order

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
```
Report overall and broken down by `variant_type`. This directly answers the "false positive / false negative / precision" feedback point.

## Step 5 — Paired Delta Analysis (the core deliverable of this benchmark design)
This is the specific analysis the project owner asked for — using the manifest's `derived_from_train_id` link, compute the **score delta between each Y-type test answer and its X-type source answer's score**, per `variant_type`:

```python
train_scores = {ex["answer_id"]: ex["human_scores"] for ex in train_examples}  # reload if needed

def paired_delta(row):
    source_human = train_scores[row["derived_from_train_id"]]
    source_pred_score = ...  # re-run model on the source train example, or cache its prediction from training eval
    test_pred_mean = sum(row["pred_scores"]) / len(row["pred_scores"])
    source_pred_mean = sum(source_pred_score.values()) / len(source_pred_score)
    return test_pred_mean - source_pred_mean

df["paired_delta"] = df.apply(paired_delta, axis=1)
```

Expected directional results, to check explicitly and report whether each was observed:
| `variant_type` | Expected paired_delta direction |
|---|---|
| `paraphrase` | ~0 (score should barely move — same meaning) |
| `scattered_evidence` | ~0 (attention should still find and combine the evidence) |
| `diffuse_padded` | ~0 to slightly negative (padding shouldn't fool the model into a lower score, but shouldn't inflate it either) |
| `partial_credit_shift` | Negative, proportional to the removed content |
| `negation_flipped` | Strongly negative |
| `confidently_wrong` | Strongly negative, paired with **low spread** (confident, not ambiguous) |
| `typo_injected` | ~0 (small negative at most — graceful degradation) |
| `genuinely_ambiguous` | Variable score, but **high spread** specifically |

Report actual vs. expected for every category — **do not omit or soften a category where the result doesn't match expectation.** A mismatch here is the single most useful finding for the presenter to know before the pitch, not something to bury.

## Step 6 — Generate the Report (`results/benchmark_report.md`)
Auto-generate a markdown report (via `src/report.py`) containing:
1. Overall accuracy (MAE, % within 1 mark, precision/recall).
2. Full per-`variant_type` breakdown table (from Step 3).
3. Paired-delta table (from Step 5) with actual vs. expected direction, flagged where mismatched.
4. 3–5 concrete example rows (question, X-type source answer, Y-type test answer, human score, predicted score, attention spread) chosen to be demo-usable — prioritize one clear success case and one clear failure/limitation case per major category (scattered evidence, negation, ambiguity, typo).
5. A short auto-generated "weakest categories" section: sort `variant_type` groups by `mean_mae` descending, list the bottom 2 explicitly.

## Checkpoint
- `results/benchmark_results.csv` contains one row per test example with all required fields.
- `results/benchmark_report.md` exists, contains all 5 sections above, and every `variant_type` from the dataset appears in both the accuracy table and the paired-delta table.
- The report explicitly states whether each expected-direction hypothesis in the Step 5 table held or didn't — no category silently omitted.
