# 04 — Training Pipeline

## Objective
Implement `src/train.py`: train RCAJ-X on `data/train_embedded.pt` (X-type answers only), with a small hyperparameter sweep, and save the best checkpoint to `checkpoints/rcaj_x_best.pt`. **Never touch `data/test_embedded.pt` in this file** — the test set is reserved entirely for `05_benchmarking_and_reporting.md`.

## Step 1 — Train/Validation Split (within training data only)
```python
from sklearn.model_selection import train_test_split

train_examples, val_examples = train_test_split(
    all_train_examples, test_size=0.2, random_state=42,
    stratify=[ex["question_id"] for ex in all_train_examples]
)
```
This validation split is carved **only from `data/train/`** — it is used for hyperparameter selection and early stopping, and is distinct from the Y-type benchmark test set. Do not conflate the two.

## Step 2 — Training Loop
```python
import torch
import torch.nn as nn

def train_one_config(n_heads, d_k, lr, weight_decay, epochs=40):
    model = RCAJ_X(n_heads=n_heads, d_k=d_k, d_v=d_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for ex in train_examples:
            optimizer.zero_grad()
            out = model(ex["R"], ex["A"], ex["negation_flags"])
            target = torch.tensor(list(ex["human_scores"].values()), dtype=torch.float32)
            loss = loss_fn(out["per_criterion_scores"], target)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_examples, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    return best_state, best_val_loss

def evaluate_loss(model, examples, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ex in examples:
            out = model(ex["R"], ex["A"], ex["negation_flags"])
            target = torch.tensor(list(ex["human_scores"].values()), dtype=torch.float32)
            total += loss_fn(out["per_criterion_scores"], target).item()
    return total / len(examples)
```

Note: since `R`/`A` are ragged (variable `n_criteria`/`n_chunks` per example), process one example at a time rather than building padded batches — acceptable at this dataset scale (150–300 examples, CPU trains in well under a minute per epoch).

## Step 3 — Hyperparameter Sweep
Run `train_one_config` across this grid, log every result to `results/ablation_results.csv`:
```python
grid = [
    {"n_heads": h, "d_k": dk, "lr": lr, "weight_decay": wd}
    for h in [2, 4, 6]
    for dk in [32, 64]
    for lr in [1e-3, 5e-4]
    for wd in [1e-4]
]
```
For each config, record: `n_heads, d_k, lr, weight_decay, best_val_loss`. Select the config with the lowest `best_val_loss`.

**This ablation table is a required deliverable** — it is the direct, numeric answer to a specific piece of feedback the project has already received about justifying hyperparameter choices. Do not skip it or run only one configuration.

## Step 4 — Save Best Model
```python
torch.save({
    "state_dict": best_state,
    "config": best_config,
    "val_loss": best_val_loss,
}, "checkpoints/rcaj_x_best.pt")
```

## Step 5 — Sanity Checks Before Moving On
- Print 5 example predictions vs. human scores from the validation set, confirm they're in a plausible range (not all identical, not wildly off).
- Confirm `results/ablation_results.csv` has one row per grid config (12 rows for the grid above).

## Checkpoint
- `checkpoints/rcaj_x_best.pt` exists and loads without error.
- `results/ablation_results.csv` contains the full sweep with all configs and their validation losses.
- Best config's validation loss is meaningfully below the loss of a "predict the mean training score" baseline (compute this baseline explicitly and report both numbers).
