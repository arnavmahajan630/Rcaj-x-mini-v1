# 01 — Score Bounding & Explanation-Score Consistency Fix

## The Bug (Confirmed Structurally in an Earlier Session)
`src/model.py`'s `ScoringHead.forward` ends with a bare `nn.Linear(hidden, 1)` — an unbounded linear regression output. Nothing in the architecture stops a prediction from landing below 0 or above `max_marks`. A reproduction using the exact same architecture and comparable data volume showed **19 of 48 predictions (~40%) landing outside `[0, max_marks]`** on training data alone, with the worst case at 2.55 for a criterion capped at 2. This is almost certainly the root cause of "scores sometimes exceed the max" and is independent of data quality — fix the architecture regardless of what happens with the data upgrade in `02`.

## Fix 1 — Bound the Output (Required, Do This First)
```python
class ScoringHead(nn.Module):
    def __init__(self, d_model=384, n_heads=4, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + n_heads + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, context, weights, negation_flags, max_marks):
        max_w = weights.max(dim=-1).values
        mean_w = weights.mean(dim=-1)
        spread = (max_w - mean_w).transpose(0, 1)
        x = torch.cat([context, spread, negation_flags.unsqueeze(-1)], dim=-1)
        raw = self.mlp(x).squeeze(-1)
        # Bound to [0, max_marks] via sigmoid scaling — smooth, differentiable, no dead zones.
        bounded_score = torch.sigmoid(raw) * max_marks
        return bounded_score, spread
```
Notes:
- `max_marks` must now be passed into the forward call — thread it through from `RCAJ_X.forward` (it's already available per-criterion from the rubric, just wasn't being used at this point in the current implementation).
- Sigmoid is preferred over a hard `torch.clamp` — clamping alone still lets gradients vanish/explode near the boundary during training and doesn't fix the underlying unboundedness that caused the drift in the first place; sigmoid scaling constrains the function's *range*, not just the output post-hoc.
- Retrain from scratch after this change — do not attempt to patch an existing checkpoint's behavior with post-hoc clamping alone and call it fixed; the model needs to learn within the new bounded output space.

## Fix 2 — Loss Function Should Match the Bounded Target Space
If `train.py` currently uses plain `MSELoss` against raw `human_scores`, this still works correctly with the bounded output (targets naturally fall in `[0, max_marks]` too), but confirm:
- Targets are never negative and never exceed `max_marks` in the training data itself (a data validation step, not a model change — add an assertion in `train.py` that raises if any `human_scores` value falls outside `[0, max_marks]` for its criterion, catching a bad label before it corrupts training).

## Fix 3 — Explanation-Score Consistency Check (New, Addresses "Reasoning Correct But Score Wrong")
The project owner's observation — reasoning correctly identifies an issue, but the score doesn't reflect it — is a symptom worth guarding against structurally, not just fixing via bounding. Add an automated consistency check in `src/explain.py`:

```python
def check_explanation_score_consistency(explanation):
    """
    Flags cases where the qualitative reasoning and the quantitative score disagree.
    This doesn't fix a disagreement — it surfaces it, which is what was missing before.
    """
    pct = explanation["score"] / explanation["max_marks"] if explanation["max_marks"] else 0
    flags = []
    if explanation["confidence"] == "review_recommended" and pct > 0.85:
        flags.append("HIGH SPREAD (ambiguous) but score is near-max — inconsistent")
    if explanation["negation_flag"] and pct > 0.5:
        flags.append("Negation mismatch flagged but score is above half-credit — inconsistent")
    if "does not fully satisfy" in explanation["reason_text"] and pct > 0.85:
        flags.append("Reason text says partial match but score is near-max — inconsistent")
    return flags
```
Run this over every prediction in `benchmark.py` and report a `results/consistency_flags.csv` — a nonzero flag count is a genuine bug signal (either the scoring head or the explanation logic is wrong, and this tells you which examples to inspect first) rather than something to silently average away in an aggregate accuracy number.

## Verification (Required Before Declaring This Fixed)
1. Retrain with the bounded head on the current dataset.
2. Re-run the same out-of-bounds check from the earlier diagnostic session: assert `0 <= pred <= max_marks` for every prediction across the full training and test sets — should now be **0 violations**, not a reduced count. Any violation after this fix means the bounding wasn't wired correctly (e.g. `max_marks` not passed through) — treat as a blocking bug, not a residual to tolerate.
3. Run the new consistency check (`Fix 3`) and confirm the flagged-count drops substantially versus a pre-fix baseline run (keep the pre-fix numbers as a comparison point — this quantifies whether the reported symptom actually improved, not just whether bounds are technically satisfied).

## Checkpoint
- `0` out-of-bounds predictions on both train and test sets, verified by an explicit assertion, not eyeballing.
- `results/consistency_flags.csv` generated and reviewed; flagged count is low and each flagged case is individually inspected, not just counted.
