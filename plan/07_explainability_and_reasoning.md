# 07 — Explainability and Score Reasoning

## Objective
For every score RCAJ-X produces, generate a **human-readable explanation** grounded directly in the model's own internal attention weights — not a separate post-hoc justification model, and not an LLM call. This is the core differentiator to test against the teammate's NLP approach: can each system explain *why* it gave the score it gave, using its own actual computation, not a plausible-sounding story bolted on afterward.

This directly answers the requirement: *"provide reasoning for why a specific score was given."*

## Why Attention-Based Explanation Is Legitimate (Not Just a Nice Story)
The attention weights (`out["attn_weights"]`, shape `(n_heads, n_criteria, n_chunks)`) are literally the values the scoring MLP consumed to produce the score — `context_i` is a weighted sum of chunk representations using exactly these weights. So "the model attended most to chunk 3 and chunk 7 for this criterion" is not an approximation of the model's reasoning — it **is** the model's reasoning, in a form a human can read. This is the concrete advantage over a flat-pooled MLP score, which has no such structure to expose. Make this point explicitly in the comparison report (`09`) if the teammate's approach cannot produce an equivalently grounded explanation.

## Step 1 — Implement `src/explain.py`

```python
def generate_explanation(model, R, A, raw_chunks, criteria, negation_flags, spread_threshold=0.4):
    out = model(R, A, negation_flags)
    weights = out["attn_weights"]              # (h, n_c, n_a)
    scores = out["per_criterion_scores"]
    spread = out["spread"]                      # (n_c, h)

    explanations = []
    for i, criterion in enumerate(criteria):
        # average attention across heads for this criterion, for evidence selection
        avg_weights = weights[:, i, :].mean(dim=0)     # (n_a,)
        top_k = min(2, len(raw_chunks))
        top_idx = avg_weights.topk(top_k).indices.tolist()
        top_chunks = [raw_chunks[j] for j in top_idx]
        top_weights = [round(avg_weights[j].item(), 3) for j in top_idx]

        mean_spread_i = spread[i].mean().item()
        is_ambiguous = mean_spread_i < spread_threshold  # low max-mean spread => diffuse attention

        neg_flagged = bool(negation_flags[i].item())

        reason = build_reason_text(
            criterion_text=criterion["text"],
            score=scores[i].item(),
            max_marks=criterion["max_marks"],
            top_chunks=top_chunks,
            top_weights=top_weights,
            is_ambiguous=is_ambiguous,
            neg_flagged=neg_flagged,
        )

        explanations.append({
            "criterion_id": criterion["criterion_id"],
            "criterion_text": criterion["text"],
            "score": round(scores[i].item(), 2),
            "max_marks": criterion["max_marks"],
            "evidence_chunks": top_chunks,
            "evidence_weights": top_weights,
            "confidence": "review_recommended" if is_ambiguous else "high_confidence",
            "negation_flag": neg_flagged,
            "reason_text": reason,
        })
    return explanations
```

## Step 2 — Implement the Template-Based Reason Generator

Keep this **rule-based/templated**, not LLM-generated — the explanation must be directly traceable to the numeric evidence (weights, spread, negation flag) with no risk of the explanation text saying something the model's actual computation doesn't support. Template logic:

```python
def build_reason_text(criterion_text, score, max_marks, top_chunks, top_weights, is_ambiguous, neg_flagged):
    pct = score / max_marks if max_marks else 0
    evidence_str = "; ".join(f'"{c}" (weight={w})' for c, w in zip(top_chunks, top_weights))

    if neg_flagged:
        return (f"Awarded {score:.1f}/{max_marks}. The model flagged a possible negation/contradiction "
                f"mismatch between the criterion and the most relevant answer text: {evidence_str}. "
                f"This lowers confidence in a straightforward match and the score reflects that.")

    if is_ambiguous:
        return (f"Awarded {score:.1f}/{max_marks}, but flagged for review. Evidence for this criterion "
                f"was spread across multiple parts of the answer rather than concentrated in one place "
                f"({evidence_str}), which the model treats as a sign of a partial or ambiguous match "
                f"rather than a confident one.")

    if pct >= 0.75:
        return (f"Awarded {score:.1f}/{max_marks} with high confidence. The answer directly addresses "
                f"'{criterion_text}' — most relevant evidence: {evidence_str}.")

    return (f"Awarded {score:.1f}/{max_marks}. The model found some relevant content "
            f"({evidence_str}) but it does not fully satisfy '{criterion_text}'.")
```

Adjust wording as needed, but preserve the structural rule: **every branch must reference the actual numeric evidence** (weights, spread, negation flag) — never emit generic filler text disconnected from the specific example's computed values.

## Step 3 — Generate a Sample Explanation Set for Manual Review

Run `generate_explanation` over 15–20 examples spanning all `variant_type` categories from the Y-type test set, save to `results/explanations_sample.json`. Manually spot-check (or have the presenter spot-check) whether the generated reasoning text actually matches what a human would say looking at the same answer — this is a qualitative check, not a metric, but it's required before moving to the Streamlit app.

## Step 4 — Explanation Quality Checks (Automated)
Implement two automated sanity checks in `src/explain.py`:
```python
def check_evidence_grounding(explanation, raw_chunks):
    # every evidence_chunk must be verbatim present in raw_chunks — catches any drift/hallucination bug
    return all(c in raw_chunks for c in explanation["evidence_chunks"])

def check_confidence_consistency(explanation, spread_threshold=0.4):
    # confidence label must match the spread value that produced it
    ...
```
Run these over the full sample set and report the pass rate — should be 100% by construction, and a failure indicates a bug in the pipeline, not a model quality issue.

## Checkpoint
- `src/explain.py` implemented and runs over the full Y-type test set without error.
- `results/explanations_sample.json` contains 15–20 explanations spanning all `variant_type` categories.
- `check_evidence_grounding` and `check_confidence_consistency` both pass 100% over the sample set.
- At least one explanation from each of these categories manually reviewed and judged sensible: `fully_correct`-derived, `negation_flipped`, `genuinely_ambiguous`, `scattered_evidence`.
