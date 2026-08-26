# 03 — Model Architecture

## Objective
Implement `src/model.py` containing the full RCAJ-X model: multi-head cross-attention + ambiguity/negation-aware scoring head + aggregation. Implement exactly as specified below — this is the locked architecture, not a design decision left to the agent.

## Required Classes

```python
import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, d_k: int = 64, d_v: int = 64):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_k
        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, R: torch.Tensor, A: torch.Tensor):
        # R: (n_criteria, d_model), A: (n_chunks, d_model)
        n_c, n_a = R.shape[0], A.shape[0]
        Q = self.W_q(R).view(n_c, self.n_heads, self.d_k).transpose(0, 1)   # (h, n_c, d_k)
        K = self.W_k(A).view(n_a, self.n_heads, self.d_k).transpose(0, 1)   # (h, n_a, d_k)
        V = self.W_v(A).view(n_a, self.n_heads, self.d_k).transpose(0, 1)   # (h, n_a, d_v)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)   # (h, n_c, n_a)
        weights = torch.softmax(scores, dim=-1)                            # (h, n_c, n_a), sums to 1 per criterion per head
        context = torch.matmul(weights, V)                                 # (h, n_c, d_v)

        context = context.transpose(0, 1).reshape(n_c, -1)                 # (n_c, h*d_v)
        out = self.W_o(context)                                            # (n_c, d_model)
        return out, weights


class ScoringHead(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, hidden: int = 32):
        super().__init__()
        # input = context (d_model) + per-head spread (n_heads) + negation flag (1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + n_heads + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, context: torch.Tensor, weights: torch.Tensor, negation_flags: torch.Tensor):
        # weights: (h, n_c, n_a) -> spread per head per criterion
        max_w = weights.max(dim=-1).values      # (h, n_c)
        mean_w = weights.mean(dim=-1)            # (h, n_c)
        spread = (max_w - mean_w).transpose(0, 1)  # (n_c, h)
        x = torch.cat([context, spread, negation_flags.unsqueeze(-1)], dim=-1)
        return self.mlp(x).squeeze(-1), spread     # return spread too, needed for benchmark analysis


class RCAJ_X(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, d_k: int = 64, d_v: int = 64, hidden: int = 32):
        super().__init__()
        self.attn = MultiHeadCrossAttention(d_model, n_heads, d_k, d_v)
        self.score_head = ScoringHead(d_model, n_heads, hidden)

    def forward(self, R: torch.Tensor, A: torch.Tensor, negation_flags: torch.Tensor, criterion_weights: torch.Tensor = None):
        context, weights = self.attn(R, A)
        per_criterion_scores, spread = self.score_head(context, weights, negation_flags)
        final_score = None
        if criterion_weights is not None:
            final_score = (per_criterion_scores * criterion_weights).sum()
        return {
            "per_criterion_scores": per_criterion_scores,
            "final_score": final_score,
            "attn_weights": weights,
            "spread": spread,
        }
```

## Design Constraints (Do Not Deviate)
- `n_heads` must be a constructor parameter, not hardcoded — `04_training_pipeline.md` requires sweeping `n_heads ∈ {2, 4, 6}` for the ablation.
- The model must return `attn_weights` and `spread` in its output dict, not just the score — the benchmarking step depends on inspecting these directly per example, not just the final scalar.
- Do not add additional layers (no second attention layer, no LayerNorm/residual stacking). This is a deliberate one-layer, cost-bounded architecture — see the exclusion notes for why depth is capped.
- Negation flags must be computed upstream (in `02_preprocessing_pipeline.md`'s pipeline) and passed in as a tensor of shape `(n_criteria,)` — one flag per criterion, computed against whichever chunk that criterion's top attention weight (averaged across heads) points to.

## Unit Tests to Implement (`tests/test_model.py`)
```python
def test_softmax_normalization():
    model = RCAJ_X(n_heads=4)
    R, A = torch.randn(3, 384), torch.randn(6, 384)
    neg = torch.zeros(3)
    out = model(R, A, neg)
    assert torch.allclose(out["attn_weights"].sum(dim=-1), torch.ones(4, 3), atol=1e-5)

def test_output_shapes():
    model = RCAJ_X(n_heads=4)
    R, A = torch.randn(5, 384), torch.randn(10, 384)
    neg = torch.zeros(5)
    weights = torch.ones(5)
    out = model(R, A, neg, weights)
    assert out["per_criterion_scores"].shape == (5,)
    assert out["final_score"].dim() == 0

def test_head_count_configurable():
    for h in [2, 4, 6]:
        model = RCAJ_X(n_heads=h)
        R, A = torch.randn(3, 384), torch.randn(4, 384)
        out = model(R, A, torch.zeros(3))
        assert out["attn_weights"].shape[0] == h
```

## Checkpoint
- All three unit tests pass.
- `RCAJ_X(n_heads=4)` instantiated and run once on real preprocessed data from `data/train_embedded.pt` without shape errors.
