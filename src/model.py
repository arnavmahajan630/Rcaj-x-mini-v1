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
