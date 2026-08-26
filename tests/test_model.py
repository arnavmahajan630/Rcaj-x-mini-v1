import torch
import pytest
import os
import sys

# Add the root directory to the sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import RCAJ_X

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
