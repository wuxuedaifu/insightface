import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import pytest


def test_mamba_s_forward_shape():
    from backbones import get_model
    model = get_model("mamba_s", num_features=512)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 512), f"expected (2,512) got {out.shape}"


def test_mamba_b_forward_shape():
    from backbones import get_model
    model = get_model("mamba_b", num_features=512)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 512)


def test_mamba_backward_runs():
    from backbones import get_model
    model = get_model("mamba_s", num_features=64)
    model.train()
    x = torch.randn(2, 3, 112, 112)
    weight = F.normalize(torch.randn(64, 64), dim=1)
    emb = model(x)
    logits = F.normalize(emb, dim=1) @ weight.T
    loss = F.cross_entropy(logits, torch.zeros(2, dtype=torch.long))
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g < 1e6 for g in grad_norms), "suspicious gradient explosion"


def test_mamba_eval_deterministic():
    from backbones import get_model
    model = get_model("mamba_s", num_features=512)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "eval mode must be deterministic"
