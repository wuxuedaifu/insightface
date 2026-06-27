import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backbones.iresnet import iresnet50


def test_iresnet50_norm_output_shape():
    model = iresnet50(False, fp16=False, num_features=512, norm_output=True)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    emb, norm = model(x)
    assert emb.shape == (2, 512), f"Expected (2,512), got {emb.shape}"
    assert norm.shape == (2, 1),  f"Expected (2,1),   got {norm.shape}"


def test_iresnet50_norm_output_is_unit_norm():
    # Use train mode so BN normalises using actual batch stats, keeping
    # features in a finite range even for a randomly-initialised model.
    model = iresnet50(False, fp16=False, num_features=512, norm_output=True)
    model.train()
    with torch.no_grad():
        x = torch.randn(4, 3, 112, 112)
        emb, norm = model(x)
    l2 = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(l2, torch.ones(4), atol=1e-4), \
        f"Embeddings not unit-normed: {l2}"


def test_iresnet50_default_output_unchanged():
    """norm_output=False (default) must return a plain tensor, not a tuple."""
    model = iresnet50(False, fp16=False, num_features=512)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    assert isinstance(out, torch.Tensor), "Default output should be a Tensor"
    assert out.shape == (2, 512)
