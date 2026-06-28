import sys
sys.path.insert(0, ".")
import torch
import pytest


def test_fft_mix_output_shape():
    from augmentation.fft_mix import amplitude_spectrum_mix
    src = torch.rand(4, 3, 112, 112)
    ref = torch.rand(4, 3, 112, 112)
    out = amplitude_spectrum_mix(src, ref, ratio=0.1)
    assert out.shape == src.shape


def test_fft_mix_blending_affects_output():
    """Blending with ratio>0 and a different ref must change the output."""
    from augmentation.fft_mix import amplitude_spectrum_mix
    torch.manual_seed(0)
    src = torch.rand(2, 3, 112, 112)
    ref = torch.rand(2, 3, 112, 112)
    out = amplitude_spectrum_mix(src, ref, ratio=0.5)
    assert not torch.allclose(out, src, atol=1e-4), \
        "blending with ratio=0.5 must change the output"


def test_fft_mix_ratio_zero_preserves_src():
    """ratio=0 means no blending region, output should equal input."""
    from augmentation.fft_mix import amplitude_spectrum_mix
    src = torch.rand(2, 3, 112, 112)
    ref = torch.rand(2, 3, 112, 112)
    out = amplitude_spectrum_mix(src, ref, ratio=0.0)
    # Phase preserved means reconstruction should be near-identical to src
    assert torch.allclose(out, src, atol=1e-4)


def test_transface_vit_train_returns_tuple():
    from backbones import get_model
    model = get_model("transface_vit_b", num_features=512)
    model.train()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, tuple), "train mode must return (emb, entropy) tuple"
    emb, entropy = out
    assert emb.shape == (2, 512), f"expected (2,512) got {emb.shape}"
    assert entropy.shape == (2,), f"expected (2,) got {entropy.shape}"


def test_transface_vit_eval_returns_tensor():
    from backbones import get_model
    model = get_model("transface_vit_b", num_features=512)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor), "eval mode must return plain tensor"
    assert out.shape == (2, 512)


def test_transface_vit_entropy_nonnegative():
    from backbones import get_model
    model = get_model("transface_vit_b", num_features=512)
    model.train()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        emb, entropy = model(x)
    assert (entropy >= 0).all(), "entropy must be non-negative"


def test_transface_vit_l_shape():
    from backbones import get_model
    model = get_model("transface_vit_l", num_features=512)
    model.train()
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        emb, entropy = model(x)
    assert emb.shape == (2, 512)
    assert entropy.shape == (2,)
