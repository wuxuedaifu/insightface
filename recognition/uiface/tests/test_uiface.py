"""
Smoke tests for the UIFace generation module.

All tests run on CPU without pre-trained weights, using tiny model configs so
each test finishes in a few seconds.  Run from the repo root:

    cd recognition/uiface
    python -m pytest tests/test_uiface.py -v
"""
import sys
import os

# Make uiface importable without package installation
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest

from diffusion.ddpm import (
    DenoisingDiffusionProbabilisticModel,
    compute_beta_schedule,
    precompute_schedule_constants,
)
from models.autoencoder.modules import Decoder, Encoder
from models.autoencoder.quantization import VectorQuantizer2
from models.diffusion.unet import ConditionalUNet


# ---------------------------------------------------------------------------
# Tiny model configs used across tests
# ---------------------------------------------------------------------------

VQGAN_CFG = dict(
    ch=32,  # must be divisible by GroupNorm's 32 groups
    out_ch=3,
    ch_mult=(1, 2),
    num_res_blocks=1,
    attn_resolutions=[],
    dropout=0.0,
    in_channels=3,
    resolution=16,
    z_channels=3,
    double_z=False,
)

UNET_UNCOND_CFG = dict(
    input_channels=3,
    initial_channels=32,  # must be divisible by GroupNorm's 32 groups
    channel_multipliers=(1, 2),
    is_attention=(False, False),
    attention_heads=2,
    attention_head_channels=16,
    n_blocks_per_resolution=1,
    condition_type="AddPlusGN",
    context_input_channels=8,
    context_channels=32,
    is_context_conditional=False,
)

UNET_COND_CFG = dict(
    **{k: v for k, v in UNET_UNCOND_CFG.items() if k != "is_context_conditional"},
    is_context_conditional=True,
    context_dropout_probability=0.1,
    unconditioned_probability=0.1,
    learn_empty_context=True,
)


# ---------------------------------------------------------------------------
# 1. Beta schedule
# ---------------------------------------------------------------------------


def test_beta_schedule_linear_range():
    betas = compute_beta_schedule(100, "linear")
    assert betas.shape == (100,)
    assert float(betas.min()) > 0
    assert float(betas.max()) < 1
    # monotonically increasing
    assert (betas[1:] >= betas[:-1]).all()


def test_beta_schedule_cosine_range():
    betas = compute_beta_schedule(100, "cosine")
    assert betas.shape == (100,)
    assert float(betas.min()) >= 0
    assert float(betas.max()) <= 1  # last beta may reach 1.0 at the cosine tail


def test_precompute_constants_shapes_and_range():
    betas = compute_beta_schedule(50, "linear")
    consts = precompute_schedule_constants(betas)
    for key in ("alpha_bars", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"):
        assert consts[key].shape == (50,), key
    # alpha_bars must be in (0, 1) and strictly decreasing
    ab = consts["alpha_bars"]
    assert float(ab.min()) > 0
    assert float(ab.max()) <= 1
    assert (ab[:-1] > ab[1:]).all()


# ---------------------------------------------------------------------------
# 2. VQ-GAN Encoder
# ---------------------------------------------------------------------------


def test_encoder_output_shape():
    enc = Encoder(**VQGAN_CFG)
    enc.eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        z = enc(x)
    # ch=16, ch_mult=(1,2) → 2 downsample levels → 16//2=8 spatial dim
    assert z.shape == (2, 3, 8, 8), z.shape


def test_encoder_output_is_finite():
    enc = Encoder(**VQGAN_CFG)
    enc.eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        z = enc(x)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# 3. VQ-GAN Decoder
# ---------------------------------------------------------------------------


def test_decoder_output_shape():
    dec = Decoder(**VQGAN_CFG)
    dec.eval()
    z = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        out = dec(z)
    assert out.shape == (2, 3, 16, 16), out.shape


# ---------------------------------------------------------------------------
# 4. VectorQuantizer
# ---------------------------------------------------------------------------


def test_vector_quantizer_output_shapes():
    vq = VectorQuantizer2(n_e=64, e_dim=3, beta=0.25)
    z = torch.randn(2, 3, 8, 8)
    quant, loss, (_, _, indices) = vq(z)
    assert quant.shape == z.shape
    assert indices.shape == (2 * 8 * 8,)
    assert loss.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# 5. Encoder → VQ → Decoder round-trip
# ---------------------------------------------------------------------------


def test_encoder_vq_decoder_roundtrip_shape():
    enc = Encoder(**VQGAN_CFG)
    vq = VectorQuantizer2(n_e=64, e_dim=3, beta=0.25)
    dec = Decoder(**VQGAN_CFG)
    # quant_conv / post_quant_conv are 1×1 projections (z_channels→embed_dim)
    quant_conv = torch.nn.Conv2d(3, 3, 1)
    post_quant_conv = torch.nn.Conv2d(3, 3, 1)

    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        h = quant_conv(enc(x))
        quant, _, _ = vq(h)
        out = dec(post_quant_conv(quant))
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 6. ConditionalUNet — unconditional forward
# ---------------------------------------------------------------------------


def test_unet_uncond_output_shape():
    unet = ConditionalUNet(**UNET_UNCOND_CFG)
    unet.eval()
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 100, (2,))
    with torch.no_grad():
        noise_pred, self_attn, cross_attn = unet(x, t)
    assert noise_pred.shape == x.shape


def test_unet_uncond_output_is_finite():
    unet = ConditionalUNet(**UNET_UNCOND_CFG)
    unet.eval()
    x = torch.randn(1, 3, 16, 16)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        noise_pred, _, _ = unet(x, t)
    assert torch.isfinite(noise_pred).all()


# ---------------------------------------------------------------------------
# 7. ConditionalUNet — conditional forward (identity embedding as context)
# ---------------------------------------------------------------------------


def test_unet_cond_output_shape():
    unet = ConditionalUNet(**UNET_COND_CFG)
    unet.eval()
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 100, (2,))
    context = torch.randn(2, 8)  # identity embedding (context_input_channels=8)
    with torch.no_grad():
        noise_pred, _, _ = unet(x, t, context=context)
    assert noise_pred.shape == x.shape


def test_unet_null_context_equals_empty_embedding():
    """Passing context=None must not crash; it uses the empty_context_embedding."""
    unet = ConditionalUNet(**UNET_COND_CFG)
    unet.eval()
    x = torch.randn(1, 3, 16, 16)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out, _, _ = unet(x, t, context=None)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 8. DDPM training forward (loss computation)
# ---------------------------------------------------------------------------


def _make_ddpm(T=20):
    unet = ConditionalUNet(**UNET_UNCOND_CFG)
    return DenoisingDiffusionProbabilisticModel(unet, T=T, schedule_type="linear")


def test_ddpm_training_loss_is_positive_scalar():
    ddpm = _make_ddpm()
    ddpm.train()
    x0 = torch.randn(2, 3, 16, 16)
    loss = ddpm(x0)
    assert loss.ndim == 0
    assert float(loss) > 0


def test_ddpm_training_loss_is_finite():
    ddpm = _make_ddpm()
    ddpm.train()
    x0 = torch.randn(2, 3, 16, 16)
    loss = ddpm(x0)
    assert torch.isfinite(loss)


def test_ddpm_loss_backprop():
    ddpm = _make_ddpm()
    ddpm.train()
    x0 = torch.randn(2, 3, 16, 16)
    loss = ddpm(x0)
    loss.backward()
    for name, p in ddpm.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


# ---------------------------------------------------------------------------
# 9. DDPM DDIM sampling (2-step, CPU)
# ---------------------------------------------------------------------------


def test_ddpm_ddim_sampling_shape():
    ddpm = _make_ddpm(T=20)
    ddpm.eval()
    size = (3, 16, 16)
    x_T = torch.randn(1, *size)
    with torch.no_grad():
        # ddim_step=2 → only 2 denoising iterations
        x0_hat = ddpm.sample_ddim(
            n_samples=1, size=size, x_T=x_T, ddim_step=2
        )
    assert x0_hat.shape == (1, *size)


def test_ddpm_ddim_sampling_is_finite():
    ddpm = _make_ddpm(T=20)
    ddpm.eval()
    size = (3, 16, 16)
    x_T = torch.randn(2, *size)
    with torch.no_grad():
        x0_hat = ddpm.sample_ddim(
            n_samples=2, size=size, x_T=x_T, ddim_step=2
        )
    assert torch.isfinite(x0_hat).all()


def test_ddpm_ddim_sampling_different_noise_gives_different_output():
    ddpm = _make_ddpm(T=20)
    ddpm.eval()
    size = (3, 16, 16)
    torch.manual_seed(0)
    x_T1 = torch.randn(1, *size)
    torch.manual_seed(999)
    x_T2 = torch.randn(1, *size)
    with torch.no_grad():
        out1 = ddpm.sample_ddim(1, size, x_T=x_T1, ddim_step=2)
        out2 = ddpm.sample_ddim(1, size, x_T=x_T2, ddim_step=2)
    assert not torch.allclose(out1, out2)
