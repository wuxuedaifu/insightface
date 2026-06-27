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


from losses import AdaFaceLoss


def test_adaface_loss_output_shape():
    loss_fn = AdaFaceLoss(m=0.4, h=0.333, s=64.0, t_alpha=1.0)
    # 4 samples, 10 classes; -1 means not a local positive class
    logits = torch.rand(4, 10).clamp(-1 + 1e-3, 1 - 1e-3)
    norms  = torch.tensor([[22.0], [18.0], [25.0], [10.0]])
    labels = torch.tensor([[2], [5], [-1], [0]])
    out = loss_fn(logits, norms, labels)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"


def test_adaface_loss_scales_by_s():
    loss_fn = AdaFaceLoss(m=0.0, h=0.0, s=64.0, t_alpha=1.0)
    # With m=0 and h=0, margin_scaler=0, no target class on this shard
    logits = torch.tensor([[0.8, 0.2, 0.5]])
    norms  = torch.tensor([[20.0]])
    labels = torch.tensor([[-1]])
    out = loss_fn(logits, norms, labels)
    # acos/cos round-trip: cos(acos(x)) = x, so final = logits * s
    assert torch.allclose(out, logits * 64.0, atol=1e-4), \
        f"Expected {logits * 64.0}, got {out}"


def test_adaface_loss_ema_buffers_update():
    loss_fn = AdaFaceLoss(m=0.4, h=0.333, s=64.0, t_alpha=1.0)
    logits = torch.rand(2, 5).clamp(-1 + 1e-3, 1 - 1e-3)
    norms  = torch.tensor([[30.0], [10.0]])
    labels = torch.tensor([[0], [-1]])
    loss_fn(logits, norms, labels)
    # With t_alpha=1.0 the EMA fully adopts the batch mean (30+10)/2 = 20
    assert abs(loss_fn.batch_mean.item() - 20.0) < 1.0, \
        f"batch_mean not updated: {loss_fn.batch_mean.item()}"
