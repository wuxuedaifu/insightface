# TransFace FFT Augmentation + MambaVision Configs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TransFace's FFT amplitude-spectrum augmentation (with a patch-entropy–guided ViT backbone), large-ViT training configs (LVFace-style), MambaVision configs, and a dedicated `train_transface.py` script into `recognition/arcface_torch/`.

**Architecture:** Three independent additions on top of the existing `arcface_torch` stack. The `MambaVit` backbone (`backbones/mamba_vit.py`) and its `get_model` registrations (`mamba_s/b/l`) are **already committed** — do not re-implement them. What remains: (1) `augmentation/fft_mix.py` standalone FFT function, (2) `backbones/transface_vit.py` subclass of `VisionTransformer` that emits patch entropy, (3) `train_transface.py` training script, and (4) seven config files.

**Tech Stack:** PyTorch 2.x, `torch.fft`, `timm`, existing `arcface_torch` infra (`get_model`, `PartialFC_V2`, `CombinedMarginLoss`, `get_config`, `get_dataloader`)

---

## Context for implementers

All work lives in `recognition/arcface_torch/`. Run all commands from that directory.

Python binary: `/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9`
Run tests: `python3.9 -m pytest tests/test_transface.py tests/test_mamba_vit.py -v`

**Already done (do not touch):**
- `backbones/mamba_vit.py` — `ConvBnAct`, `MambaBlock`, `MambaVit`
- `backbones/__init__.py` lines 94–109 — `mamba_s`, `mamba_b`, `mamba_l` registrations

**Key existing files to understand:**
- `backbones/vit.py` — `VisionTransformer`, `Block`, `Attention`, `PatchEmbed`, `Mlp`, `VITBatchNorm`
- `backbones/__init__.py` — `get_model(name, **kwargs)`, add new entries before the final `else: raise ValueError()`
- `configs/base.py` — default config values; specific configs only set values that differ
- `utils/utils_config.py` — `get_config("configs/ms1mv3_r50")` reads base.py then the named module
- `train_v2.py` — canonical training script; `train_transface.py` mirrors it exactly except the train loop

**`get_config` convention:** Pass `"configs/<filename_no_ext>"`. `base.py` defaults already include `margin_list = (1.0, 0.5, 0.0)`, `interclass_filtering_threshold = 0`, `sample_rate = 1`, `embedding_size = 512`, `fp16 = False`, `optimizer = "sgd"`. Specific configs only need to set what differs.

---

## File map

```
recognition/arcface_torch/
├── augmentation/
│   ├── __init__.py           CREATE (empty)
│   └── fft_mix.py            CREATE
├── backbones/
│   ├── transface_vit.py      CREATE
│   └── __init__.py           MODIFY (add transface_vit_b, transface_vit_l entries)
├── train_transface.py        CREATE
├── configs/
│   ├── ms1mv3_vit_b.py       CREATE
│   ├── ms1mv3_vit_l.py       CREATE
│   ├── ms1mv3_vit_h.py       CREATE
│   ├── ms1mv3_transface_vit_b.py  CREATE
│   ├── ms1mv3_transface_vit_l.py  CREATE
│   ├── ms1mv3_mamba_b.py     CREATE
│   └── ms1mv3_mamba_l.py     CREATE
└── tests/
    ├── test_transface.py     CREATE
    └── test_mamba_vit.py     CREATE
```

---

## Task 1: FFT Augmentation Module

**Files:**
- Create: `augmentation/__init__.py`
- Create: `augmentation/fft_mix.py`
- Create: `tests/test_transface.py` (FFT tests only in this task)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transface.py` with only the FFT tests:

```python
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


def test_fft_mix_values_bounded():
    from augmentation.fft_mix import amplitude_spectrum_mix
    src = torch.rand(4, 3, 112, 112)
    ref = torch.rand(4, 3, 112, 112)
    out = amplitude_spectrum_mix(src, ref, ratio=0.1)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_fft_mix_ratio_zero_preserves_src():
    """ratio=0 means no blending region, output should equal input."""
    from augmentation.fft_mix import amplitude_spectrum_mix
    src = torch.rand(2, 3, 112, 112)
    ref = torch.rand(2, 3, 112, 112)
    out = amplitude_spectrum_mix(src, ref, ratio=0.0)
    # Phase preserved means reconstruction should be near-identical to src
    assert torch.allclose(out, src, atol=1e-4)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3.9 -m pytest tests/test_transface.py::test_fft_mix_output_shape -v
```

Expected: `ModuleNotFoundError: No module named 'augmentation'`

- [ ] **Step 3: Create the augmentation package**

Create `augmentation/__init__.py` (empty file).

- [ ] **Step 4: Implement `augmentation/fft_mix.py`**

```python
import torch


def amplitude_spectrum_mix(src: torch.Tensor, ref: torch.Tensor,
                           ratio: float = 0.1) -> torch.Tensor:
    """Blend low-frequency amplitude of ref into src, preserving src's phase.

    Implements the FFT augmentation from TransFace (ICCV 2023):
    2D FFT both images, blend amplitude in the central ratio×ratio region
    of the shifted spectrum, inverse FFT to get the augmented image.

    Args:
        src: float tensor (B, C, H, W), pixel values in [0, 1]
        ref: float tensor (B, C, H, W), randomly sampled reference batch
        ratio: fraction of the spectrum's spatial extent to blend (0–1)
    Returns:
        Augmented float tensor with same shape as src, values clamped to [0, 1]
    """
    B, C, H, W = src.shape

    F_src = torch.fft.fft2(src)
    F_ref = torch.fft.fft2(ref)

    # Shift DC component to center
    F_src_s = torch.fft.fftshift(F_src)
    F_ref_s = torch.fft.fftshift(F_ref)

    amp_src = F_src_s.abs()
    phase_src = F_src_s.angle()
    amp_ref = F_ref_s.abs()

    # Central blend region
    h_crop = int(H * ratio)
    w_crop = int(W * ratio)
    h0 = H // 2 - h_crop // 2
    h1 = h0 + h_crop
    w0 = W // 2 - w_crop // 2
    w1 = w0 + w_crop

    amp_mixed = amp_src.clone()
    amp_mixed[:, :, h0:h1, w0:w1] = (
        0.5 * amp_src[:, :, h0:h1, w0:w1]
        + 0.5 * amp_ref[:, :, h0:h1, w0:w1]
    )

    F_out = amp_mixed * torch.exp(1j * phase_src)
    F_out = torch.fft.ifftshift(F_out)
    out = torch.fft.ifft2(F_out).real
    return out.clamp(0.0, 1.0)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python3.9 -m pytest tests/test_transface.py::test_fft_mix_output_shape tests/test_transface.py::test_fft_mix_values_bounded tests/test_transface.py::test_fft_mix_ratio_zero_preserves_src -v
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add augmentation/__init__.py augmentation/fft_mix.py tests/test_transface.py
git commit -m "feat(transface): add FFT amplitude spectrum mixing augmentation"
```

---

## Task 2: TransFaceViT Backbone

**Files:**
- Create: `backbones/transface_vit.py`
- Modify: `backbones/__init__.py` (add `transface_vit_b`, `transface_vit_l`)
- Modify: `tests/test_transface.py` (append TransFaceViT tests)

- [ ] **Step 1: Append failing TransFaceViT tests to `tests/test_transface.py`**

Add these tests after the FFT tests:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3.9 -m pytest tests/test_transface.py::test_transface_vit_train_returns_tuple -v
```

Expected: `ValueError` — `transface_vit_b` not registered in `get_model`

- [ ] **Step 3: Create `backbones/transface_vit.py`**

```python
from typing import Optional
import torch
import torch.nn as nn

from .vit import VisionTransformer


class AttentionWithEntropy(nn.Module):
    """Drop-in replacement for vit.Attention that stores per-patch entropy.

    During training, after the attention softmax, computes row-wise entropy
    over the attention distribution per patch and stores the batch-mean
    in self._last_entropy (shape: (B,)).  At eval time _last_entropy is
    not updated (no overhead).
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._last_entropy: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(True):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self.training:
                # attn: (B, heads, patches, patches)
                # row entropy per patch: (B, heads, patches)
                row_ent = -(attn * attn.clamp(min=1e-8).log()).sum(-1)
                # mean over heads and patch positions → (B,)
                self._last_entropy = row_ent.mean(dim=(1, 2))
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class TransFaceViT(VisionTransformer):
    """VisionTransformer subclass that emits per-patch attention entropy.

    At train time:  forward(x) -> (embedding: Tensor[B,D], entropy: Tensor[B,])
    At eval time:   forward(x) -> embedding: Tensor[B,D]

    The entropy drives the FFT amplitude augmentation in train_transface.py:
    images with below-median entropy get their low-frequency amplitude mixed
    with a randomly-selected image from the same batch.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Swap last block's Attention for AttentionWithEntropy
        last_block = self.blocks[-1]
        old = last_block.attn
        new = AttentionWithEntropy(
            dim=old.qkv.in_features,
            num_heads=old.num_heads,
            qkv_bias=old.qkv.bias is not None,
            qk_scale=old.scale,
            attn_drop=old.attn_drop.p,
            proj_drop=old.proj_drop.p,
        )
        new.load_state_dict(old.state_dict())
        last_block.attn = new

    @property
    def _entropy_head(self) -> AttentionWithEntropy:
        return self.blocks[-1].attn  # type: ignore[return-value]

    def forward(self, x: torch.Tensor):
        emb = super().forward(x)
        if self.training:
            return emb, self._entropy_head._last_entropy
        return emb
```

- [ ] **Step 4: Register `transface_vit_b` and `transface_vit_l` in `backbones/__init__.py`**

Find the line `else:\n    raise ValueError()` at the end of `get_model` and insert before it:

```python
    elif name == "transface_vit_b":
        num_features = kwargs.get("num_features", 512)
        from .transface_vit import TransFaceViT
        return TransFaceViT(
            img_size=112, patch_size=9, num_classes=num_features,
            embed_dim=512, depth=24, num_heads=8,
            drop_path_rate=0.05, norm_layer="ln",
            mask_ratio=0.05, using_checkpoint=True)

    elif name == "transface_vit_l":
        num_features = kwargs.get("num_features", 512)
        from .transface_vit import TransFaceViT
        return TransFaceViT(
            img_size=112, patch_size=9, num_classes=num_features,
            embed_dim=768, depth=24, num_heads=8,
            drop_path_rate=0.05, norm_layer="ln",
            mask_ratio=0.05, using_checkpoint=True)
```

- [ ] **Step 5: Run all TransFaceViT tests**

```bash
python3.9 -m pytest tests/test_transface.py -v
```

Expected: `7 passed` (3 FFT + 4 TransFaceViT)

- [ ] **Step 6: Commit**

```bash
git add backbones/transface_vit.py backbones/__init__.py tests/test_transface.py
git commit -m "feat(transface): add TransFaceViT backbone with patch entropy output"
```

---

## Task 3: MambaVit Tests

The `MambaVit` backbone is already implemented and committed. This task adds the test suite to protect it.

**Files:**
- Create: `tests/test_mamba_vit.py`

- [ ] **Step 1: Create `tests/test_mamba_vit.py`**

```python
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
```

- [ ] **Step 2: Run tests**

```bash
python3.9 -m pytest tests/test_mamba_vit.py -v
```

Expected: `4 passed`

Note: `test_mamba_backward_runs` uses `mamba_s` with `num_features=64` (small) to keep the sequential scan under 10s on CPU. If it times out, use `num_features=16`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mamba_vit.py
git commit -m "test(mamba): add MambaVit forward/backward/determinism tests"
```

---

## Task 4: `train_transface.py`

**Files:**
- Create: `train_transface.py`

This script is identical to `train_v2.py` with three targeted changes:
1. Import `amplitude_spectrum_mix`
2. Read `cfg.fft_prob` and `cfg.fft_ratio` (set in TransFace configs)
3. Replace the one-line train loop body with the FFT-augmented version

- [ ] **Step 1: Create `train_transface.py`**

```python
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from augmentation.fft_mix import amplitude_spectrum_mix
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                resume=cfg.wandb_resume,
                name=run_name,
                notes=cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16,
        num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank],
        bucket_cap_mb=16, find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold,
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(
            os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=wandb_logger,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            # --- TransFace: entropy-guided FFT augmentation ---
            # First forward to get patch entropy (backbone is TransFaceViT)
            local_embeddings, patch_entropy = backbone(img)
            patch_entropy = patch_entropy.detach()

            if torch.rand(1).item() < cfg.fft_prob:
                median_ent = patch_entropy.median()
                low_disc = patch_entropy < median_ent          # (B,) bool mask
                if low_disc.any():
                    ref_idx = torch.randperm(img.size(0), device=img.device)
                    img_aug = img.clone()
                    # Images are in [0,1] after standard dataloader normalization;
                    # if your loader uses ImageNet stats, un-normalize before FFT
                    # and re-normalize after. For pixel-range [0,1] this is fine.
                    img_aug[low_disc] = amplitude_spectrum_mix(
                        img[low_disc],
                        img[ref_idx][low_disc],
                        ratio=cfg.fft_ratio,
                    )
                    local_embeddings, _ = backbone(img_aug)
            # -------------------------------------------------

            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch,
                    })

                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16,
                                 lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint,
                       os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed TransFace Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
```

- [ ] **Step 2: Smoke-test the import**

```bash
python3.9 -c "
import sys; sys.path.insert(0, '.')
# patch out the dist init so it doesn't block
import train_transface
print('train_transface imports OK')
"
```

Expected: `train_transface imports OK`

If you see a `dist init` error, that's fine — the script tries to initialize NCCL at import time. Patch the test:

```bash
python3.9 -c "
import os, sys
os.environ['RANK'] = '0'; os.environ['LOCAL_RANK'] = '0'; os.environ['WORLD_SIZE'] = '1'
# check just the top-level imports
import importlib.util, ast
src = open('train_transface.py').read()
ast.parse(src)
print('train_transface syntax OK')
"
```

Expected: `train_transface syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_transface.py
git commit -m "feat(transface): add train_transface.py with FFT-guided augmentation loop"
```

---

## Task 5: All Config Files

**Files:**
- Create: `configs/ms1mv3_vit_b.py`, `configs/ms1mv3_vit_l.py`, `configs/ms1mv3_vit_h.py`
- Create: `configs/ms1mv3_transface_vit_b.py`, `configs/ms1mv3_transface_vit_l.py`
- Create: `configs/ms1mv3_mamba_b.py`, `configs/ms1mv3_mamba_l.py`

`base.py` defaults (do NOT repeat these in configs unless overriding):
- `margin_list = (1.0, 0.5, 0.0)`, `interclass_filtering_threshold = 0`
- `sample_rate = 1`, `embedding_size = 512`, `fp16 = False`, `optimizer = "sgd"`

- [ ] **Step 1: Create LVFace large-ViT configs**

`configs/ms1mv3_vit_b.py`:
```python
from easydict import EasyDict as edict

config = edict()
config.network = "vit_b_dp005_mask_005"
config.resume = False
config.output = None

config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.lr = 1e-4
config.verbose = 2000
config.dali = False
config.optimizer = "adamw"

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 40
config.warmup_epoch = 4
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
```

`configs/ms1mv3_vit_l.py` — identical, but:
```python
config.network = "vit_l_dp005_mask_005"
```

`configs/ms1mv3_vit_h.py` — identical, but:
```python
config.network = "vit_h"
```

- [ ] **Step 2: Create TransFace configs**

`configs/ms1mv3_transface_vit_b.py`:
```python
from easydict import EasyDict as edict

config = edict()
config.network = "transface_vit_b"
config.resume = False
config.output = None

config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.lr = 1e-4
config.verbose = 2000
config.dali = False
config.optimizer = "adamw"

# FFT augmentation (TransFace-specific)
config.fft_prob = 0.2   # probability of applying FFT aug each step
config.fft_ratio = 0.1  # fraction of spectrum spatial extent to blend

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 40
config.warmup_epoch = 4
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
```

`configs/ms1mv3_transface_vit_l.py` — identical, but:
```python
config.network = "transface_vit_l"
```

- [ ] **Step 3: Create MambaVit configs**

`configs/ms1mv3_mamba_b.py`:
```python
from easydict import EasyDict as edict

config = edict()
config.network = "mamba_b"
config.resume = False
config.output = None

config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.05
config.batch_size = 128
config.lr = 1e-4
config.verbose = 2000
config.dali = False
config.optimizer = "adamw"

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 30
config.warmup_epoch = 3
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
```

`configs/ms1mv3_mamba_l.py` — identical, but:
```python
config.network = "mamba_l"
```

- [ ] **Step 4: Smoke-test all configs load correctly**

```bash
python3.9 -c "
import sys; sys.path.insert(0, '.')
from utils.utils_config import get_config

for name in [
    'configs/ms1mv3_vit_b',
    'configs/ms1mv3_vit_l',
    'configs/ms1mv3_vit_h',
    'configs/ms1mv3_transface_vit_b',
    'configs/ms1mv3_transface_vit_l',
    'configs/ms1mv3_mamba_b',
    'configs/ms1mv3_mamba_l',
]:
    cfg = get_config(name)
    print(f'{name:45s}  network={cfg.network}  ok')
"
```

Expected output (7 lines, all ending in `ok`):
```
configs/ms1mv3_vit_b                           network=vit_b_dp005_mask_005  ok
configs/ms1mv3_vit_l                           network=vit_l_dp005_mask_005  ok
configs/ms1mv3_vit_h                           network=vit_h  ok
configs/ms1mv3_transface_vit_b                 network=transface_vit_b  ok
configs/ms1mv3_transface_vit_l                 network=transface_vit_l  ok
configs/ms1mv3_mamba_b                         network=mamba_b  ok
configs/ms1mv3_mamba_l                         network=mamba_l  ok
```

- [ ] **Step 5: Verify TransFace configs have fft_prob and fft_ratio**

```bash
python3.9 -c "
import sys; sys.path.insert(0, '.')
from utils.utils_config import get_config
cfg = get_config('configs/ms1mv3_transface_vit_b')
assert hasattr(cfg, 'fft_prob'), 'missing fft_prob'
assert hasattr(cfg, 'fft_ratio'), 'missing fft_ratio'
print(f'fft_prob={cfg.fft_prob}  fft_ratio={cfg.fft_ratio}  OK')
"
```

Expected: `fft_prob=0.2  fft_ratio=0.1  OK`

- [ ] **Step 6: Commit**

```bash
git add configs/ms1mv3_vit_b.py configs/ms1mv3_vit_l.py configs/ms1mv3_vit_h.py \
        configs/ms1mv3_transface_vit_b.py configs/ms1mv3_transface_vit_l.py \
        configs/ms1mv3_mamba_b.py configs/ms1mv3_mamba_l.py
git commit -m "feat: add MS1MV3 training configs for ViT-B/L/H, TransFaceViT, and MambaVit"
```

---

## Task 6: Final Test Run

- [ ] **Step 1: Run the full test suite**

```bash
python3.9 -m pytest tests/test_transface.py tests/test_mamba_vit.py -v
```

Expected:
```
tests/test_transface.py::test_fft_mix_output_shape           PASSED
tests/test_transface.py::test_fft_mix_values_bounded         PASSED
tests/test_transface.py::test_fft_mix_ratio_zero_preserves_src PASSED
tests/test_transface.py::test_transface_vit_train_returns_tuple PASSED
tests/test_transface.py::test_transface_vit_eval_returns_tensor PASSED
tests/test_transface.py::test_transface_vit_entropy_nonnegative PASSED
tests/test_transface.py::test_transface_vit_l_shape          PASSED
tests/test_mamba_vit.py::test_mamba_s_forward_shape          PASSED
tests/test_mamba_vit.py::test_mamba_b_forward_shape          PASSED
tests/test_mamba_vit.py::test_mamba_backward_runs            PASSED
tests/test_mamba_vit.py::test_mamba_eval_deterministic       PASSED
11 passed
```

- [ ] **Step 2: Run the existing AdaFace tests to confirm no regressions**

```bash
python3.9 -m pytest tests/test_adaface.py -v
```

Expected: `6 passed`

- [ ] **Step 3: Tag and commit if not already clean**

```bash
git tag transface-mamba-v1
```
