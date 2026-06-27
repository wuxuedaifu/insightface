# AdaFace + UIFace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate AdaFace quality-adaptive margin loss into `recognition/arcface_torch/` and port the UIFace diffusion-based face generation module from TFace into `recognition/uiface/`.

**Architecture:** AdaFace adds a `norm_output` flag to `IResNet` so backbones return `(embedding, norm)`, a new `AdaFaceLoss` class with EMA-based adaptive margin, and a `PartialFC_V2_AdaFace` variant that passes norms through to the margin function. UIFace is a self-contained port of TFace's `generation/uiface/` diffusion pipeline with import paths fixed for the new location.

**Tech Stack:** PyTorch (DDP, AMP), easydict (configs), hydra-core + omegaconf + pytorch-lightning (UIFace), torchvision

---

## File Map

### Phase 1 — AdaFace

| Action | File |
|--------|------|
| Modify | `recognition/arcface_torch/backbones/iresnet.py` |
| Modify | `recognition/arcface_torch/losses.py` |
| Modify | `recognition/arcface_torch/partial_fc_v2.py` |
| Create | `recognition/arcface_torch/train_adaface.py` |
| Create | `recognition/arcface_torch/configs/ms1mv3_adaface_r50.py` |
| Create | `recognition/arcface_torch/configs/ms1mv3_adaface_r100.py` |
| Create | `recognition/arcface_torch/tests/__init__.py` |
| Create | `recognition/arcface_torch/tests/test_adaface.py` |

### Phase 2 — UIFace

| Action | File |
|--------|------|
| Create | `recognition/uiface/main.py` |
| Create | `recognition/uiface/sample.py` |
| Create | `recognition/uiface/requirements.txt` |
| Create | `recognition/uiface/diffusion/__init__.py` |
| Create | `recognition/uiface/diffusion/ddpm.py` |
| Create | `recognition/uiface/models/__init__.py` |
| Create | `recognition/uiface/models/autoencoder/__init__.py` |
| Create | `recognition/uiface/models/autoencoder/modules.py` |
| Create | `recognition/uiface/models/autoencoder/quantization.py` |
| Create | `recognition/uiface/models/autoencoder/vqgan.py` |
| Create | `recognition/uiface/models/autoencoder/first_stage_config.yaml` |
| Create | `recognition/uiface/models/diffusion/__init__.py` |
| Create | `recognition/uiface/models/diffusion/nn.py` |
| Create | `recognition/uiface/models/diffusion/unet.py` |
| Create | `recognition/uiface/models/diffusion/util.py` |
| Create | `recognition/uiface/utils/__init__.py` |
| Create | `recognition/uiface/utils/helpers.py` |
| Create | `recognition/uiface/utils/ema.py` |
| Create | `recognition/uiface/utils/CASIA_dataset.py` |
| Create | `recognition/uiface/utils/checkpoint.py` |
| Create | `recognition/uiface/utils/colored.py` |
| Create | `recognition/uiface/configs/train_config.yaml` |
| Create | `recognition/uiface/configs/diffusion/ddpm.yaml` |
| Create | `recognition/uiface/configs/model/unet_cond_ca_cpd25_uncond20.yaml` |
| Create | `recognition/uiface/configs/dataset/CASIA_file.yaml` |
| Create | `recognition/uiface/configs/sample_ddim_config.yaml` |

---

## Phase 1 — AdaFace

---

### Task 1: Add `norm_output` to IResNet backbone

**Files:**
- Modify: `recognition/arcface_torch/backbones/iresnet.py`
- Create: `recognition/arcface_torch/tests/__init__.py`
- Create: `recognition/arcface_torch/tests/test_adaface.py`

- [ ] **Step 1: Create test file with a failing test for norm_output**

Create `recognition/arcface_torch/tests/__init__.py` (empty).

Create `recognition/arcface_torch/tests/test_adaface.py`:

```python
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
    model = iresnet50(False, fp16=False, num_features=512, norm_output=True)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    emb, norm = model(x)
    l2 = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(l2, torch.ones(2), atol=1e-5), \
        f"Embeddings not unit-normed: {l2}"


def test_iresnet50_default_output_unchanged():
    """norm_output=False (default) must return a plain tensor, not a tuple."""
    model = iresnet50(False, fp16=False, num_features=512)
    model.eval()
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    assert isinstance(out, torch.Tensor), "Default output should be a Tensor"
    assert out.shape == (2, 512)
```

- [ ] **Step 2: Run the test — expect failures**

```bash
cd recognition/arcface_torch && python -m pytest tests/test_adaface.py::test_iresnet50_norm_output_shape tests/test_adaface.py::test_iresnet50_norm_output_is_unit_norm tests/test_adaface.py::test_iresnet50_default_output_unchanged -v
```

Expected: `FAILED` with `TypeError: __init__() got an unexpected keyword argument 'norm_output'`

- [ ] **Step 3: Add `norm_output` to `IResNet.__init__()` and `forward()`**

In `recognition/arcface_torch/backbones/iresnet.py`, edit `IResNet.__init__()` (line 68) to accept the new parameter:

```python
# Old signature (line 68-71):
class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
```

Replace with:

```python
class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False,
                 norm_output=False):
```

Then inside `__init__`, after the existing `nn.init.constant_(self.features.weight, 1.0)` line, add:

```python
        self.norm_output = norm_output
```

Now edit `forward()` (currently lines 148–162):

```python
# Old forward (lines 148-162):
    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x
```

Replace with:

```python
    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        if self.norm_output:
            norm = torch.norm(x, 2, 1, True)
            x = torch.div(x, norm)
            return x, norm
        return x
```

- [ ] **Step 4: Run the tests — expect all pass**

```bash
cd recognition/arcface_torch && python -m pytest tests/test_adaface.py::test_iresnet50_norm_output_shape tests/test_adaface.py::test_iresnet50_norm_output_is_unit_norm tests/test_adaface.py::test_iresnet50_default_output_unchanged -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add recognition/arcface_torch/backbones/iresnet.py \
        recognition/arcface_torch/tests/__init__.py \
        recognition/arcface_torch/tests/test_adaface.py
git commit -m "feat(adaface): add norm_output option to IResNet backbone"
```

---

### Task 2: Add `AdaFaceLoss` to losses.py

**Files:**
- Modify: `recognition/arcface_torch/losses.py`
- Modify: `recognition/arcface_torch/tests/test_adaface.py`

- [ ] **Step 1: Add failing tests for AdaFaceLoss**

Append to `recognition/arcface_torch/tests/test_adaface.py`:

```python
from losses import AdaFaceLoss


def test_adaface_loss_output_shape():
    loss_fn = AdaFaceLoss(m=0.4, h=0.333, s=64.0, t_alpha=1.0)
    # 4 samples, 10 classes; labels -1 means not a local class
    logits = torch.rand(4, 10).clamp(-1 + 1e-3, 1 - 1e-3)
    norms  = torch.tensor([[22.0], [18.0], [25.0], [10.0]])
    labels = torch.tensor([[2], [5], [-1], [0]])
    out = loss_fn(logits, norms, labels)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"


def test_adaface_loss_scales_by_s():
    loss_fn = AdaFaceLoss(m=0.0, h=0.0, s=64.0, t_alpha=1.0)
    # With m=0 and h=0, margin_scaler=0, so loss is just logits * s
    logits = torch.tensor([[0.8, 0.2, 0.5]])
    norms  = torch.tensor([[20.0]])
    labels = torch.tensor([[-1]])   # no positive on this shard
    out = loss_fn(logits, norms, labels)
    expected = logits * 64.0
    assert torch.allclose(out, expected, atol=1e-4), \
        f"Expected {expected}, got {out}"


def test_adaface_loss_ema_buffers_update():
    loss_fn = AdaFaceLoss(m=0.4, h=0.333, s=64.0, t_alpha=1.0)
    logits = torch.rand(2, 5).clamp(-1 + 1e-3, 1 - 1e-3)
    norms  = torch.tensor([[30.0], [10.0]])
    labels = torch.tensor([[0], [-1]])
    loss_fn(logits, norms, labels)
    # With t_alpha=1.0 the EMA fully adopts the batch stats
    assert abs(loss_fn.batch_mean.item() - 20.0) < 1.0, \
        f"batch_mean not updated: {loss_fn.batch_mean.item()}"
```

- [ ] **Step 2: Run the tests — expect failures**

```bash
cd recognition/arcface_torch && python -m pytest tests/test_adaface.py::test_adaface_loss_output_shape tests/test_adaface.py::test_adaface_loss_scales_by_s tests/test_adaface.py::test_adaface_loss_ema_buffers_update -v
```

Expected: `ImportError: cannot import name 'AdaFaceLoss'`

- [ ] **Step 3: Add `AdaFaceLoss` to losses.py**

Append to `recognition/arcface_torch/losses.py`:

```python

class AdaFaceLoss(torch.nn.Module):
    """Adaptive margin loss from AdaFace (https://arxiv.org/abs/2204.09949).

    Adjusts angular and additive margins per-sample based on the feature norm,
    which serves as a proxy for image quality. High-norm (high-quality) samples
    receive a tighter margin; low-norm (low-quality) samples are penalised less.

    Args:
        m: base margin value (default 0.4)
        h: margin scaler coefficient, controls sensitivity to norm variation (default 0.333)
        s: feature scale (default 64.0)
        t_alpha: EMA momentum for batch norm statistics (default 0.01)
    """

    def __init__(self, m: float = 0.4, h: float = 0.333,
                 s: float = 64.0, t_alpha: float = 0.01):
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.eps = 1e-3
        # Running statistics — persisted in checkpoints, not updated by gradient
        self.register_buffer('batch_mean', torch.ones(1) * 20.0)
        self.register_buffer('batch_std',  torch.ones(1) * 100.0)

    def forward(self, logits: torch.Tensor, norms: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: cosine similarities, shape (N, C), values in (-1, 1)
            norms:  feature L2 norms from backbone, shape (N, 1)
            labels: class indices, shape (N, 1); -1 means no local positive class
        Returns:
            scaled logits with adaptive margin applied, shape (N, C)
        """
        index_positive = torch.where(labels.view(-1) != -1)[0]

        safe_norms = torch.clip(norms.view(-1), min=0.001, max=100).detach()

        # EMA update of batch norm statistics
        with torch.no_grad():
            mean = safe_norms.mean()
            std  = safe_norms.std()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std  = std  * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        # Margin scaler: z-score of norm clamped to [-1, 1], then scaled by h
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = torch.clip(margin_scaler * self.h, -1, 1)

        # --- Angular margin (reduces effective angle for low-quality samples) ---
        g_angular = self.m * margin_scaler[index_positive] * -1
        m_arc = torch.zeros_like(logits)
        m_arc[index_positive, labels[index_positive].view(-1)] = g_angular
        theta   = logits.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        logits  = theta_m.cos()

        # --- Additive margin (increases penalty for high-quality samples) ---
        g_add = self.m + self.m * margin_scaler[index_positive]
        m_cos = torch.zeros_like(logits)
        m_cos[index_positive, labels[index_positive].view(-1)] = g_add
        logits = logits - m_cos

        return logits * self.s
```

- [ ] **Step 4: Run the tests — expect all pass**

```bash
cd recognition/arcface_torch && python -m pytest tests/test_adaface.py::test_adaface_loss_output_shape tests/test_adaface.py::test_adaface_loss_scales_by_s tests/test_adaface.py::test_adaface_loss_ema_buffers_update -v
```

Expected: `3 passed`

- [ ] **Step 5: Run the full test suite to check no regressions**

```bash
cd recognition/arcface_torch && python -m pytest tests/ -v
```

Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add recognition/arcface_torch/losses.py \
        recognition/arcface_torch/tests/test_adaface.py
git commit -m "feat(adaface): add AdaFaceLoss with EMA-adaptive margin"
```

---

### Task 3: Add `PartialFC_V2_AdaFace` to partial_fc_v2.py

**Files:**
- Modify: `recognition/arcface_torch/partial_fc_v2.py`

`PartialFC_V2_AdaFace` requires a live distributed process group, so it cannot be unit-tested without launching multi-process DDP. The correctness of its logic is verified in Task 4 via a single-GPU smoke-run.

- [ ] **Step 1: Append `PartialFC_V2_AdaFace` to `partial_fc_v2.py`**

Add after the `AllGather = AllGatherFunc.apply` line at the bottom of the file:

```python


class PartialFC_V2_AdaFace(torch.nn.Module):
    """PartialFC variant for AdaFace that all-gathers feature norms and passes
    them to the margin loss alongside logits.

    Usage is identical to PartialFC_V2 except forward() takes three arguments:
        loss = module_pfc(local_embeddings, local_norms, local_labels)
    """
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        super(PartialFC_V2_AdaFace, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0
        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_local, embedding_size))
        )

        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def sample(self, labels, index_positive):
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local]).cuda()
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].cuda()
                index = index.sort()[0].cuda()
            else:
                index = positive
            self.weight_index = index
            labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_norms: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """
        Args:
            local_embeddings: L2-normalised embeddings on this GPU, shape (B, D)
            local_norms:      feature norms from backbone on this GPU, shape (B, 1)
            local_labels:     class labels on this GPU, shape (B,)
        Returns:
            scalar loss
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: "
            f"{self.last_batch_size} vs {batch_size}"
        )

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda()
            for _ in range(self.world_size)
        ]
        _gather_norms = [
            torch.zeros((batch_size, 1)).cuda()
            for _ in range(self.world_size)
        ]

        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)
        distributed.all_gather(_gather_norms, local_norms)

        embeddings = torch.cat(_list_embeddings)
        labels     = torch.cat(_gather_labels)
        norms      = torch.cat(_gather_norms)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings      = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, norms, labels)
        loss   = self.dist_cross_entropy(logits, labels)
        return loss
```

- [ ] **Step 2: Verify the existing tests still pass (regression check)**

```bash
cd recognition/arcface_torch && python -m pytest tests/ -v
```

Expected: `6 passed`

- [ ] **Step 3: Commit**

```bash
git add recognition/arcface_torch/partial_fc_v2.py
git commit -m "feat(adaface): add PartialFC_V2_AdaFace with norm all-gather"
```

---

### Task 4: Create `train_adaface.py`

**Files:**
- Create: `recognition/arcface_torch/train_adaface.py`

- [ ] **Step 1: Create the training script**

Create `recognition/arcface_torch/train_adaface.py`:

```python
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import AdaFaceLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2_AdaFace
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "torch >= 1.12.0 required"

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

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
    )

    # Backbone returns (l2_normalized_embedding, l2_norm) when norm_output=True
    backbone = get_model(
        cfg.network,
        dropout=0.0,
        fp16=cfg.fp16,
        num_features=cfg.embedding_size,
        norm_output=True,
    ).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[local_rank],
        bucket_cap_mb=16,
        find_unused_parameters=True,
    )
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.train()
    backbone._set_static_graph()

    margin_loss = AdaFaceLoss(
        m=cfg.adaface_m,
        h=cfg.adaface_h,
        s=cfg.adaface_s,
        t_alpha=cfg.adaface_t_alpha,
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2_AdaFace(
            margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False
        )
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2_AdaFace(
            margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False
        )
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step  = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step,
    )

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(
            os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
        )
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
        val_targets=cfg.val_targets,
        rec_prefix=cfg.rec,
        summary_writer=summary_writer,
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
            # Backbone returns (normalized_embedding, norm) when norm_output=True
            local_embeddings, local_norms = backbone(img)
            loss: torch.Tensor = module_partial_fc(
                local_embeddings, local_norms, local_labels
            )

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
                loss_am.update(loss.item(), 1)
                callback_logging(
                    global_step, loss_am, epoch,
                    cfg.fp16, lr_scheduler.get_last_lr()[0], amp
                )
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
            torch.save(
                checkpoint,
                os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"),
            )

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="AdaFace Training in PyTorch (Distributed)"
    )
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
```

- [ ] **Step 2: Verify the script can be imported (checks syntax)**

```bash
cd recognition/arcface_torch && python -c "import train_adaface; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add recognition/arcface_torch/train_adaface.py
git commit -m "feat(adaface): add train_adaface.py training entry point"
```

---

### Task 5: Add AdaFace configs

**Files:**
- Create: `recognition/arcface_torch/configs/ms1mv3_adaface_r50.py`
- Create: `recognition/arcface_torch/configs/ms1mv3_adaface_r100.py`

- [ ] **Step 1: Create R50 config**

Create `recognition/arcface_torch/configs/ms1mv3_adaface_r50.py`:

```python
from easydict import EasyDict as edict

config = edict()

# AdaFace margin hyperparameters
config.adaface_m      = 0.4    # base margin
config.adaface_h      = 0.333  # margin scaler coefficient
config.adaface_s      = 64.0   # feature scale
config.adaface_t_alpha = 0.01  # EMA momentum for norm statistics

# Backbone
config.network        = "r50"
config.embedding_size = 512

# Dataset — update rec path and class/image counts for your dataset
config.rec         = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image   = 5179510

# Training schedule
config.num_epoch    = 20
config.warmup_epoch = 0
config.batch_size   = 128
config.lr           = 0.1
config.momentum     = 0.9
config.weight_decay = 5e-4
config.optimizer    = "sgd"

# Partial FC
config.sample_rate = 1.0

# Misc
config.resume          = False
config.save_all_states = False
config.output          = "ms1mv3_adaface_r50"
config.fp16            = True
config.gradient_acc    = 1
config.verbose         = 2000
config.frequent        = 10
config.seed            = 2048
config.num_workers     = 2
config.dali            = False
config.dali_aug        = False
config.val_targets     = ['lfw', 'cfp_fp', 'agedb_30']

# WandB (disabled by default)
config.using_wandb     = False
config.wandb_key       = ""
config.suffix_run_name = None
config.wandb_entity    = ""
config.wandb_project   = ""
config.wandb_log_all   = False
config.save_artifacts  = False
config.wandb_resume    = False
config.notes           = ""
```

- [ ] **Step 2: Create R100 config**

Create `recognition/arcface_torch/configs/ms1mv3_adaface_r100.py`:

```python
from easydict import EasyDict as edict

config = edict()

# AdaFace margin hyperparameters
config.adaface_m      = 0.4
config.adaface_h      = 0.333
config.adaface_s      = 64.0
config.adaface_t_alpha = 0.01

# Backbone
config.network        = "r100"
config.embedding_size = 512

# Dataset
config.rec         = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image   = 5179510

# Training schedule
config.num_epoch    = 20
config.warmup_epoch = 0
config.batch_size   = 128
config.lr           = 0.1
config.momentum     = 0.9
config.weight_decay = 5e-4
config.optimizer    = "sgd"

# Partial FC
config.sample_rate = 1.0

# Misc
config.resume          = False
config.save_all_states = False
config.output          = "ms1mv3_adaface_r100"
config.fp16            = True
config.gradient_acc    = 1
config.verbose         = 2000
config.frequent        = 10
config.seed            = 2048
config.num_workers     = 2
config.dali            = False
config.dali_aug        = False
config.val_targets     = ['lfw', 'cfp_fp', 'agedb_30']

config.using_wandb     = False
config.wandb_key       = ""
config.suffix_run_name = None
config.wandb_entity    = ""
config.wandb_project   = ""
config.wandb_log_all   = False
config.save_artifacts  = False
config.wandb_resume    = False
config.notes           = ""
```

- [ ] **Step 3: Verify configs import cleanly**

```bash
cd recognition/arcface_torch && python -c "
from utils.utils_config import get_config
cfg = get_config('configs/ms1mv3_adaface_r50')
assert hasattr(cfg, 'adaface_m'), 'adaface_m missing'
assert hasattr(cfg, 'adaface_h'), 'adaface_h missing'
cfg2 = get_config('configs/ms1mv3_adaface_r100')
assert cfg2.network == 'r100'
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add recognition/arcface_torch/configs/ms1mv3_adaface_r50.py \
        recognition/arcface_torch/configs/ms1mv3_adaface_r100.py
git commit -m "feat(adaface): add ms1mv3 AdaFace R50 and R100 configs"
```

---

## Phase 2 — UIFace Generation Module

UIFace is ported from `/tmp/tface/generation/uiface/` (already cloned at `/tmp/tface`). Each step copies files and adjusts the one import that relied on `sys.path.insert(0, "uiface/")`.

---

### Task 6: Create directory scaffold and port utility files

**Files:** All `recognition/uiface/utils/` files

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p recognition/uiface/diffusion \
         recognition/uiface/models/autoencoder \
         recognition/uiface/models/diffusion \
         recognition/uiface/utils \
         recognition/uiface/configs/diffusion \
         recognition/uiface/configs/model \
         recognition/uiface/configs/dataset
touch recognition/uiface/__init__.py \
      recognition/uiface/diffusion/__init__.py \
      recognition/uiface/models/__init__.py \
      recognition/uiface/models/autoencoder/__init__.py \
      recognition/uiface/models/diffusion/__init__.py \
      recognition/uiface/utils/__init__.py
```

- [ ] **Step 2: Copy utility files verbatim**

```bash
cp /tmp/tface/generation/uiface/utils/helpers.py      recognition/uiface/utils/helpers.py
cp /tmp/tface/generation/uiface/utils/ema.py           recognition/uiface/utils/ema.py
cp /tmp/tface/generation/uiface/utils/CASIA_dataset.py recognition/uiface/utils/CASIA_dataset.py
cp /tmp/tface/generation/uiface/utils/checkpoint.py    recognition/uiface/utils/checkpoint.py
cp /tmp/tface/generation/uiface/utils/colored.py       recognition/uiface/utils/colored.py
```

- [ ] **Step 3: Verify utility imports**

```bash
cd recognition/uiface && python -c "
from utils.helpers import denormalize_to_zero_to_one, normalize_to_neg_one_to_one, ensure_path_join
from utils.ema import EMAModel
print('utils OK')
"
```

Expected: `utils OK`

- [ ] **Step 4: Copy requirements.txt**

```bash
cp /tmp/tface/generation/uiface/requirements.txt recognition/uiface/requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add recognition/uiface/
git commit -m "feat(uiface): scaffold directory and port utility modules"
```

---

### Task 7: Port model files (autoencoder + diffusion)

**Files:** All `recognition/uiface/models/` files

- [ ] **Step 1: Copy autoencoder model files**

```bash
cp /tmp/tface/generation/uiface/models/autoencoder/modules.py        recognition/uiface/models/autoencoder/modules.py
cp /tmp/tface/generation/uiface/models/autoencoder/quantization.py   recognition/uiface/models/autoencoder/quantization.py
cp /tmp/tface/generation/uiface/models/autoencoder/vqgan.py          recognition/uiface/models/autoencoder/vqgan.py
cp /tmp/tface/generation/uiface/models/autoencoder/first_stage_config.yaml \
   recognition/uiface/models/autoencoder/first_stage_config.yaml
```

- [ ] **Step 2: Copy diffusion model files**

```bash
cp /tmp/tface/generation/uiface/models/diffusion/nn.py   recognition/uiface/models/diffusion/nn.py
cp /tmp/tface/generation/uiface/models/diffusion/unet.py recognition/uiface/models/diffusion/unet.py
cp /tmp/tface/generation/uiface/models/diffusion/util.py recognition/uiface/models/diffusion/util.py
```

- [ ] **Step 3: Verify model imports**

```bash
cd recognition/uiface && python -c "
from models.autoencoder.quantization import VectorQuantizer2 as VQ
from models.autoencoder.modules import Encoder, Decoder
from models.diffusion.nn import SpatialSelfAttentionBlock
from models.diffusion.util import TimestepEmbedSequential
print('models OK')
"
```

Expected: `models OK`

- [ ] **Step 4: Commit**

```bash
git add recognition/uiface/models/
git commit -m "feat(uiface): port VQ-GAN and diffusion model files"
```

---

### Task 8: Port DDPM and fix entry-point imports

**Files:**
- Create: `recognition/uiface/diffusion/ddpm.py`
- Create: `recognition/uiface/main.py`
- Create: `recognition/uiface/sample.py`

- [ ] **Step 1: Copy DDPM file**

```bash
cp /tmp/tface/generation/uiface/diffusion/ddpm.py recognition/uiface/diffusion/ddpm.py
```

- [ ] **Step 2: Copy main.py and remove the sys.path hack**

The original `main.py` has `sys.path.insert(0, "uiface/")` which was needed when running from TFace's `generation/` directory. This is not needed now that the module is self-contained.

```bash
cp /tmp/tface/generation/uiface/main.py recognition/uiface/main.py
```

Then edit `recognition/uiface/main.py` — remove the `sys.path.insert` line:

```python
# Remove this line (it appears near the top after the imports):
sys.path.insert(0, "uiface/")
```

- [ ] **Step 3: Copy sample.py and fix its sys.path hack**

```bash
cp /tmp/tface/generation/uiface/sample.py recognition/uiface/sample.py
```

Edit `recognition/uiface/sample.py` — remove the path hack:

```python
# Remove this line:
sys.path.insert(1, "../")
```

- [ ] **Step 4: Copy hydra config files**

```bash
cp /tmp/tface/generation/uiface/configs/train_config.yaml           recognition/uiface/configs/train_config.yaml
cp /tmp/tface/generation/uiface/configs/diffusion/ddpm.yaml         recognition/uiface/configs/diffusion/ddpm.yaml
cp /tmp/tface/generation/uiface/configs/model/unet_cond_ca_cpd25_uncond20.yaml \
   recognition/uiface/configs/model/unet_cond_ca_cpd25_uncond20.yaml
cp /tmp/tface/generation/uiface/configs/dataset/CASIA_file.yaml     recognition/uiface/configs/dataset/CASIA_file.yaml
cp /tmp/tface/generation/uiface/configs/sample_ddim_config.yaml     recognition/uiface/configs/sample_ddim_config.yaml
```

- [ ] **Step 5: Verify DDPM and main imports**

```bash
cd recognition/uiface && python -c "
from diffusion.ddpm import DenoisingDiffusionProbabilisticModel
from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
print('ddpm + vqgan OK')
"
```

Expected: `ddpm + vqgan OK`

- [ ] **Step 6: Full import smoke-test**

```bash
cd recognition/uiface && python -c "
import main   # imports hydra, pytorch_lightning, diffusion, models, utils
print('main import OK')
"
```

Expected: `main import OK`

- [ ] **Step 7: Commit**

```bash
git add recognition/uiface/diffusion/ \
        recognition/uiface/main.py \
        recognition/uiface/sample.py \
        recognition/uiface/configs/
git commit -m "feat(uiface): port DDPM, entry points, and hydra configs"
```

---

### Task 9: Final integration check

**Files:** No new files

- [ ] **Step 1: Run the full AdaFace test suite one final time**

```bash
cd recognition/arcface_torch && python -m pytest tests/ -v
```

Expected: `6 passed`

- [ ] **Step 2: Confirm the UIFace module tree is complete**

```bash
find recognition/uiface -type f | sort
```

Expected output (at minimum):

```
recognition/uiface/__init__.py
recognition/uiface/configs/dataset/CASIA_file.yaml
recognition/uiface/configs/diffusion/ddpm.yaml
recognition/uiface/configs/model/unet_cond_ca_cpd25_uncond20.yaml
recognition/uiface/configs/sample_ddim_config.yaml
recognition/uiface/configs/train_config.yaml
recognition/uiface/diffusion/__init__.py
recognition/uiface/diffusion/ddpm.py
recognition/uiface/main.py
recognition/uiface/models/__init__.py
recognition/uiface/models/autoencoder/__init__.py
recognition/uiface/models/autoencoder/first_stage_config.yaml
recognition/uiface/models/autoencoder/modules.py
recognition/uiface/models/autoencoder/quantization.py
recognition/uiface/models/autoencoder/vqgan.py
recognition/uiface/models/diffusion/__init__.py
recognition/uiface/models/diffusion/nn.py
recognition/uiface/models/diffusion/unet.py
recognition/uiface/models/diffusion/util.py
recognition/uiface/requirements.txt
recognition/uiface/sample.py
recognition/uiface/utils/CASIA_dataset.py
recognition/uiface/utils/__init__.py
recognition/uiface/utils/checkpoint.py
recognition/uiface/utils/colored.py
recognition/uiface/utils/ema.py
recognition/uiface/utils/helpers.py
```

- [ ] **Step 3: Tag the completed implementation**

```bash
git tag adaface-uiface-v1
```

---

## How to Train AdaFace

Once you have a dataset in MXNet RecordIO format at `/train_tmp/ms1m-retinaface-t1`:

```bash
cd recognition/arcface_torch
# Single GPU (dev / smoke test)
python train_adaface.py configs/ms1mv3_adaface_r50

# Multi-GPU DDP (4 GPUs)
torchrun --nproc_per_node=4 train_adaface.py configs/ms1mv3_adaface_r50
```

## How to Use UIFace

1. Train (or download) a face recognition backbone (e.g., ArcFace R100).
2. Extract identity embeddings from your training set and save as `contexts.npy` of shape `(N, 512)`.
3. Train the VQ-GAN autoencoder (follow TFace's UIFace README).
4. Train the DDPM conditioned on identity embeddings:
   ```bash
   cd recognition/uiface
   python main.py training.checkpoint.VQEncoder=path/to/enc.pt \
                  training.checkpoint.VQDecoder=path/to/dec.pt
   ```
5. Generate diverse synthetic faces per identity:
   ```bash
   python sample.py sampling.contexts_file=contexts.npy \
                    checkpoint.path=path/to/diffusion.ckpt \
                    VQDecoder_path=path/to/dec.pt \
                    VQEncoder_path=path/to/enc.pt
   ```
6. Add generated images to your recognition training set and retrain.
