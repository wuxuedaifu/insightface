# TransFace FFT Augmentation + MambaVision Backbone Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate TransFace's FFT amplitude augmentation and a MambaVision hybrid backbone into `recognition/arcface_torch/`, alongside large-ViT (vit_b/l/h) MS1MV3 training configs that match what LVFace uses at scale.

**Architecture:** Three independent additions layered onto the existing `arcface_torch` codebase — (1) LVFace-style large-ViT configs (no new backbone code), (2) TransFace FFT augmentation wired into a modified training loop, and (3) a new MambaVision CNN+SSM hybrid backbone. All three use the existing `PartialFC_V2` + `CombinedMarginLoss` training stack.

**Tech Stack:** PyTorch 2.x, existing `arcface_torch` infrastructure (`get_model`, `PartialFC_V2`, `CombinedMarginLoss`, `get_dataloader`, `get_config`)

---

## Source Analysis

### LVFace (ByteDance)

LVFace's repository is inference-only. Its `VisionTransformer` class is architecturally identical to the one already in `backbones/vit.py`. All ViT size variants (T/S/B/L/H) are already registered in `backbones/__init__.py::get_model()`. What is missing is a set of MS1MV3-targeting training configs that mirror the TransFace paper's AdamW/gradient-accumulation setup for large ViT models.

### TransFace (ICCV 2023 — DanJun6737)

The method's novel contribution is **FFT amplitude spectrum mixing**, an online augmentation applied selectively to "low-discriminability" patches. During each training step:

1. A forward pass through the ViT yields per-patch attention entropy at the final attention block (mean entropy across heads per patch position).
2. Patches with **below-median** attention entropy are identified (less discriminative = more augmentation benefit).
3. For those image samples, `amplitude_spectrum_mix(src, ref, ratio)` is applied: both images are FFT'd, a central `ratio`-sized square in the amplitude spectrum is blended (α=0.5), and the result is inverse-FFT'd back to spatial domain.
4. The augmented batch re-enters the backbone for the actual gradient update.

This requires the ViT backbone to emit patch entropy alongside embeddings during training — a second return value only active in training mode.

TransFace uses standard `CombinedMarginLoss` (ArcFace mode: m1=1.0, m2=0.5, m3=0.0) + `PartialFC_V2` — no new loss function.

### MambaVision

Neither LVFace nor TransFace implement MambaVision. We design a face-adapted hybrid CNN+Mamba backbone from the NVIDIA MambaVision architecture (NeurIPS 2024):

- CNN stages reduce resolution (stride 2 × 2 = 4× total), converting 112×112 input to 28×28 = 784 patch tokens
- Mamba (S6 selective scan) blocks model long-range dependencies across the token sequence
- Output head identical to existing ViT: `Linear(d_model → 512) → BN → Linear(512 → num_classes) → BN`
- Pure PyTorch SSM implementation (no `mamba_ssm` CUDA package required)

---

## Component 1: LVFace Large-ViT Configs

### What changes

**No new backbone code.** The existing `vit_b_dp005_mask_005`, `vit_l_dp005_mask_005`, `vit_h` model names in `get_model()` are the LVFace backbones.

**New config files** (three):

```
configs/ms1mv3_vit_b.py
configs/ms1mv3_vit_l.py
configs/ms1mv3_vit_h.py
```

### Config values

All three share:
```python
config.optimizer  = "adamw"
config.lr         = 1e-4
config.weight_decay = 0.1
config.warmup_epoch = 4
config.num_epoch  = 40
config.batch_size = 128
config.embedding_size = 512
config.fp16       = True
config.sample_rate = 1.0
config.rec        = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image  = 5179510
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
```

Differ only in `config.network`:
- `ms1mv3_vit_b.py` → `"vit_b_dp005_mask_005"`
- `ms1mv3_vit_l.py` → `"vit_l_dp005_mask_005"`
- `ms1mv3_vit_h.py` → `"vit_h"`

Training command (same as existing `train_v2.py`):
```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_vit_b.py
```

---

## Component 2: TransFace FFT Augmentation

### 2a. `augmentation/__init__.py`

Empty package marker.

### 2b. `augmentation/fft_mix.py`

```python
def amplitude_spectrum_mix(src: torch.Tensor, ref: torch.Tensor,
                           ratio: float = 0.1) -> torch.Tensor:
    """Blend low-frequency amplitude of ref into src, preserving src's phase.

    Args:
        src: float tensor (B, C, H, W), pixel values in [0, 1]
        ref: float tensor (B, C, H, W), randomly sampled batch
        ratio: fraction of the spectrum's spatial extent to blend (0–1)
    Returns:
        augmented image tensor same shape as src
    """
```

Implementation:
1. `F_src = torch.fft.fft2(src)` → split into amplitude (`F_src.abs()`) and phase (`F_src.angle()`)
2. Same for `ref`
3. Compute blend region: `h_crop = int(H * ratio); w_crop = int(W * ratio)`
4. Central crop: rows `[H//2 - h_crop//2 : H//2 + h_crop//2]`, same for W (shift spectrum center via `fftshift`)
5. `amp_mixed = amp_src`; `amp_mixed[:, :, rows, cols] = 0.5 * amp_src[:, :, rows, cols] + 0.5 * amp_ref[:, :, rows, cols]`
6. Reconstruct: `F_out = amp_mixed * torch.exp(1j * phase_src)`; `out = torch.fft.ifft2(F_out).real`
7. Clamp to `[0, 1]`

### 2c. `backbones/transface_vit.py`

```python
class TransFaceViT(VisionTransformer):
    """ViT that emits per-patch attention entropy during training.

    At train time: forward() returns (embedding, patch_entropy)
      where patch_entropy is shape (B,) — mean entropy across heads
      at the last attention block, averaged over patch positions.
    At eval time: forward() returns embedding only (same as VisionTransformer).
    """
```

Implementation:
- Override `forward_features()` to hook into the **last** `Block`'s attention module during training
- After the attention softmax (`attn` tensor of shape `(B, heads, patches, patches)`), compute row-entropy per patch: `H = -( attn * attn.clamp(min=1e-8).log() ).sum(-1)` → shape `(B, heads, patches)`; mean over heads and patches → scalar per image `(B,)`
- Store as `self._last_patch_entropy`; `forward()` returns `(embedding, self._last_patch_entropy)` if `self.training`, else `embedding`

Registration in `backbones/__init__.py`:
```python
elif name == "transface_vit_b":
    from .transface_vit import TransFaceViT
    return TransFaceViT(img_size=112, patch_size=9, num_classes=num_features,
                        embed_dim=512, depth=24, num_heads=8,
                        drop_path_rate=0.05, norm_layer="ln",
                        mask_ratio=0.05, using_checkpoint=True)

elif name == "transface_vit_l":
    from .transface_vit import TransFaceViT
    return TransFaceViT(img_size=112, patch_size=9, num_classes=num_features,
                        embed_dim=768, depth=24, num_heads=8,
                        drop_path_rate=0.05, norm_layer="ln",
                        mask_ratio=0.05, using_checkpoint=True)
```

### 2d. `train_transface.py`

Mirrors `train_v2.py` with these changes:

1. Import `TransFaceViT` for backbone (via `get_model(cfg.network)` — model name handles dispatch)
2. Import `amplitude_spectrum_mix` from `augmentation.fft_mix`
3. In the train loop, after `backbone(img)` returns `(local_embeddings, patch_entropy)`:
   ```python
   local_embeddings, patch_entropy = backbone(img)
   # FFT augmentation with cfg.fft_prob probability
   if torch.rand(1).item() < cfg.fft_prob:
       median_entropy = patch_entropy.median()
       low_disc_mask = patch_entropy < median_entropy   # (B,) bool
       if low_disc_mask.any():
           ref_idx = torch.randperm(img.size(0), device=img.device)
           img_aug = img.clone()
           img_aug[low_disc_mask] = amplitude_spectrum_mix(
               img[low_disc_mask], img[ref_idx][low_disc_mask],
               ratio=cfg.fft_ratio)
           local_embeddings, _ = backbone(img_aug)
   loss = module_partial_fc(local_embeddings, local_labels)
   ```
4. Uses `PartialFC_V2` + `CombinedMarginLoss` (not AdaFace variants)

### 2e. TransFace configs

`configs/ms1mv3_transface_vit_b.py`:
```python
config.network     = "transface_vit_b"
config.optimizer   = "adamw"
config.lr          = 1e-4
config.weight_decay = 0.1
config.warmup_epoch = 4
config.num_epoch   = 40
config.batch_size  = 128
config.embedding_size = 512
config.fp16        = True
config.fft_prob    = 0.2      # probability of applying FFT aug per step
config.fft_ratio   = 0.1     # fraction of spectrum to blend
config.margin_list = (1.0, 0.5, 0.0)
config.interclass_filtering_threshold = 0
config.rec         = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image   = 5179510
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
```

`configs/ms1mv3_transface_vit_l.py` — same with `config.network = "transface_vit_l"`.

---

## Component 3: MambaVision Backbone

### Architecture

Face-adapted hybrid CNN+SSM backbone (`backbones/mamba_vit.py`):

```
Input: (B, 3, 112, 112)
  ↓
Stage 1: ConvBnRelu(3→C, 3×3, stride=2) × 2   → (B, C, 56, 56)
Stage 2: ConvBnRelu(C→2C, 3×3, stride=2) × 2  → (B, 2C, 28, 28)
Reshape: flatten spatial → (B, 784, 2C)         patch tokens
Stage 3: N × MambaBlock(2C)                     sequence modeling
  ↓
GlobalAvgPool: (B, 2C)
Linear(2C → 512) → BN → Linear(512 → num_classes) → BN
Output: (B, num_classes)
```

**MambaBlock** (pure PyTorch selective scan):
- Input projection: `x_proj = Linear(d_model, d_inner + 2*d_state + dt_rank)` — splits into `x`, `B`, `C`, `Δ` (dt)
- Selective scan (causal): discretize `(A, B)` → `(Ā, B̄)` using ZOH with softplus-activated `Δ`; compute state accumulation via `torch.cumsum` in log-space (parallel scan, O(L log L))
- Output projection: `Linear(d_inner, d_model)` + residual
- No CUDA extensions required

**Sizes:**

| Name | C (stage1) | 2C (SSM) | d_inner | N blocks | Params |
|------|-----------|----------|---------|----------|--------|
| `mamba_s` | 128 | 256 | 512 | 12 | ~25M |
| `mamba_b` | 256 | 512 | 1024 | 24 | ~90M |
| `mamba_l` | 384 | 768 | 1536 | 24 | ~190M |

Registration in `backbones/__init__.py`:
```python
elif name == "mamba_s":
    from .mamba_vit import MambaVit
    return MambaVit(stage_dims=(128, 256), num_mamba_blocks=12,
                    d_state=16, dt_rank=8, num_classes=num_features)

elif name == "mamba_b":
    from .mamba_vit import MambaVit
    return MambaVit(stage_dims=(256, 512), num_mamba_blocks=24,
                    d_state=16, dt_rank=16, num_classes=num_features)

elif name == "mamba_l":
    from .mamba_vit import MambaVit
    return MambaVit(stage_dims=(384, 768), num_mamba_blocks=24,
                    d_state=16, dt_rank=24, num_classes=num_features)
```

**Configs:**

`configs/ms1mv3_mamba_b.py`:
```python
config.network      = "mamba_b"
config.optimizer    = "adamw"
config.lr           = 1e-4
config.weight_decay = 0.05
config.warmup_epoch = 3
config.num_epoch    = 30
config.batch_size   = 128
config.embedding_size = 512
config.fp16         = True
config.sample_rate  = 1.0
config.margin_list  = (1.0, 0.5, 0.0)
config.rec          = "/train_tmp/ms1m-retinaface-t1"
config.num_classes  = 93431
config.num_image    = 5179510
config.val_targets  = ['lfw', 'cfp_fp', 'agedb_30']
```

`configs/ms1mv3_mamba_l.py` — same with `config.network = "mamba_l"`.

---

## Tests

### `tests/test_transface.py`

1. `test_fft_mix_output_shape` — `amplitude_spectrum_mix(src, ref, 0.1)` returns same shape as input
2. `test_fft_mix_values_bounded` — output values in `[0, 1]`
3. `test_transface_vit_train_returns_tuple` — `transface_vit_b` in train mode returns `(emb, entropy)` tuple; `emb.shape == (B, 512)`, `entropy.shape == (B,)`
4. `test_transface_vit_eval_returns_tensor` — eval mode returns plain tensor `(B, 512)`
5. `test_entropy_is_nonnegative` — all entropy values ≥ 0

### `tests/test_mamba_vit.py`

1. `test_mamba_s_forward_shape` — output shape `(B, num_classes)` for B=2, num_classes=512
2. `test_mamba_b_forward_shape` — same for mamba_b
3. `test_mamba_backward_runs` — `loss.backward()` completes without error (grad flow through SSM)
4. `test_mamba_train_eval_consistent` — same output in eval mode with `torch.no_grad()` across two calls (deterministic)

---

## File Map Summary

```
recognition/arcface_torch/
├── augmentation/
│   ├── __init__.py                        CREATE
│   └── fft_mix.py                         CREATE
├── backbones/
│   ├── transface_vit.py                   CREATE
│   ├── mamba_vit.py                       CREATE
│   └── __init__.py                        MODIFY  (add transface_vit_b/l, mamba_s/b/l)
├── train_transface.py                     CREATE
├── configs/
│   ├── ms1mv3_vit_b.py                    CREATE
│   ├── ms1mv3_vit_l.py                    CREATE
│   ├── ms1mv3_vit_h.py                    CREATE
│   ├── ms1mv3_transface_vit_b.py          CREATE
│   ├── ms1mv3_transface_vit_l.py          CREATE
│   ├── ms1mv3_mamba_b.py                  CREATE
│   └── ms1mv3_mamba_l.py                  CREATE
└── tests/
    ├── test_transface.py                  CREATE
    └── test_mamba_vit.py                  CREATE
```

**Total: 2 modified, 14 created.**

---

## Non-Goals

- Loading LVFace pretrained weights (inference-only; no public training code to replicate exactly)
- `mamba_ssm` CUDA kernel integration (pure PyTorch implementation covers training correctness; users can swap in CUDA kernels later)
- UIFace / AdaFace changes (separate feature branch, already shipped)
