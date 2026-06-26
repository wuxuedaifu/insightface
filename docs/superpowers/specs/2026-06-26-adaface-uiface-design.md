# AdaFace + UIFace Migration Design

**Date:** 2026-06-26  
**Scope:** `recognition/arcface_torch/` (AdaFace) + `recognition/uiface/` (UIFace generation)

---

## Goal

Integrate two face recognition research contributions into the InsightFace `recognition/` folder:

1. **AdaFace** — quality-adaptive margin loss that uses the backbone's feature norm as an image-quality proxy to dynamically adjust per-sample margins during training.
2. **UIFace** — a diffusion-based synthetic face generator (VQ-GAN + DDPM) conditioned on identity embeddings; used to generate diverse intra-class training images.

---

## Part 1 — AdaFace in `recognition/arcface_torch/`

### Overview

AdaFace modifies two pieces of the existing pipeline:

1. The **backbone** must additionally return the L2 norm of its pre-normalized output (the norm serves as a quality score).
2. The **loss head** applies an EMA-normalised norm signal to scale both angular and additive margins per sample.

All other infrastructure (DDP, `PartialFC_V2`, dataset, eval, `PolynomialLRWarmup`) stays unchanged.

### 1a. Backbone — `backbones/iresnet.py`

Add `norm_output: bool = False` to `IResNet.__init__()`.

When `norm_output=True`, the `forward()` method changes its tail:
```
# existing tail
x = self.fc(x)
x = self.features(x)       # BatchNorm1d — outputs unnormalised embedding
return x

# new tail when norm_output=True
x = self.fc(x)
x = self.features(x)
norm = torch.norm(x, 2, 1, True)          # L2 norm per sample, shape (B, 1)
x = torch.div(x, norm)                    # L2-normalised embedding
return x, norm
```

When `norm_output=False` (default) behaviour is identical to today — zero breakage for existing configs.

### 1b. Backbone registry — `backbones/__init__.py`

`get_model()` already passes `**kwargs` through to the iresnet constructors. Add explicit handling so `norm_output=True` can be passed via config:

```python
# train_adaface.py will call:
backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16,
                     num_features=cfg.embedding_size,
                     norm_output=True).cuda()
```

No changes needed to `get_model()` itself — `**kwargs` already flows through.

### 1c. Loss — `losses.py`

Add `AdaFaceLoss` class. It holds only the **margin computation** (no weight matrix — that stays in `PartialFC_V2_AdaFace`).

Signature: `forward(logits, norms, labels) -> logits`

Key parameters: `m=0.4`, `h=0.333`, `s=64.0`, `t_alpha=0.01`

Algorithm:
```
# EMA update of batch norm statistics
batch_mean = t_alpha * norms.mean() + (1 - t_alpha) * self.batch_mean
batch_std  = t_alpha * norms.std()  + (1 - t_alpha) * self.batch_std

# Adaptive margin scaler: [-1, 1] range
margin_scaler = clip((norms - batch_mean) / (batch_std + eps) * h, -1, 1)

# Angular margin: reduce margin for low-quality images
g_angular = m * margin_scaler * -1          # high quality → smaller angle
m_arc[labels] = g_angular
theta = acos(cosine)
cosine = cos(clip(theta + m_arc, eps, π-eps))

# Additive margin: increase penalty for high-quality images
g_add = m + m * margin_scaler
m_cos[labels] = g_add
cosine = cosine - m_cos

return cosine * s
```

The `batch_mean` and `batch_std` are registered as buffers (persist across steps, not updated via gradient).

### 1d. PartialFC — `partial_fc_v2.py`

Add `PartialFC_V2_AdaFace` alongside the existing class. Changes vs `PartialFC_V2`:

1. `forward(local_embeddings, local_norms, local_labels)` — accepts norms.
2. All-gathers `local_norms` in the same pass as embeddings.
3. Calls `self.margin_softmax(logits, norms, labels)` instead of `self.margin_softmax(logits, labels)`.

Everything else (sampling, `DistCrossEntropy`, weight sharding) is identical.

### 1e. Training script — `train_adaface.py`

New file, parallel to `train_v2.py`. Differences:

- Backbone created with `norm_output=True`.
- Loss is `AdaFaceLoss(m, h, s, t_alpha)` from config.
- Module is `PartialFC_V2_AdaFace(margin_loss, ...)`.
- Forward step unpacks `(local_embeddings, local_norms) = backbone(img)`.
- Passes both to `module_partial_fc(local_embeddings, local_norms, local_labels)`.

Config keys added (on top of existing base.py keys):
```
adaface_m      = 0.4
adaface_h      = 0.333
adaface_s      = 64.0
adaface_t_alpha = 0.01
```

### 1f. Configs

Two new example configs:
- `configs/ms1mv3_adaface_r50.py`
- `configs/ms1mv3_adaface_r100.py`

Both inherit the existing ms1mv3 settings but set `adaface_*` hyperparameters and point to `train_adaface.py`.

---

## Part 2 — UIFace generation in `recognition/uiface/`

### Overview

UIFace is a **standalone data-generation module** — it does not touch `arcface_torch/`. It produces diverse synthetic face images by conditioning a latent diffusion model on identity embeddings extracted from a pretrained recognition backbone.

Training workflow:
1. Extract identity embeddings from a pretrained recognition model (e.g., ArcFace R100).
2. Train UIFace's VQ-GAN autoencoder on the face dataset.
3. Train UIFace's DDPM on VQ-GAN latents, conditioned on identity embeddings.
4. At inference, provide any identity embedding → sample diverse face images.
5. Add generated images to the recognition training set and retrain.

### File layout — `recognition/uiface/`

```
recognition/uiface/
├── main.py                          # training entry point (hydra)
├── sample.py                        # inference / sampling
├── diffusion/
│   └── ddpm.py                      # DDPM denoising model
├── models/
│   ├── diffusion/
│   │   ├── unet.py                  # conditional UNet (time + identity context)
│   │   ├── nn.py                    # attention blocks (self + cross)
│   │   └── util.py                  # TimestepEmbedSequential, conv_nd helpers
│   └── autoencoder/
│       ├── vqgan.py                 # VQ-GAN encoder/decoder interfaces
│       ├── modules.py               # ResNet encoder/decoder blocks
│       └── quantization.py          # vector quantization layer
├── utils/
│   ├── helpers.py                   # misc utilities
│   ├── ema.py                       # exponential moving average model wrapper
│   ├── CASIA_dataset.py             # dataset (replace with any face dataset)
│   ├── checkpoint.py                # save/restore helpers
│   ├── colored.py                   # colored terminal output
│   └── __init__.py
└── configs/                         # hydra configs (ported from TFace)
    └── train_config.yaml
```

### Key model components

**UNet (conditional):**  
- Standard DDPM UNet with residual + attention blocks.
- Time embedding via sinusoidal positional encoding → MLP → injected at each ResBlock.
- Identity conditioning via cross-attention: identity embedding projected to key/value, noisy latent as query.
- Supports classifier-free guidance (random context dropout during training).

**VQ-GAN:**  
- Encoder compresses 112×112 face images to a compact latent (e.g., 14×14×4).
- Vector quantization layer with codebook.
- Decoder reconstructs from quantized latent.
- Trained separately (perceptual + adversarial loss) before DDPM training.

**DDPM:**  
- Wraps UNet + noise schedule.
- Forward: adds noise to VQ latent at sampled timestep, predicts noise.
- Sampling: DDIM deterministic sampler for fast inference (50 steps instead of 1000).

### Dependencies added

```
hydra-core>=1.2
omegaconf
pytorch_lightning>=1.8
torchvision
torchmetrics
```

These are UIFace-only and do not affect `arcface_torch/`.

---

## What is NOT changed

- `arcface_torch/train_v2.py` — untouched; existing ArcFace/CosFace training unaffected.
- `arcface_torch/partial_fc_v2.py::PartialFC_V2` — untouched.
- `arcface_torch/losses.py` existing classes — untouched.
- `arcface_torch/backbones/iresnet.py` — existing `forward()` path unchanged when `norm_output=False`.
- All other `recognition/` subfolders — untouched.

---

## File change summary

| File | Action |
|------|--------|
| `arcface_torch/backbones/iresnet.py` | Add `norm_output` flag and conditional `(emb, norm)` return |
| `arcface_torch/losses.py` | Add `AdaFaceLoss` class |
| `arcface_torch/partial_fc_v2.py` | Add `PartialFC_V2_AdaFace` class |
| `arcface_torch/train_adaface.py` | New training script |
| `arcface_torch/configs/ms1mv3_adaface_r50.py` | New config |
| `arcface_torch/configs/ms1mv3_adaface_r100.py` | New config |
| `recognition/uiface/` (all files) | New UIFace generation module (ported from TFace) |
