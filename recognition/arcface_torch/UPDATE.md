# arcface_torch Updates

## 2026-06-28 — AdaFace + UIFace + TransFace + MambaVision

### AdaFace (quality-adaptive margin loss)

Paper: [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.09949)

**What was added:**
- `backbones/iresnet.py` — `norm_output=True` flag on IResNet; backbone returns `(embedding, norm)` tuple when enabled. Feature norm serves as image quality proxy.
- `losses.py` — `AdaFaceLoss`: EMA-normalized adaptive margin loss. Per-sample angular and additive margins are scaled by feature norm relative to the running batch distribution.
- `partial_fc_v2.py` — `PartialFC_V2_AdaFace`: variant of `PartialFC_V2` that all-gathers feature norms across GPUs and passes them to the loss.
- `train_adaface.py` — training script for AdaFace.
- `configs/ms1mv3_adaface_r50.py`, `configs/ms1mv3_adaface_r100.py` — MS1MV3 configs with AdaFace hyperparameters (`adaface_m`, `adaface_h`, `adaface_s`, `adaface_t_alpha`).
- `tests/test_adaface.py` — 6 unit tests.

**Usage:**
```bash
torchrun --nproc_per_node=8 train_adaface.py configs/ms1mv3_adaface_r50
```

---

### UIFace (diffusion-based synthetic face generation)

Source: [TFace](https://github.com/Tencent/TFace)

**What was added:**
- `recognition/uiface/` — full VQ-GAN + DDPM generation module ported from TFace. Generates diverse synthetic training faces conditioned on identity embeddings.
  - `models/autoencoder/` — VQ-GAN encoder/decoder + vector quantization
  - `models/diffusion/` — conditional UNet with cross-attention on identity embeddings
  - `diffusion/ddpm.py` — DDPM noise schedule + DDIM sampler
  - `utils/` — EMA wrapper, dataset loader, checkpoint helpers
  - `configs/` — YAML configs for training (`train_config.yaml`) and sampling (`sample_ddim_config.yaml`)

**Usage:**
```bash
# Train the diffusion model
python recognition/uiface/main.py

# Sample synthetic faces
python recognition/uiface/sample.py
```

---

### TransFace (FFT amplitude augmentation)

Paper: [TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective](https://arxiv.org/abs/2308.10133) (ICCV 2023)

**What was added:**
- `augmentation/fft_mix.py` — `amplitude_spectrum_mix(src, ref, ratio)`: blends the low-frequency amplitude spectrum of a reference image into the source, preserving the source phase. Pure PyTorch, no extra dependencies.
- `backbones/transface_vit.py` — `TransFaceViT`: subclass of `VisionTransformer` that emits per-patch attention entropy alongside the embedding in train mode. Used to identify low-discriminability image regions for targeted augmentation.
  - Train mode: `forward(x)` → `(embedding, patch_entropy)` where `patch_entropy` shape is `(B,)`
  - Eval mode: `forward(x)` → `embedding` (identical to base `VisionTransformer`)
- `train_transface.py` — training script. Each step: run backbone to get patch entropy → apply FFT mix to below-median-entropy images (with probability `cfg.fft_prob`) → re-run backbone on augmented batch → standard ArcFace loss.
- `configs/ms1mv3_transface_vit_b.py`, `configs/ms1mv3_transface_vit_l.py`

**New `get_model` names:** `transface_vit_b`, `transface_vit_l`

**Usage:**
```bash
torchrun --nproc_per_node=8 train_transface.py configs/ms1mv3_transface_vit_b
```

---

### MambaVision backbone

Source: Inspired by [MambaVision](https://github.com/NVlabs/MambaVision) (NeurIPS 2024)

**What was added:**
- `backbones/mamba_vit.py` — `MambaVit`: hybrid CNN + Mamba (S6 selective state space) backbone adapted for 112×112 face recognition.
  - Stage 1–2: strided CNN blocks reduce 112×112 → 28×28 patch tokens
  - Stage 3: N Mamba blocks (pure PyTorch selective scan, no `mamba_ssm` CUDA package required)
  - Head: global average pool → Linear → BN → Linear → BN → 512-dim embedding
- `configs/ms1mv3_mamba_b.py`, `configs/ms1mv3_mamba_l.py`

**New `get_model` names and sizes:**

| Name | CNN dims | SSM dim | Mamba blocks | ~Params |
|------|----------|---------|-------------|---------|
| `mamba_s` | 128→256 | 256 | 12 | 7M |
| `mamba_b` | 256→512 | 512 | 24 | 90M |
| `mamba_l` | 384→768 | 768 | 24 | 190M |

**Note:** The pure-PyTorch sequential scan is correct for training but slow on CPU. For production GPU training, replace `_selective_scan` in `mamba_vit.py` with `mamba_ssm` CUDA kernels for 10–100× speedup.

**Usage:**
```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_mamba_b
```

---

### Large-ViT configs (LVFace-style)

The existing `VisionTransformer` backbone already covers ViT-B/L/H. New MS1MV3 configs are added for training large ViT models with AdamW (matching the TransFace paper setup):

| Config | Backbone | Embed dim | Depth |
|--------|----------|-----------|-------|
| `ms1mv3_vit_b` | `vit_b_dp005_mask_005` | 512 | 24 |
| `ms1mv3_vit_l` | `vit_l_dp005_mask_005` | 768 | 24 |
| `ms1mv3_vit_h` | `vit_h` | 1024 | 48 |

```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_vit_h
```

---

### Contact

wuxuedaifu@gmail.com
