# Training with one image per identity

Operational identity corpora — ID photos, visa applications, enrolment databases — usually hold
exactly one image per person. That breaks the assumption margin-softmax training is built on. This
document covers what actually happens in that regime, how to fold such a corpus into training
without damaging a model, and the three failure modes that make the attempt backfire.

## 1. What one image per class does to the loss

ArcFace / CosFace / AdaFace do two things at once: pull a sample towards **its own class centre**
(intra-class compactness) and push it away from **other class centres** (inter-class separation).

With a single sample in a class, the PartialFC weight `W_i` converges onto that sample's embedding
direction and the first term is satisfied trivially — **there is no intra-class supervision left**.
What remains is pure inter-class repulsion, which makes the objective a form of *instance
discrimination*: the same thing self-supervised methods optimise, with as many "classes" as images.

That is not worthless — instance discrimination learns good representations. But it comes with a hard
consequence:

> Under one image per class, the intra-class invariance the model can learn is **exactly** the set of
> augmentations you apply. Variation you do not synthesise is variation the model never becomes
> invariant to.

So the augmentation pipeline stops being a regulariser and becomes the primary source of signal.

## 2. Recipe: two-stage mixed fine-tuning

Do **not** train on the single-image corpus alone. Without multi-image identities nothing constrains
intra-class compactness and the model rapidly loses what it learned from a multi-image dataset. Mix
the two, and do it as a second stage on top of a converged base model.

```python
# stage 2: base model + in-domain single-image corpus
config.pretrained   = "<stage-1 output>/model.pt"   # backbone only
config.num_classes  = 2059907 + N_SINGLE            # e.g. WebFace42M + your corpus
config.sample_rate  = 0.2                           # class count grew; sample fewer negatives
config.lr           = 1e-4                          # ~0.1x the stage-1 LR
config.num_epoch    = 5
config.warmup_epoch = 1
config.dali_aug     = True                          # mandatory here, see §3
```

`num_classes` changes, so the PartialFC weight shape changes and **stage 2 cannot `resume` from
stage 1** — it initialises the backbone from `model.pt` and trains a fresh head.

### Give the backbone a smaller learning rate

A freshly initialised head over millions of classes produces large early gradients that flow straight
into an already-converged backbone. The optimizer already has two parameter groups, so this is a
two-line change:

```python
opt = torch.optim.AdamW(
    params=[{"params": backbone.parameters(),          "lr": cfg.lr * 0.1},
            {"params": module_partial_fc.parameters(), "lr": cfg.lr}],
    lr=cfg.lr, weight_decay=cfg.weight_decay)
```

`PolynomialLRWarmup` records `base_lrs` **per parameter group**, so two different LRs are preserved
and decayed correctly with no scheduler changes. (`lr_scheduler.py`'s `__main__` demo shows the same
pattern with `lr_pfc_weight = 1/3`.)

### Oversample the single-image corpus

Image counts are the wrong thing to balance. Balance **positive gradients per class**:

| | images | identities | positives per class per epoch |
|---|---|---|---|
| WebFace42M | 42.5M | 2.06M | ~20 |
| single-image corpus | N | N | **1** |

A 20x deficit means those class centres converge far more slowly. Repeat each single-image entry
5–10x when packing the RecordIO (`scripts/shuffle_rec.py` then shuffles it). The repeats are not
redundant: each pass draws different random augmentation.

## 3. Augmentation is the whole ball game

`config.dali_aug = True` enables random resize (p=0.1), Gaussian blur (p=0.2), HSV jitter (p=0.2) and
grayscale (p=0.1); the horizontal flip is unconditional. That is a generic degradation set. For a
single-image corpus you want augmentation that models the **specific gap between how the enrolment
image was captured and how the probe image will be captured**:

| Covariate | Enrolment image | Probe image | Synthesisable with |
|---|---|---|---|
| Resolution / compression | high-res, lightly compressed | small crop from a camera stream, re-compressed | 2D — extend the DALI pipeline |
| Illumination | even studio light | overhead, backlit, mixed colour temperature | 2D |
| Motion / defocus blur | none | common | 2D |
| **Pose** | frontal by regulation | ±15–30° yaw/pitch | **3D warp or generative** |
| **Occlusion** | none | glasses, masks, head coverings | overlay compositing |
| **Age** | at issue | up to the document's validity period later | **generative** |

The first three are DALI pipeline work. Pose and age need 3D fitting or a generative model —
`recognition/uiface/` (VQ-GAN + DDPM conditioned on identity embeddings) is the in-repo option, but
generated-data domain shift is a research problem in its own right; treat it as the last thing to
try, not the first.

Occlusion is worth calling out separately: if the deployment population routinely wears something the
enrolment photo forbids, that covariate dominates real-world FNMR and no amount of backbone capacity
substitutes for synthesising it.

## 4. Three ways this backfires

### 4.1 Inconsistent alignment

Every source must be detected and aligned with the **same landmark set and the same affine template**
before packing. Mix two alignment conventions and the model learns the convention as an identity
cue — training loss looks fine, cross-domain evaluation collapses. This is the single most common
failure when merging corpora.

### 4.2 Duplicate identities are negative supervision

If one person appears under two labels, margin softmax **explicitly pushes those two classes apart**.
It is not a missed learning opportunity; it is the loss actively teaching the model that one person is
two people. Operational corpora routinely contain such conflicts (renewals, reissues, name changes).

The README's Noisy Datasets table quantifies the damage on a dataset with deliberate identity
conflicts:

| Dataset | Backbone | MFR-ALL | IJB-C(1e-4) |
|---|---|---|---|
| WF12M-Conflict | r50 | 79.93 | 95.30 |
| WF12M-Conflict + abnormal inter-class filtering | r50 | **91.68** | **97.28** |

11.75 MFR-ALL points, purely from handling identity conflicts. Deduplicate offline before packing —
extract embeddings, nearest-neighbour search (faiss handles millions of 512-d vectors in minutes),
review the high-similarity pairs, and cross-check against record metadata where it exists. The
recovered duplicates are valuable twice over: as merged training identities *and* as the genuine
pairs an otherwise single-image corpus cannot provide (see [eval.md](eval.md#5-building-an-fmr--1e-6-test-set-of-your-own)).

### 4.3 Inter-class filtering is unavailable under AdaFace

The repo's built-in defence against identity conflicts is
`config.interclass_filtering_threshold` (see `configs/wf12m_conflict_r50_pfc03_filter04.py`, which
uses 0.4). It is implemented in `CombinedMarginLoss.forward` (`losses.py`): non-target logits above
the threshold are zeroed, so the loss stops pushing away classes that look suspiciously like the
sample.

`AdaFaceLoss.__init__` **does not take that parameter**, so selecting `--loss adaface` silently gives
up the filter. Options: port the masking block into `AdaFaceLoss.forward` (it applies to the logits
before the margin, roughly eight lines), rely entirely on offline deduplication, or use CosFace
`margin_list = (1.0, 0.0, 0.4)` for the mixed stage. Doing both the port and the offline pass is the
safe choice.

## 5. Measuring whether it worked

A/B the fine-tuned model against the stage-1 base **on an in-domain test set** — see
[eval.md](eval.md). Expect an asymmetric result:

- **In-domain**: this is where the gain should appear, and it can be large.
- **General benchmarks (IJB-C, LFW)**: flat or slightly worse. You are specialising the decision
  boundary towards one domain; a small generalisation cost is the expected price.

A general-benchmark regression beyond a few tenths of a point usually means oversampling was too
aggressive or the alignment conventions did not actually match.
