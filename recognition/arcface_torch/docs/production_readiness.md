# Production readiness: recipe pitfalls, reference numbers, and the gap to commercial systems

Notes collected while bringing up a WebFace42M TransFace-L run on this repo. Part 1 is a list of
recipe mistakes that are easy to make and expensive to discover late. Part 2 is what "good" looks
like inside this repo. Part 3 is what "good" looks like *outside* it — how the numbers in our README
relate to what commercial face recognition vendors are measured on, and where the remaining distance
actually sits.

---

## Part 1 — Recipe pitfalls

### 1. Learning rate does not scale with batch size here

The linear scaling rule (`lr ∝ total batch`) is a **SGD/CNN** convention. The official AdamW ViT
configs in this repo do not follow it:

| Config | Total batch | `config.lr` |
|---|---|---|
| `wf42m_pfc03_40epoch_8gpu_vit_b.py` | 8 x 256 = 2048 | 0.001 |
| `wf42m_pfc03_40epoch_8gpu_vit_t.py` | 8 x 512 = 4096 | 0.001 |
| `wf42m_pfc03_40epoch_64gpu_vit_l.py` | 64 x 384 = 24576 | 0.001 |

`lr = 1e-3` across a 12x span of batch sizes. `launch.py`'s `recommend()` scales the LR linearly from
a reference total batch, which is right for the CNN/SGD recipes it also covers but pushes AdamW ViT
runs above anything the upstream authors use. Override it with `--lr` (or edit the generated config).

Symptoms of an over-high LR here: the AMP `GradScaler` settles at a low value (frequent inf/NaN
backoffs) instead of recovering to 16384+, and the loss climbs as warmup ramps rather than falling.

### 2. AdaFace without augmentation is close to pointless

`config.dali_aug = False` leaves only the horizontal flip (`crop_mirror_normalize(mirror=...)` is
unconditional in `dataset.py`); it disables random resize (p=0.1), Gaussian blur (p=0.2), HSV jitter
(p=0.2) and grayscale (p=0.1).

AdaFace's entire mechanism is an **image-quality-adaptive margin** driven by the feature norm — low
norm gets a negative margin, high norm a positive one. Train only on clean aligned crops and the norm
distribution is narrow, the running `batch_mean` / `batch_std` lose discriminative power, and the
adaptive margin degenerates towards a fixed one. The four augmentations above are close to the
crop / rescale / photometric set the AdaFace paper trains with. **If you select `--loss adaface`,
turn `dali_aug` on.**

### 3. `val_targets = []` means no evaluation at all

`CallBackVerification` built with an empty `val_targets` has an empty `ver_list`, so `ver_test()`
iterates zero times. `config.verbose` then fires a no-op every N steps and the run produces a loss
curve and nothing else. Note the official WebFace42M configs also ship `val_targets = []` — upstream
evaluates separately on MFR / IJB-C after training — so copying an official config does not fix this.

To monitor during training, put `lfw.bin`, `cfp_fp.bin`, `agedb_30.bin` in the directory
`config.rec` points at and list them in `config.val_targets`. These `.bin` files ship with the
MS1MV3 / WebFace4M packages; the WebFace42M RecordIO release does not include them.

### 4. Margin choice is not free, and `margin_list` is ignored under AdaFace

The official WebFace42M ViT configs use **CosFace** `margin_list = (1.0, 0.0, 0.4)`. When
`config.loss == "adaface"` the training scripts build `AdaFaceLoss` and `PartialFC_V2_AdaFace`, and
`margin_list` is not read at all — the margin comes from `adaface_m / adaface_h / adaface_s /
adaface_t_alpha`. A leftover ArcFace-looking `margin_list` in an AdaFace config is cosmetic, not a
bug, but it misleads anyone reading the config.

Combining TransFace (DPAP + EHSM, which weights samples by patch entropy) with AdaFace (which weights
the margin by feature norm) stacks two quality-adaptive mechanisms that were never published
together. It may help or cancel out — there is no way to know without Part 3's evaluation.

### 5. Pretrained backbone + randomly initialised PartialFC head

`config.pretrained` loads only the backbone; the `(num_classes, 512)` classifier starts from
`N(0, 0.01)`. For 2M classes the early gradients from that random head are large and flow straight
into an already-converged backbone. Both parameter groups share one LR. Freezing the backbone (or
giving it a smaller LR group) for the first epoch is the standard mitigation; this repo does not
implement it, so expect the first epochs to partly undo the pretrained weights.

Check the load actually matched before trusting it:

```
pretrained backbone .../glint360k_model_TransFace_L.pt: loaded 284 tensors, skipped 0 (name/shape mismatch), 0 left at init
```

### 6. TransFace runs an extra forward pass per step

`train_transface.py` calls `backbone(img)` once under `no_grad` to obtain the SE patch weights for
DPAP, then again for the real step. This happens on **every** step — `dpap_prob` controls how many
samples get perturbed, not whether the extra forward runs. Cost is roughly +30% compute. It is the
paper's design, not a bug; set `dpap_prob = 0` to trade DPAP away for throughput.

---

## Part 2 — Reference numbers inside this repo

From the README results tables — these are the targets a WebFace42M run should be compared against:

| Datasets | Backbone(bs x gpus) | MFR-ALL | IJB-C(1e-4) | IJB-C(1e-5) |
|---|---|---|---|---|
| WF42M-PFC-0.3 | ViT-S(384x64) | 95.87 | 97.73 | 96.57 |
| WF42M-PFC-0.3 | ViT-B(384x64) | 97.42 | 97.90 | 97.04 |
| WF42M-PFC-0.3 | **ViT-L(384x64)** | **97.85** | **98.00** | **97.23** |
| WF42M-PFC-0.3 | ViT-B(256x8) | 97.16 | 97.91 | 97.05 |
| WF42M-PFC-0.3 | r200(128x32) | 97.70 | 97.97 | 96.93 |

Two things follow.

**Small clusters reproduce these recipes well.** The same ViT-B at 8 GPUs (2048 total) versus 64 GPUs
(24576 total) scores 97.91 vs 97.90 on IJB-C(1e-4) — identical — and 97.16 vs 97.42 on MFR-ALL, a
0.26 point cost for a 12x smaller batch. `wf42m_pfc03_40epoch_64gpu_vit_l.py` run on 6-8 GPUs with
`lr` left at 1e-3 should land near the ViT-L row.

**Every one of these numbers comes from an offline evaluation**, not from the training loop. Upstream
publishes the full training logs alongside them, which makes loss curves comparable step by step —
the cheapest sanity check available for a long run.

---

## Part 3 — The gap to commercial systems

### Different coordinate systems

The single most common misreading is comparing our number to a vendor's directly. They are measured
at different operating points:

| | This repo | NIST FRTE / commercial |
|---|---|---|
| Headline metric | IJB-C TAR @ **FAR = 1e-4** = 98.00 | FNMR @ **FMR = 1e-6** |
| As an error rate | 2% miss rate | 0.06% – 0.5% miss rate |
| Impostor operating point | 1 in 10,000 | **1 in 1,000,000** |

NIST's FRTE 1:1 report card fixes an FMR per dataset (1e-6 for the visa and mugshot categories). A
model at 98% TAR at FAR=1e-4 says nothing about where it sits at FMR=1e-6 — the ROC tail is exactly
where architectures separate. Any comparison that does not state the operating point is meaningless.

### Where the commercial state of the art sits (NIST FRTE, August 2026 update)

**1:1 verification**, FNMR (lower is better):

| Category | Best | Vendors |
|---|---|---|
| Visa–Visa | 0.06% | Cloudwalk Moontime, Recognito |
| Visa–Border | 0.14% | QazSmartVision.AI, TrueSight Laboratories |
| Border–Border | 0.16% | Cloudwalk Moontime |
| Mugshot–Mugshot | 0.20% | Paravision, QazSmartVision.AI, TrueSight Laboratories |
| Kiosk–Border | 0.53% | Sparktek International (2nd place: 3.73%) |

The mugshot leaders are followed at 0.21% by ROC, Innovatrics, Incode, Panasonic, Idemia, Sensetime
and others — the top tier has converged, and leadership is fragmented by use case rather than held by
one vendor. The Kiosk–Border column is the informative one: 0.53% for the leader, 3.73% for second
place. **In uncontrolled capture, the data domain decides the ranking, not the architecture.**

**1:N identification**: Idemia v13 ranks first at every gallery size from 640k to 12M identities at
FPIR = 0.1%; NEC reports a 0.06% error rate on a 12M-person gallery and leads 4 of 8 major
categories; top-tier algorithms reach FNIR 0.0017 at FPIR 0.001 on a 12M gallery.

### How big is the gap, per InsightFace's own guidance

From the [InsightFace model selection guide](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate):

> At FMR = 1e-6 on 1:1 protocols, commercial models reduce FNMR by a factor of **2–5x** compared to
> the strongest open-source pack — for example from 5–8% down to 1–2% on hard subsets such as masked
> or low-resolution faces.

The same guide recommends moving to a commercial model when any of these hold: operating at
FMR ≤ 1e-6 (border control, payments, regulated KYC); galleries beyond 100k–1M identities; a fairness
requirement across demographic groups; or heavy occlusion / low resolution. It puts the open-source
water line at LFW 99.50–99.85, CFP-FP 96–99, IJB-C@1e-4 96–97.5 for R100-class models — so a
successful WebFace42M ViT-L run lands at the top of the open-source range, and the 2–5x gap remains.

### Where the distance actually is

1. **Training data domain, not architecture.** WebFace42M is web-crawled celebrity imagery. Vendors
   train on operational captures — visa applications, border gates, law-enforcement mugshots, ATM
   cameras — with real age spans, capture-device diversity and pose/illumination distributions. Our
   models have never seen a border kiosk frame. This is the largest single term and no backbone swap
   closes it.
2. **Demographic fairness.** NIST publishes differentials by region, age and sex; vendors optimise
   for them explicitly with balanced data. Web-crawled corpora are skewed by construction. For
   regulated deployments this is a gate, not a bonus.
3. **The deliverable is an SDK, not a model.** NIST evaluates detection → quality assessment →
   alignment → template extraction → template compression → matching, under time and template-size
   constraints. We produce an ONNX embedding model; the detector (SCRFD / RetinaFace) and alignment
   often move the end-to-end number more than a backbone generation does.
4. **Presentation-attack and injection detection are absent.** PAD (iBeta L1/L2), morph detection
   (NIST FATE MORPH), injection-attack detection — none of it exists here. In deployment the attack
   surface usually costs more than the 2% miss rate does.
5. **1:N at scale.** Million-scale galleries need indexing and score normalisation, and FPIR degrades
   with N. IJB-C's gallery is a few thousand identities.
6. **Threshold governance.** Preprocessing changes silently shift FMR. Production practice is to
   freeze preprocessing, model hash, threshold and metrics as a single release unit and re-measure
   against fresh impostor sets on a schedule.

### A path to actually measuring the gap

1. **Run IJB-C** (`eval_ijbc.py`). It is the only bridge between our numbers and published ones.
2. **Build an FMR = 1e-6 test set.** Statistical significance at 1e-6 needs impostor pairs on the
   order of 10^7 — easy to generate by cross-pairing a 2M-identity corpus. This is the first point at
   which a number is comparable to a vendor's.
3. **Submit to NIST FRTE.** Participation is free and open to any developer worldwide: sign the
   participation agreement, wrap the model behind the published C++ API, run the validation package,
   encrypt and submit. It converts an estimate into a fact for the price of engineering time.

Honest summary: the model layer can be brought to within 2–5x of commercial at the operating points
that matter, but the detector, quality assessment, PAD, fairness testing and certification layers are
collectively more work than training the model. For self-hosted, lower-risk use — internal access
control, photo clustering, non-regulated identity checks — a well-trained WebFace42M model of this
class is genuinely sufficient.

## Sources

- [NIST FRTE 1:1 Verification](https://pages.nist.gov/frvt/html/frvt11.html) ·
  [NIST FRTE 1:N Identification](https://pages.nist.gov/frvt/html/frvt1N.html) ·
  [FRVT/FRTE participation agreement](https://www.nist.gov/system/files/documents/2021/01/13/FRVT_1N_participation_agreement.pdf)
- [NIST FRTE 1:1 shows facial verification accuracy converging](https://www.biometricupdate.com/202608/nist-frte-11-shows-facial-verification-accuracy-converging) (Biometric Update, Aug 2026)
- [NIST FRTE 1:N shows face recognition race shifting beyond accuracy](https://www.biometricupdate.com/202608/nist-frte-1n-shows-face-recognition-race-shifting-beyond-accuracy) (Biometric Update, Aug 2026)
- [NEC face recognition ranks first in NIST accuracy testing](https://www.nec.com/en/press/202603/global_20260309_02.html) (Mar 2026)
- [Choosing a Face Recognition Model: 1:1, 1:N Testing, and Threshold Selection](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate) (InsightFace)
