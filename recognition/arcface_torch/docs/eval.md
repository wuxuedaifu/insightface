# Evaluation: metrics, operating points, and building your own test set

Training this repo produces a loss curve. A loss curve is not an accuracy number, and an accuracy
number at the wrong operating point is worse than none at all. This document covers what the metrics
mean, how to monitor during training, and how to build a test set that can actually measure the
operating points production systems run at.

## 1. Genuine and impostor pairs

Face verification compares two images, produces a similarity score, and applies a threshold. A test
set is therefore a set of *pairs*, and every pair is one of two kinds:

| | NIST term | The two images | Correct answer | Error is called |
|---|---|---|---|---|
| **Genuine pair** | mated | the same person | match | **FNMR** — false non-match, a legitimate user rejected |
| **Impostor pair** | non-mated | different people | non-match | **FMR** — false match, someone else accepted |

- **FMR = 1e-6** means: of a million impostor pairs, at most one scores above the threshold.
- **FNMR = 0.06%** means: of ten thousand genuine comparisons, six are wrongly rejected.

The two trade off against each other through the threshold, so **neither number means anything
alone**. "FNMR 0.06% at FMR 1e-6" is a complete statement; "99.94% accurate" is not.

## 2. Academic operating points are not commercial operating points

| | Typical academic report | NIST FRTE / commercial |
|---|---|---|
| Headline | IJB-C TAR @ **FAR = 1e-4** | FNMR @ **FMR = 1e-6** |
| Impostor operating point | 1 in 10,000 | **1 in 1,000,000** |

Two orders of magnitude apart. A model at 98% TAR at FAR=1e-4 tells you nothing about where it sits
at FMR=1e-6 — that is exactly the part of the ROC tail where models separate. Papers stop at 1e-5 or
1e-6 largely because public benchmarks do not contain enough comparisons to measure further (see
below). Any accuracy claim that does not state the operating point should be treated as unstated.

## 3. Monitoring during training

`config.val_targets` drives `CallBackVerification`, which runs every `config.verbose` steps:

```python
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
config.verbose = 2000
```

The named `.bin` files must sit in the directory `config.rec` points at. They ship with the
MS1MV3 / WebFace4M packages; some RecordIO releases (WebFace42M among them) do not include them.

**An empty `val_targets` silently disables evaluation entirely** — `ver_list` is empty, `ver_test()`
iterates zero times, and `verbose` fires a no-op forever. Most official configs in `configs/` ship
with `val_targets = []` because upstream evaluates offline; copying one does not give you monitoring.

These three sets saturate near the top of the range (LFW above 99.8% is routine) and are useful as
*regression detectors* during a long run, not as evidence of production readiness.

## 4. Offline evaluation

- **IJB-C** — `eval_ijbc.py`. The only benchmark that makes our numbers directly comparable to the
  published table in the README. Requires a separate dataset request.
- **ICCV21-MFR (MFR-ALL)** — submission-based, see the
  [MFR challenge](https://github.com/deepinsight/insightface/blob/master/challenges/mfr/README.md).
- **NIST FRTE** — free and open to any developer worldwide. Sign the participation agreement, wrap
  the model behind the published C++ API, run the validation package, encrypt and submit. This is the
  only way to obtain a number directly comparable to commercial vendors.

## 5. Building an FMR = 1e-6 test set of your own

Public benchmarks cannot reach 1e-6 because they do not contain enough impostor comparisons. Your own
data usually can, and it has a second advantage explained in §6.

### How many pairs are needed

To observe a rate of 1e-6 you need at least ~10^6 impostor pairs to see a single error, which is
statistically meaningless. At **10^7 pairs** you expect ~10 false matches and a relative error around
30% — the practical minimum. 10^8 gets you to ~10%.

### Where the pairs come from

A corpus with **one image per identity** produces impostor pairs for free: any two images are, by
construction, different people. For `N` identities:

```
C(N, 2) = N x (N - 1) / 2
```

which for N = 1,000,000 is 5 x 10^11 and for N = 4,000,000 is 8 x 10^12 — orders of magnitude more
than the 10^7 needed, so sampling is trivial. Genuine pairs are the scarce resource, not impostor
pairs; see [single_image_datasets.md](single_image_datasets.md) for recovering genuine pairs from a
nominally single-image corpus.

### Deduplication is a prerequisite, not an optimisation

The impostor set is only valid if every sampled pair really is two different people. Operational
identity corpora routinely contain the same person under multiple records (renewals, reissues,
re-registrations). Sampled as "impostors", those pairs score high and are counted as false matches,
which **inflates the right tail and pushes the computed threshold too high** — costing FNMR for no
security benefit.

The same deduplication pass produces both halves of the test set: a clean impostor set (FMR) and a
set of genuine pairs (FNMR). Run it once, use it twice.

## 6. Why the threshold must be calibrated on the deployment population

The impostor score distribution depends on who is in it. A population that is homogeneous — same
ethnic group, same age band, same capture device, same background and pose convention — produces
*higher* impostor scores and a **thicker right tail** than a diverse benchmark of web imagery and
video frames.

The threshold is read off that right tail: sort impostor scores descending, cut at the one-in-a-
million position. So a thin-tailed calibration set yields a threshold that is too permissive for a
thick-tailed deployment:

```
threshold from a diverse public benchmark at FMR=1e-6   ->  0.42
threshold the real deployment population requires        ->  0.51
```

Ship 0.42 and the system believes its false-match rate is one in a million while the true rate may be
one in ten thousand — two orders of magnitude worse, **with no error, no alarm and no log line**. It
simply admits people it should not. This is among the most common ways a face recognition deployment
fails silently, and it is why an in-domain impostor set matters more than a larger public one.

## 7. A practical order of work

1. Put the three `.bin` sets in place and set `val_targets` — cheap regression monitoring.
2. Deduplicate the in-domain corpus; keep both the impostor and the recovered genuine pairs.
3. Calibrate the threshold at FMR = 1e-6 on that impostor set, and freeze preprocessing, model hash
   and threshold together as one release unit.
4. Run IJB-C for comparability with published numbers.
5. Submit to NIST FRTE for comparability with commercial ones.
