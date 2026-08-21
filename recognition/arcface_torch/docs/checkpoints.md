# Checkpoints, resuming and changing the GPU count

All four training scripts (`train_v2.py`, `train_adaface.py`, `train_transface.py`,
`train_transface_pp.py`) share `utils/utils_checkpoint.py`. This document covers where state is
written, how to resume, how to bound disk usage, and how to move a run to a different number of
GPUs — which needs an explicit re-shard because PartialFC splits the classifier across ranks.

## 1. What is written, and where

Everything goes to `config.output`. Files are written at the **end of every epoch**:

| File | Contents | Lifetime |
|---|---|---|
| `checkpoint_gpu_{rank}.pt` | backbone, PartialFC shard, optimizer, LR scheduler, AMP `GradScaler`, `epoch`, `global_step` | overwritten every epoch |
| `checkpoint_epoch{E}_gpu_{rank}.pt` | a copy of the above, `E` = the epoch that just finished (0-based) | kept while `keep_epoch_checkpoints` is on |
| `model.pt` | backbone `state_dict` only, written by rank 0 | overwritten every epoch |
| `training.log`, `tensorboard/` | logs | append |

Relevant config keys:

```python
config.save_all_states = True          # write the full resumable checkpoint at all
config.keep_epoch_checkpoints = True   # also keep the per-epoch copies
config.keep_last_epochs = 4            # ... but only the newest 4 (0 = keep every epoch)
```

Two properties worth knowing:

- **`checkpoint_gpu_{rank}.pt` is crash-safe.** It is written to `.tmp` and then `os.replace`d, so an
  interrupted save never leaves a half-written "latest".
- **`epoch` in the file is the *next* epoch to run.** `save_checkpoint` stores `epoch + 1`, matching
  the upstream scripts. A file reporting `epoch 3` was written after epoch 2 finished.

### Disk usage

One epoch snapshot across all ranks costs roughly

```
  3 x num_classes x embedding_size x 4 B          (PFC weight + AdamW exp_avg + exp_avg_sq)
+ world_size x 3 x backbone size                  (replicated on every rank)
```

For WebFace42M (2,059,907 classes, 512-d) with a ViT-L backbone that is ≈25 GB on 4 GPUs and ≈32 GB
on 6. Keeping all 35 epochs is over a terabyte, hence `keep_last_epochs`.

`prune_epoch_checkpoints()` runs after each save. It orders by the **numeric** epoch (so
`epoch10` is not treated as older than `epoch9`), only ever touches the calling rank's own files, and
logs a warning instead of killing training if a file cannot be removed.

## 2. Resuming

```python
config.resume = True                  # latest checkpoint in config.output
config.resume_epoch = 7               # ... or the one written after epoch 7
config.resume_from = "work_dirs/x"    # ... or from another directory
config.resume_from = "/ck/e{epoch}_r{rank}.pt"   # ... or an explicit {rank}/{epoch} pattern
```

`load_checkpoint` restores backbone, PartialFC, optimizer, LR scheduler and `GradScaler` in place and
returns `(start_epoch, global_step)`. Confirm it worked by looking for this line:

```
resumed from .../checkpoint_gpu_0.pt: next epoch 3, global_step 41479
```

Notes:

- **`config.pretrained` is irrelevant when resuming.** `load_pretrained_backbone` runs first and
  `load_checkpoint` overwrites it, so drop the flag rather than wondering which wins.
- **The LR scheduler state wins over the config.** `_LRScheduler.load_state_dict` restores
  `total_iters`, `warmup_iters` and `base_lrs` from the checkpoint, so changing `num_epoch` (or the
  total batch — see below) in the config does *not* change the restored schedule.
  `load_checkpoint` logs a warning when `num_epoch` differs.
- **The GPU count must match** the run that wrote the checkpoint, unless you re-shard first.

## 3. Changing the number of GPUs

`PartialFC_V2` gives rank `r` the class range

```python
num_local = num_classes // world_size + int(r < num_classes % world_size)
start     = num_classes // world_size * r + min(r, num_classes % world_size)
```

so each rank's `state_dict_softmax_fc.weight` — and the `exp_avg` / `exp_avg_sq` of that parameter in
the optimizer state — is a *different slice* of the full `(num_classes, embedding_size)` matrix.
Loading a 4-rank checkpoint on 8 ranks fails on shape, and any manual reshuffle that does not
reproduce the formula above silently mis-aligns class centres against labels.

`scripts/reshard_checkpoint.py` concatenates the shards, re-splits them with the same formula, and
validates that the source shard sizes really match the claimed source world size before writing
anything. The backbone, the optimizer state of every backbone parameter, the `GradScaler` and the
AdaFace running statistics are identical on all ranks (DDP-synchronised, or computed from
all-gathered embeddings), so they are copied from rank 0.

### Case A — the total batch stays the same

Halve the per-GPU batch as you double the GPUs and nothing else changes: steps per epoch, the LR
curve and `global_step` all stay valid, and the optimisation trajectory is identical.

```shell
# 4 x 512 -> 8 x 256, both 2048 total
python scripts/reshard_checkpoint.py --src work_dirs/run --dst work_dirs/run_8gpu --dst-gpus 8
```

Then point `config.output` at the new directory, set `config.resume = True`, and launch with the new
`--nproc_per_node`.

### Case B — the total batch changes

The LR schedule is **step-based**, and the restored scheduler would keep the old `total_iters`. Pass
the new total batch so the schedule is rebuilt:

```shell
# 4 x 512 (2048) -> 6 x 512 (3072)
python scripts/reshard_checkpoint.py \
    --src work_dirs/run --dst work_dirs/run_6gpu --dst-gpus 6 \
    --new-total-batch 3072 --num-image 42474629 --warmup-epoch 3
```

which rewrites `total_iters`, `warmup_iters`, `last_epoch` and `global_step` to
`epoch x (num_image // new_total_batch)`, keeping the run at the same *fraction* of the schedule.

`--lr-scale` additionally multiplies `base_lrs`, `_last_lr` and the optimizer's `lr` / `initial_lr`.
**Do not use it for AdamW ViT training on this codebase** — see
[production_readiness.md](production_readiness.md#1-learning-rate-does-not-scale-with-batch-size-here):
the official WebFace42M ViT configs use `lr = 1e-3` at every total batch from 2048 to 24576.
It exists for recipes that genuinely follow the linear scaling rule (SGD/CNN).

Other flags: `--epoch E` re-shards `checkpoint_epoch{E}_gpu_*.pt` instead of the latest,
`--src-gpus` overrides the auto-detected source world size, `--dry-run` prints the split without
writing. `--src` and `--dst` must differ (every source rank is read before anything is written); to
patch a checkpoint in place, write to a temporary directory and move the files back.

### Worked example

A 4 x 512 WebFace42M TransFace-L run moved to 6 GPUs at the same per-GPU batch:

```
  rank 0..3: (514977, 512) (514977, 512) (514977, 512) (514976, 512)   epoch=2 step=41480
  -> 2,059,907 classes x 512 dims
  LR schedule: steps/epoch 20739 -> 13826, total_iters 725865 -> 483910,
               warmup 62217 -> 41478, global_step 41480 -> 27652
  rank 0..4: 343,318 rows each, rank 5: 343,317 rows
```

The resumed run logged `LearningRate 0.001001` where the 4-GPU run had `0.000667` at the same point —
exactly the `--lr-scale 1.5` that was passed, at an unchanged fraction of warmup. Without
`--lr-scale` the LR continues unchanged, which is what you normally want.
