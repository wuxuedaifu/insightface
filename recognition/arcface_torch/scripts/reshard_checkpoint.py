#!/usr/bin/env python
"""Re-shard a PartialFC training checkpoint for a different number of GPUs.

PartialFC splits the classifier weight across ranks, so `{output}/checkpoint_gpu_{rank}.pt`
holds a *different* slice of the class centres on every GPU (see PartialFC_V2.__init__:
rank r owns classes [num_classes//W*r + min(r, num_classes%W), ... + num_local)).
Resuming with a different world size therefore needs the shards concatenated and split again.
This also re-shards the AdamW moments (exp_avg / exp_avg_sq) of that weight.

    # 4 GPUs -> 8 GPUs, same total batch (per-GPU batch halved): nothing else changes
    python scripts/reshard_checkpoint.py --src <output> --dst <output>_8gpu --dst-gpus 8

    # 4 GPUs -> 8 GPUs keeping the per-GPU batch (total batch 2048 -> 4096):
    # the LR schedule is step-based, so it has to be rebuilt for the new steps/epoch
    python scripts/reshard_checkpoint.py --src <output> --dst <output>_8gpu --dst-gpus 8 \
        --new-total-batch 4096 --num-image 42474629 --warmup-epoch 3 --lr-scale 2.0

The backbone, the optimizer state of every backbone parameter, the GradScaler and the AdaFace
running stats are identical on all ranks (DDP / all-gathered), so they are copied from rank 0.
"""
import argparse
import glob
import os
import re

import torch

LATEST = "checkpoint_gpu_{rank}.pt"
EPOCH = "checkpoint_epoch{epoch}_gpu_{rank}.pt"


def shard_range(num_classes, world_size, rank):
    """Exactly PartialFC_V2's split."""
    num_local = num_classes // world_size + int(rank < num_classes % world_size)
    start = num_classes // world_size * rank + min(rank, num_classes % world_size)
    return start, num_local


def src_path(src, rank, epoch):
    name = LATEST.format(rank=rank) if epoch is None else EPOCH.format(epoch=epoch, rank=rank)
    return os.path.join(src, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir holding the checkpoints to re-shard")
    ap.add_argument("--dst", required=True, help="dir to write the new checkpoints into (must differ from --src)")
    ap.add_argument("--dst-gpus", type=int, required=True, help="world size to resume with")
    ap.add_argument("--src-gpus", type=int, help="world size that wrote --src (default: count the files)")
    ap.add_argument("--epoch", type=int, help="use checkpoint_epoch{E}_gpu_*.pt instead of the latest")
    ap.add_argument("--new-total-batch", type=int,
                    help="rebuild the LR schedule for this total batch (batch_size * dst_gpus). "
                         "Only needed when the total batch changes; requires --num-image")
    ap.add_argument("--num-image", type=int, help="config.num_image, for --new-total-batch")
    ap.add_argument("--num-epoch", type=int, help="config.num_epoch (default: the value in the checkpoint)")
    ap.add_argument("--warmup-epoch", type=int, default=3, help="config.warmup_epoch, for --new-total-batch")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiply the learning rate by this (linear scaling rule: 2.0 when the total batch doubles)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        raise SystemExit("--dst must differ from --src (the re-shard reads every source rank)")
    src_gpus = a.src_gpus
    if src_gpus is None:
        pat = os.path.join(a.src, LATEST.format(rank="*") if a.epoch is None else EPOCH.format(epoch=a.epoch, rank="*"))
        found = glob.glob(pat)
        ranks = {int(re.search(r"_gpu_(\d+)\.pt$", f).group(1)) for f in found}
        if not ranks or sorted(ranks) != list(range(len(ranks))):
            raise SystemExit(f"cannot infer --src-gpus from {pat} (found ranks {sorted(ranks)})")
        src_gpus = len(ranks)
    print(f"source: {src_gpus} ranks in {a.src}" + (f" (epoch {a.epoch})" if a.epoch is not None else " (latest)"))

    # --- pass 1: collect the classifier shards (only the 3 big tensors are kept in RAM)
    weights, exp_avgs, exp_avg_sqs, metas = [], [], [], []
    pfc_id = None
    for r in range(src_gpus):
        path = src_path(a.src, r, a.epoch)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        pid = ck["state_optimizer"]["param_groups"][1]["params"][0]
        if pfc_id is None:
            pfc_id = pid
        elif pid != pfc_id:
            raise SystemExit("the PartialFC parameter has different optimizer ids across ranks")
        st = ck["state_optimizer"]["state"][pid]
        weights.append(ck["state_dict_softmax_fc"]["weight"])
        exp_avgs.append(st["exp_avg"])
        exp_avg_sqs.append(st["exp_avg_sq"])
        metas.append((ck["epoch"], ck["global_step"]))
        print(f"  rank {r}: {tuple(weights[-1].shape)}  epoch={ck['epoch']} step={ck['global_step']}")
        if r == 0:
            template = ck                      # everything else is rank-independent
        else:
            del ck
    if len(set(metas)) != 1:
        raise SystemExit(f"ranks disagree on (epoch, global_step): {metas}")
    epoch, global_step = metas[0]

    num_classes = sum(w.shape[0] for w in weights)
    for r in range(src_gpus):                  # the shards must line up with PartialFC's own split
        _, expect = shard_range(num_classes, src_gpus, r)
        if weights[r].shape[0] != expect:
            raise SystemExit(f"rank {r} holds {weights[r].shape[0]} classes, PartialFC would give it {expect}; "
                             f"is --src-gpus really {src_gpus}?")
    weight = torch.cat(weights); del weights
    exp_avg = torch.cat(exp_avgs); del exp_avgs
    exp_avg_sq = torch.cat(exp_avg_sqs); del exp_avg_sqs
    print(f"  -> {num_classes:,} classes x {weight.shape[1]} dims, next epoch {epoch}, global_step {global_step}")

    # --- optional: rebuild the step-based LR schedule for a new total batch
    sched = template["state_lr_scheduler"]
    new_step = global_step
    if a.new_total_batch:
        if not a.num_image:
            raise SystemExit("--new-total-batch needs --num-image")
        num_epoch = a.num_epoch or template.get("num_epoch")
        steps = a.num_image // a.new_total_batch
        old_spe = sched["total_iters"] // num_epoch
        new_step = epoch * steps                       # epoch = the next epoch to run
        print(f"  LR schedule: steps/epoch {old_spe} -> {steps}, total_iters {sched['total_iters']} -> "
              f"{steps * num_epoch}, warmup {sched['warmup_iters']} -> {steps * a.warmup_epoch}, "
              f"global_step {global_step} -> {new_step}")
        sched["total_iters"] = steps * num_epoch
        sched["warmup_iters"] = steps * a.warmup_epoch
        sched["last_epoch"] = new_step
        sched["_step_count"] = new_step + 1
    if a.lr_scale != 1.0:
        sched["base_lrs"] = [lr * a.lr_scale for lr in sched["base_lrs"]]
        sched["_last_lr"] = [lr * a.lr_scale for lr in sched["_last_lr"]]
        for g in template["state_optimizer"]["param_groups"]:
            g["lr"] *= a.lr_scale
            if "initial_lr" in g:
                g["initial_lr"] *= a.lr_scale
        print(f"  lr x{a.lr_scale}: base_lrs {sched['base_lrs']}, current lr {sched['_last_lr']}")
    template["global_step"] = new_step

    # --- pass 2: write one checkpoint per new rank
    if a.dry_run:
        for r in range(a.dst_gpus):
            s, n = shard_range(num_classes, a.dst_gpus, r)
            print(f"  [dry-run] rank {r}: classes [{s:,}, {s + n:,})  {n:,} rows")
        return
    os.makedirs(a.dst, exist_ok=True)
    for r in range(a.dst_gpus):
        s, n = shard_range(num_classes, a.dst_gpus, r)
        template["state_dict_softmax_fc"]["weight"] = weight[s:s + n].clone()
        opt_state = template["state_optimizer"]["state"][pfc_id]
        opt_state["exp_avg"] = exp_avg[s:s + n].clone()
        opt_state["exp_avg_sq"] = exp_avg_sq[s:s + n].clone()
        out = os.path.join(a.dst, LATEST.format(rank=r))
        torch.save(template, out + ".tmp")
        os.replace(out + ".tmp", out)
        print(f"  wrote {out}: classes [{s:,}, {s + n:,})  {n:,} rows")
    print(f"done. resume with config.resume = True and config.resume_from = \"{os.path.abspath(a.dst)}\"")


if __name__ == "__main__":
    main()
