"""Train one model for one (source -> target) dose pair and one fold.

Example:
    python train.py --data data/piglet --model cut --src 100_FBP --tgt 10_FBP --fold 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
from datasets import PatchDataset, body_slices, load_pair, slab_split
from models import build_model


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--model", required=True, choices=["cyclegan", "cut", "vae", "vaegan"])
    ap.add_argument("--src", default="100_FBP")
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gap", type=int, default=5, help="slices dropped around the test slab")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--d_hp", type=float, default=30.0,
                    help="noise-aware discriminator: weight of the high-pass input (0 = standard D)")
    ap.add_argument("--beta", type=float, default=1e-2, help="VAE KL weight")
    ap.add_argument("--lambda_idt", type=float, default=0.5, help="CycleGAN identity-loss weight")
    ap.add_argument("--hu_range", type=float, nargs=2, default=[-1024.0, 2048.0],
                    help="HU range mapped to [-1,1] for the networks (narrow = noise more visible)")
    ap.add_argument("--quick", action="store_true", help="6 res-blocks, 5k iters: fast first pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--log_every", type=int, default=200)
    return ap.parse_args(argv)


def run_name(a):
    return f"{a.model}_{a.src}_to_{a.tgt}_f{a.fold}_s{a.seed}"


def main(argv=None):
    args = get_args(argv)
    if args.quick and args.iters == 20000:
        args.iters = 5000
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(args.out, run_name(args))
    os.makedirs(out, exist_ok=True)

    datasets.set_hu_range(*args.hu_range)
    src, tgt, z, spacing = load_pair(args.data, args.src, args.tgt)
    tr, te = slab_split(len(z), args.n_folds, args.fold, args.gap, z=z)
    n0, m0 = len(tr), len(te)
    tr, te = body_slices(src, tr), body_slices(src, te)
    print(f"[data] dropped {n0 - len(tr)} train / {m0 - len(te)} test slices without body")
    print(f"[data] {len(z)} aligned slices -> train {len(tr)}, test {len(te)} (fold {args.fold}/{args.n_folds})")
    paired = args.model in ("vae", "vaegan")
    ds = PatchDataset(src, tgt, tr, patch=args.patch, paired=paired, length=args.iters * args.batch)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, drop_last=True,
                    pin_memory=dev == "cuda", persistent_workers=args.workers > 0)

    model_kw = dict(residual=not args.no_residual, d_hp=args.d_hp, beta=args.beta, lambda_idt=args.lambda_idt)
    model = build_model(args.model, quick=args.quick, lr=args.lr, **model_kw).to(dev)
    model.make_schedulers(args.iters)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(vars(args), train_idx=tr.tolist(), test_idx=te.tolist(), spacing=spacing,
                       paired_training=paired, device=dev), f, indent=1)

    t0, hist, run = time.time(), [], {}
    for it, (a, b) in enumerate(dl, 1):
        logs = model.train_step(a.to(dev, non_blocking=True), b.to(dev, non_blocking=True))
        model.step_schedulers()
        for k, v in logs.items():
            run[k] = run.get(k, 0) + v
        if it % args.log_every == 0 or it == args.iters:
            avg = {k: v / args.log_every for k, v in run.items()}
            el = time.time() - t0
            print(f"[{it}/{args.iters}] " + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                  + f"  {el / it:.3f}s/it  eta {el / it * (args.iters - it) / 60:.1f} min", flush=True)
            hist.append(dict(it=it, **avg))
            run = {}
        if it >= args.iters:
            break

    torch.save({"model": args.model, "quick": args.quick, "residual": not args.no_residual,
                "kw": model_kw, "state": model.state_dict()}, os.path.join(out, "model.pt"))
    with open(os.path.join(out, "train_log.json"), "w") as f:
        json.dump(dict(history=hist, minutes=(time.time() - t0) / 60,
                       params_G=sum(p.numel() for p in model.G.parameters())), f)
    print(f"[done] {out}  ({(time.time() - t0) / 60:.1f} min)")
    return out


if __name__ == "__main__":
    main()
