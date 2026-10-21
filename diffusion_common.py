"""Shared helpers for the diffusion models (CADM and the classical per-dose DDPM)."""
import json
import os

import numpy as np
import torch

from datasets import body_slices, drop_subjects, load_pair, slab_split

DEFAULT_WINDOW = (-150.0, 250.0)   # HU range the diffusion models work in (thesis Fig. 3.13)


def hu_to_unit(x, window):          # HU -> [-1, 1]
    lo, hi = window
    return ((x - lo) / (hi - lo)).clamp(0, 1) * 2 - 1 if torch.is_tensor(x) else \
        np.clip((x - lo) / (hi - lo), 0, 1) * 2 - 1


def unit_to_hu(x, window):
    lo, hi = window
    return (x + 1) / 2 * (hi - lo) + lo


def dose_code(r):
    """log-dose code in [0, 1]: 5 % -> 0, 100 % -> 1 (extrapolates for r < 5 %)."""
    return float((np.log(r) - np.log(0.05)) / (0 - np.log(0.05)))


def load_multi_dose(data, src, recon, doses, n_folds, fold, gap, exclude=()):
    """Per target dose: aligned NDCT/LDCT volumes (float16 HU) and the SAME slab split
    the GAN benchmark uses for that (src, tgt) pair."""
    with open(os.path.join(data, "index.json")) as f:
        index = json.load(f)
    out = {}
    for d in doses:
        tgt = f"{d}_{recon}"
        if tgt not in index:
            print(f"[skip] {tgt} not in {data}/index.json")
            continue
        s, t, z, spacing = load_pair(data, src, tgt)
        tr, te = slab_split(len(z), n_folds, fold, gap, z=z)
        tr, te = body_slices(s, tr), body_slices(s, te)
        if exclude:
            n0 = len(tr)
            tr = drop_subjects(tr, z, index[tgt].get("subjects"), list(exclude))
            print(f"[data] excluded {list(exclude)} from training: {n0} -> {len(tr)} slices")
        out[d] = dict(tgt=tgt, src=s.astype(np.float16), ld=t.astype(np.float16), z=z, train=tr, test=te,
                      dx=float(spacing[0]), spacing=spacing)
        print(f"[data] {src} -> {tgt}: {len(z)} slices, train {len(tr)}, test {len(te)}")
    if not out:
        raise SystemExit("no target dose available")
    return out


def write_pred_run(runs_dir, model_name, src, tgt, fold, n_folds, seed, data, train_idx, test_idx,
                   pred_hu, window, extra=None):
    """Store predictions so evaluate.py scores them exactly like the trained GAN runs."""
    rd = os.path.join(runs_dir, f"{model_name}_{src}_to_{tgt}_f{fold}_s{seed}")
    os.makedirs(rd, exist_ok=True)
    np.save(os.path.join(rd, "pred.npy"), np.asarray(pred_hu, np.float16))
    cfg = dict(model=model_name, data=data, src=src, tgt=tgt, fold=fold, n_folds=n_folds, seed=seed,
               train_idx=[int(i) for i in train_idx], test_idx=[int(i) for i in test_idx],
               hu_range=list(window), prediction_file="pred.npy")
    cfg.update(extra or {})
    with open(os.path.join(rd, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"[saved] {rd}/pred.npy  {np.asarray(pred_hu).shape}")
    return rd


def lf_anchor(pred_hu, nd_hu, window, sigma):
    """Optional post-processing: keep the model's high frequencies (noise, streaks) but
    take the low frequencies (mean HU, anatomy contrast) from the NDCT:
        out = pred - G_sigma(pred) + G_sigma(NDCT)
    Removes any global brightness/contrast drift of the generator. sigma in pixels."""
    from scipy import ndimage
    lo, hi = window
    nd = np.clip(np.asarray(nd_hu, np.float32), lo, hi)
    out = []
    for p, n in zip(np.asarray(pred_hu, np.float32), nd):
        out.append(p - ndimage.gaussian_filter(p, sigma) + ndimage.gaussian_filter(n, sigma))
    return np.clip(np.stack(out), lo, hi)
