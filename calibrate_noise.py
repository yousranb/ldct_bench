"""Per-dose noise-amplitude calibration of existing predictions (CPU only, no re-sampling).

Problem it fixes: a dose-conditioned generator often compresses the dose range
(too little noise at 5 %, too much at 25 %). We keep the model's noise TEXTURE and
anatomy, and rescale only the inserted high-frequency noise so its amplitude matches
the noise level measured on the TRAINING slices of that dose (no test ground truth
is used - same principle as the I0 calibration of the Zeng baseline).

    inserted noise   n = HF(pred - NDCT)          HF = x - G_sigma(x)
    output           out = NDCT + k_d * n          (also removes brightness drift, like _lf)
    k_d = std[ HF(LDCT_train - NDCT_train) ] / std[ HF(pred_test - NDCT_test) ]   on flat soft tissue

    python calibrate_noise.py --runs runs_v3 --model cadm_long
    -> runs_v3/cadm_long_cal_<src>_to_<tgt>_f0_s0/pred.npy   (k_d stored in config.json)
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage

from datasets import load_pair
from diffusion_common import write_pred_run


def hf(x, sigma):
    return x - ndimage.gaussian_filter(x, sigma)


def flat_mask(nd, lo=-50, hi=150, max_std=20.0):
    """flat soft tissue: HU in [lo, hi] and low local std in the NDCT."""
    m = ndimage.uniform_filter(nd, 9)
    s = np.sqrt(np.maximum(ndimage.uniform_filter(nd ** 2, 9) - m ** 2, 0))
    return (m > lo) & (m < hi) & (s < max_std)


def noise_std(diffs, nds, sigma, window):
    lo, hi = window
    vals = []
    for d, n in zip(diffs, nds):
        n = np.clip(n, lo, hi)
        msk = ndimage.binary_erosion(flat_mask(n), iterations=3)
        if msk.sum() > 500:
            vals.append(hf(d, sigma)[msk])
    if not vals:
        raise SystemExit("no flat soft-tissue pixels found")
    return float(np.concatenate(vals).std())


ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_v3")
ap.add_argument("--model", required=True, help="e.g. cadm_long (the raw predictions, not _lf)")
ap.add_argument("--sigma", type=float, default=4.0)
ap.add_argument("--n_train", type=int, default=40, help="training slices used to measure the target noise")
ap.add_argument("--suffix", default="_cal")
a = ap.parse_args()

found = 0
for cfg_path in sorted(glob.glob(os.path.join(a.runs, f"{a.model}_*", "config.json"))):
    cfg = json.load(open(cfg_path))
    if cfg.get("model") != a.model or "prediction_file" not in cfg:
        continue
    rd = os.path.dirname(cfg_path)
    lo, hi = window = cfg["hu_range"]
    pred = np.load(os.path.join(rd, cfg["prediction_file"])).astype(np.float32)
    s, t, _, _ = load_pair(cfg["data"], cfg["src"], cfg["tgt"])
    te = np.asarray(cfg["test_idx"])
    tr = np.asarray(cfg["train_idx"])
    tr = tr[np.linspace(0, len(tr) - 1, min(a.n_train, len(tr))).astype(int)]
    nd_te = np.clip(s[te], lo, hi)
    assert len(nd_te) == len(pred), f"{rd}: {len(pred)} predictions vs {len(nd_te)} test slices"

    target = noise_std([np.clip(t[i], lo, hi) - np.clip(s[i], lo, hi) for i in tr], s[tr], a.sigma, window)
    current = noise_std(pred - nd_te, nd_te, a.sigma, window)
    k = target / max(current, 1e-6)
    out = np.stack([np.clip(n + k * hf(p - n, a.sigma), lo, hi) for p, n in zip(pred, nd_te)])
    print(f"{cfg['tgt']}: target noise {target:.1f} HU (train LDCT) | model {current:.1f} HU | k = {k:.2f}")
    write_pred_run(a.runs, a.model + a.suffix, cfg["src"], cfg["tgt"], cfg["fold"], cfg["n_folds"], cfg["seed"],
                   cfg["data"], cfg["train_idx"], cfg["test_idx"], out, window,
                   extra=dict(noise_gain=k, target_noise=target, model_noise=current, sigma=a.sigma, from_run=rd))
    found += 1
print(f"done: {found} run(s)")
if not found:
    raise SystemExit(f"no pred.npy run named '{a.model}' in {a.runs}")
