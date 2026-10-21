"""Per-dose NOISE-TEXTURE calibration of existing predictions (CPU only, no re-sampling).

Generalises calibrate_noise.py: instead of one gain k per dose, a 2-D frequency filter
H_d(fx, fy) reshapes the inserted noise so its 2-D NPS (texture, grain size AND
orientation) matches the NPS measured on the TRAINING slices of that dose.
No test ground truth is used.

    n        = HF_sigma(pred - NDCT)                    inserted noise (drift removed)
    H_d      = sqrt( NPS_train(LDCT - NDCT) / NPS_model(n) )   on flat soft-tissue ROIs
    out      = NDCT + IFFT( FFT(n) * H_d )

    python nps_match.py --runs runs_v3 --model cadm_long
    -> runs_v3/cadm_long_nps_<src>_to_<tgt>_f0_s0/pred.npy
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
    m = ndimage.uniform_filter(nd, 9)
    s = np.sqrt(np.maximum(ndimage.uniform_filter(nd ** 2, 9) - m ** 2, 0))
    return ndimage.binary_erosion((m > lo) & (m < hi) & (s < max_std), iterations=3)


def nps2d(noises, nds, window, sigma, roi=64, stride=32, apply_hf=True):
    """mean 2-D NPS (fftshifted, roi x roi) of the HF noise over flat ROIs."""
    lo, hi = window
    acc, n = np.zeros((roi, roi)), 0
    win = np.outer(np.hanning(roi), np.hanning(roi))     # reduces spectral leakage
    win /= np.sqrt((win ** 2).mean())
    for d, nd in zip(noises, nds):
        msk = flat_mask(np.clip(nd, lo, hi))
        h = hf(d, sigma) if apply_hf else d
        H, W = h.shape
        for y in range(0, H - roi + 1, stride):
            for x in range(0, W - roi + 1, stride):
                if msk[y:y + roi, x:x + roi].all():
                    r = h[y:y + roi, x:x + roi]
                    acc += np.abs(np.fft.fftshift(np.fft.fft2((r - r.mean()) * win))) ** 2
                    n += 1
    if n == 0:
        raise SystemExit("no flat soft-tissue ROI found")
    return acc / n, n


def build_filter(nps_t, nps_m, size, smooth=1.0, gmax=10.0):
    t = ndimage.gaussian_filter(nps_t, smooth)
    m = ndimage.gaussian_filter(nps_m, smooth)
    g = np.sqrt(t / np.maximum(m, 1e-3 * m.max()))
    g = np.clip(g, 0, gmax)
    # roi frequency grid -> full-image frequency grid (exact frequency coordinates)
    from scipy.interpolate import RegularGridInterpolator
    f_roi = np.fft.fftshift(np.fft.fftfreq(g.shape[0]))
    interp = RegularGridInterpolator((f_roi, f_roi), g, bounds_error=False, fill_value=None)
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    return np.clip(interp(np.stack([fy, fx], -1)), 0, gmax)          # unshifted, size x size


ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_v3")
ap.add_argument("--model", required=True, help="e.g. cadm_long (raw predictions)")
ap.add_argument("--sigma", type=float, default=4.0)
ap.add_argument("--n_train", type=int, default=60)
ap.add_argument("--suffix", default="_nps")
ap.add_argument("--gmax", type=float, default=6.0, help="max per-frequency gain")
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
    te, tr = np.asarray(cfg["test_idx"]), np.asarray(cfg["train_idx"])
    tr = tr[np.linspace(0, len(tr) - 1, min(a.n_train, len(tr))).astype(int)]
    nd_te = np.clip(s[te], lo, hi)
    assert len(nd_te) == len(pred)

    nps_t, nt = nps2d([np.clip(t[i], lo, hi) - np.clip(s[i], lo, hi) for i in tr], s[tr], window, a.sigma)
    nps_m, nm = nps2d(pred - nd_te, nd_te, window, a.sigma)
    Hf = build_filter(nps_t, nps_m, pred.shape[-1], gmax=a.gmax)
    out = []
    for p, nd in zip(pred, nd_te):
        n = hf(p - nd, a.sigma)
        out.append(np.clip(nd + np.real(np.fft.ifft2(np.fft.fft2(n) * Hf)), lo, hi))
    out = np.stack(out)
    nps_o, _ = nps2d(out - nd_te, nd_te, window, a.sigma, apply_hf=False)   # out is already HF
    k = float(np.sqrt(nps_t.sum() / nps_m.sum()))
    err = lambda q: float(np.abs(q / q.sum() - nps_t / nps_t.sum()).sum())
    print(f"{cfg['tgt']}: ROIs train {nt} / test {nm} | noise gain {k:.2f} | "
          f"NPS-shape L1 error before {err(nps_m):.3f} -> after {err(nps_o):.3f}")
    write_pred_run(a.runs, a.model + a.suffix, cfg["src"], cfg["tgt"], cfg["fold"], cfg["n_folds"], cfg["seed"],
                   cfg["data"], cfg["train_idx"], cfg["test_idx"], out, window,
                   extra=dict(nps_match=True, sigma=a.sigma, from_run=rd))
    found += 1
print(f"done: {found} run(s)")
if not found:
    raise SystemExit(f"no pred.npy run named '{a.model}' in {a.runs}")
