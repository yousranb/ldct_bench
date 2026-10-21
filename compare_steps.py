"""Quick visual + numeric comparison of a few prediction runs on their common test slices
(e.g. residual CADM sampled with 25 vs 50 DDIM steps). CPU only, ~1 min.

    python compare_steps.py --runs runs_norsvin --names cadm_res cadm_res_50s --dose 10

Numbers (flat soft tissue, texture = image - Gaussian(2 px)):
  texture std      : noise level (compare with the real LDCT)
  NPS error        : relative error of the radial NPS vs the real LDCT (lower = better)
  stripe index     : local orientation coherence of the texture (structure tensor);
                     ~0.2-0.3 for isotropic grain, higher = oriented stripes (compare with the real LDCT)
"""
import argparse
import json
import os

import numpy as np
from scipy import ndimage

from datasets import load_pair

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_norsvin")
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--dose", type=int, default=10)
ap.add_argument("--recon", default="IR")
ap.add_argument("--n_fig", type=int, default=4)
a = ap.parse_args()

preds, cfg = {}, None
for n in a.names:
    rd = os.path.join(a.runs, f"{n}_100_{a.recon}_to_{a.dose}_{a.recon}_f0_s0")
    c = json.load(open(os.path.join(rd, "config.json")))
    preds[n] = (np.load(os.path.join(rd, "pred.npy"), mmap_mode="r"), c["test_idx"])
    cfg = cfg or c
nmin = min(len(t) for _, t in preds.values())
idx = preds[a.names[0]][1][:nmin]
for n, (_, t) in preds.items():
    assert t[:nmin] == idx, f"{n}: different test slices"
src, ld, _, _ = load_pair(cfg["data"], cfg["src"], cfg["tgt"])
print(f"{a.dose}%: comparing {len(idx)} common test slices")


def hp(x):
    return x - ndimage.gaussian_filter(x, 2)


def flat_mask(ref):
    m = ndimage.uniform_filter(ref, 9)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(ref ** 2, 9) - m ** 2, 0))
    return ndimage.binary_erosion((m > -50) & (m < 150) & (sd < 40), iterations=3)


def coherence(h, s=4):
    gy, gx = np.gradient(h)
    jxx, jyy, jxy = (ndimage.gaussian_filter(v, s) for v in (gx * gx, gy * gy, gx * gy))
    return np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)


def radial_nps(h, msk, P=64):
    acc, n = np.zeros((P, P)), 0
    for i in range(0, h.shape[0] - P, P // 2):
        for j in range(0, h.shape[1] - P, P // 2):
            if msk[i:i + P, j:j + P].mean() > 0.9:
                p = h[i:i + P, j:j + P] - h[i:i + P, j:j + P].mean()
                acc += np.abs(np.fft.fftshift(np.fft.fft2(p * np.hanning(P)[:, None] * np.hanning(P)))) ** 2
                n += 1
    yy, xx = np.mgrid[:P, :P] - P // 2
    r = np.hypot(yy, xx).astype(int).ravel()
    return np.bincount(r, (acc / max(n, 1)).ravel())[:P // 2] / np.maximum(np.bincount(r)[:P // 2], 1), n


sub = np.linspace(0, len(idx) - 1, min(60, len(idx))).astype(int)      # slices used for the numbers
res = {k: dict(std=[], coh=[], nps=0) for k in ["real LDCT"] + a.names}
for k in sub:
    i = idx[k]
    nd = np.asarray(src[i], np.float32)
    msk = flat_mask(nd)
    if msk.sum() < 2000:
        continue
    imgs = {"real LDCT": np.asarray(ld[i], np.float32)}
    imgs.update({n: np.asarray(preds[n][0][k], np.float32) for n in a.names})
    for n, x in imgs.items():
        h = hp(x)
        res[n]["std"].append(h[msk].std())
        res[n]["coh"].append(coherence(h)[msk].mean())
        N, cnt = radial_nps(h, msk)
        if cnt:
            res[n]["nps"] = res[n]["nps"] + N
ref = res["real LDCT"]["nps"]
print(f"{'':22s} texture std   NPS error   stripe index")
for n, r in res.items():
    err = np.abs(r["nps"] - ref).sum() / ref.sum()
    print(f"{n:22s} {np.mean(r['std']):8.1f}     {err:8.3f}     {np.mean(r['coh']):8.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cols = ["NDCT", "real LDCT"] + a.names
show = np.linspace(len(idx) * 0.15, len(idx) * 0.85, a.n_fig).astype(int)
fig, ax = plt.subplots(a.n_fig, len(cols), figsize=(3.2 * len(cols), 3.2 * a.n_fig), squeeze=False)
for r, k in enumerate(show):
    i = idx[k]
    imgs = [src[i], ld[i]] + [preds[n][0][k] for n in a.names]
    c0 = src.shape[-1] // 2 - 96
    for c, x in enumerate(imgs):
        ax[r, c].imshow(np.asarray(x, np.float32)[c0:c0 + 192, c0:c0 + 192], cmap="gray", vmin=-100, vmax=200)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[r, 0].set_ylabel(f"slice {i}")
for c, t in enumerate(cols):
    ax[0, c].set_title(t)
fig.suptitle(f"{a.dose}% dose, center 192x192 crop, window [-100, 200] HU")
fig.tight_layout()
out = f"compare_steps_{a.dose}.png"
fig.savefig(out, dpi=110)
print(f"[saved] {out}")
