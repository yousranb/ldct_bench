"""Is the striped texture of residual CADM caused by the SD autoencoder (VAE) decoder?

No diffusion involved. On real test slices:
  recon     = D(E(LDCT))                       -> can the VAE reproduce real LDCT at all?
  oracle    = NDCT + D(E(LDCT)) - D(E(NDCT))   -> the BEST residual CADM could ever do
                                                  (a perfect diffusion model predicting the exact latent residual)
If 'oracle' already shows the stripes, they come from the decoder, not from the diffusion model.

    python -u vae_test.py --data data/norsvin --recon IR --dose 10 --sd_path ~/models/sd15
Outputs: vae_test_<dose>.png and printed numbers.
"""
import argparse

import numpy as np
import torch
from scipy import ndimage

from diffusion_common import DEFAULT_WINDOW, hu_to_unit, load_multi_dose, unit_to_hu

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/norsvin")
ap.add_argument("--recon", default="IR")
ap.add_argument("--dose", type=int, default=10)
ap.add_argument("--sd_path", required=True)
ap.add_argument("--n", type=int, default=6, help="number of test slices")
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
win = DEFAULT_WINDOW

from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(a.sd_path, subfolder="vae").to(dev).eval()

D = load_multi_dose(a.data, f"100_{a.recon}", a.recon, [a.dose], 3, 0, 5)[a.dose]
te = D["test"]
pick = te[np.linspace(len(te) * 0.1, len(te) * 0.9, a.n).astype(int)]


@torch.no_grad()
def ed(x_hu):                                   # HU slice -> D(E(x)) in HU (same path as CADM: mode encoding, 3ch mean)
    x = hu_to_unit(torch.tensor(x_hu.astype(np.float32), device=dev), win)[None, None]
    z = vae.encode(x.repeat(1, 3, 1, 1)).latent_dist.mode()
    y = vae.decode(z).sample.mean(1, keepdim=True).clamp(-1, 1)
    return unit_to_hu(y[0, 0].float().cpu().numpy(), win)


def hp(x):                                      # noise/texture = image minus its smooth version
    return x - ndimage.gaussian_filter(x, 2)


def flat_mask(ref):
    m = ndimage.uniform_filter(ref, 9)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(ref ** 2, 9) - m ** 2, 0))
    return ndimage.binary_erosion((m > -50) & (m < 150) & (sd < 40), iterations=3)


def nps2d(x, msk, P=64):                        # mean 2D NPS of the texture over flat patches
    h = hp(x)
    acc, n = np.zeros((P, P)), 0
    for i in range(0, x.shape[0] - P, P // 2):
        for j in range(0, x.shape[1] - P, P // 2):
            if msk[i:i + P, j:j + P].mean() > 0.9:
                p = h[i:i + P, j:j + P] - h[i:i + P, j:j + P].mean()
                acc += np.abs(np.fft.fftshift(np.fft.fft2(p * np.hanning(P)[:, None] * np.hanning(P)))) ** 2
                n += 1
    return acc / max(n, 1), n


def anisotropy(N):                              # 0 = isotropic, 1 = all power along one direction
    P = N.shape[0]
    yy, xx = np.mgrid[:P, :P] - P // 2
    ang = np.arctan2(yy, xx)
    r = np.hypot(yy, xx)
    sel = (r > 3) & (r < P // 2)
    c = np.sum(N[sel] * np.exp(2j * ang[sel])) / np.sum(N[sel])
    return float(np.abs(c))


rows, stats = [], {k: [] for k in ("ld", "rec", "orc", "nd")}
NPS = {k: 0 for k in stats}
for i in pick:
    nd = D["src"][i].astype(np.float32)
    ld = D["ld"][i].astype(np.float32)
    rec_ld, rec_nd = ed(ld), ed(nd)
    orc = nd + rec_ld - rec_nd
    msk = flat_mask(nd)
    for k, x in (("ld", ld), ("rec", rec_ld), ("orc", orc), ("nd", nd)):
        if msk.sum() > 500:
            stats[k].append(hp(x)[msk].std())
            N, n = nps2d(x, msk)
            if n:
                NPS[k] = NPS[k] + N
    rows.append((i, nd, ld, rec_ld, orc))

names = {"nd": "NDCT", "ld": "real LDCT", "rec": "D(E(LDCT))", "orc": "oracle residual"}
print(f"\n{a.dose}% dose, {len(pick)} test slices, window {win}")
print(f"{'':18s} texture std (HU)   NPS anisotropy (0=round grain)")
for k in ("nd", "ld", "rec", "orc"):
    print(f"{names[k]:18s} {np.mean(stats[k]):8.1f}           {anisotropy(NPS[k]) if np.ndim(NPS[k]) else float('nan'):.3f}")
for i, nd, ld, rec, orc in rows[:1]:
    m = ld > -500
    psnr = 10 * np.log10((win[1] - win[0]) ** 2 / np.mean((np.clip(rec, *win) - np.clip(ld, *win))[m] ** 2))
    print(f"VAE reconstruction PSNR of the LDCT (window): {psnr:.1f} dB")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cols = ["NDCT", "real LDCT", "D(E(LDCT))", "oracle residual", "LDCT texture", "oracle texture"]
fig, ax = plt.subplots(len(rows) + 1, 6, figsize=(18, 3 * (len(rows) + 1)))
for r, (i, nd, ld, rec, orc) in enumerate(rows):
    c0 = nd.shape[0] // 2 - 96
    crop = (slice(c0, c0 + 192), slice(c0, c0 + 192))      # zoom so the grain is visible
    for c, x in enumerate([nd, ld, rec, orc]):
        ax[r, c].imshow(x[crop], cmap="gray", vmin=-100, vmax=200)
    for c, x in ((4, ld), (5, orc)):
        ax[r, c].imshow(hp(x)[crop], cmap="gray", vmin=-60, vmax=60)
    ax[r, 0].set_ylabel(f"slice {i}")
for c, k in enumerate(("nd", "ld", "rec", "orc")):
    if np.ndim(NPS[k]):
        ax[-1, c].imshow(np.log(NPS[k] + 1e-6), cmap="magma")
        ax[-1, c].set_xlabel(f"2D NPS, anisotropy {anisotropy(NPS[k]):.2f}")
ax[-1, 4].axis("off"); ax[-1, 5].axis("off")
for c, t in enumerate(cols):
    ax[0, c].set_title(t)
for x in ax.ravel():
    x.set_xticks([]); x.set_yticks([])
fig.suptitle(f"SD autoencoder test, {a.dose}% dose (center 192x192 crop, window [-100, 200] HU)")
fig.tight_layout()
out = a.out or f"vae_test_{a.dose}.png"
fig.savefig(out, dpi=110)
print(f"[saved] {out}")
