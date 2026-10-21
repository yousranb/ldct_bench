"""Are the low/ultralow scans spatially aligned with the normal-dose scan?

Separates three effects that all inflate a raw difference image:
  * noise            -> std of the high-pass difference in flat soft tissue
  * misregistration  -> std of the LOW-PASS difference in the body (should be ~0 if aligned)
  * shift            -> best slice offset (z) and in-plane shift (rows, cols) by correlation

    python check_alignment.py --root ~/CT_norsvin_data_segmented --subject BJ01100185
"""
import argparse

import numpy as np
from scipy import ndimage

import prepare_nifti as pn


def lowpass(x, s=3.0):
    return ndimage.gaussian_filter(x, s)


def flat_noise(x, ref):
    m = ndimage.uniform_filter(ref, 9)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(ref ** 2, 9) - m ** 2, 0))
    msk = ndimage.binary_erosion((m > -50) & (m < 150) & (sd < 25), iterations=3)
    h = x - ndimage.gaussian_filter(x, 2)
    return float(h[msk].std()) if msk.sum() > 300 else np.nan


def inplane_shift(refs, movs, max_shift=12):
    """integer (dy, dx) minimising the low-pass difference in the body, over several slices."""
    A = [lowpass(r) for r in refs]
    B = [lowpass(m) for m in movs]
    M = [r > -500 for r in refs]
    best, arg = np.inf, (0, 0)
    for step, rng_ in ((3, range(-max_shift, max_shift + 1, 3)), (1, None)):
        if rng_ is None:                                   # refine around the coarse optimum
            cy, cx = arg
            cand = [(cy + i, cx + j) for i in range(-2, 3) for j in range(-2, 3)]
        else:
            cand = [(i, j) for i in rng_ for j in rng_]
        for dy, dx in cand:
            e = np.mean([np.std((np.roll(b, (dy, dx), (0, 1)) - a)[m]) for a, b, m in zip(A, B, M)])
            if e < best:
                best, arg = e, (dy, dx)
    return arg


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--subject", required=True)
ap.add_argument("--series", nargs="+", default=["low", "ultralow"])
ap.add_argument("--n", type=int, default=12, help="number of slices tested")
ap.add_argument("--max_dz", type=int, default=8)
a = ap.parse_args()

ref, zr, sp = pn.load_nifti(pn.series_path(a.root, a.subject, "normal"))
body = np.array([(s[::8, ::8] > -500).mean() for s in ref])
cand = np.where(body > 0.3)[0]
idx = cand[np.linspace(a.max_dz, len(cand) - 1 - a.max_dz, a.n).astype(int)]
print(f"{a.subject}: normal {ref.shape}, testing {len(idx)} slices {idx.tolist()}")
print(f"  normal: flat-tissue noise {np.nanmedian([flat_noise(ref[i], ref[i]) for i in idx]):.1f} HU")

for name in a.series:
    v, z, _ = pn.load_nifti(pn.series_path(a.root, a.subject, name))
    if v.shape != ref.shape:
        print(f"  {name}: shape {v.shape} != normal {ref.shape}")
    noise = np.nanmedian([flat_noise(v[i], ref[i]) for i in idx])
    lp0 = np.median([np.std((lowpass(v[i]) - lowpass(ref[i]))[ref[i] > -500]) for i in idx])
    mean0 = np.median([np.mean((v[i] - ref[i])[ref[i] > -500]) for i in idx])
    # best slice offset
    errs = {}
    for dz in range(-a.max_dz, a.max_dz + 1):
        errs[dz] = np.median([np.std((lowpass(v[i + dz]) - lowpass(ref[i]))[ref[i] > -500]) for i in idx])
    dz_best = min(errs, key=errs.get)
    dy, dx = inplane_shift([ref[i] for i in idx], [v[i + dz_best] for i in idx])
    lp_best = np.median([np.std((lowpass(np.roll(v[i + dz_best], (dy, dx), (0, 1))) - lowpass(ref[i]))[ref[i] > -500])
                         for i in idx])
    print(f"  {name}: flat-tissue noise {noise:.1f} HU | mean diff {mean0:+.1f} HU | "
          f"low-pass diff std {lp0:.1f} HU as stored")
    print(f"     best slice offset dz = {dz_best:+d}, in-plane shift (rows, cols) = ({dy:+d}, {dx:+d}) "
          f"-> low-pass diff std {lp_best:.1f} HU")
print("\nInterpretation: aligned data give a low-pass diff std of a few HU (< ~15) with dz = 0 and shift (0, 0).\n"
      "A large value that drops after the shift = rigid offset (fixable); one that stays large = the animal\n"
      "moved/breathed between scans (non-rigid misalignment): only unpaired methods and noise metrics are then safe.")
