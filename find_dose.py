"""Look for dose information for each series, and estimate the noise-equivalent dose.

1) metadata: NIfTI header text fields and extensions, JSON sidecars (dcm2niix/BIDS),
   DICOM files anywhere under the subject folder (mAs, CTDIvol, kernel, recon).
2) noise-equivalent dose: with FBP, noise variance ~ 1 / dose, so
       dose_low / dose_normal ~ (sigma_normal / sigma_low)^2
   With iterative reconstruction this UNDER-estimates the dose reduction (IR removes
   noise), so treat it as an upper bound, not as the true dose.

    python find_dose.py --root ~/CT_norsvin_data_segmented --subject BJ01100185
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--subject", required=True)
ap.add_argument("--series", nargs="+", default=["normal", "low", "ultralow"])
a = ap.parse_args()
d = os.path.join(a.root, a.subject)

print("== 1) metadata")
import nibabel as nib
for name in a.series:
    p = glob.glob(os.path.join(d, f"*_{name}.nii*"))
    if not p:
        continue
    h = nib.load(p[0]).header
    txt = {k: h[k].tobytes().decode(errors="ignore").strip("\x00 ") for k in ("descrip", "aux_file", "intent_name", "db_name")
           if k in h}
    print(f"  {name}: {txt}")
    for e in nib.load(p[0]).header.extensions:
        print(f"     extension code {e.get_code()}: {str(e.get_content())[:300]}")
side = glob.glob(os.path.join(a.root, "**", "*.json"), recursive=True)[:20]
dcm = glob.glob(os.path.join(a.root, "**", "*.dcm"), recursive=True)[:5]
other = sorted({os.path.splitext(f)[1] for f in glob.glob(os.path.join(a.root, "**", "*"), recursive=True)
                if os.path.isfile(f)})
print(f"  JSON sidecars: {side or 'none'}")
print(f"  DICOM files:   {dcm or 'none'}")
print(f"  file types in the dataset: {other}")
for js in side[:6]:
    try:
        meta = json.load(open(js))
        keys = [k for k in meta if any(s in k.lower() for s in ("exposure", "current", "ctdi", "kvp", "kernel",
                                                                  "convolution", "recon", "dose", "mas"))]
        print(f"  {os.path.basename(js)}: " + ", ".join(f"{k}={meta[k]}" for k in keys))
    except Exception as e:
        print(f"  {js}: unreadable ({e})")

print("\n== 2) noise-equivalent dose (FBP assumption; upper bound with IR)")
import prepare_nifti as pn
pn.OPT.update(stride=4)
sig = {}
for name in a.series:
    p = pn.series_path(a.root, a.subject, name)
    if p is None:
        continue
    v, _, _ = pn.load_nifti(p)
    body = np.array([(s[::8, ::8] > -500).mean() for s in v])
    idx = np.where(body > 0.3)[0][::max(1, len(v) // 40)]
    vals = []
    for i in idx:
        x = v[i]
        m = ndimage.uniform_filter(x, 15)
        sd = np.sqrt(np.maximum(ndimage.uniform_filter(x ** 2, 15) - m ** 2, 0))
        msk = ndimage.binary_erosion((m > 0) & (m < 100), iterations=5)
        if msk.sum() > 500:
            vals.append(np.median(sd[msk]))           # local std in soft tissue (robust to edges)
    sig[name] = float(np.median(vals))
    print(f"  {name:9s} soft-tissue noise {sig[name]:.1f} HU")
ref = sig.get("normal")
if ref:
    for name, s in sig.items():
        if name != "normal":
            print(f"  {name:9s} noise-equivalent dose ~ {100 * (ref / s) ** 2:.0f} % of normal")
