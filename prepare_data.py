"""
Convert the piglet DICOM tree into aligned HU volumes.

Expected layout (names are parsed, so small inconsistencies are fine):
    piglet_sorted/
        5%/   5% DOSE 20% ASIR/IM0 IM1 ...
        10%/  10% DOSE FBP/...
        25%/  25 DOSE FBP/  25% DOSE FBP/ ...   (duplicates are detected)
        100%/ FULL DOSE FBP/ ...
        EXTRA/  (skipped)

Output:
    <out>/<dose>_<recon>.npy   int16 HU volume, shape (Z, H, W), sorted by z
    <out>/index.json           z positions, pixel spacing, slice count per series

Usage:
    python prepare_data.py --root "E:/piglet_sorted/piglet_sorted" --out data/piglet
"""
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict

import numpy as np
import pydicom


def parse_series_name(name):
    """'5% DOSE 20% ASIR' -> (5, 'ASIR20'); 'FULL DOSE FBP' -> (100, 'FBP')."""
    n = name.upper().strip()
    if n.startswith("FULL"):
        dose = 100
    else:
        m = re.match(r"^\s*(\d+)\s*%?\s*DOSE", n)
        if not m:
            return None
        dose = int(m.group(1))
    m = re.search(r"(\d+)\s*%?\s*ASIR", n)
    if m:
        recon = f"ASIR{int(m.group(1))}"
    elif "MBIR" in n or "VEO" in n:
        recon = "MBIR"
    elif "FBP" in n:
        recon = "FBP"
    else:
        return None
    return dose, recon


def read_series(folder):
    slices = []
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if not os.path.isfile(p) or f.upper() == "DICOMDIR":
            continue
        try:
            ds = pydicom.dcmread(p, force=True)
            arr = ds.pixel_array
        except Exception:
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", -1024.0))
        hu = arr.astype(np.float32) * slope + inter
        if hasattr(ds, "ImagePositionPatient"):
            z = float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            z = float(ds.SliceLocation)
        else:
            z = float(getattr(ds, "InstanceNumber", len(slices)))
        sp = [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0])]
        thick = float(getattr(ds, "SliceThickness", 0) or 0)
        slices.append((z, hu, sp, thick))
    if not slices:
        return None
    slices.sort(key=lambda s: s[0])
    vol = np.stack([s[1] for s in slices]).clip(-32768, 32767).astype(np.int16)
    return dict(vol=vol, z=[round(s[0], 3) for s in slices],
                spacing=slices[0][2], thickness=slices[0][3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/piglet")
    ap.add_argument("--map", nargs="*", default=[], metavar="FOLDER=DOSE_RECON",
                    help='assign a folder explicitly, overriding name parsing, e.g. '
                         '--map "EXTRA=5_FBP" "25% DOSE FBP=25_FBP". FOLDER may be a dose '
                         'folder (all files directly inside) or a series sub-folder.')
    args = ap.parse_args()
    manual = {}
    for m in args.map:
        folder, key = m.rsplit("=", 1)
        d, r = key.split("_", 1)
        manual[folder.strip().upper()] = (int(d), r.upper())
    os.makedirs(args.out, exist_ok=True)

    found = defaultdict(list)
    forced = set()  # (dose, recon) slots filled through --map: they win over parsed folders
    for dose_dir in sorted(os.listdir(args.root)):
        dpath = os.path.join(args.root, dose_dir)
        if not os.path.isdir(dpath):
            continue
        if dose_dir.upper() in manual:  # whole dose-level folder mapped (e.g. EXTRA holds files)
            found[manual[dose_dir.upper()]].append(dpath)
            forced.add(manual[dose_dir.upper()])
            continue
        for s in sorted(os.listdir(dpath)):
            spath = os.path.join(dpath, s)
            if not os.path.isdir(spath):
                continue
            if s.upper() in manual:
                found[manual[s.upper()]].append(spath)
                forced.add(manual[s.upper()])
                continue
            parsed = parse_series_name(s)
            if parsed is None:
                print(f"[skip] unrecognised series name: {dose_dir}/{s}")
                continue
            found[parsed].append(spath)

    # a --map assignment replaces any automatically parsed folder for that slot
    for slot in forced:
        keep = [p for p in found[slot] if os.path.basename(p).upper() in manual
                and manual[os.path.basename(p).upper()] == slot]
        for p in found[slot]:
            if p not in keep:
                print(f"[map] {p} ignored: slot {slot[0]}_{slot[1]} assigned via --map")
        found[slot] = keep

    index = {}
    for (dose, recon), paths in sorted(found.items()):
        loaded = []
        for p in paths:
            r = read_series(p)
            if r is None:
                print(f"[warn] no readable DICOM in {p}")
                continue
            r["path"] = p
            r["md5"] = hashlib.md5(r["vol"].tobytes()).hexdigest()
            loaded.append(r)
        # de-duplicate (e.g. '25 DOSE FBP' and '25% DOSE FBP')
        uniq = []
        for r in loaded:
            if any(r["md5"] == u["md5"] for u in uniq):
                print(f"[dedup] {r['path']} identical to an earlier folder, ignored")
                continue
            uniq.append(r)
        if len(uniq) > 1:
            print(f"[warn] {len(uniq)} DIFFERENT series for dose={dose} recon={recon}:")
            for i, r in enumerate(uniq):
                print(f"        [{i}] {r['path']}  ({len(r['z'])} slices)")
            print("        keeping the one with most slices as main; others saved as _alt*")
            uniq.sort(key=lambda r: -len(r["z"]))
        for i, r in enumerate(uniq):
            key = f"{dose}_{recon}" + ("" if i == 0 else f"_alt{i}")
            np.save(os.path.join(args.out, key + ".npy"), r["vol"])
            index[key] = dict(dose=dose, recon=recon, path=r["path"], z=r["z"],
                              spacing=r["spacing"], thickness=r["thickness"],
                              shape=list(r["vol"].shape))
            print(f"[ok] {key:14s} {r['vol'].shape}  z {r['z'][0]:.1f}..{r['z'][-1]:.1f}  "
                  f"px {r['spacing'][0]:.3f}mm  from {r['path']}")

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f)

    # alignment sanity check against full dose
    for ref in [k for k in index if k.startswith("100_") and "alt" not in k]:
        recon = index[ref]["recon"]
        zr = set(index[ref]["z"])
        for k, v in index.items():
            if v["recon"] != recon or k == ref:
                continue
            common = len(zr & set(v["z"]))
            print(f"[align] {k:14s} vs {ref}: {common}/{len(v['z'])} slices share a z position")
    print(f"\nWrote {len(index)} series to {args.out}")


if __name__ == "__main__":
    main()
