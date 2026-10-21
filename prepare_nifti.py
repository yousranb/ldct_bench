"""Convert a multi-subject NIfTI dataset (one folder per animal/patient) into the format
used by the benchmark (<out>/<dose>_<recon>.npy + index.json).

Expected layout (e.g. CT_norsvin_data_segmented):
    <root>/<SUBJECT>/<SUBJECT>_normal.nii.gz
    <root>/<SUBJECT>/<SUBJECT>_low.nii.gz
    <root>/<SUBJECT>/<SUBJECT>_ultralow.nii.gz
    (mask files are ignored)

All subjects are stacked along z. The z key of every slice encodes the subject:
    z = (subject_index + 1) * 1_000_000 + 500_000 + round(z_mm * 10)
so that (1) the dose levels are matched slice by slice on their physical position and
(2) datasets.slab_split() makes SUBJECT-WISE folds (a subject is never in train and test).

    # 1) look at the data first (no files written)
    python prepare_nifti.py --root ~/CT_norsvin_data_segmented --inspect
    # 2) convert (give the real dose of each series in % of the normal dose)
    python prepare_nifti.py --root ~/CT_norsvin_data_segmented --out data/norsvin \\
        --series normal=100 low=25 ultralow=10 --recon FBP
"""
import argparse
import glob
import json
import os

import numpy as np

SUBJECT_Z = 1_000_000


OPT = dict(slice_axis=None, hu_offset="auto", stride=1)   # set from the command line


def load_nifti(path, verbose=False):
    """Returns (vol (n_slices, rows, cols) in HU, slice positions in mm, [row_mm, col_mm, slice_mm])."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))          # RAS+ orientation
    vol = np.asarray(img.dataobj, dtype=np.float32)        # applies scl_slope / scl_inter
    if vol.ndim == 4:
        vol = vol[..., 0]
    aff = img.affine
    zooms = [float(v) for v in img.header.get_zooms()[:3]]
    # slice (acquisition) axis: the one with the largest voxel size unless given explicitly
    ax = OPT["slice_axis"] if OPT["slice_axis"] is not None else int(np.argmax(zooms))
    pos = []
    for k in range(vol.shape[ax]):
        v = np.zeros(4); v[ax] = k; v[3] = 1
        pos.append(float((aff @ v)[ax]))
    pos = np.array(pos)
    plane = [i for i in range(3) if i != ax]                 # remaining two axes, in RAS order
    vol = np.moveaxis(vol, ax, 0)                            # (slices, plane[0], plane[1])
    # display orientation: second plane axis upwards, radiological left/right
    vol = np.ascontiguousarray(np.flip(np.rot90(vol, axes=(1, 2)), axis=2))
    sp = [zooms[plane[1]], zooms[plane[0]], zooms[ax]]
    # HU offset: many exports store CT numbers as HU + 1024 (air = 0, water = 1024)
    off = OPT["hu_offset"]
    if off == "auto":
        # real HU always contains air at about -1000; HU+1024 storage has air at about 0
        p1 = float(np.percentile(vol[::4, ::4, ::4], 1))
        off = -1024.0 if p1 > -200 else 0.0
        if verbose and off:
            print(f"   [auto] air at {p1:.0f} -> values stored as HU+1024: subtracting 1024")
    vol += float(off)
    if OPT["stride"] > 1:
        vol, pos = vol[::OPT["stride"]], pos[::OPT["stride"]]
    if verbose:
        print(f"   slice axis {ax} (voxel {np.round(zooms, 3).tolist()} mm) -> {vol.shape[0]} slices of {vol.shape[1]}x{vol.shape[2]}")
    return vol, pos, sp


def series_path(root, subj, name):
    p = os.path.join(root, subj, f"{subj}_{name}.nii.gz")
    if os.path.exists(p):
        return p
    alt = glob.glob(os.path.join(root, subj, f"*_{name}.nii*"))
    return alt[0] if alt else None


def resize_slices(vol, size):
    if vol.shape[1] == size and vol.shape[2] == size:
        return vol
    from scipy import ndimage
    f = (1, size / vol.shape[1], size / vol.shape[2])
    return ndimage.zoom(vol, f, order=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/norsvin")
    ap.add_argument("--series", nargs="+", default=["normal=100", "low=50", "ultralow=10"],
                    help="file suffix = dose in %% of the normal dose (the normal dose must be 100)")
    ap.add_argument("--recon", default="FBP", help="reconstruction label used in the series keys")
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of subject folders (default: all)")
    ap.add_argument("--size", type=int, default=512, help="in-plane size (resampled if different; 0 = keep)")
    ap.add_argument("--inspect", action="store_true", help="only print shapes, spacing, HU range, alignment")
    ap.add_argument("--slice_axis", type=int, default=None, choices=[0, 1, 2],
                    help="axis (after RAS reorientation) along which slices are taken; default: largest voxel size")
    ap.add_argument("--hu_offset", default="auto", help="added to the stored values; 'auto' detects HU+1024 storage")
    ap.add_argument("--stride", type=int, default=1, help="keep every k-th slice (saves disk/RAM)")
    ap.add_argument("--register", action="store_true",
                    help="deformable registration of each low-dose series onto the normal-dose one "
                         "(B-spline, nearest-neighbour resampling keeps the real LDCT noise)")
    ap.add_argument("--reg_mesh_mm", type=float, default=50.0, help="B-spline control-point spacing (mm)")
    a = ap.parse_args()

    OPT.update(slice_axis=a.slice_axis, stride=a.stride,
               hu_offset=a.hu_offset if a.hu_offset == "auto" else float(a.hu_offset))
    series = dict(s.split("=") for s in a.series)
    series = {k: int(v) for k, v in series.items()}
    if 100 not in series.values():
        raise SystemExit("one series must be the full dose (=100)")
    subjects = a.subjects or sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    print(f"{len(subjects)} subjects: {subjects}")
    print(f"series: {series}")

    # ---------------- pass 1: headers / inspection
    info = {}
    for si, subj in enumerate(subjects):
        info[subj] = {}
        for name in series:
            p = series_path(a.root, subj, name)
            if p is None:
                print(f"[missing] {subj}: no *_{name}.nii.gz")
                continue
            info[subj][name] = p
        if not a.inspect:
            continue
        vols = {}
        for name, p in info[subj].items():
            v, z, sp = load_nifti(p, verbose=True)
            vols[name] = (v, z)
            c = v[len(v) // 2]
            print(f"{subj:12s} {name:9s} shape {v.shape}  spacing {np.round(sp, 3).tolist()}  "
                  f"z {z.min():.1f}..{z.max():.1f} mm  HU[min/med/max] {v.min():.0f}/{np.median(c):.0f}/{v.max():.0f}")
        full = [n for n in vols if series[n] == 100]
        if full:
            vf, zf = vols[full[0]]
            for name, (v, z) in vols.items():
                if name == full[0]:
                    continue
                common = sorted(set(np.round(zf, 1)) & set(np.round(z, 1)))
                if not common or v.shape[1:] != vf.shape[1:]:
                    print(f"   !! {name}: {len(common)} common z positions, in-plane {v.shape[1:]} vs {vf.shape[1:]}")
                    continue
                zm = common[len(common) // 2]
                i, j = int(np.argmin(abs(zf - zm))), int(np.argmin(abs(z - zm)))
                body = vf[i] > -500
                d = (v[j] - vf[i])[body]
                print(f"   {name} vs {full[0]}: {len(common)} common slices, central slice: mean diff "
                      f"{d.mean():+.1f} HU, std diff {d.std():.1f} HU (std ~ noise level is OK; a large mean or a std >> noise = misaligned)")
                if a.register and v.shape == vf.shape:
                    from register import register_to_reference
                    print(f"   registering {name} -> {full[0]} ...")
                    register_to_reference(vf, v, (sp[2], sp[0], sp[1]), mesh_mm=a.reg_mesh_mm)
    if a.inspect:
        return

    # ---------------- pass 2: convert (subject by subject, so the low-dose series can be
    # registered onto the normal-dose volume of the same animal)
    os.makedirs(a.out, exist_ok=True)
    full = [n for n in series if series[n] == 100][0]
    store = {n: dict(parts=[], zs=[], sp=[], reg={}) for n in series}
    for si, subj in enumerate(subjects):
        if full not in info[subj]:
            print(f"[skip] {subj}: no normal-dose series")
            continue
        ref, z_ref, sp_ref = load_nifti(info[subj][full])
        for name in series:
            if name not in info[subj]:
                continue
            if name == full:
                v, z_mm, sp = ref, z_ref, sp_ref
            else:
                v, z_mm, sp = load_nifti(info[subj][name])
                if a.register:
                    if v.shape != ref.shape:
                        raise SystemExit(f"{subj} {name}: shape {v.shape} != normal {ref.shape}, cannot register")
                    from register import register_to_reference
                    print(f"[{subj}] registering {name} -> {full}")
                    v, rep = register_to_reference(ref, v, (sp_ref[2], sp_ref[0], sp_ref[1]), mesh_mm=a.reg_mesh_mm)
                    z_mm, sp = z_ref, sp_ref
                    store[name]["reg"][subj] = rep
            if a.size:
                if v.shape[1] != a.size:
                    sp = [sp[0] * v.shape[1] / a.size, sp[1] * v.shape[2] / a.size, sp[2]]
                v = resize_slices(v, a.size)
            store[name]["parts"].append(np.clip(np.rint(v), -1024, 3071).astype(np.int16))
            store[name]["zs"] += [int((si + 1) * SUBJECT_Z + 500_000 + round(zm * 10)) for zm in z_mm]
            store[name]["sp"].append(sp)
            print(f"[{name}] {subj}: {v.shape[0]} slices, spacing {np.round(sp, 3).tolist()}")
        del ref
    index = {}
    for name, dose in series.items():
        key = f"{dose}_{a.recon}"
        st = store[name]
        if not st["parts"]:
            continue
        vol = np.concatenate(st["parts"])
        zs = st["zs"]
        order = np.argsort(zs, kind="stable")
        vol, zs = vol[order], [zs[i] for i in order]
        spacings = np.array(st["sp"])
        if spacings[:, 0].max() / spacings[:, 0].min() > 1.05:
            print(f"[warn] in-plane spacing varies across subjects: {spacings[:, 0].min():.3f}-"
                  f"{spacings[:, 0].max():.3f} mm (median used for the physics priors)")
        np.save(os.path.join(a.out, key + ".npy"), vol)
        index[key] = dict(dose=dose, recon=a.recon, z=zs, spacing=np.median(spacings, 0).tolist(),
                          subjects=subjects, source=a.root, series=name,
                          registered=bool(a.register and name != full), registration=st["reg"])
        print(f"[saved] {a.out}/{key}.npy {vol.shape} int16")
        del vol
        st["parts"] = []
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"[saved] {a.out}/index.json  keys: {sorted(index)}")


if __name__ == "__main__":
    main()
