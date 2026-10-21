"""Data loading: aligned NDCT/LDCT slices, slab-wise splits, patch sampling."""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

HU_MIN, HU_MAX = -1024.0, 2048.0  # mapped linearly to [-1, 1]; change with set_hu_range


def set_hu_range(lo, hi):
    """Narrow ranges (e.g. -160 240) make the noise ~8x larger in network units,
    but everything outside the range is clipped (bone/air are lost)."""
    global HU_MIN, HU_MAX
    HU_MIN, HU_MAX = float(lo), float(hi)


def hu_to_net(x):
    return (np.clip(x, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN) * 2 - 1


def net_to_hu(x):
    return (x + 1) / 2 * (HU_MAX - HU_MIN) + HU_MIN


def load_pair(data_dir, src_key, tgt_key):
    """Return (src_vol, tgt_vol, z, spacing) restricted to z positions present in both."""
    with open(os.path.join(data_dir, "index.json")) as f:
        index = json.load(f)
    for k in (src_key, tgt_key):
        if k not in index:
            raise KeyError(f"series '{k}' not found. Available: {sorted(index)}")
    zs, zt = index[src_key]["z"], index[tgt_key]["z"]
    common = sorted(set(zs) & set(zt))
    if len(common) < 0.9 * min(len(zs), len(zt)):
        print(f"[warn] only {len(common)} common z positions between {src_key} and {tgt_key}")
    si = [zs.index(z) for z in common]
    ti = [zt.index(z) for z in common]
    src = np.load(os.path.join(data_dir, src_key + ".npy"), mmap_mode="r")
    tgt = np.load(os.path.join(data_dir, tgt_key + ".npy"), mmap_mode="r")
    return (np.asarray(src[si], np.float32), np.asarray(tgt[ti], np.float32),
            np.array(common), index[tgt_key]["spacing"])


def body_slices(vol, idx, min_frac=0.05):
    """Keep only slices where the body (> -500 HU) covers at least `min_frac` of the
    image: the first/last slices of the scan are empty air and would inflate the
    metrics (every method is 'perfect' on air) and waste training samples."""
    idx = np.asarray(idx)
    frac = np.array([(np.asarray(vol[i][::8, ::8]) > -500).mean() for i in idx])
    return idx[frac >= min_frac]


SUBJECT_Z = 1_000_000   # multi-subject datasets: z = (subject+1) * SUBJECT_Z + 500000 + z_mm*10


def subject_of(z):
    return (np.asarray(z) // SUBJECT_Z).astype(int)


def drop_subjects(idx, z, names, exclude):
    """Remove from `idx` the slices whose subject (encoded in z) is listed in `exclude`
    (folder names, e.g. ER01035833). Used to leave badly registered animals out of TRAINING
    only; the test split is never changed."""
    if not exclude:
        return idx
    if names is None or not len(z) or np.min(z) < SUBJECT_Z:
        raise SystemExit("--exclude_subjects needs a multi-subject dataset (prepare_nifti.py)")
    unknown = set(exclude) - set(names)
    if unknown:
        raise SystemExit(f"unknown subject(s) {sorted(unknown)}; available: {names}")
    subj = subject_of(np.asarray(z)[idx]) - 1
    keep = np.array([names[s] not in exclude for s in subj], bool)
    return np.asarray(idx)[keep]


def slab_split(n, n_folds, fold, gap, z=None):
    """Single volume: contiguous z-slabs. Test = slab `fold`; `gap` slices on each side are
    dropped from training so neighbouring (highly correlated) slices don't leak.
    Multi-subject data (z encodes the subject, see prepare_nifti.py): SUBJECT-WISE split,
    the subjects are dealt into n_folds groups and the whole subject goes to train or test."""
    if z is not None and len(z) and np.min(z) >= SUBJECT_Z:
        subj = subject_of(z)
        groups = np.array_split(np.unique(subj), n_folds)
        test_s = set(groups[fold].tolist())
        test = np.array([i for i in range(n) if subj[i] in test_s])
        train = np.array([i for i in range(n) if subj[i] not in test_s])
        return train, test
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    lo, hi = edges[fold], edges[fold + 1]
    test = np.arange(lo, hi)
    train = np.array([i for i in range(n) if i < lo - gap or i >= hi + gap])
    return train, test


class PatchDataset(Dataset):
    """Yields (A=NDCT patch, B=LDCT patch).

    paired=True  -> B is the co-located LDCT crop (used by the cVAE).
    paired=False -> B comes from a random other slice/location (CycleGAN/CUT),
                    so the unpaired methods really are trained without correspondence.
    """

    def __init__(self, src, tgt, idx, patch=128, paired=False, length=10000, fg_thresh=-500):
        self.src = hu_to_net(src[idx]).astype(np.float32)
        self.tgt = hu_to_net(tgt[idx]).astype(np.float32)
        self.patch, self.paired, self.length = patch, paired, length
        # foreground centres (body), so we don't train on air
        fg = src[idx] > fg_thresh
        self.fg = [np.argwhere(m) for m in fg]

    def __len__(self):
        return self.length

    def _crop(self, vol, rng):
        s = rng.integers(len(vol))
        y, x = self._centre(s, vol.shape[1:], rng)
        return s, y, x

    def _centre(self, s, shape, rng):
        p = self.patch
        H, W = shape
        pts = self.fg[s]
        if len(pts) and rng.random() < 0.9:
            cy, cx = pts[rng.integers(len(pts))]
        else:
            cy, cx = rng.integers(H), rng.integers(W)
        y = int(np.clip(cy - p // 2, 0, H - p))
        x = int(np.clip(cx - p // 2, 0, W - p))
        return y, x

    def __getitem__(self, i):
        rng = np.random.default_rng()
        p = self.patch
        s, y, x = self._crop(self.src, rng)
        a = self.src[s, y:y + p, x:x + p]
        # only left-right flips: rotations would alter the orientation of
        # CT noise/streak texture, which is exactly what we want to learn.
        if self.paired:
            b = self.tgt[s, y:y + p, x:x + p]
            if rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
        else:
            s2, y2, x2 = self._crop(self.tgt, rng)
            b = self.tgt[s2, y2:y2 + p, x2:x2 + p]
            if rng.random() < 0.5:
                a = a[:, ::-1]
            if rng.random() < 0.5:
                b = b[:, ::-1]
        return (torch.from_numpy(a.copy())[None], torch.from_numpy(b.copy())[None])
