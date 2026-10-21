"""Apply the low-frequency anchor to EXISTING predictions (no GPU, no re-sampling).

    python apply_lf.py --runs runs_v3 --model cadm_long --sigma 4
    -> writes runs_v3/cadm_long_lf_<src>_to_<tgt>_f0_s0/pred.npy for every dose found

out = pred - G_sigma(pred) + G_sigma(NDCT)   (see diffusion_common.lf_anchor)
"""
import argparse
import glob
import json
import os

import numpy as np

from datasets import load_pair
from diffusion_common import lf_anchor, write_pred_run

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_v3")
ap.add_argument("--model", required=True, help="method name, e.g. cadm_long")
ap.add_argument("--sigma", type=float, default=4.0, help="Gaussian sigma in pixels")
ap.add_argument("--suffix", default="_lf")
a = ap.parse_args()

found = 0
for cfg_path in sorted(glob.glob(os.path.join(a.runs, f"{a.model}_*", "config.json"))):
    cfg = json.load(open(cfg_path))
    if cfg.get("model") != a.model or "prediction_file" not in cfg:
        continue                      # e.g. skips cadm_long_lf when --model cadm_long
    rd = os.path.dirname(cfg_path)
    pred = np.load(os.path.join(rd, cfg["prediction_file"])).astype(np.float32)
    s, _, _, _ = load_pair(cfg["data"], cfg["src"], cfg["tgt"])
    nd = s[np.asarray(cfg["test_idx"])]
    assert len(nd) == len(pred), f"{rd}: {len(pred)} predictions vs {len(nd)} test slices"
    out = lf_anchor(pred, nd, cfg["hu_range"], a.sigma)
    write_pred_run(a.runs, a.model + a.suffix, cfg["src"], cfg["tgt"], cfg["fold"], cfg["n_folds"], cfg["seed"],
                   cfg["data"], cfg["train_idx"], cfg["test_idx"], out, cfg["hu_range"],
                   extra=dict(lf_anchor=a.sigma, from_run=rd))
    found += 1
print(f"done: {found} run(s)")
if not found:
    raise SystemExit(f"no pred.npy run named '{a.model}' in {a.runs}")