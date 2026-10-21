"""Ablation study of PhysHVS-LDCT (CADM).

Each configuration is trained (one dose-controllable model for all --doses), sampled on
the test slices of every dose, and scored with exactly the same evaluation as the
benchmark. Finished steps are skipped, so the script can be stopped and restarted.

    python run_ablation.py --data data/piglet --recon FBP --doses 5 10 25 \
        --sd_path ~/models/sd15 --iters 3000 --steps 25 --runs runs_v3
    python run_ablation.py ... --configs full no_pm no_parm      # a subset

Configurations (method name in the table -> what changes vs the full model):
    cadm            full model: NDCT + PARM(+dose) + perceptual map, lambda_n=1, lambda_c=0.5
    abl_no_pm       no perceptual branch                              (value of the HVS prior)
    abl_no_parm     no physics branch; dose code moved to the NDCT branch (value of the physics prior)
    abl_ndct_only   NDCT branch + dose code only                      (both priors removed)
    abl_no_dose     full model without the explicit log-dose map      (does PARM alone carry the dose?)
    abl_parm_noise  PARM = FBP(noisy) - FBP(clean) instead of Eq. 3.23 (prior definition)
    abl_lc_0        lambda_c = 0 at the same architecture              (fusion weight)
    abl_lc_1        lambda_c = 1                                       (fusion weight)
    abl_per_dose    full model, but one model trained per dose        (value of dose-controllable training)

Output: <results>/ablation_table.csv (mean per config and dose) and
        <results>/ablation_delta.csv (difference to the full model; sign so that > 0 = better).
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

CONFIGS = {
    "full":       dict(name="cadm",           args=[]),
    "no_pm":      dict(name="abl_no_pm",      args=["--branches", "s", "n"]),
    "no_parm":    dict(name="abl_no_parm",    args=["--branches", "s", "c"]),
    "ndct_only":  dict(name="abl_ndct_only",  args=["--branches", "s"]),
    "no_dose":    dict(name="abl_no_dose",    args=["--dose_channel", "none"]),
    "parm_noise": dict(name="abl_parm_noise", args=["--parm_mode", "noise"]),
    "lc_0":       dict(name="abl_lc_0",       args=["--lambda_c", "0"]),
    "lc_1":       dict(name="abl_lc_1",       args=["--lambda_c", "1"]),
    "per_dose":   dict(name="abl_per_dose",   args=[], per_dose=True),
}
HIGHER_BETTER = {"psnr", "ssim", "psnr_01", "ssim_01", "lf_psnr"}
KEY = ["rmse", "psnr", "ssim", "lpips", "dists", "nps_mae", "nps_l2", "df_peak",
       "rmse_01", "psnr_01", "ssim_01", "noise_std_err", "abs_hu_bias"]


def sh(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--recon", default="FBP")
    ap.add_argument("--doses", type=int, nargs="+", default=[5, 10, 25])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--sd_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--runs", default="runs_v3")
    ap.add_argument("--results", default=None, help="default: <runs>_ablation -> results_<...>")
    ap.add_argument("--ckpt_dir", default="cadm_runs")
    ap.add_argument("--suffix", default="", help="e.g. a3k")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra arguments passed to every train_cadm.py train call")
    a = ap.parse_args()
    results = a.results or a.runs.replace("runs", "results", 1) + "_ablation" + (f"_{a.suffix}" if a.suffix else "")
    sfx = f"_{a.suffix}" if a.suffix else ""
    name_of = lambda k: CONFIGS[k]["name"] + sfx
    py = sys.executable
    common = ["--data", a.data, "--recon", a.recon, "--fold", str(a.fold), "--n_folds", str(a.n_folds)]
    src = f"100_{a.recon}"

    for key in a.configs:
        cfg = CONFIGS[key]
        groups = [[d] for d in a.doses] if cfg.get("per_dose") else [a.doses]
        for doses in groups:
            todo = [d for d in doses if not os.path.exists(
                os.path.join(a.runs, f"{name_of(key)}_{src}_to_{d}_{a.recon}_f{a.fold}_s0", "pred.npy"))]
            if not todo:
                print(f"[done] {name_of(key)} {doses}")
                continue
            tag = "" if key == "full" else key + ("_" + "-".join(map(str, doses)) if cfg.get("per_dose") else "")
            tag = "_".join(x for x in (tag, a.suffix) if x)
            branches = cfg["args"][cfg["args"].index("--branches") + 1:] if "--branches" in cfg["args"] else ["s", "n", "c"]
            branches = [b for b in branches if b in ("s", "n", "c")]
            ckpt = os.path.join(a.ckpt_dir, f"cadm_{a.recon}_{'-'.join(branches)}_f{a.fold}"
                                + (f"_{tag}" if tag else ""), "final")
            if not os.path.exists(os.path.join(ckpt, "lora.pt")):
                sh([py, "train_cadm.py", "train", *common, "--doses", *map(str, doses), "--sd_path", a.sd_path,
                    "--iters", str(a.iters), "--out", a.ckpt_dir, *(["--tag", tag] if tag else []),
                    *cfg["args"], *a.extra])
            sh([py, "train_cadm.py", "sample", "--ckpt", ckpt, "--doses", *map(str, todo), "--steps", str(a.steps),
                "--runs", a.runs, "--name", name_of(key)])

    names = [name_of(k) for k in a.configs]
    sh([py, "run_benchmark.py", "--data", a.data, "--recon", a.recon, "--doses", *map(str, a.doses),
        "--folds", str(a.fold), "--n_folds", str(a.n_folds), "--runs", a.runs, "--results", results,
        "--models", *names, "--baselines"])
    summarize(results, names, ref_name=name_of("full"))


def summarize(results, names, ref_name="cadm"):
    t = pd.read_csv(os.path.join(results, "benchmark_table.csv"))
    t = t[t["method"].isin(names)]
    for k in KEY:   # strings "m ± s" when several folds -> keep the mean
        if k in t and t[k].dtype == object:
            t[k] = t[k].str.split(" ").str[0].astype(float)
    keep = ["recon", "dose", "method"] + [k for k in KEY if k in t]
    t = t[keep].sort_values(["dose", "method"])
    t.to_csv(os.path.join(results, "ablation_table.csv"), index=False)
    if ref_name in set(t["method"]):
        ref = t[t["method"] == ref_name].set_index("dose")
        rows = []
        for _, r in t[t["method"] != ref_name].iterrows():
            row = {"dose": r["dose"], "method": r["method"]}
            for k in keep[3:]:
                diff = r[k] - ref.loc[r["dose"], k]
                row[k] = diff if k in HIGHER_BETTER else -diff      # > 0 : ablation is better than full
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(results, "ablation_delta.csv"), index=False)
    print("\n================ ABLATION ================")
    print(t.round(4).to_string(index=False))
    print(f"\nsaved {results}/ablation_table.csv and ablation_delta.csv (> 0 = better than the full model)")


if __name__ == "__main__":
    main()
