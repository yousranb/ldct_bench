"""Full benchmark: every model x every target dose x every fold, then evaluate and aggregate.

Examples
    # default = FBP, target doses 5/10/25 %, fold 0. Fast first pass:
    python run_benchmark.py --data data/piglet --quick
    # full run with 3-fold slab cross-validation
    python run_benchmark.py --data data/piglet --folds 0 1 2
    # several reconstructions in one go (each: 100% recon X -> low-dose recon X)
    python run_benchmark.py --data data/piglet --recon FBP ASIR20 ASIR40 ASIR60
    # cross-reconstruction: FBP full dose -> ASIR40 low dose
    python run_benchmark.py --data data/piglet --src_recon FBP --recon ASIR40

Finished runs (model.pt present) are skipped, so the script can be restarted.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import evaluate
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--recon", nargs="+", default=["FBP"],
                    help="target reconstruction(s): FBP ASIR20 ASIR40 ASIR60 MBIR (several = one after the other)")
    ap.add_argument("--src_recon", default=None,
                    help="reconstruction of the full-dose input (default: same as the target). "
                         "e.g. --src_recon FBP --recon ASIR40 learns FBP full dose -> ASIR40 low dose")
    ap.add_argument("--src_dose", type=int, default=100)
    ap.add_argument("--doses", type=int, nargs="*", default=[5, 10, 25],
                    help="target doses (default 5 10 25; pass --doses with no value for all available)")
    ap.add_argument("--models", nargs="+", default=["cyclegan", "cut", "vae"],
                    help="also: cadm, cddpm, cadm_sn ... (diffusion predictions must exist in --runs)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--baselines", nargs="*", default=["identity", "gaussian", "zeng"],
                    choices=["identity", "gaussian", "zeng", "zeng_nominal"],
                    help="zeng = I0 calibrated on training slices; zeng_nominal = fixed I0 (--zeng_I0), no calibration")
    ap.add_argument("--zeng_I0", type=float, default=1e7,
                    help="full-dose photons per ray for zeng_nominal (default 1e7 = thesis Eq. 3.19 / CADM physics prior)")
    ap.add_argument("--zeng_sigma_e2", type=float, default=10.0, help="electronic noise variance for zeng_nominal")
    ap.add_argument("--n_examples", type=int, default=8, help="test slices shown in the figures")
    ap.add_argument("--d_hp", type=float, default=30.0)
    ap.add_argument("--beta", type=float, default=1e-2)
    ap.add_argument("--tag", default="", help="suffix for the runs/results folders, e.g. v2")
    ap.add_argument("--hu_range", type=float, nargs=2, default=[-1024.0, 2048.0],
                    help="HU range the networks see, e.g. -160 240 to make noise more visible")
    ap.add_argument("--unit_range", type=float, nargs=2, default=None,
                    help="HU range mapped to [0,1] for the *_01 metrics (default: common range of the methods)")
    ap.add_argument("--win", type=float, nargs=2, default=[-100.0, 200.0],
                    help="HU window for PSNR/SSIM/LPIPS (e.g. -1024 2048 for full range)")
    a = ap.parse_args()
    if a.tag:
        a.runs, a.results = f"{a.runs}_{a.tag}", f"{a.results}_{a.tag}"

    index = json.load(open(os.path.join(a.data, "index.json")))
    for recon in a.recon:
        run_recon(a, index, recon)
    aggregate(a.results)


def run_recon(a, index, recon):
    src = f"{a.src_dose}_{a.src_recon or recon}"
    if src not in index:
        print(f"[skip] source {src} not in index.json")
        return
    doses = a.doses or sorted({v["dose"] for k, v in index.items()
                               if v["recon"] == recon and v["dose"] != a.src_dose and "alt" not in k})
    print(f"source {src}; targets {[f'{d}_{recon}' for d in doses]}; models {a.models}; folds {a.folds}")
    for d in doses:
        if f"{d}_{recon}" not in index:
            print(f"[skip] {d}_{recon} not available. If the series exists under another folder name "
                  f"(e.g. EXTRA), re-run prepare_data.py with --map \"<folder>={d}_{recon}\"")
    doses = [d for d in doses if f"{d}_{recon}" in index]

    for d in doses:
        tgt = f"{d}_{recon}"
        for fold in a.folds:
            run_dirs = []
            for m in a.models:
                if m not in ("cyclegan", "cut", "vae", "vaegan"):
                    # diffusion models: trained/sampled by train_cadm.py / train_cddpm.py
                    rd = os.path.join(a.runs, f"{m}_{src}_to_{tgt}_f{fold}_s0")
                    if os.path.exists(os.path.join(rd, "pred.npy")):
                        run_dirs.append(rd)
                    else:
                        print(f"[skip] {rd}/pred.npy not found (train/sample it first: train_cadm.py, train_cddpm.py "
                              f"or run_ablation.py)")
                    continue
                for s in a.seeds:
                    argv = ["--data", a.data, "--model", m, "--src", src, "--tgt", tgt, "--fold", str(fold),
                            "--n_folds", str(a.n_folds), "--seed", str(s), "--batch", str(a.batch), "--patch", str(a.patch),
                            "--workers", str(a.workers), "--out", a.runs]
                    argv += ["--d_hp", str(a.d_hp), "--beta", str(a.beta), "--hu_range", *map(str, a.hu_range)]
                    if a.quick:
                        argv.append("--quick")
                    if a.iters:
                        argv += ["--iters", str(a.iters)]
                    rd = os.path.join(a.runs, train.run_name(train.get_args(argv)))
                    if not os.path.exists(os.path.join(rd, "model.pt")):
                        print(f"\n===== train {m} {src} -> {tgt} fold {fold} seed {s} =====")
                        train.main(argv)
                    run_dirs.append(rd)
            print(f"\n===== evaluate {src} -> {tgt} fold {fold} =====")
            evaluate.main(["--runs", *run_dirs, "--out", a.results, "--win", *map(str, a.win),
                           "--n_examples", str(a.n_examples), "--baselines", *a.baselines,
                           "--zeng_I0", str(a.zeng_I0), "--zeng_sigma_e2", str(a.zeng_sigma_e2)]
                          + (["--unit_range", *map(str, a.unit_range)] if a.unit_range else []))



def aggregate(results):
    files = glob.glob(os.path.join(results, "*", "summary.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dose"] = df["tgt"].str.split("_").str[0].astype(int)
    df["recon"] = df["tgt"].str.split("_", n=1).str[1]
    src_r = df["src"].str.split("_", n=1).str[1]
    df.loc[src_r != df["recon"], "recon"] = src_r + "->" + df["recon"]     # cross-reconstruction runs
    keys = ["rmse", "psnr", "ssim", "lpips", "dists", "nps_mae", "nps_l2", "df_peak",
            "rmse_01", "psnr_01", "ssim_01", "lpips_01", "dists_01", "nps_mae_01", "nps_l2_01", "nps_mae_rel",
            "noise_std_err", "nps_shape_err", "f_mean_err", "abs_hu_bias", "lf_psnr",
            "nps_mae_abs", "nps_l2_abs", "nps_mae_rel_abs", "df_peak_abs", "nps_shape_err_abs", "f_mean_err_abs",
            "noise_std_err_abs", "noise_map_corr", "noise_map_err", "dprime_err_5mm", "dprime_err_10mm"]
    keys = [k for k in keys if k in df.columns]          # older summaries lack the *_abs columns
    g = df.groupby(["recon", "dose", "method"])[keys]
    table = g.mean().round(4)
    if df["fold"].nunique() > 1:
        std = g.std().round(4)
        table = table.astype(str) + " ± " + std.astype(str)
    table.to_csv(os.path.join(results, "benchmark_table.csv"))
    print("\n================ BENCHMARK (mean over folds) ================")
    print(table.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = df.groupby(["recon", "dose", "method"])[keys].mean().reset_index()
    for recon, sub in m.groupby("recon"):
        panels = ["rmse", "psnr", "ssim", "lpips", "dists", "nps_mae", "nps_l2", "df_peak"]
        up = {"psnr", "ssim", "lf_psnr", "psnr_01", "ssim_01"}
        fig, ax = plt.subplots(2, 4, figsize=(16, 7))
        ax = ax.ravel()
        for meth, s in sub.groupby("method"):
            s = s.sort_values("dose")
            ls = "--" if meth in ("identity", "gaussian", "zeng", "zeng_nominal") else "-"
            for j, k in enumerate(panels):
                ax[j].plot(s["dose"], s[k], marker="o", ls=ls, label=meth)
                ax[j].set(title=k + (" ↑" if k in up else " ↓"), xlabel="target dose (%)")
        for x in ax:
            x.set_xscale("log")
            x.set_xticks(sorted(sub["dose"].unique()))
            x.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[0].legend(fontsize=8)
        fig.suptitle(f"LDCT synthesis benchmark — {recon}")
        fig.tight_layout()
        fig.savefig(os.path.join(results, f"benchmark_{recon.replace('->', '_to_')}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
