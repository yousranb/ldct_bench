"""Evaluate trained runs (+ baselines) for one source->target pair and one fold.

All methods are scored on the SAME held-out slab of full 512x512 slices.

Why not just PSNR/SSIM?  A synthetic LDCT is a *different noise realisation* from
the real one, so pixel-wise metrics against the real LDCT mostly reward smooth
outputs. The primary metrics here therefore compare noise *statistics*, using
the noise image  n = image - NDCT  inside flat soft tissue:

  noise_std_err   |std(n_gen) - std(n_real)| / std(n_real)        (magnitude)
  nps_shape_err   L1 distance of area-normalised radial NPS curves (texture)
  f_mean_err      |mean NPS frequency gen - real| (cycles/mm)      (grain size)
  hu_bias         mean(gen - real LDCT) in soft tissue (HU)        (CT numbers)
  lf_psnr         PSNR after Gaussian blur (sigma=3 px)            (anatomy kept,
                  noise-agnostic)
  nps_mae         mean |NPS_gen(f) - NPS_real(f)| over radial bins (HU^2 mm^2)
  nps_l2          Euclidean norm ||NPS_gen - NPS_real||_2           (HU^2 mm^2)
  df_peak         |f_peak gen - f_peak real| (cycles/mm), peak located with
                  sub-bin parabolic interpolation
  rmse            RMSE in HU vs real LDCT (whole image)
  psnr / ssim     vs real LDCT, window --win (default [-100, 200] HU = level 50, width 300)
  lpips           LPIPS-AlexNet on the windowed image (needs `pip install lpips`)
                  -- pixel/perceptual metrics vs a *different* noise realisation:
                  interpret with care

NPS-based metrics are computed on the NPS averaged over all ROIs of all test
slices (one value per method and fold); the others are averaged over slices.

Baselines:  identity (NDCT unchanged) and gaussian (white noise with the
training-set noise std) — a good model must beat both on texture metrics.

Usage:
    python evaluate.py --runs runs/cut_100_FBP_to_10_FBP_f0_s0 runs/cyclegan_... --out results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import datasets
from datasets import body_slices, hu_to_net, load_pair, net_to_hu
from models import build_model

ROI = 64
WIN = [-100.0, 200.0]   # display/metric window (level 50, width 300); --win


def masks(ndct):
    body = ndimage.binary_fill_holes(ndct > -500)
    body = ndimage.binary_erosion(body, iterations=8)
    sm = ndimage.gaussian_filter(ndct, 2)
    grad = np.hypot(ndimage.sobel(sm, 0), ndimage.sobel(sm, 1)) / 8
    flat = body & (sm > -150) & (sm < 250) & (grad < 15)
    return body, flat


def roi_corners(flat, stride=32):
    H, W = flat.shape
    pts = []
    for y in range(0, H - ROI + 1, stride):
        for x in range(0, W - ROI + 1, stride):
            if flat[y:y + ROI, x:x + ROI].mean() > 0.95:
                pts.append((y, x))
    return pts


def nps2d(noise, pts, dx):
    acc, n = np.zeros((ROI, ROI)), 0
    for y, x in pts:
        r = noise[y:y + ROI, x:x + ROI]
        r = r - r.mean()
        acc += np.abs(np.fft.fftshift(np.fft.fft2(r))) ** 2
        n += 1
    return acc * dx * dx / (ROI * ROI), n


def radial(nps, dx, nbins=32):
    f = np.fft.fftshift(np.fft.fftfreq(ROI, d=dx))
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.hypot(fy, fx)
    edges = np.linspace(0, 0.5 / dx, nbins + 1)
    idx = np.digitize(fr.ravel(), edges) - 1
    prof = np.array([nps.ravel()[idx == i].mean() if np.any(idx == i) else 0 for i in range(nbins)])
    return 0.5 * (edges[1:] + edges[:-1]), prof


def psnr_ssim(x, ref):
    lo, hi = WIN
    xw, rw = np.clip(x, lo, hi), np.clip(ref, lo, hi)
    return (peak_signal_noise_ratio(rw, xw, data_range=hi - lo),
            structural_similarity(rw, xw, data_range=hi - lo))


def peak_freq(f, c):
    """frequency of the NPS maximum, refined by fitting a parabola through the
    peak bin and its two neighbours."""
    if np.sum(c) <= 0:
        return np.nan
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(f[k] + np.clip(off, -0.5, 0.5) * (f[1] - f[0]))
    return float(f[k])


class LPIPS:
    def __init__(self, dev):
        try:
            import lpips
            self.net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
            self.dev = dev
        except Exception as e:  # package missing or weights not downloadable
            print(f"[warn] LPIPS disabled ({e.__class__.__name__}: {e}). `pip install lpips` to enable.")
            self.net = None

    @torch.no_grad()
    def __call__(self, x, ref, rng=None):
        if self.net is None:
            return np.nan
        lo, hi = rng or WIN
        def t(a):
            a = (np.clip(a, lo, hi) - lo) / (hi - lo) * 2 - 1
            return torch.from_numpy(a.astype(np.float32))[None, None].repeat(1, 3, 1, 1).to(self.dev)
        return float(self.net(t(x), t(ref)).item())


class DISTS:
    """DISTS (Ding et al., TPAMI 2020): structure + texture similarity in VGG feature
    space, designed to tolerate texture resampling - well suited to comparing two
    different noise realisations. Lower = more similar. Needs `pip install piq`."""

    def __init__(self, dev):
        try:
            import piq
            self.net = piq.DISTS(reduction="none").to(dev).eval()
            self.dev = dev
        except Exception as e:
            print(f"[warn] DISTS disabled ({e.__class__.__name__}: {e}). `pip install piq` to enable.")
            self.net = None

    @torch.no_grad()
    def __call__(self, x, ref, rng=None):
        if self.net is None:
            return np.nan
        lo, hi = rng or WIN
        def t(a):
            a = np.clip((np.clip(a, lo, hi) - lo) / (hi - lo), 0, 1)
            return torch.from_numpy(a.astype(np.float32))[None, None].repeat(1, 3, 1, 1).to(self.dev)
        try:
            return float(self.net(t(x), t(ref)).item())
        except Exception as e:
            print(f"[warn] DISTS failed ({e}); disabling")
            self.net = None
            return np.nan


def unit_metrics(x, ref, rng):
    """RMSE / PSNR / SSIM on images mapped to [0, 1] with the full HU range `rng`."""
    lo, hi = rng
    a = (np.clip(x, lo, hi) - lo) / (hi - lo)
    b = (np.clip(ref, lo, hi) - lo) / (hi - lo)
    return (float(np.sqrt(np.mean((a - b) ** 2))), peak_signal_noise_ratio(b, a, data_range=1.0),
            structural_similarity(b, a, data_range=1.0))


def lf_psnr(x, ref, body):
    a, b = ndimage.gaussian_filter(x, 3), ndimage.gaussian_filter(ref, 3)
    mse = np.mean((a[body] - b[body]) ** 2)
    return 10 * np.log10((WIN[1] - WIN[0]) ** 2 / max(mse, 1e-8))


def load_run(path, dev):
    """Trained network (model.pt) or stored predictions (pred.npy, diffusion models)."""
    if not os.path.exists(os.path.join(path, "model.pt")) and os.path.exists(os.path.join(path, "pred.npy")):
        return np.load(os.path.join(path, "pred.npy"))
    ck = torch.load(os.path.join(path, "model.pt"), map_location=dev)
    kw = ck.get("kw", {"residual": ck["residual"], "d_hp": 0.0, "beta": 1e-3})
    m = build_model(ck["model"], quick=ck["quick"], **kw)
    m.load_state_dict(ck["state"])
    return m.to(dev).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--baselines", nargs="*", default=["identity", "gaussian", "zeng"],
                    choices=["identity", "gaussian", "zeng"], help="which baselines to include")
    ap.add_argument("--n_examples", type=int, default=8, help="test slices shown in the figures")
    ap.add_argument("--win", type=float, nargs=2, default=[-100.0, 200.0],
                    help="HU window for PSNR/SSIM/LPIPS, e.g. --win -1024 2048 for full range")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--no_dists", action="store_true")
    ap.add_argument("--unit_range", type=float, nargs=2, default=None,
                    help="HU range mapped to [0,1] for the *_01 metrics "
                         "(default: the common HU range of the evaluated methods)")
    ap.add_argument("--no_zeng", action="store_true", help="skip the Zeng-style projection-noise baseline")
    ap.add_argument("--zeng_angles", type=int, default=720)
    args = ap.parse_args(argv)
    WIN[:] = args.win
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = LPIPS(dev) if not args.no_lpips else None
    dists = DISTS(dev) if not args.no_dists else None

    cfgs = [json.load(open(os.path.join(r, "config.json"))) for r in args.runs]
    c0 = cfgs[0]
    for c in cfgs[1:]:
        assert (c["src"], c["tgt"], c["fold"], c["n_folds"]) == (c0["src"], c0["tgt"], c0["fold"], c0["n_folds"]), \
            "all runs passed together must share src/tgt/fold"
    # each method keeps its own HU range for its input; all outputs AND the reference
    # are scored on the common (intersected) range, so methods working in a narrow
    # window (diffusion models) are compared fairly with full-range models.
    ranges = [c.get("hu_range", [-1024.0, 2048.0]) for c in cfgs]
    lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
    if len({tuple(r) for r in ranges}) > 1:
        print(f"[eval] runs use different HU ranges {sorted({tuple(r) for r in ranges})}; "
              f"scoring everything on the common range [{lo:g}, {hi:g}] HU")
    src, tgt, z, spacing = load_pair(c0["data"], c0["src"], c0["tgt"])
    # methods receive the raw NDCT (Zeng needs real air/bone values); every output
    # and the reference are clipped to the networks' HU range before scoring. This
    # also removes the -3024 HU padding outside the reconstruction circle.
    clip = lambda x: np.clip(x, lo, hi)
    urng = tuple(args.unit_range) if args.unit_range else (lo, hi)
    print(f"[eval] window metrics in [{WIN[0]:g}, {WIN[1]:g}] HU; *_01 metrics on [0,1] = [{urng[0]:g}, {urng[1]:g}] HU")
    dx = float(spacing[0])
    te, tr = np.array(c0["test_idx"]), np.array(c0["train_idx"])
    for c in cfgs[1:]:   # only slices every method has (diffusion runs may store a subset)
        te = np.array([i for i in te if i in set(c["test_idx"])])
    assert len(te), "no common test slices between the runs"
    n0 = len(te)
    te = body_slices(src, te)        # also filters runs trained before this check existed
    if len(te) < n0:
        print(f"[eval] excluded {n0 - len(te)} test slices without body (empty air)")
    tag = f"{c0['src']}_to_{c0['tgt']}_f{c0['fold']}"
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)

    # methods: name -> fn(ndct_hu) -> hu
    methods = {}
    if not args.no_baselines:
        # train-set noise std for the gaussian baseline
        stds = []
        for i in tr[:: max(1, len(tr) // 40)]:
            _, flat = masks(src[i])
            if flat.sum() > 500:
                stds.append(np.std((clip(tgt[i]) - clip(src[i]))[flat]))
        g_std = float(np.median(stds)) if stds else 10.0
        if "identity" in args.baselines:
            methods["identity"] = lambda x, i=None: clip(x)
        if not args.no_zeng and "zeng" in args.baselines:
            from zeng_baseline import calibrate, dose_fraction_from_keys
            zsim, zpar = calibrate(src, tgt, tr, dose_fraction_from_keys(c0["src"], c0["tgt"]), dx,
                                   n_angles=args.zeng_angles)
            json.dump(zpar, open(os.path.join(out, "zeng_params.json"), "w"), indent=1)
            methods["zeng"] = lambda x, i=None, sim=zsim: clip(sim(x, dx))
        if "gaussian" in args.baselines:
            methods["gaussian"] = lambda x, i=None, s=g_std: clip(x + np.random.normal(0, s, x.shape).astype(np.float32))
    for r, c, rng in zip(args.runs, cfgs, ranges):
        m = load_run(r, dev)
        name = c["model"] + ("" if c.get("seed", 0) == 0 else f"_s{c['seed']}")
        if isinstance(m, np.ndarray):           # stored predictions
            pos = {int(k): j for j, k in enumerate(c["test_idx"])}

            def fn(x, i, P=m, pos=pos):
                return clip(P[pos[int(i)]].astype(np.float32))
        else:
            def fn(x, i=None, m=m, rng=rng):
                datasets.set_hu_range(*rng)
                t = torch.from_numpy(hu_to_net(x).astype(np.float32))[None, None].to(dev)
                return clip(net_to_hu(m.translate(t)[0, 0].float().cpu().numpy()))
        methods[name] = fn

    rows = []
    nps_acc = {k: [np.zeros((ROI, ROI)), 0] for k in ["real"] + list(methods)}
    # example slices: evenly spread over test slices that actually contain the body
    body_frac = np.array([(src[i][::8, ::8] > -500).mean() for i in te])
    cand = te[body_frac > 0.5 * body_frac.max()] if len(te) else te
    ex_ids = cand[np.linspace(0, len(cand) - 1, min(args.n_examples, len(cand))).astype(int)] if len(cand) else []
    examples = {}
    for i in te:
        body, flat = masks(src[i])          # masks from raw HU (air/body boundary)
        nd_raw = src[i]
        nd, ld = clip(src[i]), clip(tgt[i])
        pts = roi_corners(flat)
        n_real = ld - nd
        p, n = nps2d(n_real, pts, dx)
        nps_acc["real"][0] += p
        nps_acc["real"][1] += n
        std_real = np.std(n_real[flat]) if flat.sum() > 100 else np.nan
        if i in ex_ids:
            examples[int(i)] = {"NDCT": nd, "real LDCT": ld}
        for name, fn in methods.items():
            g = fn(nd_raw, i)
            ng = g - nd
            p, n = nps2d(ng, pts, dx)
            nps_acc[name][0] += p
            nps_acc[name][1] += n
            ps, ss = psnr_ssim(g, ld)
            std_g = np.std(ng[flat]) if flat.sum() > 100 else np.nan
            rows.append(dict(method=name, slice=int(i), rmse=float(np.sqrt(np.mean((g - ld) ** 2))),
                             psnr=ps, ssim=ss, lpips=lp(g, ld) if lp else np.nan,
                             dists=dists(g, ld) if dists else np.nan,
                             **dict(zip(["rmse_01", "psnr_01", "ssim_01"], unit_metrics(g, ld, urng))),
                             lpips_01=lp(g, ld, urng) if lp else np.nan,
                             dists_01=dists(g, ld, urng) if dists else np.nan,
                             lf_psnr=lf_psnr(g, ld, body),
                             noise_std=std_g, noise_std_real=std_real,
                             noise_std_err=abs(std_g - std_real) / std_real if std_real else np.nan,
                             hu_bias=float(np.mean((g - ld)[flat])) if flat.sum() > 100 else np.nan))
            if i in ex_ids:
                examples[int(i)][name] = g

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_slice.csv"), index=False)

    # NPS curves + texture metrics
    curves = {}
    f = radial(np.zeros((ROI, ROI)), dx)[0]
    if nps_acc["real"][1] == 0:
        print("[warn] no flat soft-tissue ROI found in the test slices: NPS metrics will be NaN")
    for k, (acc, n) in nps_acc.items():
        curves[k] = radial(acc / n, dx)[1] if n else np.zeros(len(f))
    curves["freq"] = f
    pd.DataFrame(curves).to_csv(os.path.join(out, "nps_radial.csv"), index=False)
    real = curves["real"]
    fmean_real = np.sum(f * real) / np.sum(real)
    fpeak_real = peak_freq(f, real)
    summ = df.groupby("method").agg(rmse=("rmse", "mean"), psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                                    lpips=("lpips", "mean"), dists=("dists", "mean"),
                                    rmse_01=("rmse_01", "mean"), psnr_01=("psnr_01", "mean"),
                                    ssim_01=("ssim_01", "mean"), lpips_01=("lpips_01", "mean"),
                                    dists_01=("dists_01", "mean"), lf_psnr=("lf_psnr", "mean"),
                                    noise_std=("noise_std", "mean"), noise_std_real=("noise_std_real", "mean"),
                                    noise_std_err=("noise_std_err", "mean"),
                                    abs_hu_bias=("hu_bias", lambda s: np.nanmean(np.abs(s))))
    for k in methods:
        c = curves[k]
        summ.loc[k, "nps_mae"] = float(np.mean(np.abs(c - real)))
        summ.loc[k, "nps_l2"] = float(np.linalg.norm(c - real))
        u2 = (urng[1] - urng[0]) ** 2          # NPS of the [0,1]-normalised images
        summ.loc[k, "nps_mae_01"] = summ.loc[k, "nps_mae"] / u2
        summ.loc[k, "nps_l2_01"] = summ.loc[k, "nps_l2"] / u2
        summ.loc[k, "nps_mae_rel"] = float(np.mean(np.abs(c - real)) / max(np.mean(real), 1e-12))
        fp = peak_freq(f, c)
        summ.loc[k, "f_peak"] = fp
        summ.loc[k, "df_peak"] = abs(fp - fpeak_real) if np.isfinite(fp) else np.nan
        if np.sum(c) <= 0:
            summ.loc[k, ["nps_shape_err", "f_mean", "f_mean_err"]] = [np.nan, np.nan, np.nan]
            continue
        fm = np.sum(f * c) / np.sum(c)
        summ.loc[k, "nps_shape_err"] = 0.5 * np.sum(np.abs(c / c.sum() - real / real.sum()))
        summ.loc[k, "f_mean"] = fm
        summ.loc[k, "f_mean_err"] = abs(fm - fmean_real)
    summ["f_mean_real"] = fmean_real
    summ["f_peak_real"] = fpeak_real
    summ["win"] = f"{WIN[0]:g},{WIN[1]:g}"
    summ["hu_range"] = f"{lo:g},{hi:g}"
    summ["unit_range"] = f"{urng[0]:g},{urng[1]:g}"
    summ["n_rois"] = nps_acc["real"][1]
    summ["n_test_slices"] = len(te)
    summ = summ.reset_index()
    summ.insert(1, "src", c0["src"])
    summ.insert(2, "tgt", c0["tgt"])
    summ.insert(3, "fold", c0["fold"])
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(summ.round(4).to_string(index=False))

    _plots(out, curves, examples, tag)
    return out


def _grid(out, examples, tag):
    """One overview figure: rows = test slices, columns = NDCT, real LDCT, methods."""
    import matplotlib.pyplot as plt
    sls = sorted(examples)
    names = list(examples[sls[0]])
    H, W = examples[sls[0]]["NDCT"].shape
    crop = (slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4))
    fig, ax = plt.subplots(len(sls), len(names), figsize=(2.2 * len(names), 2.2 * len(sls)), squeeze=False)
    for r, sl in enumerate(sls):
        for c, k in enumerate(names):
            ax[r, c].imshow(examples[sl][k][crop], cmap="gray", vmin=WIN[0], vmax=WIN[1])
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(k, fontsize=10)
            if c == 0:
                ax[r, c].set_ylabel(f"slice {sl}", fontsize=9)
    fig.suptitle(f"{tag}   window [{WIN[0]:g}, {WIN[1]:g}] HU (L {(WIN[0] + WIN[1]) / 2:g} / W {WIN[1] - WIN[0]:g})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "examples_grid.png"), dpi=120)
    plt.close(fig)


def _plots(out, curves, examples, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = curves["freq"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, c in curves.items():
        if k == "freq":
            continue
        style = dict(color="k", lw=2.5) if k == "real" else dict(lw=1.5, ls="--" if k in ("identity", "gaussian", "zeng") else "-")
        ax[0].plot(f, c, label=k, **style)
        if c.sum() > 0:
            ax[1].plot(f, c / c.sum(), label=k, **style)
    ax[0].set(title="Noise power spectrum", xlabel="spatial frequency (cycles/mm)", ylabel="NPS (HU² mm²)")
    ax[1].set(title="Area-normalised NPS (texture)", xlabel="spatial frequency (cycles/mm)")
    ax[0].legend(fontsize=8)
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nps.png"), dpi=150)
    plt.close(fig)

    if examples:
        _grid(out, examples, tag)
    for sl, ims in examples.items():
        names = list(ims)
        H, W = ims["NDCT"].shape
        cy, cx = H // 2, W // 2
        crop = (slice(cy - H // 4, cy + H // 4), slice(cx - W // 4, cx + W // 4))
        fig, ax = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.4))
        for j, k in enumerate(names):
            ax[0, j].imshow(ims[k][crop], cmap="gray", vmin=WIN[0], vmax=WIN[1])
            ax[0, j].set_title(k, fontsize=9)
            if k != "NDCT":
                ax[1, j].imshow((ims[k] - ims["NDCT"])[crop], cmap="RdBu_r", vmin=-100, vmax=100)
            for a in ax[:, j]:
                a.axis("off")
        ax[1, 0].text(0.5, 0.5, "noise =\nimage − NDCT\n[−100, 100] HU", ha="center", va="center",
                      transform=ax[1, 0].transAxes, fontsize=9)
        fig.suptitle(f"{tag}  slice {sl}   (window [{WIN[0]:g}, {WIN[1]:g}] HU)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"example_slice{sl}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
