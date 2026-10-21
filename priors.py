"""Conditioning priors of PhysHVS-LDCT (thesis Section 3.5).

PANM  - Physics-Aware Noise Modeling -> Physics-Aware Residual Map (Eqs. 3.17-3.23)
    mu   = mu_water (1 + HU/1000)                       (3.17)
    p    = Radon(mu)                                    (3.18)
    I0_r = r * I0                                       (3.20)
    I    ~ Poisson(I0_r exp(-p)) + N(0, sigma_e^2)      (3.21)
    p~   = -ln(I / I0_r)                                (3.22)
    PARM = NDCT - FBP(p~)                               (3.23)
  mu_water is 0.0192 / mm (= 0.192 / cm) with the pixel size in mm.
  parm_mode="thesis"  : exactly Eq. 3.23 (includes the slight blur of the
                        reprojection/FBP round trip at edges)
  parm_mode="noise"   : FBP(p~) - FBP(p), i.e. the dose-dependent noise/streaks only.

PSA   - Perceptual Sensitivity Adaptation -> perceptual map (Eqs. 3.24-3.25)
    PM = | F^-1( F(I_NDCT) * CSF ) |, Barten (1999) contrast sensitivity function,
  computed on the NDCT shown in the display window (what the observer sees).
  Pixel frequencies are converted to cycles/degree with `ppd` (pixels per degree
  of visual angle, ~30 for a 512 image viewed at a typical reading distance).
"""
import numpy as np
import torch

from zeng_baseline import MU_WATER, TorchParallelBeam


# ----------------------------------------------------------------------- PANM
class PhysicsPrior:
    def __init__(self, n, I0=1e7, sigma_e=10.0, n_angles=720, filter_name="hann",
                 parm_mode="thesis", device=None, seed=0):
        self.tp = TorchParallelBeam(n, np.linspace(0, 180, n_angles, endpoint=False), filter_name,
                                    device=device)
        self.dev = self.tp.dev
        self.I0, self.se, self.mode = I0, sigma_e, parm_mode
        self.n = n
        yy, xx = torch.meshgrid(torch.arange(n, device=self.dev), torch.arange(n, device=self.dev), indexing="ij")
        self.inside = ((yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2) <= (n / 2) ** 2
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    def reseed(self, seed):
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def parm(self, ndct_hu, dose_ratio, dx):
        """ndct_hu: (n,n) torch/numpy HU; dose_ratio r in (0,1]; dx pixel size (mm).
        Returns PARM (n,n) torch, in HU, zero outside the reconstruction circle."""
        x = torch.as_tensor(np.asarray(ndct_hu) if not torch.is_tensor(ndct_hu) else ndct_hu,
                            dtype=torch.float32, device=self.dev)
        mu = (MU_WATER * (1 + x.clamp(min=-1024) / 1000.0)).clamp(min=0) * self.inside
        p = self.tp.radon(mu) * dx
        I0r = dose_ratio * self.I0
        lam = I0r * torch.exp(-p)
        counts = torch.poisson(lam, generator=self.gen)
        counts = counts + self.se * torch.randn(counts.shape, device=self.dev, generator=self.gen)
        p_noisy = -torch.log(counts.clamp(min=1.0) / I0r)
        mu_ld = self.tp.fbp(p_noisy) / dx
        ld_hu = (mu_ld / MU_WATER - 1) * 1000.0
        if self.mode == "noise":
            nd_rec = (self.tp.fbp(p) / dx / MU_WATER - 1) * 1000.0
            out = nd_rec - ld_hu          # sign as in Eq. 3.23 (high - low)
        else:
            out = x - ld_hu
        return out * self.inside


# ------------------------------------------------------------------------ PSA
def barten_csf(u, L=100.0, X0=20.0, k=3.0, T=0.1, eta=0.03, p=1.2274e6, phi0=3e-8, u0=7.0,
               Xmax=12.0, Nmax=15.0, sigma0=0.0083, Cab=0.0013):
    """Barten (1999) CSF; u in cycles/degree. Standard psychovisual parameters:
    k signal-to-noise ratio, T integration time (s), eta quantum efficiency,
    p photon conversion factor, phi0 neural noise, u0 lateral-inhibition cut-off,
    L luminance (cd/m2), X0 object size (deg)."""
    d = 5 - 3 * np.tanh(0.4 * np.log10(L * X0 ** 2 / 40 ** 2))       # pupil diameter (mm)
    E = np.pi * d ** 2 / 4 * L                                        # retinal illuminance (Td)
    sigma = np.sqrt(sigma0 ** 2 + (Cab * d) ** 2)
    Mopt = np.exp(-2 * np.pi ** 2 * sigma ** 2 * u ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lat = 1 - np.exp(-(u / u0) ** 2)
        S = (Mopt / k) / np.sqrt((2 / T) * (1 / X0 ** 2 + 1 / Xmax ** 2 + u ** 2 / Nmax ** 2)
                                 * (1 / (eta * p * E) + phi0 / lat))
    return np.nan_to_num(S, nan=0.0, posinf=0.0)


class PerceptualPrior:
    def __init__(self, n, window=(-150.0, 250.0), ppd=30.0, L=100.0, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        f = np.fft.fftfreq(n)                                          # cycles / pixel
        fy, fx = np.meshgrid(f, f, indexing="ij")
        u = np.hypot(fy, fx) * ppd                                     # cycles / degree
        csf = barten_csf(u, L=L, X0=n / ppd)
        csf[0, 0] = 0.0
        self.csf = torch.tensor(csf / csf.max(), dtype=torch.float32, device=self.dev)
        self.window = window

    @torch.no_grad()
    def pm(self, ndct_hu):
        """Perceptual sensitivity map in [0, 1] (Eq. 3.25, magnitude, robust-normalised)."""
        x = torch.as_tensor(np.asarray(ndct_hu) if not torch.is_tensor(ndct_hu) else ndct_hu,
                            dtype=torch.float32, device=self.dev)
        lo, hi = self.window
        x = ((x - lo) / (hi - lo)).clamp(0, 1)                          # displayed luminance
        pm = torch.fft.ifft2(torch.fft.fft2(x) * self.csf).real.abs()
        q = torch.quantile(pm.flatten()[::7], 0.99).clamp(min=1e-6)
        return (pm / q).clamp(0, 1)


if __name__ == "__main__":
    # quick visual check:  python priors.py --data data/piglet --key 100_FBP --slice 400
    import argparse
    import json
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--key", default="100_FBP")
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--doses", type=float, nargs="+", default=[25, 10, 5])
    ap.add_argument("--I0", type=float, default=1e7)
    ap.add_argument("--parm_mode", default="thesis", choices=["thesis", "noise"])
    ap.add_argument("--out", default="priors_example.png")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.data, "index.json")))
    vol = np.load(os.path.join(a.data, a.key + ".npy"), mmap_mode="r")
    s = a.slice if a.slice is not None else len(vol) // 2
    x = np.asarray(vol[s], np.float32)
    dx = float(idx[a.key]["spacing"][0])
    P = PhysicsPrior(x.shape[0], I0=a.I0, parm_mode=a.parm_mode)
    C = PerceptualPrior(x.shape[0])
    ims = [("NDCT", x, dict(cmap="gray", vmin=-150, vmax=250))]
    for d in a.doses:
        r = P.parm(x, d / 100, dx).cpu().numpy()
        print(f"dose {d:g}%: PARM std inside body = {r[x > -500].std():.1f} HU")
        ims.append((f"{d:g}% PARM", r, dict(cmap="gray", vmin=-100, vmax=100)))
    ims.append(("Perceptual map", C.pm(x).cpu().numpy(), dict(cmap="magma", vmin=0, vmax=1)))
    fig, ax = plt.subplots(1, len(ims), figsize=(3.2 * len(ims), 3.4))
    for k, (t, im, kw) in enumerate(ims):
        ax[k].imshow(im, **kw)
        ax[k].set_title(t)
        ax[k].axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)
