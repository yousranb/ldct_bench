"""Zeng-style low-dose simulation (image-based approximation).

Reference: D. Zeng et al., "A simple low-dose x-ray CT simulation from high-dose
scan", IEEE TNS 62(5), 2015.

Zeng works on raw sinograms. We only have reconstructed DICOM images, so:
  1. NDCT (HU) -> attenuation mu = mu_water * (1 + HU/1000)
  2. parallel-beam forward projection  p = Radon(mu) * pixel_size
  3. Zeng's noise model for a dose fraction a (0 < a < 1):
        var = (1-a)/a * exp(p)/I0 * (1 + (1+a)/a * sigma_e^2 * exp(p)/I0)
        p_low = p + sqrt(var) * N(0, 1)
     (this already accounts for the quantum + electronic noise present in the NDCT)
  4. FBP is linear, so the noise image is  FBP(p_low - p); it is added to the
     ORIGINAL NDCT so anatomy is not blurred by the reproject/reconstruct round trip.
  5. I0 (photon count) is calibrated on TRAINING slices so the noise std in flat
     soft tissue matches the real LDCT of that dose. sigma_e is fixed.

Differences from the real scanner (state them in the paper): parallel instead of
helical fan-beam geometry, no bowtie filter, Hann/ramp FBP instead of GE kernel.

Usage
  standalone (calibrate + simulate the test slab, save .npy):
    python zeng_baseline.py --data data/piglet --src 100_FBP --tgt 10_FBP --fold 0 --save
  inside the benchmark: evaluate.py adds it automatically as the 'zeng' baseline.
"""
import argparse
import json
import os

import numpy as np
from skimage.transform import iradon, radon

MU_WATER = 0.0192  # 1/mm, ~ effective energy of a 120 kVp beam


class TorchParallelBeam:
    """Parallel-beam Radon transform and FBP on the GPU (falls back to CPU).
    Projection = rotate image by theta and sum along rows; backprojection is the
    exact adjoint (smear + rotate back), so forward/back geometry always match.
    Conventions and scaling follow skimage.transform.radon / iradon(circle=True)."""

    def __init__(self, n, theta_deg, filter_name="hann", device=None, chunk=90):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n, self.chunk = n, chunk
        th = torch.tensor(np.deg2rad(-np.asarray(theta_deg)), dtype=torch.float32, device=self.dev)
        self.cos, self.sin = torch.cos(th), torch.sin(th)
        # ramp * window, as in skimage (padded to a power of 2 >= 2n)
        P = max(64, int(2 ** np.ceil(np.log2(2 * n))))
        f = np.fft.fftfreq(P)
        filt = 2 * np.abs(f)
        if filter_name == "hann":
            filt *= np.fft.fftshift(np.hanning(P))
        elif filter_name == "hamming":
            filt *= np.fft.fftshift(np.hamming(P))
        elif filter_name == "shepp-logan":
            w = np.pi * f[1:] / 2
            filt[1:] *= np.sin(w) / w
        elif filter_name == "cosine":
            filt *= np.fft.fftshift(np.sin(np.linspace(0, np.pi, P, endpoint=False)))
        self.P, self.filt = P, torch.tensor(filt, dtype=torch.float32, device=self.dev)

    def _rot(self, img, c, s):
        # img (B,1,n,n); rotate each by its own angle
        B = c.shape[0]
        A = self.torch.zeros(B, 2, 3, device=self.dev)
        A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = c, -s, s, c
        grid = self.F.affine_grid(A, (B, 1, self.n, self.n), align_corners=False)
        return self.F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def radon(self, img):
        """img (n,n) numpy or torch -> sinogram (n, n_angles) torch, like skimage.radon."""
        if self.torch.is_tensor(img):
            t = img.to(self.dev, self.torch.float32)[None, None]
        else:
            t = self.torch.tensor(img, dtype=self.torch.float32, device=self.dev)[None, None]
        out = []
        for i in range(0, len(self.cos), self.chunk):
            c, s = self.cos[i:i + self.chunk], self.sin[i:i + self.chunk]
            out.append(self._rot(t.expand(len(c), 1, self.n, self.n), c, s).sum(2)[:, 0])  # (B, n)
        return self.torch.cat(out).T  # (n, A)

    def fbp(self, sino):
        """sino (n, n_angles) torch -> image (n, n), like skimage.iradon(circle=True)."""
        torch = self.torch
        n, A = sino.shape
        q = torch.fft.ifft(torch.fft.fft(sino, n=self.P, dim=0) * self.filt[:, None], dim=0).real[:n]
        img = torch.zeros(1, 1, n, n, device=self.dev)
        for i in range(0, A, self.chunk):
            c, s = self.cos[i:i + self.chunk], self.sin[i:i + self.chunk]
            smear = q[:, i:i + self.chunk].T[:, None, None, :].expand(len(c), 1, n, n).contiguous()
            img += self._rot(smear, c, -s).sum(0, keepdim=True)  # inverse rotation = adjoint
        img = img[0, 0] * np.pi / (2 * A)
        yy, xx = torch.meshgrid(torch.arange(n, device=self.dev), torch.arange(n, device=self.dev), indexing="ij")
        img[(yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 > (n / 2) ** 2] = 0
        return img


class ZengSimulator:
    def __init__(self, dose_fraction, I0=1e5, sigma_e2=10.0, n_angles=720, filter_name="hann", seed=0,
                 backend="auto"):
        assert 0 < dose_fraction < 1, "dose_fraction must be target/source, e.g. 0.10"
        self.a, self.I0, self.se2 = dose_fraction, I0, sigma_e2
        self.theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
        self.filter = filter_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.backend = backend   # "auto" -> torch (GPU if available), "skimage" -> CPU reference
        self._tp = None

    def _mu(self, hu):
        mu = MU_WATER * (1 + np.clip(hu, -1024, None) / 1000.0)
        mu = np.clip(mu, 0, None).astype(np.float64)
        H, W = mu.shape
        yy, xx = np.mgrid[:H, :W]
        mu[(yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2 > (min(H, W) / 2) ** 2] = 0  # circle=True
        return mu

    def _zeng_var(self, p, exp):
        a, I0, se2 = self.a, self.I0, self.se2
        e = exp(p) / I0
        return (1 - a) / a * e * (1 + (1 + a) / a * se2 * e)

    def noise_image(self, ndct_hu, dx):
        """Noise to add to the NDCT (in HU)."""
        mu = self._mu(ndct_hu)
        if self.backend == "skimage":
            p = radon(mu, theta=self.theta, circle=True) * dx
            dp = np.sqrt(self._zeng_var(p, np.exp)) * self.rng.standard_normal(p.shape)
            dmu = iradon(dp, theta=self.theta, circle=True, filter_name=self.filter,
                         output_size=ndct_hu.shape[0]) / dx
            return (dmu / MU_WATER * 1000.0).astype(np.float32)
        if self._tp is None or self._tp.n != mu.shape[0]:
            self._tp = TorchParallelBeam(mu.shape[0], self.theta, self.filter)
            self._gen = self._tp.torch.Generator(device=self._tp.dev).manual_seed(self.seed)
        torch = self._tp.torch
        p = self._tp.radon(mu) * dx
        var = self._zeng_var(p, torch.exp)
        dp = torch.sqrt(var) * torch.randn(p.shape, device=p.device, generator=self._gen)
        dmu = self._tp.fbp(dp) / dx
        return (dmu / MU_WATER * 1000.0).float().cpu().numpy()

    def __call__(self, ndct_hu, dx):
        return ndct_hu + self.noise_image(ndct_hu, dx)


def calibrate(src, tgt, train_idx, dose_fraction, dx, n_slices=12, iters=3, **kw):
    """Find I0 so that std(sim - NDCT) == std(LDCT - NDCT) in flat soft tissue."""
    from evaluate import masks
    idx = train_idx[np.linspace(0, len(train_idx) - 1, n_slices).astype(int)]
    sel, target = [], []
    for i in idx:
        _, flat = masks(src[i])
        if flat.sum() > 500:
            sel.append((i, flat))
            target.append(np.std((tgt[i] - src[i])[flat]))
    if not sel:
        raise RuntimeError("no flat soft-tissue region found for calibration")
    target = float(np.median(target))
    sim = ZengSimulator(dose_fraction, **kw)
    for _ in range(iters):
        s = float(np.median([np.std(sim.noise_image(src[i], dx)[f]) for i, f in sel]))
        # quantum-noise dominated: std ~ 1/sqrt(I0)
        sim.I0 *= (s / target) ** 2
    s = float(np.median([np.std(sim.noise_image(src[i], dx)[f]) for i, f in sel]))
    print(f"[zeng] a={dose_fraction:.3f}  I0={sim.I0:.3g}  noise std: sim {s:.2f} HU vs real {target:.2f} HU")
    return sim, dict(dose_fraction=dose_fraction, I0=sim.I0, sigma_e2=sim.se2, n_angles=len(sim.theta),
                     filter=sim.filter, std_sim=s, std_real=target)


def dose_fraction_from_keys(src_key, tgt_key):
    return int(tgt_key.split("_")[0]) / int(src_key.split("_")[0])


def main():
    from datasets import load_pair, slab_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--src", default="100_FBP")
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--n_angles", type=int, default=720)
    ap.add_argument("--filter", default="hann", choices=["ramp", "shepp-logan", "cosine", "hamming", "hann"])
    ap.add_argument("--sigma_e2", type=float, default=10.0)
    ap.add_argument("--backend", default="auto", choices=["auto", "skimage"],
                    help="auto = PyTorch (GPU if available); skimage = slow CPU reference")
    ap.add_argument("--save", action="store_true", help="save simulated test slab as .npy (HU)")
    ap.add_argument("--out", default="zeng_out")
    a = ap.parse_args()

    src, tgt, z, spacing = load_pair(a.data, a.src, a.tgt)
    # raw HU on purpose: the projector needs real air/bone attenuation
    tr, te = slab_split(len(z), a.n_folds, a.fold, a.gap, z=z)
    dx = float(spacing[0])
    sim, params = calibrate(src, tgt, tr, dose_fraction_from_keys(a.src, a.tgt), dx,
                            n_angles=a.n_angles, filter_name=a.filter, sigma_e2=a.sigma_e2, backend=a.backend)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.src}_to_{a.tgt}_f{a.fold}"
    json.dump(params, open(os.path.join(a.out, tag + "_params.json"), "w"), indent=1)
    if a.save:
        vol = np.stack([sim(src[i], dx) for i in te]).astype(np.float32)
        np.save(os.path.join(a.out, tag + "_test.npy"), vol)
        print(f"[zeng] saved {vol.shape} to {a.out}/{tag}_test.npy")


if __name__ == "__main__":
    main()
