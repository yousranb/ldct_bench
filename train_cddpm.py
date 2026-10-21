"""Classical conditional diffusion baseline: one image-conditioned DDPM PER DOSE.

Standard SR3/Palette-style conditional DDPM in pixel space: the U-Net receives
[x_t, NDCT] (channel concatenation) and predicts the noise epsilon. No physics or
perceptual prior, no dose conditioning -> a separate model per target dose, like
CycleGAN/CUT/VAE. Trained on random 256x256 paired crops, sampled on full slices
(DDIM). Works in the same HU window as CADM.

    python train_cddpm.py --data data/piglet --recon FBP --doses 10 25 50
    (trains each dose, then samples its test slab -> runs/cddpm_<src>_to_<tgt>_f0_s0/pred.npy)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_common import DEFAULT_WINDOW, hu_to_unit, lf_anchor, load_multi_dose, unit_to_hu, write_pred_run

RES_SCALE = 200.0   # HU: residual target (LDCT - NDCT) / RES_SCALE


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--src_dose", type=int, default=100)
    ap.add_argument("--recon", default="FBP", help="target reconstruction (FBP, ASIR20, ASIR40, ASIR60, MBIR)")
    ap.add_argument("--src_recon", default=None,
                    help="reconstruction of the full-dose input (default: same as --recon)")
    ap.add_argument("--doses", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--window", type=float, nargs=2, default=list(DEFAULT_WINDOW))
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256, 256])
    ap.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    ap.add_argument("--sample_batch", type=int, default=4)
    ap.add_argument("--max_slices", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cddpm_runs")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--sample_only", action="store_true", help="skip training, load existing model.pt")
    ap.add_argument("--target", default="residual", choices=["residual", "image"],
                    help="residual: diffuse LDCT-NDCT (standard for LDCT synthesis, brightness anchored "
                         "to the NDCT); image: diffuse the LDCT image itself")
    ap.add_argument("--lf_anchor", type=float, default=0.0, help="see train_cadm.py (0 = off)")
    ap.add_argument("--exclude_subjects", nargs="*", default=[],
                    help="animals left out of TRAINING (multi-subject data), e.g. badly registered ones")
    ap.add_argument("--tiny", action="store_true")
    return ap.parse_args()


def build_unet(a):
    from diffusers import UNet2DModel
    ch = (16, 32) if a.tiny else tuple(a.channels)
    n = len(ch)
    return UNet2DModel(in_channels=2, out_channels=1, block_out_channels=ch, layers_per_block=1 if a.tiny else 2,
                       down_block_types=("DownBlock2D",) * (n - 1) + ("AttnDownBlock2D",),
                       up_block_types=("AttnUpBlock2D",) + ("UpBlock2D",) * (n - 1),
                       norm_num_groups=8 if a.tiny else 32)


def crops(D, idx, rng, patch, n):
    H = D["src"].shape[-1]
    p = min(patch, H)
    xs, ys = [], []
    for _ in range(n):
        i = rng.choice(idx)
        nd, ld = D["src"][i].astype(np.float32), D["ld"][i].astype(np.float32)
        fg = np.argwhere(nd[::4, ::4] > -500) * 4
        cy, cx = fg[rng.integers(len(fg))] if len(fg) else (H // 2, H // 2)
        y, x = int(np.clip(cy - p // 2, 0, H - p)), int(np.clip(cx - p // 2, 0, H - p))
        a, b = nd[y:y + p, x:x + p], ld[y:y + p, x:x + p]
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        xs.append(a.copy())
        ys.append(b.copy())
    return np.stack(xs), np.stack(ys)


def train_one(a, d, D, dev, out):
    from diffusers import DDPMScheduler
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = build_unet(a).to(dev)
    sch = DDPMScheduler(num_train_timesteps=1000, clip_sample=a.target == "image")
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    print(f"[cddpm {d}%] {sum(p.numel() for p in net.parameters()) / 1e6:.0f} M params, device {dev}")
    t0, logl = time.time(), []
    for it in range(1, a.iters + 1):
        nd, ld = crops(D, D["train"], rng, a.patch, a.batch)
        c = hu_to_unit(torch.tensor(nd, device=dev), a.window)[:, None]
        if a.target == "residual":
            lo, hi = a.window
            x0 = ((torch.tensor(ld, device=dev).clamp(lo, hi) - torch.tensor(nd, device=dev).clamp(lo, hi))
                  / RES_SCALE)[:, None]
        else:
            x0 = hu_to_unit(torch.tensor(ld, device=dev), a.window)[:, None]
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (x0.shape[0],), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            pred = net(torch.cat([sch.add_noise(x0, noise, t), c], 1), t).sample
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        logl.append(loss.item())
        if it % a.log_every == 0 or it == a.iters:
            el = time.time() - t0
            print(f"[{d}% {it}/{a.iters}] loss={np.mean(logl):.4f}  {el / it:.3f}s/it  "
                  f"eta {el / it * (a.iters - it) / 60:.1f} min", flush=True)
            logl = []
    torch.save({"state": net.state_dict(), "target": a.target}, os.path.join(out, "model.pt"))
    return net


@torch.no_grad()
def sample_one(a, d, D, net, dev):
    from diffusers import DDIMScheduler, DDPMScheduler
    sch = DDIMScheduler.from_config(DDPMScheduler(num_train_timesteps=1000, clip_sample=a.target == "image").config,
                                    timestep_spacing="trailing")
    sch.set_timesteps(a.steps)
    net.eval()
    te = D["test"][:a.max_slices] if a.max_slices else D["test"]
    g = torch.Generator(device=dev).manual_seed(a.seed + d)
    preds, t0 = [], time.time()
    for k in range(0, len(te), a.sample_batch):
        idx = te[k:k + a.sample_batch]
        c = hu_to_unit(torch.tensor(D["src"][idx].astype(np.float32), device=dev), a.window)[:, None]
        x = torch.randn(c.shape, generator=g, device=dev)
        for t in sch.timesteps:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                e = net(torch.cat([x, c], 1), t.to(dev).expand(len(idx))).sample
            x = sch.step(e.float(), t, x).prev_sample
        if a.target == "residual":
            lo, hi = a.window
            nd = D["src"][idx].astype(np.float32).clip(lo, hi)
            preds.append(np.clip(nd + x[:, 0].float().cpu().numpy() * RES_SCALE, lo, hi))
        else:
            preds.append(unit_to_hu(x.clamp(-1, 1)[:, 0].cpu().numpy(), a.window))
        done = k + len(idx)
        print(f"[sample {d}%] {done}/{len(te)}  eta {(time.time() - t0) / done * (len(te) - done) / 60:.1f} min",
              flush=True)
    return te, np.concatenate(preds)


def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not a.tiny:
        print("[warn] no GPU visible: this will be extremely slow (are you on the frontend?)")
    src = f"{a.src_dose}_{a.src_recon or a.recon}"
    data = load_multi_dose(a.data, src, a.recon, a.doses, a.n_folds, a.fold, a.gap, exclude=a.exclude_subjects)
    for d, D in data.items():
        out = os.path.join(a.out, f"cddpm_{src}_to_{D['tgt']}_f{a.fold}" + ("_res" if a.target == "residual" else "")
                           + (f"_{a.tag}" if a.tag else ""))
        os.makedirs(out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
        if a.sample_only or os.path.exists(os.path.join(out, "model.pt")):
            net = build_unet(a).to(dev)
            ck = torch.load(os.path.join(out, "model.pt"), map_location=dev)
            if isinstance(ck, dict) and "state" in ck:
                a.target = ck.get("target", "image")
                ck = ck["state"]
            else:
                a.target = "image"            # models saved before the residual option
            net.load_state_dict(ck)
            print(f"[cddpm {d}%] loaded {out}/model.pt")
        else:
            net = train_one(a, d, D, dev, out)
        te, pred = sample_one(a, d, D, net, dev)
        name = "cddpm" + (f"_{a.tag}" if a.tag else "")
        if a.lf_anchor > 0:
            pred = lf_anchor(pred, D["src"][te], a.window, a.lf_anchor)
            name += "_lf"
        write_pred_run(a.runs, name, src, D["tgt"], a.fold, a.n_folds, 0, a.data, D["train"], te, pred, a.window,
                       extra=dict(steps=a.steps, iters=a.iters, target=a.target, lf_anchor=a.lf_anchor))


if __name__ == "__main__":
    main()
