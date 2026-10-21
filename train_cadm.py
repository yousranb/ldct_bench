"""Train and sample PhysHVS-LDCT (CADM): ONE dose-controllable model for all target doses (3 folds, as the GAN benchmark).

Train (one model for 5, 10, 25 %):
    python train_cadm.py train --data data/piglet --recon FBP --doses 10 25 50
Sample the test slab of every dose -> runs/cadm_<src>_to_<tgt>_f<fold>_s0/pred.npy
    python train_cadm.py sample --ckpt cadm_runs/cadm_FBP_s-n-c_f0/final --doses 10 25 50
    (a dose NOT used in training can also be sampled, to test dose control)
Ablations (same script):  --branches s n   (no perceptual prior)
                          --branches s c   (no physics prior)
                          --branches s     (plain ControlNet on NDCT)
                          --tag <name>     (keep runs separate)

Before the first real run, download Stable Diffusion 1.5 once (on a machine with internet):
    huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
        --include "vae/*" "unet/*" "text_encoder/*" "tokenizer/*" "scheduler/*" "model_index.json" \
        --local-dir ~/models/sd15
and pass  --sd_path ~/models/sd15
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from diffusion_common import (DEFAULT_WINDOW, dose_code, hu_to_unit, lf_anchor, load_multi_dose, unit_to_hu,
                              write_pred_run)
from priors import PerceptualPrior, PhysicsPrior


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "sample"])
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--src_dose", type=int, default=100)
    ap.add_argument("--recon", default="FBP", help="target reconstruction (FBP, ASIR20, ASIR40, ASIR60, MBIR)")
    ap.add_argument("--src_recon", default=None,
                    help="reconstruction of the full-dose input (default: same as --recon)")
    ap.add_argument("--doses", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--window", type=float, nargs=2, default=list(DEFAULT_WINDOW),
                    help="HU range the diffusion model works in")
    # model
    ap.add_argument("--sd_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--branches", nargs="+", default=["s", "n", "c"], choices=["s", "n", "c"])
    ap.add_argument("--lambda_n", type=float, default=1.0)
    ap.add_argument("--lambda_c", type=float, default=0.5)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--dose_channel", default="auto", choices=["auto", "none"],
                    help="auto: log-dose map added to the physics branch (or to the NDCT branch when the "
                         "physics branch is removed); none: no explicit dose code (ablation)")
    ap.add_argument("--name", default=None, help="method name in the results table (sample mode)")
    # priors
    ap.add_argument("--I0", type=float, default=1e7, help="full-dose incident photons (Eq. 3.19)")
    ap.add_argument("--sigma_e", type=float, default=10.0, help="electronic noise std (counts)")
    ap.add_argument("--n_angles", type=int, default=720)
    ap.add_argument("--parm_mode", default="thesis", choices=["thesis", "noise"])
    ap.add_argument("--csf_ppd", type=float, default=30.0, help="display pixels per degree for the CSF")
    # training
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr_control", type=float, default=1e-5)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--no_resume", action="store_true",
                    help="start from scratch even if checkpoints of this run exist")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cadm_runs")
    ap.add_argument("--tag", default="")
    # sampling
    ap.add_argument("--ckpt", help="checkpoint folder (sample mode)")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps")
    ap.add_argument("--sample_batch", type=int, default=4)
    ap.add_argument("--runs", default="runs", help="where prediction run folders go (evaluate.py input)")
    ap.add_argument("--max_slices", type=int, default=0, help="debug: only sample this many test slices")
    ap.add_argument("--lf_anchor", type=float, default=0.0,
                    help="sigma (px) of optional low-frequency anchoring to the NDCT (0 = off); "
                         "the method name gets the suffix _lf")
    ap.add_argument("--tiny", action="store_true", help="tiny random model, for code testing only")
    ap.add_argument("--exclude_subjects", nargs="*", default=[],
                    help="animals left out of TRAINING (multi-subject data), e.g. badly registered ones")
    ap.add_argument("--target", default="image", choices=["image", "residual"],
                    help="image: diffuse the LDCT latent (original CADM). residual: diffuse the latent "
                         "difference E(LDCT)-E(NDCT); the output is NDCT + [D(E(NDCT)+r) - D(E(NDCT))], so the "
                         "anatomy comes from the NDCT pixels and only the dose-dependent change is generated")
    return ap.parse_args()


def dose_branch(a):
    """Which branch carries the log-dose map (None = no explicit dose code)."""
    if getattr(a, "dose_channel", "auto") == "none":
        return None
    return "n" if "n" in a.branches else "s"


def branch_channels(a):
    db = dose_branch(a)
    return {b: 1 + (b == db) for b in ("s", "n", "c")}


class Conditioner:
    """Builds the three conditioning images for a batch of NDCT slices (on the GPU)."""

    def __init__(self, a, n, dev):
        self.a, self.dev = a, dev
        self.phys = PhysicsPrior(n, I0=a.I0, sigma_e=a.sigma_e, n_angles=a.n_angles, parm_mode=a.parm_mode,
                                 device=dev, seed=a.seed)
        self.perc = PerceptualPrior(n, window=a.window, ppd=a.csf_ppd, device=dev)
        self.half = (a.window[1] - a.window[0]) / 2

    def __call__(self, nd_hu, doses, dx):
        """nd_hu (B,H,W) torch HU; doses list of percent; dx pixel size."""
        db = dose_branch(self.a)
        out = {b: [] for b in self.a.branches}
        for x, d in zip(nd_hu, doses):
            maps = {"s": (hu_to_unit(x, self.a.window) + 1) / 2}
            if "n" in self.a.branches:
                maps["n"] = (self.phys.parm(x, d / 100.0, dx) / self.half).clamp(-1, 1)
            if "c" in self.a.branches:
                maps["c"] = self.perc.pm(x)
            for b in self.a.branches:
                chans = [maps[b]] + ([torch.full_like(maps[b], dose_code(d / 100.0))] if b == db else [])
                out[b].append(torch.stack(chans))
        return {b: torch.stack(v) for b, v in out.items()}


def run_name(a):
    rec = a.recon if not a.src_recon or a.src_recon == a.recon else f"{a.src_recon}-to-{a.recon}"
    return f"cadm_{rec}_{'-'.join(a.branches)}_f{a.fold}" + (f"_{a.tag}" if a.tag else "")


def build(a, dev):
    from cadm import CADM
    return CADM(sd_path=a.sd_path, branches=a.branches, lambda_n=a.lambda_n, lambda_c=a.lambda_c,
                lora_rank=a.lora_rank, tiny=a.tiny, channels=branch_channels(a),
                residual=getattr(a, "target", "image") == "residual",
                res_scale=getattr(a, "res_scale", None) or 1.0).to(dev)


@torch.no_grad()
def vae_check(model, data, a, dev, n=6):
    """How much of the real LDCT noise survives the frozen SD VAE (upper bound for CADM)."""
    from evaluate import masks
    ratios = []
    for d, D in data.items():
        for i in D["train"][np.linspace(0, len(D["train"]) - 1, n).astype(int)]:
            nd, ld = D["src"][i].astype(np.float32), D["ld"][i].astype(np.float32)
            _, flat = masks(nd)
            if flat.sum() < 500:
                continue
            x = torch.tensor(hu_to_unit(ld, a.window), dtype=torch.float32, device=dev)[None, None]
            rec = unit_to_hu(model.decode(model.encode(x))[0, 0].float().cpu().numpy(), a.window)
            ndc = np.clip(nd, *a.window)
            r = np.std((rec - ndc)[flat]) / max(np.std((np.clip(ld, *a.window) - ndc)[flat]), 1e-6)
            ratios.append(r)
        print(f"[vae-check] {d}%: VAE(LDCT) keeps {np.median(ratios) * 100:.0f}% of the real noise std "
              f"(100% = perfect; this bounds what CADM can reproduce)")
        ratios = []


@torch.no_grad()
def residual_scale(model, data, a, dev, n=16):
    """1 / std of the latent residual E(LDCT) - E(NDCT) on training slices, so the diffused
    residual has unit variance (like SD latents)."""
    rng = np.random.default_rng(1234)
    vals = []
    for d, D in data.items():
        for i in rng.choice(D["train"], size=min(n, len(D["train"])), replace=False):
            nd = torch.tensor(hu_to_unit(D["src"][i].astype(np.float32), a.window), dtype=torch.float32, device=dev)
            ld = torch.tensor(hu_to_unit(D["ld"][i].astype(np.float32), a.window), dtype=torch.float32, device=dev)
            r = model.encode(ld[None, None], sample=False) - model.encode(nd[None, None], sample=False)
            vals.append(r.float().std().item())
    s = 1.0 / max(float(np.median(vals)), 1e-4)
    print(f"[residual] latent residual std {1 / s:.4f} -> res_scale {s:.3f}")
    return s


def train(a, dev):
    data = load_multi_dose(a.data, f"{a.src_dose}_{a.src_recon or a.recon}", a.recon, a.doses, a.n_folds, a.fold, a.gap,
                           exclude=a.exclude_subjects)
    doses = list(data)
    out = os.path.join(a.out, run_name(a))
    os.makedirs(out, exist_ok=True)
    json.dump(dict(vars(a), doses=doses, src=f"{a.src_dose}_{a.src_recon or a.recon}",
                   split={d: dict(train=data[d]["train"].tolist(), test=data[d]["test"].tolist()) for d in doses}),
              open(os.path.join(out, "config.json"), "w"), indent=1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = build(a, dev)
    n = data[doses[0]]["src"].shape[-1]
    cond = Conditioner(a, n, dev)
    vae_check(model, data, a, dev)
    if a.target == "residual":
        a.res_scale = residual_scale(model, data, a, dev)
        model.res_scale = a.res_scale
    D0 = data[doses[0]]
    p = (cond.phys.parm(D0["src"][D0["train"][len(D0["train"]) // 2]].astype(np.float32), doses[0] / 100,
                        D0["dx"])).cpu().numpy()
    print(f"[prior-check] PARM std at {doses[0]}% = {p[D0['src'][D0['train'][len(D0['train']) // 2]] > -500].std():.1f} HU "
          f"(I0={a.I0:g}; it is a conditioning signal, its scale need not match the real noise)")

    start = 0
    if not a.no_resume:      # continue from the latest checkpoint of this run (e.g. after a walltime kill)
        its = sorted(int(d[2:]) for d in os.listdir(out) if d.startswith("it") and d[2:].isdigit()
                     and os.path.exists(os.path.join(out, d, "lora.pt")))
        if os.path.exists(os.path.join(out, "final", "lora.pt")):
            print(f"[resume] {out}/final already exists: nothing to train")
            return
        for n in reversed(its):                # newest first; skip damaged checkpoints
            ck = os.path.join(out, f"it{n}")
            try:
                load_weights(model, ck, dev)
            except Exception as e:
                print(f"[resume] {ck} is unreadable ({e.__class__.__name__}), deleting it")
                import shutil
                shutil.rmtree(ck, ignore_errors=True)
                continue
            start = n
            print(f"[resume] continuing from {ck} (iteration {start}); optimizer state restarts")
            break
    opt = torch.optim.AdamW(model.trainable_groups(a.lr_control, a.lr_lora), weight_decay=1e-2)
    rng = np.random.default_rng(a.seed + start)
    n_train = sum(p.numel() for g in model.trainable_groups() for p in g["params"])
    print(f"[model] branches {a.branches}, trainable params {n_train / 1e6:.0f} M, device {dev}")
    use_amp = dev == "cuda"
    model.train()
    t0, logl = time.time(), []
    for it in range(start + 1, a.iters + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            ds = rng.choice(doses, size=a.batch)
            nd, ld = [], []
            for d in ds:
                D = data[d]
                i = rng.choice(D["train"])
                x, y = D["src"][i].astype(np.float32), D["ld"][i].astype(np.float32)
                if rng.random() < 0.5:               # left-right flip only (keeps noise orientation)
                    x, y = x[:, ::-1], y[:, ::-1]
                nd.append(np.ascontiguousarray(x))
                ld.append(np.ascontiguousarray(y))
            nd = torch.tensor(np.stack(nd), device=dev)
            ld = hu_to_unit(torch.tensor(np.stack(ld), device=dev), a.window)[:, None]
            conds = cond(nd, [int(d) for d in ds], data[ds[0]]["dx"])
            x_nd = hu_to_unit(nd, a.window)[:, None] if a.target == "residual" else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model.loss(ld, conds, x_nd=x_nd) / a.accum
            loss.backward()
            logl.append(loss.item() * a.accum)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        if it % a.log_every == 0 or it == a.iters:
            el = time.time() - t0
            done = it - start
            print(f"[{it}/{a.iters}] loss={np.mean(logl):.4f}  {el / done:.2f}s/it  "
                  f"eta {el / done * (a.iters - it) / 3600:.1f} h", flush=True)
            logl = []
        if it % a.save_every == 0 or it == a.iters:
            save(model, os.path.join(out, "final" if it == a.iters else f"it{it}"), a)
            # keep only the newest checkpoint: each one is ~2 GB (3 ControlNet copies)
            import shutil
            keep = "final" if it == a.iters else f"it{it}"
            for d in os.listdir(out):
                if d.startswith("it") and d[2:].isdigit() and d != keep:
                    shutil.rmtree(os.path.join(out, d), ignore_errors=True)
    print(f"[done] {out}/final")


def load_weights(model, ckpt, dev):
    for b, c in model.controlnets.items():
        c.load_state_dict({k: v.float() for k, v in
                           torch.load(os.path.join(ckpt, f"controlnet_{b}.pt"), map_location=dev).items()})
    model.load_lora_state_dict(torch.load(os.path.join(ckpt, "lora.pt"), map_location=dev))


def save(model, path, a):
    os.makedirs(path, exist_ok=True)
    for b, c in model.controlnets.items():
        torch.save({k: v.half() for k, v in c.state_dict().items()}, os.path.join(path, f"controlnet_{b}.pt"))
    torch.save(model.lora_state_dict(), os.path.join(path, "lora.pt"))
    json.dump(vars(a), open(os.path.join(path, "args.json"), "w"), indent=1)
    print(f"[ckpt] {path}")


@torch.no_grad()
def sample(a, dev):
    targs = json.load(open(os.path.join(a.ckpt, "args.json")))
    for k in ["branches", "lambda_n", "lambda_c", "lora_rank", "window", "I0", "sigma_e", "n_angles",
              "parm_mode", "csf_ppd", "recon", "src_recon", "src_dose", "n_folds", "fold", "gap", "data", "tiny",
              "dose_channel", "target", "res_scale"]:
        setattr(a, k, targs.get(k))      # model/prior settings must match training
    a.dose_channel = a.dose_channel or "auto"
    a.target = a.target or "image"
    model = build(a, dev)
    for b, c in model.controlnets.items():
        c.load_state_dict({k: v.float() for k, v in
                           torch.load(os.path.join(a.ckpt, f"controlnet_{b}.pt"), map_location=dev).items()})
    model.load_lora_state_dict(torch.load(os.path.join(a.ckpt, "lora.pt"), map_location=dev))
    model.eval()
    data = load_multi_dose(a.data, f"{a.src_dose}_{a.src_recon or a.recon}", a.recon, a.doses, a.n_folds, a.fold, a.gap)
    trained = targs.get("doses", targs["doses"])
    for d, D in data.items():
        if d not in trained:
            print(f"[note] {d}% was not a training dose: this tests dose controllability (interpolation)")
        cond = Conditioner(a, D["src"].shape[-1], dev)
        cond.phys.reseed(1234 + d)
        te = D["test"][:a.max_slices] if a.max_slices else D["test"]
        g = torch.Generator(device=dev).manual_seed(a.seed + d)
        preds, t0 = [], time.time()
        for k in range(0, len(te), a.sample_batch):
            idx = te[k:k + a.sample_batch]
            nd = torch.tensor(D["src"][idx].astype(np.float32), device=dev)
            conds = cond(nd, [d] * len(idx), D["dx"])
            x_nd = hu_to_unit(nd, a.window)[:, None] if a.target == "residual" else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                x = model.sample(conds, nd.shape[-2:], steps=a.steps, generator=g, x_nd=x_nd)
            lo, hi = a.window
            preds.append(np.clip(unit_to_hu(x[:, 0].float().cpu().numpy(), a.window), lo, hi))
            done = k + len(idx)
            print(f"[sample {d}%] {done}/{len(te)}  eta {(time.time() - t0) / done * (len(te) - done) / 60:.1f} min",
                  flush=True)
        name = "cadm" if a.branches == ["s", "n", "c"] else "cadm_" + "".join(a.branches)
        name = a.name or (name + (f"_{a.tag}" if a.tag else ""))
        pred = np.concatenate(preds)
        if a.lf_anchor > 0:
            pred = lf_anchor(pred, D["src"][te], a.window, a.lf_anchor)
            name += "_lf"
        write_pred_run(a.runs, name, f"{a.src_dose}_{a.src_recon or a.recon}", D["tgt"],
                       a.fold, a.n_folds, 0, a.data, D["train"], te, pred, a.window,
                       extra=dict(ckpt=a.ckpt, steps=a.steps, trained_doses=trained, lf_anchor=a.lf_anchor))


def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not a.tiny:
        print("[warn] no GPU visible: this will be extremely slow (are you on the frontend?)")
    train(a, dev) if a.mode == "train" else sample(a, dev)


if __name__ == "__main__":
    main()
