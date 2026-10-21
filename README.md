# LDCT synthesis benchmark: CycleGAN vs CUT vs VAE (piglet data)

**Paper:** [PhysHVS-LDCT: Dose-Controllable Physics-Based Low-Dose CT Synthesis for Benchmarking and Data Augmentation](https://ieeexplore.ieee.org/document/11515637)

The task is to learn **full-dose (100 %) → low-dose** CT for each target dose. All models are scored on the same held-out slices.

Default setup: **FBP reconstruction, target doses 5 / 10 / 25 %**. Other doses and reconstructions (ASIR20/40/60, MBIR, 50 %) are available as command-line options.

## Install
```
pip install torch pydicom scikit-image scipy pandas matplotlib
```

## 1. Prepare the data (once)
```
python prepare_data.py --root "E:/piglet_sorted/piglet_sorted" --out data/piglet
```
This reads every series, converts it to HU, sorts slices by z position and saves one volume per `<dose>_<recon>`. It also prints an alignment check against full dose; every line should show (almost) all slices in common.

**Two things to check in your folders:**

1. **There is no 5 % FBP folder.** `5%` only has ASIR 20/40/60 and MBIR. The `EXTRA` folder was modified at 00:30, right after `5% DOSE MBIR` (00:28), so it may well be the missing 5 % FBP. Look at its DICOM header (`SeriesDescription`, `ConvolutionKernel`). If it is the 5 % FBP series, map it:
   ```
   python prepare_data.py --root "E:/piglet_sorted/piglet_sorted" --out data/piglet --map "EXTRA=5_FBP"
   ```
   If it isn't, 5 % is skipped for FBP. You can run 5 % with `--recon ASIR20` instead.
2. **25 % has two FBP folders** (`25 DOSE FBP` and `25% DOSE FBP`; the second was modified 10 days later). Identical copies are merged automatically. If they differ, the script warns you and keeps the one with more slices. To choose explicitly:
   `--map "25% DOSE FBP=25_FBP"` (the mappings can be combined in one `--map`).

## 2. Run the benchmark
```
python run_benchmark.py --data data/piglet --quick          # first pass, ~1 h on one GPU
python run_benchmark.py --data data/piglet --folds 0 1 2    # final numbers, 3-fold
```
Later, for other doses and reconstructions:
```
python run_benchmark.py --data data/piglet --doses 5 10 25 50 --recon ASIR40
python run_benchmark.py --data data/piglet --doses --recon MBIR     # all doses
```
Finished runs are skipped, so you can stop and restart. To train or evaluate a single model:
`python train.py --model cut --src 100_FBP --tgt 10_FBP --fold 0`
then `python evaluate.py --runs runs/<run1> runs/<run2> ...`.

Rough GPU time per model per dose (RTX 3090 class, 20k iters, 128-px patches):
CycleGAN ≈ 1–2 h, CUT ≈ 0.5–1 h, VAE ≈ 20–40 min. With `--quick` (5k iters, 6 res-blocks) it is about 4× less.

## Design choices (worth stating in the paper)

* **Split by contiguous z-slabs, not random slices.** Neighbouring slices are almost identical, so a random split leaks test data into training. `--n_folds 3` makes 3 slabs, and `--gap 5` drops 5 slices on each side of the test slab from training.
* **Same generator for all methods.** CycleGAN, CUT and the VAE decoder all use a ResNet generator, so differences come from the training objective rather than capacity. The generator is residual (output = NDCT + learned residual) and starts as the identity. `--no_residual` gives the original tanh output.
* **Supervision differs.** CycleGAN and CUT train **unpaired** (the LDCT patch comes from a random other slice and location). The **conditional VAE is paired**: it learns p(LDCT | NDCT), and at test time each call samples new noise. Report this difference, or add `vaegan` (VAE + adversarial loss) as a 4th method: `--models cyclegan cut vae vaegan`.
* **Augmentation is left–right flips only.** Rotations would change the orientation of CT noise and streaks, which is part of what is being learned.
* **Baselines.** `identity` (NDCT unchanged) and `gaussian` (white noise with the training-set noise level) are included. A useful model has to beat `gaussian` on texture (NPS), not only on noise magnitude.

## Metrics (`results/benchmark_table.csv`, `benchmark_FBP.png`)

**Main table metrics**

| metric | definition | better |
|---|---|---|
| `rmse` | RMSE in HU vs real LDCT, whole image | ↓ |
| `psnr` | PSNR vs real LDCT in HU window `--win` (default soft tissue [-160, 240]) | ↑ |
| `ssim` | SSIM vs real LDCT, same window | ↑ |
| `lpips` | LPIPS (AlexNet) on the windowed image, needs `pip install lpips` | ↓ |
| `nps_mae` | mean \|NPS_gen(f) − NPS_real(f)\| over radial frequency bins (HU²·mm²) | ↓ |
| `nps_l2` | ‖NPS_gen − NPS_real‖₂ over radial bins (HU²·mm²) | ↓ |
| `df_peak` | \|f_peak,gen − f_peak,real\| (cycles/mm), peak located with sub-bin parabolic fit | ↓ |

**Extra diagnostics:** `noise_std_err` (relative noise-level error), `nps_shape_err` (texture only, NPS normalised to unit area), `f_mean_err` (mean-frequency error), `abs_hu_bias` (HU shift in soft tissue), and `lf_psnr` (PSNR after Gaussian blur, so it checks anatomy and ignores noise).

How these are computed:
* **NPS** is measured on the noise image `n = image − NDCT`, using 64×64 ROIs placed in flat soft tissue. The same ROIs are used for every method and for the real LDCT. The radial NPS is averaged over all test slices, which gives one NPS value per method and fold. Use `--folds 0 1 2` to get mean ± std.
* **Pixel and perceptual metrics** (RMSE/PSNR/SSIM/LPIPS) are averaged over test slices. A synthetic LDCT carries a *different noise realisation* from the real one, so these metrics favour smooth outputs; the `identity` baseline often scores well on them. That is why NPS-based metrics belong in the same table.
* **PSNR/SSIM window:** `run_benchmark.py --win -1024 2048` computes them on the full HU range instead of soft tissue. State in your paper which window you used.

Each `results/<pair>_f<fold>/` folder also holds `nps.png` (NPS curves of all methods vs real), `example_slice*.png` (images and noise maps side by side) and per-slice CSVs.

## Diffusion models

| name | script | what it is |
|---|---|---|
| `cadm` | `train_cadm.py` | **PhysHVS-LDCT**. One dose-controllable model for all target doses. Stable Diffusion 1.5 (VAE, text encoder and U-Net frozen) + 3 ControlNet branches: NDCT, PARM_d + log-dose channel, CSF perceptual map. Fused as r_t = g_s + λn·g_n + λc·g_c, with a LoRA in the U-Net attention and no text conditioning. |
| `cddpm` | `train_cddpm.py` | Classical conditional DDPM (pixel space, [x_t, NDCT] → ε). No priors; **one model per dose**. |

Priors (`priors.py`):
* **PARM:** NDCT → µ → Radon → Poisson(r·I0·e^(−p)) + electronic noise → −log → FBP; PARM = NDCT − LDCT_sim. This runs on the GPU and is recomputed with fresh noise at every training step.
* **Perceptual map:** |F⁻¹(F(NDCT) · CSF_Barten)|, computed on the windowed NDCT.

To look at the priors for one slice: `python priors.py --data data/piglet --slice 400`.

### Setup (new environment; the diffusion libraries need Python ≥ 3.10)
```bash
conda create -n ldctdiff python=3.10 -y && conda activate ldctdiff
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# once, on a machine with internet (e.g. the frontend):
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --include "vae/*" "unet/*" "text_encoder/*" "tokenizer/*" "scheduler/*" "model_index.json" \
  --local-dir ~/models/sd15
```

### Run (same doses, recon and fold as the GAN benchmark)
```bash
# PhysHVS-LDCT: one model for 10/25/50 %, then sample each dose's test slices
python train_cadm.py train  --data data/piglet --recon FBP --doses 10 25 50 --sd_path ~/models/sd15
python train_cadm.py sample --ckpt cadm_runs/cadm_FBP_s-n-c_f0/final --doses 10 25 50
# classical diffusion, one model per dose (trains then samples)
python train_cddpm.py --data data/piglet --recon FBP --doses 10 25 50
# evaluate everything together (GAN models are not retrained if runs/ already has them)
python run_benchmark.py --data data/piglet --recon FBP --doses 10 25 50 --quick \
    --models cyclegan cut vae cadm cddpm --baselines zeng
```
Use the same `--runs` folder (and `--tag`) as your GAN runs. The predictions go to `runs/cadm_*` and `runs/cddpm_*`.

**Ablations of the priors:** `train_cadm.py train ... --branches s n --tag noPM` trains without the perceptual map; `--branches s` is a plain ControlNet on NDCT. The resulting predictions appear as `cadm_sn`, `cadm_s`, etc., which you add to `--models` when running the benchmark.

**Dose controllability:** train with `--doses 10 25`, then sample `--doses 50` to test a dose the model never saw.

**Scoring notes:**
* The diffusion models work in the HU window [-150, 250]. When you evaluate them together with the GAN models, everything (all methods and the real LDCT) is scored on the common range [-150, 250], which is fair to every method.
* At start-up, `train_cadm.py` prints a **vae-check**: how much of the real LDCT noise survives the frozen SD VAE. This is an upper bound on what CADM can reproduce.

### Normalised [0,1] metrics and DISTS
Every summary also contains `*_01` columns: `rmse_01`, `psnr_01`, `ssim_01`, `lpips_01`, `dists_01`, `nps_mae_01` and `nps_l2_01`. For these, images are mapped linearly to [0,1] over the evaluated HU range (by default the common range of the methods: [-1024, 2048] for the GANs, [-150, 250] when diffusion runs are included), with no display window. Set the range with `--unit_range -1024 2048`. `nps_mae_rel` is the NPS-MAE divided by the mean real NPS, a unitless error (0.1 = 10 %).

**DISTS** (`dists`, `dists_01`, lower is better) comes from `piq`. It compares structure and texture statistics, so it tolerates a different noise realisation better than pixel metrics do.

LPIPS and DISTS download their network weights on first use. If the GPU nodes have no internet, cache the weights once on the frontend:
`python -c "import piq, lpips; piq.DISTS(); lpips.LPIPS(net='alex')"`

## Other reconstructions (ASIR, MBIR)
`prepare_data.py` already converts every series it finds (FBP, ASIR20/40/60, MBIR).

**Same reconstruction for input and target** (100% ASIR40 → 10% ASIR40, etc.): pass several values to `--recon`:
```bash
python run_benchmark.py --data data/piglet --recon FBP ASIR20 ASIR40 ASIR60 --doses 5 10 25 --baselines zeng
python train_cadm.py train  --data data/piglet --recon ASIR40 --doses 5 10 25 --sd_path ~/models/sd15
python train_cddpm.py --data data/piglet --recon ASIR40 --doses 5 10 25
```
**Cross-reconstruction** (FBP full dose → ASIR low dose: synthesize what an iterative reconstruction would show at low dose, starting from a standard FBP scan):
```bash
python run_benchmark.py --data data/piglet --src_recon FBP --recon ASIR40 --doses 5 10 25 --baselines zeng
python train_cadm.py train --data data/piglet --src_recon FBP --recon ASIR40 --doses 5 10 25 --sd_path ~/models/sd15
```
`benchmark_table.csv` gets one row group per reconstruction (`ASIR40`, or `FBP->ASIR40` for cross-reconstruction), and one plot per group (`benchmark_ASIR40.png`, `benchmark_FBP_to_ASIR40.png`).

Two notes for interpreting ASIR/MBIR results:
* **The Zeng baseline and the CADM physics prior both reconstruct their noise with FBP.** For ASIR/MBIR targets they get the noise *level* right after calibration, but with FBP *texture*. The gap between Zeng's NPS error on FBP and on ASIR measures how much the reconstruction itself changes the noise.
* **One model per reconstruction:** CADM learns the reconstruction-specific texture from the data, but each reconstruction needs its own model. A reconstruction-conditioned CADM (one model for all doses and reconstructions) is a natural extension.

## Ablation study of PhysHVS-LDCT
`run_ablation.py` trains, samples and scores each variant with the same split and metrics as the benchmark. The full model (`cadm`) is reused if it already exists.

| name in table | change vs the full model | question it answers |
|---|---|---|
| `abl_no_pm` | no perceptual branch | what does the HVS prior add? |
| `abl_no_parm` | no physics branch (dose code moves to the NDCT branch) | what does the physics prior add? |
| `abl_ndct_only` | NDCT + dose code only | both priors removed |
| `abl_no_dose` | no explicit log-dose map | does PARM alone carry the dose? |
| `abl_parm_noise` | PARM = FBP(noisy) − FBP(clean) | definition of the physics prior |
| `abl_lc_0`, `abl_lc_1` | λc = 0 / 1 (default 0.5) | fusion weight of the perceptual branch |
| `abl_per_dose` | one model per dose instead of one for all | value of dose-controllable training |

```bash
python run_ablation.py --data data/piglet --recon FBP --doses 5 10 25 --sd_path ~/models/sd15 \
    --iters 3000 --steps 25 --runs runs_v3
python run_ablation.py ... --configs no_pm no_parm ndct_only no_dose      # a subset first
```
Outputs (in `results_v3_ablation/`):
* `ablation_table.csv`: all metrics per variant and dose
* `ablation_delta.csv`: difference to the full model, signed so that **> 0 means the variant is better**
* the usual figures

Use the same `--iters` as your full model, otherwise the comparison is not fair.

### Brightness of the diffusion models
* **Sampler:** both diffusion models sample with DDIM "trailing" timestep spacing, so the first step is t = T−1 and matches the pure-noise start. The SD default ("leading") starts near t ≈ 960, which biases the generated brightness (Lin et al., WACV 2024).
* **Classical DDPM:** `train_cddpm.py` diffuses the residual LDCT − NDCT by default (`--target residual`), so its HU values stay anchored to the NDCT. `--target image` gives the plain image-conditioned version.
* **Optional low-frequency anchoring:** `--lf_anchor 4`, at sampling time, for both models. It keeps the generated noise and streaks but takes the low frequencies (σ = 4 px) from the NDCT. Results are saved as `cadm_lf` / `cddpm_lf`, so you can report them next to the raw outputs.

## Citation
If you use this code, please cite our paper:

```bibtex
@inproceedings{taifour2026physhvs,
  title     = {PhysHVS-LDCT: Dose-Controllable Physics-Based Low-Dose CT Synthesis for Benchmarking and Data Augmentation},
  author    = {Taifour, Y. and Tliba, M. and Aburaed, N. and Beghdadi, A. and Cheikh, F. A. and others},
  booktitle = {2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026}
}
```
