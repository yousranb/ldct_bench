"""Constraint-Aware Diffusion Model (CADM), thesis Section 3.5.1.

* Stable Diffusion 1.5: VAE, text encoder and U-Net are frozen.
* Three ControlNet branches, each a trainable copy of the U-Net encoder:
      g_s : NDCT image                 (structural prior)
      g_n : PARM_d (+ log-dose channel) (physics prior, carries the dose)
      g_c : perceptual map PM          (HVS prior)
  fused as   r_t = g_s + lambda_n * g_n + lambda_c * g_c          (Eq. 3.26)
  and injected into the frozen U-Net as additive residuals.
* A LoRA adapter in the U-Net attention blocks is trained jointly.
* Text conditioning is disabled: the U-Net always receives the empty-prompt
  embedding (no prompt information).
* Objective: epsilon prediction in latent space (Eqs. 3.27-3.30).

`tiny=True` builds small randomly initialised components (no download) - only for
testing the code path.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BRANCH_CHANNELS = {"s": 1, "n": 2, "c": 1}


class CADM(nn.Module):
    def __init__(self, sd_path="stable-diffusion-v1-5/stable-diffusion-v1-5", branches=("s", "n", "c"),
                 lambda_n=1.0, lambda_c=0.5, lora_rank=16, tiny=False, gradient_checkpointing=True,
                 channels=None, residual=False, res_scale=1.0):
        super().__init__()
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig
        self.branches = list(branches)
        self.residual, self.res_scale = residual, float(res_scale)
        channels = channels or BRANCH_CHANNELS
        self.weights = {"s": 1.0, "n": lambda_n, "c": lambda_c}
        if tiny:
            self.vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                                     down_block_types=("DownEncoderBlock2D",) * 2,
                                     up_block_types=("UpDecoderBlock2D",) * 2,
                                     block_out_channels=(16, 32), layers_per_block=1, norm_num_groups=8)
            self.unet = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels=4,
                                             block_out_channels=(32, 64), layers_per_block=1,
                                             down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                             up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                                             cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8)
            self.register_buffer("ehs", torch.zeros(1, 4, 32))
            self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                                           beta_start=0.00085, beta_end=0.012)
            self.scaling = 1.0
        else:
            from transformers import CLIPTextModel, CLIPTokenizer
            self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
            tok = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
            te = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
            with torch.no_grad():
                ids = tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
                self.register_buffer("ehs", te(ids)[0])       # empty prompt -> no text information
            del te
            self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
            self.scaling = self.vae.config.scaling_factor
        self.vae_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        n_emb = int(math.log2(self.vae_factor)) + 1
        emb = (16, 32, 96, 256)[-n_emb:] if n_emb <= 4 else (16, 32, 96, 256)
        self.controlnets = nn.ModuleDict({
            b: ControlNetModel.from_unet(self.unet, conditioning_channels=channels[b],
                                         conditioning_embedding_out_channels=emb)
            for b in self.branches})
        # LoRA added AFTER copying the encoder into the ControlNets
        self.unet.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_rank, init_lora_weights="gaussian",
                                         target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        if gradient_checkpointing and not tiny:
            self.unet.enable_gradient_checkpointing()
            for c in self.controlnets.values():
                c.enable_gradient_checkpointing()

    # ------------------------------------------------------------------ utils
    def trainable_groups(self, lr_control=1e-5, lr_lora=1e-4):
        lora = [p for p in self.unet.parameters() if p.requires_grad]
        ctrl = [p for c in self.controlnets.values() for p in c.parameters()]
        return [dict(params=ctrl, lr=lr_control), dict(params=lora, lr=lr_lora)]

    def lora_state_dict(self):
        from peft.utils import get_peft_model_state_dict
        return get_peft_model_state_dict(self.unet)

    def load_lora_state_dict(self, sd):
        from peft.utils import set_peft_model_state_dict
        set_peft_model_state_dict(self.unet, sd)

    @torch.no_grad()
    def encode(self, x, sample=True):   # x: (B,1,H,W) in [-1,1]
        dist = self.vae.encode(x.repeat(1, 3, 1, 1)).latent_dist
        return (dist.sample() if sample else dist.mode()) * self.scaling

    @torch.no_grad()
    def decode(self, z):
        return self.vae.decode(z / self.scaling).sample.mean(1, keepdim=True).clamp(-1, 1)

    def _residuals(self, zt, t, ehs, conds):
        down, mid = None, None
        for b in self.branches:
            d, m = self.controlnets[b](zt, t, encoder_hidden_states=ehs, controlnet_cond=conds[b],
                                       return_dict=False)
            w = self.weights[b]
            if down is None:
                down, mid = [w * x for x in d], w * m
            else:
                down = [a + w * x for a, x in zip(down, d)]
                mid = mid + w * m
        return down, mid

    def eps(self, zt, t, conds):
        ehs = self.ehs.expand(zt.shape[0], -1, -1).to(zt.dtype)
        down, mid = self._residuals(zt, t, ehs, conds)
        return self.unet(zt, t, encoder_hidden_states=ehs, down_block_additional_residuals=down,
                         mid_block_additional_residual=mid).sample

    def loss(self, x_ld, conds, x_nd=None):
        if self.residual:           # diffuse the (scaled) latent change NDCT -> LDCT
            z = (self.encode(x_ld, sample=False) - self.encode(x_nd, sample=False)) * self.res_scale
        else:
            z = self.encode(x_ld)
        noise = torch.randn_like(z)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (z.shape[0],), device=z.device)
        zt = self.scheduler.add_noise(z, noise, t)
        return F.mse_loss(self.eps(zt, t, conds).float(), noise.float())

    @torch.no_grad()
    def sample(self, conds, shape_hw, steps=50, generator=None, x_nd=None):
        from diffusers import DDIMScheduler
        # "trailing" spacing: the first step is t = T-1, matching the pure-noise start.
        # The SD default ("leading") starts at t~T-T/steps, which biases the image brightness
        # (Lin et al., "Common diffusion noise schedules and sample steps are flawed", 2024).
        sch = DDIMScheduler.from_config(self.scheduler.config, timestep_spacing="trailing")
        sch.set_timesteps(steps)
        B = next(iter(conds.values())).shape[0]
        H, W = shape_hw
        dev = next(iter(conds.values())).device
        z = torch.randn((B, self.unet.config.in_channels, H // self.vae_factor, W // self.vae_factor),
                        generator=generator, device=dev)
        for t in sch.timesteps:
            e = self.eps(z, t.to(dev).expand(B), conds)
            z = sch.step(e.float(), t, z).prev_sample
        if self.residual:
            # pixel-domain residual: anatomy from the NDCT pixels, VAE reconstruction errors cancel
            z_nd = self.encode(x_nd, sample=False)
            return x_nd + self.decode(z_nd + z / self.res_scale) - self.decode(z_nd)
        return self.decode(z)
