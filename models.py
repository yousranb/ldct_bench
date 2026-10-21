"""CycleGAN, CUT and conditional VAE for NDCT -> LDCT synthesis.

All models share the same ResNet generator backbone (so differences come from the
training objective, not capacity) and the same PatchGAN discriminator.
Common interface:
    m.train_step(a, b, it, total) -> dict of floats
    m.translate(a)                -> synthetic LDCT in network units
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- blocks


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.b(x)


class ResnetGenerator(nn.Module):
    """Johnson-style generator (as in CycleGAN/CUT).
    residual=True: output = input + learned residual. This makes the task
    'add realistic dose-dependent noise' and keeps HU values anchored."""

    def __init__(self, in_c=1, out_c=1, ngf=64, n_blocks=9, residual=True, extra_in=0):
        super().__init__()
        self.residual = residual
        L = [nn.ReflectionPad2d(3), nn.Conv2d(in_c + extra_in, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        c = ngf
        for _ in range(2):
            L += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        L += [ResBlock(c) for _ in range(n_blocks)]
        for _ in range(2):
            L += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        L += [nn.ReflectionPad2d(3), nn.Conv2d(c, out_c, 7)]
        self.model = nn.ModuleList(L)
        # layers whose outputs CUT uses for PatchNCE (input, 2 downsamples, 2 resblocks)
        self.nce_layers = [0, 4, 8, 12, 16] if n_blocks >= 7 else [0, 4, 8, 11, 14]

    def forward(self, x, cond=None, layers=None):
        h = x if cond is None else torch.cat([x, cond], 1)
        feats = []
        for i, layer in enumerate(self.model):
            h = layer(h)
            if layers is not None and i in layers:
                feats.append(h)
                if i == layers[-1]:
                    return feats
        out = x + h if self.residual else torch.tanh(h)
        return out.clamp(-1, 1)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN. With hp_scale > 0 (default) it is 'noise-aware': it sees the
    image AND an amplified high-pass copy (x - 3x3 local mean). CT noise is only
    ~10-50 HU, i.e. ~1 % of the network's [-1024, 2048] HU range, and a plain
    discriminator cannot see it (it gets stuck at D(x)=0.5 and the generator
    stays the identity). hp_scale=0 gives the standard discriminator."""

    def __init__(self, in_c=1, ndf=64, n_layers=3, hp_scale=30.0):
        super().__init__()
        self.hp_scale = hp_scale
        in_c = in_c * 2 if hp_scale > 0 else in_c
        L = [nn.Conv2d(in_c, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = ndf
        for i in range(1, n_layers + 1):
            c2 = ndf * min(2 ** i, 8)
            L += [nn.Conv2d(c, c2, 4, 2 if i < n_layers else 1, 1), nn.InstanceNorm2d(c2), nn.LeakyReLU(0.2, True)]
            c = c2
        L += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        if self.hp_scale > 0:
            hp = x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
            x = torch.cat([x, self.hp_scale * hp], 1)
        return self.net(x)


def gan_loss(pred, real):  # LSGAN
    return F.mse_loss(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ImagePool:
    def __init__(self, size=50):
        self.size, self.imgs = size, []

    def query(self, x):
        out = []
        for im in x.detach():
            im = im[None]
            if len(self.imgs) < self.size:
                self.imgs.append(im)
                out.append(im)
            elif torch.rand(1).item() < 0.5:
                j = torch.randint(len(self.imgs), (1,)).item()
                out.append(self.imgs[j].clone())
                self.imgs[j] = im
            else:
                out.append(im)
        return torch.cat(out)


class BaseModel(nn.Module):
    def lr_lambda(self, total):
        # constant for first half, linear decay to 0 over second half
        return lambda it: 1.0 if it < total // 2 else max(0.0, 2.0 * (1 - it / total))

    def make_schedulers(self, total):
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, self.lr_lambda(total)) for o in self.opts]

    def step_schedulers(self):
        for s in self.scheds:
            s.step()

    @torch.no_grad()
    def translate(self, a):
        return self.G(a)


# ------------------------------------------------------------------- CycleGAN


class CycleGAN(BaseModel):
    def __init__(self, ngf=64, n_blocks=9, lr=2e-4, lambda_cyc=10.0, lambda_idt=0.5, residual=True, d_hp=30.0):
        super().__init__()
        self.G = ResnetGenerator(ngf=ngf, n_blocks=n_blocks, residual=residual)   # A(NDCT)->B(LDCT)
        self.F = ResnetGenerator(ngf=ngf, n_blocks=n_blocks, residual=residual)   # B->A
        self.DA, self.DB = PatchDiscriminator(hp_scale=d_hp), PatchDiscriminator(hp_scale=d_hp)
        self.apply(init_weights)
        self.lc, self.li = lambda_cyc, lambda_idt
        self.optG = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()), lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(itertools.chain(self.DA.parameters(), self.DB.parameters()), lr, betas=(0.5, 0.999))
        self.opts = [self.optG, self.optD]
        self.poolA, self.poolB = ImagePool(), ImagePool()

    def train_step(self, a, b):
        fb, fa = self.G(a), self.F(b)
        for p in itertools.chain(self.DA.parameters(), self.DB.parameters()):
            p.requires_grad_(False)
        l_gan = gan_loss(self.DB(fb), True) + gan_loss(self.DA(fa), True)
        l_cyc = F.l1_loss(self.F(fb), a) + F.l1_loss(self.G(fa), b)
        l_idt = F.l1_loss(self.G(b), b) + F.l1_loss(self.F(a), a) if self.li > 0 else torch.zeros((), device=a.device)
        lG = l_gan + self.lc * l_cyc + self.lc * self.li * l_idt
        self.optG.zero_grad(set_to_none=True)
        lG.backward()
        self.optG.step()
        for p in itertools.chain(self.DA.parameters(), self.DB.parameters()):
            p.requires_grad_(True)
        fb_, fa_ = self.poolB.query(fb), self.poolA.query(fa)
        lD = 0.5 * (gan_loss(self.DB(b), True) + gan_loss(self.DB(fb_), False)) \
            + 0.5 * (gan_loss(self.DA(a), True) + gan_loss(self.DA(fa_), False))
        self.optD.zero_grad(set_to_none=True)
        lD.backward()
        self.optD.step()
        return dict(G_gan=l_gan.item(), cyc=l_cyc.item(), idt=l_idt.item(), D=lD.item())


# ------------------------------------------------------------------------ CUT


class PatchSampleF(nn.Module):
    def __init__(self, in_channels, nc=256):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(c, nc), nn.ReLU(), nn.Linear(nc, nc)) for c in in_channels])

    def forward(self, feats, num_patches=256, patch_ids=None):
        out, ids = [], []
        for k, f in enumerate(feats):
            B, C, H, W = f.shape
            f = f.flatten(2).permute(0, 2, 1)  # B, HW, C
            if patch_ids is not None:
                pid = patch_ids[k]
            else:
                pid = torch.randperm(H * W, device=f.device)[:min(num_patches, H * W)]
            x = self.mlps[k](f[:, pid, :].flatten(0, 1))
            out.append(F.normalize(x, dim=1))
            ids.append(pid)
        return out, ids


def patch_nce(q, k, B, tau=0.07):
    # q,k: (B*P, C). positives = same location, negatives = other locations of same image
    P = q.shape[0] // B
    q, k = q.view(B, P, -1), k.view(B, P, -1).detach()
    logits = torch.bmm(q, k.transpose(1, 2)) / tau  # B,P,P
    target = torch.arange(P, device=q.device).repeat(B)
    return F.cross_entropy(logits.flatten(0, 1), target)


class CUT(BaseModel):
    def __init__(self, ngf=64, n_blocks=9, lr=2e-4, lambda_nce=1.0, nce_idt=True, num_patches=256, residual=True,
                 d_hp=30.0):
        super().__init__()
        self.G = ResnetGenerator(ngf=ngf, n_blocks=n_blocks, residual=residual)
        self.D = PatchDiscriminator(hp_scale=d_hp)
        self.apply(init_weights)
        with torch.no_grad():
            chans = [f.shape[1] for f in self.G(torch.zeros(1, 1, 64, 64), layers=self.G.nce_layers)]
        self.Fm = PatchSampleF(chans)
        self.Fm.apply(init_weights)
        self.ln, self.idt, self.np = lambda_nce, nce_idt, num_patches
        self.optG = torch.optim.Adam(itertools.chain(self.G.parameters(), self.Fm.parameters()), lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr, betas=(0.5, 0.999))
        self.opts = [self.optG, self.optD]

    def _nce(self, src, tgt):
        L = self.G.nce_layers
        fq = self.G(tgt, layers=L)
        fk = self.G(src, layers=L)
        k, ids = self.Fm(fk, self.np)
        q, _ = self.Fm(fq, self.np, ids)
        return sum(patch_nce(qi, ki, src.shape[0]) for qi, ki in zip(q, k)) / len(L)

    def train_step(self, a, b):
        fb = self.G(a)
        # D
        lD = 0.5 * (gan_loss(self.D(b), True) + gan_loss(self.D(fb.detach()), False))
        self.optD.zero_grad(set_to_none=True)
        lD.backward()
        self.optD.step()
        # G
        for p in self.D.parameters():
            p.requires_grad_(False)
        l_gan = gan_loss(self.D(fb), True)
        l_nce = self._nce(a, fb)
        if self.idt:
            l_nce = 0.5 * (l_nce + self._nce(b, self.G(b)))
        lG = l_gan + self.ln * l_nce
        self.optG.zero_grad(set_to_none=True)
        lG.backward()
        self.optG.step()
        for p in self.D.parameters():
            p.requires_grad_(True)
        return dict(G_gan=l_gan.item(), nce=l_nce.item(), D=lD.item())


# ----------------------------------------------------------------------- cVAE


class ConditionalVAE(BaseModel):
    """Paired conditional VAE: models p(LDCT | NDCT) through a spatial latent z.

    Encoder q(z | NDCT, LDCT-NDCT) -> z at 1/2 resolution (z_ch channels).
    Decoder = ResNet generator taking [NDCT, upsampled z] and predicting the
    LDCT as NDCT + residual. At test time z ~ N(0, I), so each call draws a new
    noise realisation. Optional adversarial term (lambda_adv > 0) gives a VAE-GAN.
    NOTE: unlike CycleGAN/CUT this uses paired (co-registered) slices.
    """

    def __init__(self, ngf=64, n_blocks=6, lr=2e-4, z_ch=4, beta=1e-2, lambda_adv=0.0, residual=True, d_hp=30.0):
        super().__init__()
        self.z_ch, self.beta, self.la = z_ch, beta, lambda_adv
        self.enc = nn.Sequential(
            nn.Conv2d(2, ngf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1), nn.InstanceNorm2d(ngf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1), nn.InstanceNorm2d(ngf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, 2 * z_ch, 3, 1, 1))
        self.G = ResnetGenerator(ngf=ngf, n_blocks=n_blocks, residual=residual, extra_in=z_ch)
        self.apply(init_weights)
        self.optG = torch.optim.Adam(itertools.chain(self.enc.parameters(), self.G.parameters()), lr, betas=(0.5, 0.999))
        self.opts = [self.optG]
        if lambda_adv > 0:
            self.D = PatchDiscriminator(hp_scale=d_hp)
            self.D.apply(init_weights)
            self.optD = torch.optim.Adam(self.D.parameters(), lr, betas=(0.5, 0.999))
            self.opts.append(self.optD)

    def _decode(self, a, z):
        return self.G(a, cond=F.interpolate(z, size=a.shape[-2:], mode="nearest"))

    def train_step(self, a, b):
        mu, logvar = self.enc(torch.cat([a, b - a], 1)).chunk(2, 1)
        logvar = logvar.clamp(-10, 10)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        out = self._decode(a, z)
        l_rec = F.l1_loss(out, b)
        l_kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        loss = l_rec + self.beta * l_kl
        logs = dict(rec=l_rec.item(), kl=l_kl.item())
        if self.la > 0:
            for p in self.D.parameters():
                p.requires_grad_(False)
            l_gan = gan_loss(self.D(out), True)
            loss = loss + self.la * l_gan
            logs["G_gan"] = l_gan.item()
        self.optG.zero_grad(set_to_none=True)
        loss.backward()
        self.optG.step()
        if self.la > 0:
            for p in self.D.parameters():
                p.requires_grad_(True)
            lD = 0.5 * (gan_loss(self.D(b), True) + gan_loss(self.D(out.detach()), False))
            self.optD.zero_grad(set_to_none=True)
            lD.backward()
            self.optD.step()
            logs["D"] = lD.item()
        return logs

    @torch.no_grad()
    def translate(self, a):
        H, W = a.shape[-2:]
        z = torch.randn(a.shape[0], self.z_ch, H // 2, W // 2, device=a.device)
        return self._decode(a, z)


def build_model(name, quick=False, **kw):
    nb = 6 if quick else 9
    name = name.lower()
    if name in ("cyclegan", "cut"):
        kw.pop("beta", None)
    if name != "cyclegan":
        kw.pop("lambda_idt", None)
    if name == "cyclegan":
        m = CycleGAN(n_blocks=nb, **kw)
    elif name == "cut":
        m = CUT(n_blocks=nb, **kw)
    elif name == "vae":
        m = ConditionalVAE(n_blocks=6, **kw)
    elif name == "vaegan":
        m = ConditionalVAE(n_blocks=6, lambda_adv=kw.pop("lambda_adv", 0.1), **kw)
    else:
        raise ValueError(name)
    # residual generators start as the identity mapping (last conv = 0):
    # training then only has to learn the dose-dependent noise.
    for g in m.modules():
        if isinstance(g, ResnetGenerator) and g.residual:
            nn.init.zeros_(g.model[-1].weight)
            nn.init.zeros_(g.model[-1].bias)
    return m
