"""
Experiment 11: CatGAN-26, a 128x128 cat-face generator trained with 2026 practice.

The generator deliberately stays in the 2018 op vocabulary (Dense -> reshape -> [UpSampling2D,
Conv2D, BatchNorm, LeakyReLU] -> Conv2D), so every weightpatch tool built for the legacy
Anime-DCGAN applies unchanged. What changes is the training around it:
  - AFHQ cats (5,153 clean 512x512 photos, downsampled to 128x128 with Lanczos)
  - residual discriminator with minibatch standard deviation, regularised by a lazy R1 gradient
    penalty (no spectral norm: with it, this discriminator's output range collapsed to ~1e-5)
  - non-saturating logistic loss, Adam(0, 0.99), batch 32, bf16 mixed precision
  - differentiable augmentation (colour, translation, cutout) on real and fake images,
    which keeps the discriminator from memorising 5k images
  - an exponential moving average (EMA) of the generator, with BatchNorm statistics recomputed
  - linear output (no tanh), so output-layer colour folds stay exact

Exports the EMA generator with BatchNorm folded into the preceding layers as a Keras-style dict
to results/exp11_out/catgan26.npz (checked against modern_np.catgan_np / catgan_torch).

  <venv python> src/exp11_catgan26.py --steps 40000        (resumes from results/exp11_out/catgan26.pt)
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modern_np as mnp

ROOT = "data/afhq/data/afhq"; OUT = "results/exp11_out"; RES, ZDIM = 128, 128
R1_GAMMA, R1_EVERY = 0.1, 16                 # gamma = 0.0002 * res^2 / batch (StyleGAN2-ADA heuristic), lazy every 16 steps
SN = lambda m: m                             # no spectral norm; kept as a hook to compare

def load_cats(split="train"):
    cache = os.path.join(OUT, f"afhq_cat_{split}_{RES}.npy")
    if os.path.exists(cache): return np.load(cache)
    d = os.path.join(ROOT, split, "cat")
    a = np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB").resize((RES, RES), Image.LANCZOS)) for f in sorted(os.listdir(d))])
    np.save(cache, a); return a

class Generator(nn.Module):
    def __init__(self, widths=(512, 512, 256, 128, 64, 32)):
        super().__init__()
        self.dense = nn.Linear(ZDIM, widths[0] * 16); self.bn0 = nn.BatchNorm2d(widths[0])
        self.blocks = nn.ModuleList(nn.ModuleList([nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o)])
                                    for i, o in zip(widths[:-1], widths[1:]))
        self.out = nn.Conv2d(widths[-1], 3, 3, padding=1)
    def forward(self, z):
        x = F.leaky_relu(self.bn0(self.dense(z).view(len(z), -1, 4, 4)), 0.2)
        for c1, b1, c2, b2 in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.leaky_relu(b2(c2(F.leaky_relu(b1(c1(x)), 0.2))), 0.2)
        return self.out(x)

class DBlock(nn.Module):
    def __init__(self, i, o, first=False):
        super().__init__(); self.first = first
        self.c1, self.c2, self.sc = SN(nn.Conv2d(i, o, 3, padding=1)), SN(nn.Conv2d(o, o, 3, padding=1)), SN(nn.Conv2d(i, o, 1))
    def forward(self, x):
        h = x if self.first else F.leaky_relu(x, 0.2)
        return (F.avg_pool2d(self.c2(F.leaky_relu(self.c1(h), 0.2)), 2) + self.sc(F.avg_pool2d(x, 2))) / 2 ** 0.5

class Discriminator(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256, 512)):
        super().__init__()
        self.blocks = nn.Sequential(*[DBlock(i, o, k == 0) for k, (i, o) in enumerate(zip((3,) + widths[:-1], widths))])
        self.conv = SN(nn.Conv2d(widths[-1] + 1, widths[-1], 3, padding=1)); self.fc = SN(nn.Linear(widths[-1] * 16, 1))
        for m in self.modules():                                                    # He init for LeakyReLU(0.2): keeps signal scale
            if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(m.weight, a=0.2); nn.init.zeros_(m.bias)
    def forward(self, x):
        h = F.leaky_relu(self.blocks(x), 0.2)
        k = 4 if len(h) % 4 == 0 else 1                                                # minibatch std, groups of 4
        sd = (h.view(k, -1, *h.shape[1:]).var(0, unbiased=False) + 1e-8).sqrt().mean((1, 2, 3)).repeat(k)
        h = torch.cat([h, sd[:, None, None, None].expand(-1, 1, *h.shape[2:])], 1)
        return self.fc(F.leaky_relu(self.conv(h), 0.2).flatten(1)).squeeze(1)

def diffaug(x):
    """Colour, translation (1/8) and cutout (1/2), with fresh randomness per image."""
    n, _, h, w = x.shape; dev = x.device; r = lambda: torch.rand(n, 1, 1, 1, device=dev)
    x = x + (r() - 0.5)
    m = x.mean(1, keepdim=True); x = (x - m) * (2 * r()) + m
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * (r() + 0.5) + m
    s = int(h * 0.125 + 0.5); ar = torch.arange(n, device=dev)[:, None, None]
    ty, tx = torch.randint(-s, s + 1, (n, 1, 1), device=dev), torch.randint(-s, s + 1, (n, 1, 1), device=dev)
    gy, gx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    xp = F.pad(x, [1, 1, 1, 1]).permute(0, 2, 3, 1)
    x = xp[ar, (gy[None] + ty + 1).clamp(0, h + 1), (gx[None] + tx + 1).clamp(0, w + 1)].permute(0, 3, 1, 2)
    c = h // 2; oy, ox = torch.randint(0, h + 1 - c % 2, (n, 1, 1), device=dev), torch.randint(0, w + 1 - c % 2, (n, 1, 1), device=dev)
    cy, cx = torch.meshgrid(torch.arange(c, device=dev), torch.arange(c, device=dev), indexing="ij")
    mask = torch.ones(n, h, w, device=dev)
    mask[ar, (cy[None] + oy - c // 2).clamp(0, h - 1), (cx[None] + ox - c // 2).clamp(0, w - 1)] = 0
    return (x * mask[:, None]).contiguous(memory_format=torch.channels_last)

@torch.no_grad()
def recalibrate_bn(g, dev, batches=20, bs=64):
    bns = [m for m in g.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns: m.reset_running_stats(); m.momentum = None                       # cumulative average
    g.train()
    for _ in range(batches): g(torch.randn(bs, ZDIM, device=dev))
    for m in bns: m.momentum = 0.1
    g.eval()

def fold_export(g):
    """Fold every eval-mode BatchNorm into the layer before it, reorder the dense columns from torch's
    view(C,4,4) to Keras' Reshape((4,4,C)), and fold y/2 + 1/2 into the output conv so images come out in
    [0,1] like the legacy generators. Returns a Keras-style dict (HWIO kernels); every step is exact."""
    def scale_shift(bn): s = bn.weight / (bn.running_var + bn.eps).sqrt(); return s, bn.bias - bn.running_mean * s
    W = {}; s, t = scale_shift(g.bn0); s16, t16 = s.repeat_interleave(16), t.repeat_interleave(16)
    C0 = g.bn0.num_features; keras_order = torch.arange(C0 * 16).view(C0, 4, 4).permute(1, 2, 0).reshape(-1)
    W["dense/kernel"] = (g.dense.weight.T * s16)[:, keras_order]; W["dense/bias"] = (g.dense.bias * s16 + t16)[keras_order]
    for i, (c1, b1, c2, b2) in enumerate(g.blocks, 1):
        for j, (c, b) in enumerate(((c1, b1), (c2, b2)), 1):
            s, t = scale_shift(b); W[f"b{i}c{j}/kernel"] = (c.weight * s[:, None, None, None]).permute(2, 3, 1, 0); W[f"b{i}c{j}/bias"] = c.bias * s + t
    W["out/kernel"], W["out/bias"] = g.out.weight.permute(2, 3, 1, 0) / 2, g.out.bias / 2 + 0.5
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in W.items()}

def save_grid(x, path, nrow=8):
    a = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy(); n, h, w, _ = a.shape
    Image.fromarray(a.reshape(n // nrow, nrow, h, w, 3).transpose(0, 2, 1, 3, 4).reshape(n // nrow * h, nrow * w, 3)).save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-only", metavar="NAME", help="export the current checkpoint's EMA generator to results/exp11_out/NAME and exit")
    a = ap.parse_args(); os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
    if a.export_only:
        G_ema = Generator().to("cuda"); c = torch.load(os.path.join(OUT, "catgan26.pt"), map_location="cuda")
        G_ema.load_state_dict(c["G_ema"]); recalibrate_bn(G_ema, "cuda", batches=50)
        np.savez(os.path.join(OUT, a.export_only), **fold_export(G_ema)); print(f"exported step {c['step']} to {a.export_only}"); return
    dev = "cuda"; torch.manual_seed(a.seed); torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    data = torch.from_numpy(load_cats()).permute(0, 3, 1, 2).contiguous().to(dev)     # uint8 on the GPU
    G, D = Generator().to(dev, memory_format=torch.channels_last), Discriminator().to(dev, memory_format=torch.channels_last)
    G_ema = copy.deepcopy(G).eval()
    oG = torch.optim.Adam(G.parameters(), a.lr, betas=(0.0, 0.99)); oD = torch.optim.Adam(D.parameters(), a.lr, betas=(0.0, 0.99))
    ck = os.path.join(OUT, "catgan26.pt"); step = 0; log = []
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev)
        for m, k in ((G, "G"), (D, "D"), (G_ema, "G_ema"), (oG, "oG"), (oD, "oD")): m.load_state_dict(c[k])
        step, log = c["step"], c["log"]; print(f"resumed at step {step}", flush=True)
    print(f"G {sum(p.numel() for p in G.parameters()):,} params, D {sum(p.numel() for p in D.parameters()):,} params, {len(data)} images", flush=True)
    z_fixed = torch.randn(64, ZDIM, generator=torch.Generator().manual_seed(123)).to(dev)
    def real(n):
        x = data[torch.randint(0, len(data), (n,), device=dev)].float() / 127.5 - 1
        return torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.5, x.flip(3), x).contiguous(memory_format=torch.channels_last)
    t0, B, run = time.time(), a.batch, np.zeros(4)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)                      # bf16 compute, fp32 weights
    while step < a.steps:
        with torch.no_grad(), amp(): fake = G(torch.randn(B, ZDIM, device=dev))
        with amp(): dr, df = D(diffaug(real(B))).float(), D(diffaug(fake)).float()
        lD = F.softplus(-dr).mean() + F.softplus(df).mean()
        if step % R1_EVERY == 0:                                                     # R1 on augmented reals, fp32
            xr = diffaug(real(B // 2)).detach().requires_grad_(True)
            g, = torch.autograd.grad(D(xr).sum(), xr, create_graph=True)
            lD = lD + R1_GAMMA / 2 * g.pow(2).sum((1, 2, 3)).mean() * R1_EVERY
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
        with amp(): lG = F.softplus(-D(diffaug(G(torch.randn(B, ZDIM, device=dev)))).float()).mean()
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step(); step += 1
        with torch.no_grad():
            beta = min(0.999, (1 + step) / (10 + step))
            for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 1 - beta)
        run = 0.98 * run + 0.02 * np.array([lD.item(), lG.item(), dr.mean().item(), df.mean().item()])
        if step % 200 == 0:
            print(f"step {step}  D {run[0]:.4f}  G {run[1]:.4f}  D(real) {run[2]:+.4f}  D(fake) {run[3]:+.4f}  gap {run[2] - run[3]:.4f}  "
                  f"{(time.time() - t0) / 200:.3f}s/step", flush=True); t0 = time.time()
        if step % 2000 == 0 or step == a.steps:
            g = copy.deepcopy(G_ema); recalibrate_bn(g, dev)
            with torch.no_grad(): save_grid(g(z_fixed), os.path.join(OUT, "samples", f"step{step:06d}.png"))
            log.append({"step": step, **{k: float(v) for k, v in zip(("D", "G", "D_real", "D_fake"), run)}})
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(),
                        "step": step, "log": log}, ck)
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False           # full fp32 from here
    g = copy.deepcopy(G_ema).float().to(memory_format=torch.contiguous_format); recalibrate_bn(g, dev, batches=50)
    W = fold_export(g); np.savez(os.path.join(OUT, "catgan26.npz"), **W)
    json.dump(mnp.catgan_ops(W), open(os.path.join(OUT, "catgan26_ops.json"), "w"), indent=1)
    z = torch.randn(4, ZDIM, device=dev)
    with torch.no_grad():
        ref = g(z) / 2 + 0.5; d_t = float((mnp.catgan_torch(z, mnp.to_torch(W)) - ref).abs().max())
    d_np = float(np.abs(mnp.catgan_np(z.cpu().numpy(), W) - ref.permute(0, 2, 3, 1).cpu().numpy()).max())
    info = {"steps": step, "params_G_folded": int(sum(v.size for v in W.values())), "export_check": {"numpy": d_np, "torch_functional": d_t}, "log": log}
    json.dump(info, open(os.path.join(OUT, "catgan26.json"), "w"), indent=1); print(json.dumps(info["export_check"]), flush=True)

if __name__ == "__main__":
    main()
