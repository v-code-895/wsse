"""
Experiment 10: TinySR-26, a super-sampler trained with 2026 practice, as the modern counterpart
of TinySR-18 (the 2018 'DLSS' generator.h5: 2x, 317,059 parameters, trained on anime faces).

Same parameter budget or smaller, and the same building blocks (3x3 convs, ReLU), but:
  - residual blocks, a pixel-shuffle upsampler and a global bicubic skip (the net only learns detail)
  - trained on AFHQ (14,630 cat / dog / wildlife photos at 512x512) with mixed downsampling kernels
    (bicubic, bilinear, box, nearest), uint8-quantised LR inputs, flips and rotations
  - L1 loss, Adam with warm-up + cosine decay, an exponential moving average (EMA) of the weights

Exports the EMA weights as a Keras-style dict (`layer/kernel` HWIO, `layer/bias`) to
results/exp10_out/tinysr26_x{scale}.npz, checked against the NumPy forward pass in modern_np.py.

  <venv python> src/exp10_tinysr26.py --scale 2
  <venv python> src/exp10_tinysr26.py --scale 4
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modern_np as mnp

ROOT = "data/afhq/data/afhq"; OUT = "results/exp10_out"; KERNELS = ["bicubic", "bilinear", "area", "nearest"]

def files(split, classes=("cat", "dog", "wild"), per_class=None):
    out = []
    for c in classes:
        fs = sorted(os.listdir(os.path.join(ROOT, split, c)))[:per_class]
        out += [os.path.join(ROOT, split, c, f) for f in fs]
    return out

class Patches(torch.utils.data.Dataset):
    """k random HR patches (uint8, CHW) per decoded image, with random flips and 90° rotations."""
    def __init__(self, paths, size, k=4): self.paths, self.size, self.k = paths, size, k
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        a = np.asarray(Image.open(self.paths[i]).convert("RGB")); rng = np.random.default_rng(); out = []
        for _ in range(self.k):
            y, x = rng.integers(0, a.shape[0] - self.size + 1), rng.integers(0, a.shape[1] - self.size + 1)
            p = a[y:y + self.size, x:x + self.size]
            if rng.random() < 0.5: p = p[:, ::-1]
            out.append(np.ascontiguousarray(np.rot90(p, rng.integers(4)).transpose(2, 0, 1)))
        return torch.from_numpy(np.stack(out))

def degrade(hr, s, kind):
    """Downsample with the same conventions as PIL's resize (used at test time)."""
    if kind == "area": lr = F.avg_pool2d(hr, s)
    elif kind == "nearest": lr = hr[:, :, s // 2::s, s // 2::s]                     # PIL NEAREST samples the block centre
    else: lr = F.interpolate(hr, size=(hr.shape[2] // s, hr.shape[3] // s), mode=kind, antialias=True, align_corners=False)
    return (lr.clamp(0, 1) * 255).round() / 255                                     # LR images are 8-bit

class TinySR(nn.Module):
    def __init__(self, s, ch=48, blocks=6):
        super().__init__(); self.s = s
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(blocks)])
        self.tail = nn.Conv2d(ch, ch, 3, padding=1)
        self.up = nn.Conv2d(ch, 3 * s * s, 3, padding=1)
    def forward(self, x):
        h = self.head(x); r = h
        for b in self.body: r = r + b(r)
        r = self.tail(r) + h
        return F.pixel_shuffle(self.up(r), self.s) + F.interpolate(x, scale_factor=self.s, mode="bicubic", align_corners=False)

def export(model):
    W = {"head/kernel": model.head.weight, "head/bias": model.head.bias, "tail/kernel": model.tail.weight,
         "tail/bias": model.tail.bias, "up/kernel": model.up.weight, "up/bias": model.up.bias}
    for i, b in enumerate(model.body):
        W[f"body{i}_a/kernel"], W[f"body{i}_a/bias"], W[f"body{i}_b/kernel"], W[f"body{i}_b/bias"] = b[0].weight, b[0].bias, b[2].weight, b[2].bias
    return {k: (v.detach().cpu().numpy().transpose(2, 3, 1, 0) if v.ndim == 4 else v.detach().cpu().numpy()).astype(np.float32) for k, v in W.items()}

def psnr(a, b): return float(10 * np.log10(1 / max(np.mean((a - b) ** 2), 1e-12)))

@torch.no_grad()
def evaluate(model, paths, s, dev):
    """Mean RGB PSNR on full validation images; LR made with PIL (bicubic and nearest), as in exp2b."""
    R = {k: {"model": [], "bicubic": []} for k in ("bicubic", "nearest")}
    for p in paths:
        im = Image.open(p).convert("RGB"); hr = np.asarray(im, np.float32) / 255
        for k, f in (("bicubic", Image.BICUBIC), ("nearest", Image.NEAREST)):
            lr = im.resize((im.width // s, im.height // s), f)
            x = torch.from_numpy(np.asarray(lr, np.float32) / 255).permute(2, 0, 1)[None].to(dev)
            y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            R[k]["model"].append(psnr(y, hr)); R[k]["bicubic"].append(psnr(np.asarray(lr.resize(im.size, Image.BICUBIC), np.float32) / 255, hr))
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in R.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2); ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-patch", type=int, default=48); ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=6); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    a = ap.parse_args(); s = a.scale; os.makedirs(OUT, exist_ok=True); name = f"tinysr26_x{s}{a.tag}"
    torch.manual_seed(a.seed); dev = "cuda"; torch.backends.cudnn.benchmark = True
    model = TinySR(s, a.ch, a.blocks).to(dev); ema = copy.deepcopy(model).eval()
    nparam = sum(p.numel() for p in model.parameters()); print(f"[{name}] {nparam:,} parameters", flush=True)
    opt = torch.optim.Adam(model.parameters(), a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))
    dl = torch.utils.data.DataLoader(Patches(files("train"), a.lr_patch * s), batch_size=a.batch // 4, shuffle=True,
                                     num_workers=a.workers, drop_last=True, persistent_workers=True, pin_memory=True)
    val = files("val", per_class=40); log = []; step = 0; t0 = time.time(); run = 0.0
    while step < a.steps:
        for batch in dl:
            hr = batch.view(-1, 3, a.lr_patch * s, a.lr_patch * s).to(dev, non_blocking=True).float() / 255
            lr = torch.cat([degrade(g, s, k) for g, k in zip(hr.chunk(4), KERNELS)])
            loss = F.l1_loss(model(lr), hr)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - min(0.999, (1 + step) / (10 + step)))
            run = 0.98 * run + 0.02 * loss.item()
            if step % 500 == 0: print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % 10000 == 0 or step == a.steps:
                ev = evaluate(ema, val, s, dev); log.append({"step": step, "loss": run, **ev}); print(json.dumps(log[-1]), flush=True)
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step}, os.path.join(OUT, name + ".pt"))
            if step >= a.steps: break
    W = export(ema); np.savez(os.path.join(OUT, name + ".npz"), **W)
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False           # full fp32 for the check
    x = torch.rand(2, 3, 40, 52, device=dev)
    with torch.no_grad(): ref = ema(x).cpu().numpy().transpose(0, 2, 3, 1)
    d_np = float(np.abs(mnp.tinysr_np(x.cpu().numpy().transpose(0, 2, 3, 1), W, s) - ref).max())
    with torch.no_grad(): d_t = float((mnp.tinysr_torch(x, mnp.to_torch(W), s) - ema(x)).abs().max())
    info = {"name": name, "params": nparam, "steps": a.steps, "seconds": time.time() - t0, "log": log, "export_check": {"numpy": d_np, "torch_functional": d_t}}
    json.dump(info, open(os.path.join(OUT, name + ".json"), "w"), indent=1); print(json.dumps(info["export_check"]), flush=True)

if __name__ == "__main__":
    main()
