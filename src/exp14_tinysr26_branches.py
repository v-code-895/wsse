"""
Experiment 14: git for fine-tunes. Two branches are fine-tuned independently from the same
TinySR-26 x2 base and stored as weight deltas (patches); merging is addition.

  branch "bicubic": fine-tune on bicubic-downsampled LR only      -> claim: better on bicubic LR
  branch "noisy":   fine-tune on mixed kernels + Gaussian LR noise  -> claim: better on noisy LR

Merges: base + d_bicubic + d_noisy (sum) and base + (d_bicubic + d_noisy) / 2 (average). Delta merges
commute exactly (addition), so any conflict shows up only in the tests, which is what wpgit's
in-context tests are for. Tests on held-out AFHQ (60 images per class, 2x from 256 to 512, the
training resolution):
  - bicubic LR             (the bicubic branch's claim)
  - noisy bicubic LR       (the noisy branch's claim)
  - nearest LR             (regression guard for both)

A first version with a cats-only branch is kept in results/exp14_out/*_v1_cats_branch.*: that branch did not
specialise (+0.08 dB on cats), and its 128->256 test favoured any smoothing edit.

  <venv python> src/exp14_tinysr26_branches.py
"""
import copy, json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp10_tinysr26 as e10
from exp2b_tinysr18_weightpatch import weights_hash

OUT = "results/exp14_out"; DEV = "cuda"; NOISE = 8 / 255; CLASSES = ("cat", "dog", "wild")
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def from_dict(W, s=2):
    m = e10.TinySR(s).to(DEV)
    names = {"head": m.head, "tail": m.tail, "up": m.up, **{f"body{i}_a": b[0] for i, b in enumerate(m.body)}, **{f"body{i}_b": b[2] for i, b in enumerate(m.body)}}
    with torch.no_grad():
        for n, layer in names.items():
            layer.weight.copy_(torch.from_numpy(W[n + "/kernel"].transpose(3, 2, 0, 1).copy())); layer.bias.copy_(torch.from_numpy(W[n + "/bias"]))
    return m

def finetune(W0, kernels, noise, steps=3000, lr=1e-4, seed=1):
    torch.manual_seed(seed); m = from_dict(W0); ema = copy.deepcopy(m).eval()
    opt = torch.optim.Adam(m.parameters(), lr, betas=(0.9, 0.99))
    dl = torch.utils.data.DataLoader(e10.Patches(e10.files("train", CLASSES), 96), batch_size=8, shuffle=True,
                                     num_workers=4, drop_last=True, persistent_workers=True)
    step = 0
    while step < steps:
        for b in dl:
            hr = b.view(-1, 3, 96, 96).to(DEV).float() / 255
            lr_ = torch.cat([e10.degrade(g, 2, k) for g, k in zip(hr.chunk(len(kernels)), kernels)])
            if noise: lr_ = ((lr_ + torch.rand(len(lr_), 1, 1, 1, device=DEV) * noise * torch.randn_like(lr_)).clamp(0, 1) * 255).round() / 255
            loss = F.l1_loss(m(lr_), hr); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), m.parameters()): pe.lerp_(p, 0.01)
            if step >= steps: break
    return e10.export(ema)

def test_images(per_class=60):
    out = []
    for c in CLASSES:
        d = os.path.join(e10.ROOT, "val", c); out += [Image.open(os.path.join(d, f)).convert("RGB") for f in sorted(os.listdir(d))[-per_class:]]
    return out

@torch.no_grad()
def psnr_on(W, imgs, kind, noise=0.0, seed=0):
    """2x from 256 to 512: HR is the original AFHQ image, LR made with PIL, optionally with 8-bit Gaussian noise."""
    m = from_dict(W).eval(); g = torch.Generator(device=DEV).manual_seed(seed); out = []
    for im in imgs:
        hr = np.asarray(im, np.float32) / 255
        lr = np.asarray(im.resize((im.width // 2, im.height // 2), {"bicubic": Image.BICUBIC, "nearest": Image.NEAREST}[kind]), np.float32) / 255
        x = torch.from_numpy(lr).permute(2, 0, 1)[None].to(DEV)
        if noise: x = ((x + noise * torch.randn(x.shape, generator=g, device=DEV)).clamp(0, 1) * 255).round() / 255
        y = m(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy(); out.append(10 * np.log10(1 / np.mean((y - hr) ** 2)))
    return float(np.mean(out))

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True)
    W0 = {k: v.astype(np.float32) for k, v in np.load("results/exp10_out/tinysr26_x2.npz").items()}; base = weights_hash(W0)
    t0 = time.time(); Wb = finetune(W0, ["bicubic"], 0.0); Wn = finetune(W0, e10.KERNELS, NOISE, seed=2)
    print(f"fine-tunes done in {time.time() - t0:.0f}s", flush=True)
    db = {k: Wb[k] - W0[k] for k in W0}; dn = {k: Wn[k] - W0[k] for k in W0}
    np.savez(os.path.join(OUT, "delta_bicubic.npz"), **db); np.savez(os.path.join(OUT, "delta_noisy.npz"), **dn)
    models = {"base": W0, "branch: bicubic": Wb, "branch: noisy": Wn,
              "merge: base + d_bicubic + d_noisy": {k: W0[k] + db[k] + dn[k] for k in W0},
              "merge: base + (d_bicubic + d_noisy) / 2": {k: W0[k] + (db[k] + dn[k]) / 2 for k in W0}}
    ab = {k: (W0[k] + db[k]) + dn[k] for k in W0}; ba = {k: (W0[k] + dn[k]) + db[k] for k in W0}
    flat = lambda d: np.concatenate([d[k].ravel() for k in sorted(d)])
    R = {"base": base, "delta_cos": float(flat(db) @ flat(dn) / np.linalg.norm(flat(db)) / np.linalg.norm(flat(dn))),
         "delta_rel_norm": {"bicubic": float(np.linalg.norm(flat(db)) / np.linalg.norm(flat(W0))), "noisy": float(np.linalg.norm(flat(dn)) / np.linalg.norm(flat(W0)))},
         "merge_order_maxabs": float(max(np.abs(ab[k] - ba[k]).max() for k in W0)), "tests": {}}
    imgs = test_images()
    for name, W in models.items():
        R["tests"][name] = {"bicubic LR": psnr_on(W, imgs, "bicubic"), "noisy bicubic LR": psnr_on(W, imgs, "bicubic", NOISE),
                            "nearest LR": psnr_on(W, imgs, "nearest")}
        print(name, R["tests"][name], flush=True)
    json.dump(R, open(os.path.join(OUT, "exp14_results.json"), "w"), indent=1)
    b = R["tests"]["base"]; cols = list(b)
    L = ["# Experiment 14: git for fine-tunes (TinySR-26 x2)\n",
         f"Base `{base}`. Two branches fine-tuned independently for 3,000 steps each; patches are weight deltas. "
         f"cos(d_bicubic, d_noisy) = {R['delta_cos']:+.3f}; relative size of each delta: bicubic {R['delta_rel_norm']['bicubic']:.4f}, "
         f"noisy {R['delta_rel_norm']['noisy']:.4f}. Merge order (A then B vs B then A): max weight difference {R['merge_order_maxabs']:.1e}.\n",
         f"PSNR (dB) on held-out AFHQ, 2x (256 -> 512), {len(imgs)} images; noise σ = 8/255; Δ vs base in brackets.\n",
         "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, t in R["tests"].items():
        L.append(f"| {name} | " + " | ".join(f"{t[c]:.2f}" + ("" if name == "base" else f" ({t[c] - b[c]:+.2f})") for c in cols) + " |")
    open(os.path.join(OUT, "exp14_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
