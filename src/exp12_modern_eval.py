"""
Experiment 12: the 2026 models against the 2018 ones, and a cat pipeline (CatGAN-26 -> TinySR-26 x4).

A. Super-sampling: TinySR-18 (2018 'DLSS' generator.h5) vs TinySR-26 vs bicubic
   - the exp2b held-out suites (natural photos, sprites, legacy v1 images), 2x, LR made by nearest and by bicubic
   - 300 AFHQ validation photos (100 cats, 100 dogs, 100 wildlife): 2x (128 -> 256) and 4x (128 -> 512;
     TinySR-18 applied twice, since it only does 2x)
B. CatGAN-26 samples judged by MobileNetV2 (ImageNet): share called a domestic cat, Fréchet distance of
   MobileNetV2 features to held-out AFHQ cats (with real-vs-real as the floor), and a memorisation check
   (nearest training image by 32x32 thumbnail correlation, compared with held-out real cats).
C. Figures in results/exp12_out/.

  <venv python> src/exp12_modern_eval.py
"""
import importlib.util, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modern_np as mnp, mobilenet_np as mn

OUT = "results/exp12_out"; AFHQ = "data/afhq/data/afhq"; DEV = "cuda"
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False      # full fp32 for every comparison
CAT_CLASSES = [281, 282, 283, 284, 285]                          # tabby, tiger cat, Persian, Siamese, Egyptian cat
FILTERS = {"bicubic": Image.BICUBIC, "nearest": Image.NEAREST}

def load_exp2b():
    import exp2b_tinysr18_weightpatch as m; return m

def u8(a): return (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
def psnr(a, b): return float(10 * np.log10(1 / max(np.mean((a - b) ** 2), 1e-12)))
def ssim(a, b): return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))

class SR:
    """The three upscalers on (h, w, 3) float images in [0, 1]; outputs clipped to [0, 1]."""
    def __init__(self, e2):
        W18, self.ops18, _ = e2.load_model("legacy/TinySR-18/generator.h5")
        self.ops18 = e2.strip_clip_equivalent_tail(self.ops18)[0]; self.T18 = mnp.to_torch(W18, DEV); self.W18, self.e2 = W18, e2
        self.T26 = {s: mnp.to_torch(dict(np.load(f"results/exp10_out/tinysr26_x{s}.npz")), DEV) for s in (2, 4)}
    @torch.no_grad()
    def _t(self, f, lr):
        x = torch.from_numpy(np.ascontiguousarray(lr, np.float32)).permute(2, 0, 1)[None].to(DEV)
        return f(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    def tinysr18(self, lr): return self._t(lambda x: mnp.legacy_torch(x, self.T18, self.ops18), lr)
    def tinysr26(self, lr, s): return self._t(lambda x: mnp.tinysr_torch(x, self.T26[s], s), lr)
    @staticmethod
    def bicubic(lr, s): return np.asarray(Image.fromarray(u8(lr)).resize((lr.shape[1] * s, lr.shape[0] * s), Image.BICUBIC), np.float32) / 255

def lr_of(hr_u8, s, f): return np.asarray(Image.fromarray(hr_u8).resize((hr_u8.shape[1] // s, hr_u8.shape[0] // s), FILTERS[f]), np.float32) / 255

def afhq_val(per_class=100):
    out = []
    for c in ("cat", "dog", "wild"):
        d = os.path.join(AFHQ, "val", c); out += [(c, os.path.join(d, f)) for f in sorted(os.listdir(d))[:per_class]]
    return out

def part_a(e2, sr):
    R = {}
    # check the GPU port of TinySR-18 against the exp2b NumPy engine once
    S = e2.build_suites("bicubic"); lr0, hr0 = S["natural_test"]["chelsea"][:2]
    R["tinysr18_gpu_vs_numpy_maxabs"] = float(np.abs(sr.tinysr18(lr0) - np.clip(e2.run(lr0, sr.W18, sr.ops18), 0, 1)).max())
    for f in FILTERS:
        S = e2.build_suites(f)
        for suite in ("natural_test", "sprite_test", "legacy"):
            rows = {"bicubic": [], "TinySR-18": [], "TinySR-26": []}
            for name, pair in S[suite].items():
                lr, hr = pair[:2]
                for k, y in (("bicubic", sr.bicubic(lr, 2)), ("TinySR-18", sr.tinysr18(lr)), ("TinySR-26", sr.tinysr26(lr, 2))):
                    rows[k].append((psnr(y, hr), ssim(y, hr)))
            R[f"exp2b {suite} 2x, LR {f}"] = {k: np.mean(v, 0).tolist() for k, v in rows.items()}
    # colour-fold headroom: the best affine colour correction y_hr ~ M y + t, fit on exp2b's calibration split
    # (nearest LR, as in exp2b) and scored on its held-out natural images; applied post hoc, not folded
    S = e2.build_suites("nearest"); calib = list(S["natural_calib"].values()) + list(S["sprite_calib"].values())
    for name, f in (("TinySR-18", sr.tinysr18), ("TinySR-26", lambda lr: sr.tinysr26(lr, 2))):
        X = np.concatenate([f(p[0]).reshape(-1, 3) for p in calib]); Y = np.concatenate([p[1].reshape(-1, 3) for p in calib])
        A = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)[0]
        gain = [psnr(np.clip(f(p[0]) @ A[:3] + A[3], 0, 1), p[1]) - psnr(f(p[0]), p[1]) for p in S["natural_test"].values()]
        R[f"colour-fold headroom, {name}"] = {"M": A[:3].T.round(3).tolist(), "t_rgb": (A[3] * 255).round(2).tolist(), "heldout_gain_db": float(np.mean(gain))}
    val = afhq_val()
    for f in FILTERS:
        rows2 = {"bicubic": [], "TinySR-18": [], "TinySR-26": []}; rows4 = {"bicubic": [], "TinySR-18 twice": [], "TinySR-26": []}
        for c, p in val:
            im = Image.open(p).convert("RGB"); hr512 = np.asarray(im); hr256 = np.asarray(im.resize((256, 256), Image.LANCZOS))
            lr = lr_of(hr256, 2, f); t = hr256 / 255.0
            for k, y in (("bicubic", sr.bicubic(lr, 2)), ("TinySR-18", sr.tinysr18(lr)), ("TinySR-26", sr.tinysr26(lr, 2))): rows2[k].append((psnr(y, t), ssim(y, t)))
            lr = lr_of(hr512, 4, f); t = hr512 / 255.0
            for k, y in (("bicubic", sr.bicubic(lr, 4)), ("TinySR-18 twice", sr.tinysr18(sr.tinysr18(lr))), ("TinySR-26", sr.tinysr26(lr, 4))): rows4[k].append((psnr(y, t), ssim(y, t)))
        R[f"AFHQ val 2x (128->256), LR {f}"] = {k: np.mean(v, 0).tolist() for k, v in rows2.items()}
        R[f"AFHQ val 4x (128->512), LR {f}"] = {k: np.mean(v, 0).tolist() for k, v in rows4.items()}
    return R

def mobilenet(imgs_u8, Wm, chunk=16):
    L, F = [], []
    for s in range(0, len(imgs_u8), chunk):
        l, f = mn.forward(np.stack([mn.preprocess(a) for a in imgs_u8[s:s + chunk]]), Wm, return_features=True); L.append(l); F.append(f)
    return np.concatenate(L), np.concatenate(F)

def frechet(F1, F2):
    """|m1 - m2|^2 + tr(C1 + C2 - 2 (C1 C2)^1/2), with tr (C1 C2)^1/2 = sum sqrt eig(C1^1/2 C2 C1^1/2)."""
    m1, m2, C1, C2 = F1.mean(0), F2.mean(0), np.cov(F1, rowvar=False), np.cov(F2, rowvar=False)
    w, V = np.linalg.eigh(C1); S1 = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    ev = np.linalg.eigvalsh(S1 @ C2 @ S1)
    return float(np.sum((m1 - m2) ** 2) + np.trace(C1) + np.trace(C2) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())

def thumbs(imgs_u8):
    T = np.stack([np.asarray(Image.fromarray(a).convert("L").resize((32, 32), Image.BILINEAR), np.float32) for a in imgs_u8]).reshape(len(imgs_u8), -1)
    return (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-6)

def generate(W, n, psi=1.0, seed=0):
    T = mnp.to_torch(W, DEV); z = torch.randn(n, 128, generator=torch.Generator().manual_seed(seed)).to(DEV) * psi; out = []
    with torch.no_grad():
        for s in range(0, n, 64): out.append(mnp.catgan_torch(z[s:s + 64], T).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
    return u8(np.concatenate(out))

def part_b(n=500):
    R = {}; Wg = dict(np.load("results/exp11_out/catgan26.npz")); Wm = mn.load()
    train = np.load("results/exp11_out/afhq_cat_train_128.npy"); vd = os.path.join(AFHQ, "val", "cat")
    val = np.stack([np.asarray(Image.open(os.path.join(vd, f)).convert("RGB").resize((128, 128), Image.LANCZOS)) for f in sorted(os.listdir(vd))[:n]])
    sets = {"AFHQ val cats (real)": val, "AFHQ train cats (real, 500)": train[np.random.default_rng(0).choice(len(train), n, replace=False)],
            "CatGAN-26, psi 1.0": generate(Wg, n), "CatGAN-26, psi 0.7 (truncation)": generate(Wg, n, 0.7)}
    feats = {}
    for k, imgs in sets.items():
        t0 = time.time(); Lg, Fe = mobilenet(imgs, Wm); feats[k] = Fe; top = Lg.argmax(1)
        R[k] = {"domestic_cat_rate": float(np.isin(top, CAT_CLASSES).mean()), "seconds": time.time() - t0}
        print(k, R[k], flush=True)
    for k in sets:
        if k != "AFHQ val cats (real)": R[k]["frechet_mobilenet_vs_val"] = frechet(feats[k], feats["AFHQ val cats (real)"])
    Tt = thumbs(train)
    for k in ("AFHQ val cats (real)", "CatGAN-26, psi 1.0", "CatGAN-26, psi 0.7 (truncation)"):
        C = thumbs(sets[k]) @ Tt.T / 1024; best = C.max(1)
        R[k]["nearest_train_thumb_corr"] = {"median": float(np.median(best)), "p95": float(np.percentile(best, 95)), "share_above_0.95": float((best > 0.95).mean())}
    return R, sets, train, Tt

def grid(rows, pad=4):
    h = max(a.shape[0] for r in rows for a in r); w = sum(a.shape[1] for a in rows[0]) + pad * (len(rows[0]) - 1)
    out = np.full((len(rows) * h + pad * (len(rows) - 1), w, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        x = 0
        for a in r: out[i * (h + pad):i * (h + pad) + a.shape[0], x:x + a.shape[1]] = a; x += a.shape[1] + pad
    return out

def up_nearest(a, s): return a.repeat(s, 0).repeat(s, 1)

def figures(e2, sr, sets, train, Tt):
    # 1. chelsea, 2x from a bicubic-downsampled 128 image: HR | bicubic | TinySR-18 | TinySR-26 (128-px crops)
    S = e2.build_suites("bicubic"); lr, hr = S["natural_test"]["chelsea"][:2]; c = (slice(40, 168), slice(40, 168))
    ys = [hr, sr.bicubic(lr, 2), sr.tinysr18(lr), sr.tinysr26(lr, 2)]
    Image.fromarray(grid([[up_nearest(u8(y[c]), 2) for y in ys]])).save(os.path.join(OUT, "fig_sr_chelsea_2x.png"))
    # 2. AFHQ val, 4x from bicubic 128: LR | bicubic | TinySR-18 twice | TinySR-26 | HR (crops of 256)
    rows = []
    for c_, p in afhq_val(per_class=2)[::2]:
        hr = np.asarray(Image.open(p).convert("RGB")); lr = lr_of(hr, 4, "bicubic"); cr = (slice(128, 384), slice(128, 384))
        ys = [up_nearest(lr, 4), sr.bicubic(lr, 4), sr.tinysr18(sr.tinysr18(lr)), sr.tinysr26(lr, 4), hr / 255.0]
        rows.append([u8(y[cr]) for y in ys])
    Image.fromarray(grid(rows)).save(os.path.join(OUT, "fig_sr_afhq_4x.png"))
    # 3. generated cats (psi 0.7), 4 x 8 at 128
    g = sets["CatGAN-26, psi 0.7 (truncation)"][:32]
    Image.fromarray(grid([list(g[i * 8:(i + 1) * 8]) for i in range(4)], pad=2)).save(os.path.join(OUT, "fig_catgan26_samples.png"))
    # 4. pipeline: 128 sample -> bicubic 512 -> TinySR-26 4x 512
    rows = [[up_nearest(a, 4), u8(sr.bicubic(a / 255.0, 4)), u8(sr.tinysr26(a / 255.0, 4))] for a in g[:3]]
    Image.fromarray(grid(rows)).save(os.path.join(OUT, "fig_pipeline_128_to_512.png"))
    # 2018 vs 2026 generators: Anime-DCGAN (64 px, shown at 128) above CatGAN-26 (128 px), 8 random samples each
    import exp3_anime_weightpatch as e3
    Wa, oa, _ = e2.load_model("legacy/Anime-DCGAN/generator50000.h5")
    anime = u8(np.clip(e3.generate(np.random.default_rng(7).standard_normal((8, 64)), Wa, oa), 0, 1))
    Image.fromarray(grid([[up_nearest(a, 2) for a in anime], list(sets["CatGAN-26, psi 1.0"][8:16])], pad=2)).save(os.path.join(OUT, "fig_2018_vs_2026_generators.png"))
    # 5. memorisation check: 8 generated cats and their nearest training images
    gen = sets["CatGAN-26, psi 1.0"][:8]; nn_ = (thumbs(gen) @ Tt.T).argmax(1)
    Image.fromarray(grid([list(gen), list(train[nn_])], pad=2)).save(os.path.join(OUT, "fig_catgan26_nearest_train.png"))

def write_md(R):
    L = ["# Experiment 12: 2026 models vs 2018 models\n", "## A. Super-sampling (PSNR dB / SSIM, means)\n",
         f"GPU port of TinySR-18 vs the exp2b NumPy engine: max difference {R['A']['tinysr18_gpu_vs_numpy_maxabs']:.1e}.\n"]
    for k, v in R["A"].items():
        if not isinstance(v, dict) or k.startswith("colour-fold"): continue
        L += [f"**{k}**\n", "| model | PSNR | SSIM |", "|---|---|---|"] + [f"| {m} | {p:.2f} | {s:.4f} |" for m, (p, s) in v.items()] + [""]
    L += ["**Colour-fold headroom** (best affine colour correction fit on exp2b's calibration split, gain on its held-out natural images)\n",
          "| model | M | t (RGB) | held-out gain |", "|---|---|---|---|"]
    L += [f"| {k.split(', ')[1]} | {v['M']} | {v['t_rgb']} | {v['heldout_gain_db']:+.2f} dB |" for k, v in R["A"].items() if k.startswith("colour-fold")] + [""]
    L += ["## B. CatGAN-26 judged by MobileNetV2 (ImageNet)\n",
          "| images (500 each) | called a domestic cat | Fréchet distance to val cats (MobileNetV2 features) | nearest-train thumbnail corr. (median / p95 / share > 0.95) |",
          "|---|---|---|---|"]
    for k, v in R["B"].items():
        nn_ = v.get("nearest_train_thumb_corr"); fd = v.get("frechet_mobilenet_vs_val")
        L.append(f"| {k} | {v['domestic_cat_rate']:.0%} | {'—' if fd is None else f'{fd:.1f}'} | "
                 + ("—" if nn_ is None else f"{nn_['median']:.2f} / {nn_['p95']:.2f} / {nn_['share_above_0.95']:.0%}") + " |")
    open(os.path.join(OUT, "exp12_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"                      # "A" (super-sampling only) or "all"
    path = os.path.join(OUT, "exp12_results.json"); R = json.load(open(path)) if os.path.exists(path) else {}
    e2 = load_exp2b(); sr = SR(e2)
    if part in ("A", "all"): R["A"] = part_a(e2, sr); print(json.dumps(R["A"], indent=1), flush=True); json.dump(R, open(path, "w"), indent=1)
    if part == "all":
        R["B"], sets, train, Tt = part_b(); figures(e2, sr, sets, train, Tt)
        json.dump(R, open(path, "w"), indent=1); write_md(R)

if __name__ == "__main__":
    main()
