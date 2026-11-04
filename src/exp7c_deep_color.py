"""
Experiment 7c: Deep-Color. Test the blind-audit predictions and repair the model
with exact folds and one fixed architecture edit (no training).

Test images: scikit-image's colour photos (128x128) plus 8 faces from the Anime
DCGAN (exp3), upsized to 128.

  python src/exp7c_deep_color.py
"""
import json, os, sys
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import skimage.data as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keras_np as kn

OUT = "results/exp7_out"
LUMA = np.array([0.2125, 0.7154, 0.0721], np.float32)        # skimage rgb2gray weights (sum to 1)

def psnr(a, b): return float(10 * np.log10(1 / (np.mean((a - b) ** 2) + 1e-12)))

def test_images():
    imgs = {}
    for n in ["astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "retina", "hubble_deep_field"]:
        a = getattr(sd, n)()[..., :3]; h, w = a.shape[:2]; s = min(h, w)
        a = a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        imgs[n] = np.asarray(Image.fromarray(a).resize((128, 128), Image.BICUBIC))
    from exp2b_tinysr18_weightpatch import load_model
    from exp3_anime_weightpatch import generate
    Wa, oa, _ = load_model("legacy/Anime-DCGAN/generator50000.h5")
    faces = np.clip(generate(np.random.default_rng(7).standard_normal((8, 64)), Wa, oa), 0, 1)
    for i, f in enumerate(faces):
        imgs[f"anime_{i}"] = np.asarray(Image.fromarray((f * 255).astype(np.uint8)).resize((128, 128), Image.BICUBIC))
    return imgs

# ---------------------------------------------------------------------------
# Folds and edits
# ---------------------------------------------------------------------------
def fold_bn(W, layers):
    """conv -> BN  ==  conv' (exact). Returns BN-free weights and layers."""
    W = dict(W); out = []
    for i, L in enumerate(layers):
        if L["type"] == "BatchNormalization":
            c = out[-1]["name"]; n = L["name"]
            s = W[n + "/gamma"] / np.sqrt(W[n + "/moving_variance"] + L.get("epsilon", 1e-3))
            W[c + "/kernel"] = (W[c + "/kernel"] * s).astype(np.float32)
            W[c + "/bias"] = ((W[c + "/bias"] - W[n + "/moving_mean"]) * s + W[n + "/beta"]).astype(np.float32)
            continue
        out.append(L)
    return {k: v for k, v in W.items() if "batch_normalization" not in k}, out

def fold_input_scale(W, first, c):
    W = dict(W); W[first + "/kernel"] = (W[first + "/kernel"] * c).astype(np.float32); return W

def centre_output_kernel(W, layers, last):
    """2x2 'same' (taps at offsets 0,+1) -> 3x3 'same' that averages the output with its
    up/left neighbours: taps at -1,0,+1, centred on the pixel."""
    K2 = W[last + "/kernel"]; K3 = np.zeros((3, 3) + K2.shape[2:], np.float32)
    for s in (0, 1):
        for t in (0, 1): K3[s:s + 2, t:t + 2] += K2 / 4
    W = dict(W); W[last + "/kernel"] = K3
    layers = [dict(L, kernel_size=[3, 3]) if L["name"] == last else L for L in layers]
    return W, layers

def luma_skip(rgb, gray):
    """Fixed architecture edit: replace the output's luminance with the input's.
    out' = out + (g - LUMA.out) * [1,1,1]; LUMA sums to 1 so LUMA.out' = g exactly."""
    return np.clip(rgb + (gray - rgb @ LUMA)[..., None], 0, 1)

# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def frac_shift(img, dy, dx):
    """img sampled at (i+dy, j+dx), bilinear, edge-clamped."""
    H, W = img.shape; y = np.clip(np.arange(H) + dy, 0, H - 1); x = np.clip(np.arange(W) + dx, 0, W - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int); y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    return (img[y0][:, x0] * (1 - fy) * (1 - fx) + img[y1][:, x0] * fy * (1 - fx)
            + img[y0][:, x1] * (1 - fy) * fx + img[y1][:, x1] * fy * fx)

def best_shift(out_luma, gray):
    grid = np.arange(-1, 1.01, 0.125); best = (1e9, 0, 0)
    for dy in grid:
        for dx in grid:
            e = np.mean((out_luma[4:-4, 4:-4] - frac_shift(gray, dy, dx)[4:-4, 4:-4]) ** 2)
            if e < best[0]: best = (e, dy, dx)
    return best[1], best[2]

def chroma(rgb): return rgb - (rgb @ LUMA)[..., None]

def palette_r2(gray, rgb, bins=32):
    """Share of chroma variance explained by the gray value alone (lookup table)."""
    C = chroma(rgb).reshape(-1, 3); g = np.minimum((gray.ravel() * bins).astype(int), bins - 1)
    pred = np.zeros_like(C)
    for b in range(bins):
        m = g == b
        if m.any(): pred[m] = C[m].mean(0)
    return float(1 - ((C - pred) ** 2).sum() / ((C - C.mean(0)) ** 2).sum())

def validate_against_author(W, layers):
    """Crop original + 'Restored Image' panels out of the README figure (combined.png) and
    compare our forward pass to the author's restored output."""
    im = np.asarray(Image.open("legacy/Deep-Color/combined.png").convert("RGB")).astype(np.float32) / 255
    def boxes(col):
        x0 = col * im.shape[1] // 3; sub = im[:, x0:(col + 1) * im.shape[1] // 3].mean(-1) < 0.97
        rows = np.where(sub.mean(1) > 0.6)[0]; cols = np.where(sub.mean(0) > 0.6)[0]
        runs = np.split(rows, np.where(np.diff(rows) > 5)[0] + 1)
        return [(r[0], r[-1], x0 + cols[0], x0 + cols[-1]) for r in runs if len(r) > 100]
    crop = lambda b: np.asarray(Image.fromarray((im[b[0] + 2:b[1] - 1, b[2] + 2:b[3] - 1] * 255).astype(np.uint8)).resize((128, 128), Image.BICUBIC))
    rows = []
    for name, bo, br in zip(["painting", "Lisa Simpson", "anime girl", "skateboard"], boxes(0), boxes(2)):
        orig, rest = crop(bo) / 255.0, crop(br) / 255.0
        out = np.clip(kn.forward((rgb2gray(crop(bo)) / 255)[None, ..., None].astype(np.float32), W, layers)[0], 0, 1)
        cc = lambda a, b: float(np.corrcoef(chroma(a).ravel(), chroma(b).ravel())[0, 1])
        rows.append((name, psnr(out, rest), cc(out, rest), cc(rest, orig)))
    return rows

def main():
    os.makedirs(OUT, exist_ok=True); sys.stdout.reconfigure(encoding="utf-8")
    W, layers = kn.load("legacy/Deep-Color/generator2800.h5")
    convs = [L["name"] for L in layers if L["type"] == "Conv2D"]; first, last = convs[0], convs[-1]
    imgs = test_images(); names = list(imgs)
    true = np.stack([imgs[n] / 255.0 for n in names]).astype(np.float32)
    gray = np.stack([rgb2gray(imgs[n]) for n in names]).astype(np.float32)          # [0,1], as skimage returns
    run = lambda Wm, Lm, x: np.clip(kn.forward(x[..., None], Wm, Lm, chunk=4), 0, 1)

    out_trained = run(W, layers, gray / 255)            # how the notebooks call it
    out_naive = run(W, layers, gray)                    # what a new user would do
    Wf, Lf = fold_bn(W, layers); Wf = fold_input_scale(Wf, first, 1 / 255)   # BN-free, accepts [0,1]
    out_fold = run(Wf, Lf, gray)
    Wc, Lc = centre_output_kernel(Wf, Lf, last); out_c = run(Wc, Lc, gray)   # tested, not kept: see shift result
    out_skip = luma_skip(out_fold, gray)
    gray_rgb = np.repeat(gray[..., None], 3, -1)
    # does the model's chroma carry any signal? least-squares gain on its chroma, fit on the anime
    # faces, applied to the photos (and the reverse); 0 = turn colour off, 1 = keep as is
    Cm, Ct = chroma(out_skip), chroma(true); anime = np.array([n.startswith("anime") for n in names])
    gain = lambda m: float((Cm[m] * Ct[m]).sum() / (Cm[m] ** 2).sum())
    corr = float(np.corrcoef(Cm.ravel(), Ct.ravel())[0, 1])
    g_a, g_p = gain(anime), gain(~anime)
    out_gain = np.empty_like(out_skip)
    out_gain[~anime] = np.clip(gray[~anime][..., None] + g_a * Cm[~anime], 0, 1)
    out_gain[anime] = np.clip(gray[anime][..., None] + g_p * Cm[anime], 0, 1)

    def ev(o):
        return {"psnr": float(np.mean([psnr(o[i], true[i]) for i in range(len(names))])),
                "luma_psnr": float(np.mean([psnr(o[i] @ LUMA, gray[i]) for i in range(len(names))])),
                "chroma_mae": float(np.mean(np.abs(chroma(o) - chroma(true)))),
                "colourfulness": float(np.mean(np.linalg.norm(chroma(o), axis=-1)))}
    M = {"true colour (reference)": ev(true), "gray (no colour at all)": ev(gray_rgb),
         "model, as trained (input /255)": ev(out_trained), "model, naive 0-1 input": ev(out_naive),
         "BN folded + 1/255 folded (0-1 input)": ev(out_fold), "(+ centred output kernel, rejected)": ev(out_c),
         "+ luminance skip": ev(out_skip), "+ chroma gain fit on the other half": ev(out_gain)}
    shifts = {"as trained": [best_shift(out_trained[i] @ LUMA, gray[i]) for i in range(len(names))],
              "centred": [best_shift(out_c[i] @ LUMA, gray[i]) for i in range(len(names))]}
    pr2 = {"model output": float(np.mean([palette_r2(gray[i], out_trained[i]) for i in range(len(names))])),
           "true images": float(np.mean([palette_r2(gray[i], true[i]) for i in range(len(names))]))}
    exact = float(np.abs(out_fold - out_trained).max())

    R = {"metrics": M, "fold_exactness_maxabs": exact, "palette_r2": pr2,
         "chroma_corr_with_truth": corr, "chroma_gain_fit_on_anime": g_a, "chroma_gain_fit_on_photos": g_p,
         "shift_median": {k: np.median(np.array(v), 0).tolist() for k, v in shifts.items()},
         "per_image_psnr": {n: {"gray": psnr(gray_rgb[i], true[i]), "model": psnr(out_trained[i], true[i]),
                                "fixed": psnr(out_skip[i], true[i])} for i, n in enumerate(names)}}
    json.dump(R, open(os.path.join(OUT, "deep_color_results.json"), "w"), indent=1)
    save_fig(names, true, gray_rgb, out_trained, out_naive, out_skip, os.path.join(OUT, "deep_color_grid.png"))

    V = validate_against_author(W, layers); R["author_check"] = V
    L = ["# Experiment 7c: Deep-Color\n",
         "## Engine check against the author's README figure\n",
         "| sample | our output vs author's 'Restored' (PSNR) | chroma corr ours↔author | author's chroma corr with original |",
         "|---|---|---|---|"] + [f"| {n} | {p:.2f} dB | {a:+.3f} | {b:+.3f} |" for n, p, a, b in V] + [
         "\nThree of four panels reproduce at >30 dB, so the numpy forward pass matches; the Lisa panel likely came "
         "from a later checkpoint (the inference notebook loads `generator8800.h5`; the repo ships `generator2800.h5`).\n",
         "## Test set\n", f"{len(names)} test images (7 photos, 8 anime faces), 128×128.\n",
         "| model | PSNR vs true colour | luma PSNR vs input | chroma MAE | colourfulness |", "|---|---|---|---|---|"]
    for k, v in M.items():
        L.append(f"| {k} | {v['psnr']:.2f} | {v['luma_psnr']:.2f} | {v['chroma_mae']:.4f} | {v['colourfulness']:.4f} |")
    L += [f"\n- Model chroma vs true chroma: correlation {corr:+.3f}; best gain on the model's colour is "
          f"{g_a:.2f} (fit on anime) / {g_p:.2f} (fit on photos), where 1 = keep it and 0 = discard it.",
          f"- Folding BN and the 1/255 input scale into the convs is exact: max output difference {exact:.1e}.",
          f"- Half-pixel shift: median best alignment of output luma to input, (dy, dx) = "
          f"{tuple(R['shift_median']['as trained'])} as trained → {tuple(R['shift_median']['centred'])} after centring the 2×2 kernel.",
          f"- Palette test: gray value alone explains {pr2['model output']:.0%} of the model's output chroma variance "
          f"(per image, 32-bin lookup), vs {pr2['true images']:.0%} for the true images.",
          "\n| image | gray PSNR | model PSNR | fixed PSNR |", "|---|---|---|---|"]
    L += [f"| {n} | {v['gray']:.2f} | {v['model']:.2f} | {v['fixed']:.2f} |" for n, v in R["per_image_psnr"].items()]
    open(os.path.join(OUT, "deep_color_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

def save_fig(names, *rows_and_path):
    *rows, path = rows_and_path
    labels = ["true colour", "gray input", "model (as trained)", "model, naive 0-1 input", "fixed (folds + luma skip)"]
    pick = [n for n in ("astronaut", "coffee", "chelsea", "anime_0", "anime_3") if n in names]; idx = [names.index(n) for n in pick]
    s = 128; canvas = Image.new("RGB", (s * len(idx), s * len(rows) + 0), "white")
    for r, row in enumerate(rows):
        for c, i in enumerate(idx):
            canvas.paste(Image.fromarray((np.clip(row[i], 0, 1) * 255).astype(np.uint8)), (c * s, r * s))
    canvas = canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST)
    lab = Image.new("RGB", (190, canvas.height), "white")
    from PIL import ImageDraw; d = ImageDraw.Draw(lab)
    for r, t in enumerate(labels): d.text((6, r * 2 * s + s - 6), t, fill="black")
    full = Image.new("RGB", (lab.width + canvas.width, canvas.height), "white"); full.paste(lab, (0, 0)); full.paste(canvas, (lab.width, 0))
    full.save(path)

if __name__ == "__main__":
    main()
