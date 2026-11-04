"""
Experiment 7d: how far can weight edits push the Anime DCGAN's style, with no training?

  1. mirror / rotate      exact: flip or rotate every kernel and the dense layer's 4x4 layout
  2. pencil sketch        compose the 3x3 output conv with a 5x5 Mexican-hat filter on luminance
                          -> one 7x7 conv (kernel size changes; the clip to [0,1] does the line inking)
  3. duotone / sepia      fold a 3x3 colour matrix into the output conv
  4. bigger canvas        architecture edit: lay several dense outputs side by side so the
                          translation-invariant conv stack paints one wide image
  5. model composition    Anime -> DLSS upscaler (256 px);  Anime -> gray -> Deep-Color

  python src/exp7d_anime_styles.py
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2b_tinysr18_weightpatch import load_model
from exp3_anime_weightpatch import apply_op_b, generate
import keras_np as kn

OUT = "results/exp7_out"
LUMA = np.array([0.299, 0.587, 0.114])

def convs(ops): return [o["name"] for o in ops if o["op"] == "conv"]

def mirror(W, ops):
    W = {k: v.copy() for k, v in W.items()}
    for c in convs(ops): W[c + "/kernel"] = W[c + "/kernel"][:, ::-1].copy()
    d = ops[0]["name"]; shp = ops[1]["shape"]                     # dense output is a (4,4,64) layout
    W[d + "/kernel"] = W[d + "/kernel"].reshape(-1, *shp)[:, :, ::-1].reshape(W[d + "/kernel"].shape).copy()
    W[d + "/bias"] = W[d + "/bias"].reshape(shp)[:, ::-1].reshape(-1).copy()
    return W

def rotate90(W, ops):
    """G'(z) = rot90(G(z)) counter-clockwise: rotate kernels and the layout by the same rotation."""
    W = {k: v.copy() for k, v in W.items()}
    for c in convs(ops): W[c + "/kernel"] = np.rot90(W[c + "/kernel"], 1, (0, 1)).copy()
    d = ops[0]["name"]; shp = ops[1]["shape"]
    W[d + "/kernel"] = np.rot90(W[d + "/kernel"].reshape(-1, *shp), 1, (1, 2)).reshape(W[d + "/kernel"].shape).copy()
    W[d + "/bias"] = np.rot90(W[d + "/bias"].reshape(shp), 1, (0, 1)).reshape(-1).copy()
    return W

def mexican_hat(size=5, sigma=0.8):
    r = np.arange(size) - size // 2; X, Y = np.meshgrid(r, r); q = (X ** 2 + Y ** 2) / (2 * sigma ** 2)
    k = (1 - q) * np.exp(-q); return k - k.mean()                 # zero-sum: flat regions -> 0

def sketch(W, ops, strength):
    """out = 1 + s * MH * luma(conv_last(h) + b)   ->  one (3+5-1)x(3+5-1) conv, all channels equal.
    Dark lines give negative MH response (dark output); flat or bright regions clip to white."""
    W = {k: v.copy() for k, v in W.items()}; c = convs(ops)[-1]
    Kl = W[c + "/kernel"].astype(np.float64) @ LUMA                # (3,3,64)
    H = mexican_hat(); kh = Kl.shape[0] + H.shape[0] - 1
    big = np.zeros((kh, kh, Kl.shape[2]))
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            big[i:i + Kl.shape[0], j:j + Kl.shape[1]] += H[i, j] * Kl  # full 2-D convolution of the two kernels
    W[c + "/kernel"] = np.repeat((strength * big)[..., None], 3, -1).astype(np.float32)
    W[c + "/bias"] = np.ones(3, np.float32)                         # MH is zero-sum, so luma bias cancels
    return W

def colour_fold(W, ops, M, t):
    W = {k: v.copy() for k, v in W.items()}; c = convs(ops)[-1]
    W[c + "/kernel"] = np.einsum("hwio,po->hwip", W[c + "/kernel"], M).astype(np.float32)
    W[c + "/bias"] = (M @ W[c + "/bias"] + t).astype(np.float32); return W

def duotone(dark, light):
    dark, light = np.array(dark) / 255, np.array(light) / 255
    return np.outer(light - dark, LUMA), dark                      # rgb' = dark + (light - dark) * luma(rgb)

SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])

def canvas(W, ops, Z, rows):
    """Architecture edit: dense(z_k) for each z, laid out as a rows x cols grid of 4x4 blocks,
    then the unchanged conv stack paints the whole canvas in one pass."""
    d = ops[0]["name"]; shp = ops[1]["shape"]
    blocks = (Z @ W[d + "/kernel"] + W[d + "/bias"]).reshape(len(Z), *shp)
    cols = len(Z) // rows
    grid = np.concatenate([np.concatenate(list(blocks[r * cols:(r + 1) * cols]), 1) for r in range(rows)], 0)
    h = grid[None].astype(np.float32)
    for op in ops[2:]: h = apply_op_b(h, op, W)
    return np.clip(h[0], 0, 1)

def main():
    os.makedirs(OUT, exist_ok=True); sys.stdout.reconfigure(encoding="utf-8")
    W, ops, _ = load_model("legacy/Anime-DCGAN/generator50000.h5")
    z = np.random.default_rng(3).standard_normal((6, 64))
    base = np.clip(generate(z, W, ops), 0, 1)
    R, rows = {}, {"base": base}

    Wm = mirror(W, ops); xm = np.clip(generate(z, Wm, ops), 0, 1)
    R["mirror_exact_maxabs"] = float(np.abs(xm - base[:, :, ::-1]).max()); rows["mirror (exact)"] = xm
    Wr = rotate90(W, ops); xr = np.clip(generate(z, Wr, ops), 0, 1)
    R["rotate90_exact_maxabs"] = float(np.abs(xr - np.rot90(base, 1, (1, 2))).max()); rows["rotate 90 (exact)"] = xr

    # sketch: compare the single 7x7 conv against applying conv_last then the filter separately
    ops7 = [dict(o) for o in ops]
    Ws = sketch(W, ops, strength=6.0); xs = np.clip(generate(z, Ws, ops7), 0, 1)
    pre = generate(z, W, ops[:-2])                                 # features entering the last conv
    y = kn.conv2d(pre, W[convs(ops)[-1] + "/kernel"], W[convs(ops)[-1] + "/bias"], "same") @ LUMA
    two_step = np.clip(1 + 6.0 * kn.conv2d(y[..., None].astype(np.float32), mexican_hat()[..., None, None].astype(np.float32), None, "same")[..., 0], 0, 1)
    R["sketch_vs_two_step_maxabs_interior"] = float(np.abs(xs[:, 3:-3, 3:-3, 0] - two_step[:, 3:-3, 3:-3]).max())
    R["sketch_vs_two_step_maxabs_border"] = float(np.abs(xs[..., 0] - two_step).max())
    rows["pencil sketch (7x7 fold)"] = xs

    M, t = duotone((30, 10, 70), (120, 255, 230)); rows["duotone (colour fold)"] = np.clip(generate(z, colour_fold(W, ops, M, t), ops), 0, 1)
    rows["sepia (colour fold)"] = np.clip(generate(z, colour_fold(W, ops, SEPIA, np.zeros(3)), ops), 0, 1)
    R["params_changed"] = {"mirror": "all conv kernels + dense (a permutation, no values change)",
                           "sketch": "last conv 3x3x64x3 -> 7x7x64x3 (architecture: kernel size)",
                           "colour folds": "last conv kernel + bias (579 numbers)"}

    # bigger canvases
    zc = np.random.default_rng(11).standard_normal((8, 64))
    strip = canvas(W, ops, zc[:6], rows=1)                       # 64 x 384
    quilt = canvas(W, ops, zc[:4], rows=2)                       # 128 x 128
    # smooth panorama: z interpolated per 4x4 block across 12 blocks
    ts = np.linspace(0, 1, 12)[:, None]; zp = (1 - ts) * zc[6] + ts * zc[7]
    pano = canvas(W, ops, zp, rows=1)

    # composition with other repos' models
    Wd, Ld = kn.load("legacy/TinySR-18/generator.h5")
    lr = np.stack([np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST)) for b in base[:4]]) / 255.0
    up = np.clip(kn.forward(lr.astype(np.float32), Wd, Ld), 0, 1)
    Wc, Lc = kn.load("legacy/Deep-Color/generator2800.h5")
    from skimage.color import rgb2gray
    g = np.stack([rgb2gray(np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((128, 128), Image.BICUBIC))) for b in base[:4]]).astype(np.float32)
    recol = np.clip(kn.forward((g / 255)[..., None], Wc, Lc), 0, 1)

    json.dump(R, open(os.path.join(OUT, "anime_styles.json"), "w"), indent=1)
    save_rows(rows, os.path.join(OUT, "anime_styles.png"))
    for name, img in [("anime_strip.png", strip), ("anime_quilt.png", quilt), ("anime_panorama.png", pano)]:
        Image.fromarray((img * 255).astype(np.uint8)).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(os.path.join(OUT, name))
    comp = np.concatenate([np.concatenate([np.asarray(Image.fromarray((base[i] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)) / 255,
                                           up[i], np.asarray(Image.fromarray((recol[i] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)) / 255], 1)
                           for i in range(4)], 0)
    Image.fromarray((comp * 255).astype(np.uint8)).save(os.path.join(OUT, "anime_composition.png"))
    print(json.dumps(R, indent=1))

def save_rows(rows, path, s=128):
    lw = 190; n = len(next(iter(rows.values())))
    im = Image.new("RGB", (lw + n * s, len(rows) * s), "white"); d = ImageDraw.Draw(im)
    for r, (k, v) in enumerate(rows.items()):
        for c in range(n): im.paste(Image.fromarray((v[c] * 255).astype(np.uint8)).resize((s, s), Image.NEAREST), (lw + c * s, r * s))
        d.text((6, r * s + s // 2 - 6), k, fill="black")
    im.save(path)

if __name__ == "__main__":
    main()
