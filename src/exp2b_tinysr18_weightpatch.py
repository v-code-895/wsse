#!/usr/bin/env python3
"""
Experiment 2b — Unit-tested, zero-gradient weight patches for the DLSS generator.

What changed from v1 (exp2_tinysr18_patch_v1.py) and why:
  * The forward pass is built from the model_config stored inside the .h5
    (layer order, LeakyReLU alpha, activations) instead of being hard-coded.
  * Baselines (nearest, bicubic) are reported, so "the model got better/worse"
    has a reference point.
  * Patches are FIT on a calibration split and EVALUATED on a held-out split.
  * Patches only touch the final conv layer, which is linear. That makes every
    edit an exact algebraic fold (verified numerically), not a guess that has to
    survive LeakyReLUs.
  * Every patch is a small JSON ".weightpatch" file that re-applies from text
    alone, is pinned to a content hash of the base weights, and can be merged.
  * A software-style gate decides PASS / FAIL:
        regression suite: no PSNR drop > 0.10 dB, no SSIM drop > 0.005
        edge-case suite : PSNR must improve and the black floor must drop
  * v1 is re-run (plus bias-only and AC-only ablations) to attribute its regression.

Usage
  python src/exp2b_tinysr18_weightpatch.py --weights legacy/TinySR-18/generator.h5
  python src/exp2b_tinysr18_weightpatch.py --weights generator.h5 --images path/to/anime_or_pokemon_pngs
  python src/exp2b_tinysr18_weightpatch.py --export-npz legacy/TinySR-18/generator.h5
  python src/exp2b_tinysr18_weightpatch.py --selftest          # harness check on a planted-defect stand-in

Needs: numpy, pillow, matplotlib; h5py for .h5 input; scikit-image for SSIM and
the natural test photos (optional, degrades gracefully).
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None

# ---------------------------------------------------------------------------
# 1. Loading weights + architecture
# ---------------------------------------------------------------------------
DLSS_FALLBACK_OPS = [
    {"op": "conv", "name": "conv2d_5", "act": "linear"}, {"op": "lrelu", "alpha": 0.2},
    {"op": "conv", "name": "conv2d_6", "act": "linear"}, {"op": "lrelu", "alpha": 0.2},
    {"op": "up", "size": [2, 2]},   # real order per generator.h5 config (v1 had this after conv2d_7)
    {"op": "conv", "name": "conv2d_7", "act": "linear"}, {"op": "lrelu", "alpha": 0.2},
    {"op": "conv", "name": "conv2d_8", "act": "linear"}, {"op": "lrelu", "alpha": 0.2},
    {"op": "conv", "name": "conv2d_9", "act": "linear"}, {"op": "lrelu", "alpha": 0.2},
]

def parse_keras_config(cfg):
    """Flatten a Keras 2.x model_config (Model/Sequential, possibly nested) into an op list."""
    ops = []
    def walk(layer):
        cn, c = layer["class_name"], layer["config"]
        if cn == "Sequential":
            for l in (c["layers"] if isinstance(c, dict) else c):
                walk(l)
        elif cn in ("Model", "Functional"):
            for l in c["layers"]:
                walk(l)
        elif cn == "InputLayer":
            pass
        elif cn == "Conv2D":
            if c.get("padding") != "same" or list(c.get("strides", [1, 1])) != [1, 1]:
                raise NotImplementedError(f"Conv2D {c['name']}: only stride-1 'same' supported")
            ops.append({"op": "conv", "name": c["name"], "act": c.get("activation", "linear")})
        elif cn == "LeakyReLU":
            ops.append({"op": "lrelu", "alpha": float(c.get("alpha", 0.3))})  # Keras default is 0.3
        elif cn == "UpSampling2D":
            if c.get("interpolation", "nearest") != "nearest":
                raise NotImplementedError("only nearest UpSampling2D supported")
            ops.append({"op": "up", "size": list(c.get("size", [2, 2]))})
        elif cn == "Dense":
            ops.append({"op": "dense", "name": c["name"], "act": c.get("activation", "linear")})
        elif cn == "Reshape":
            ops.append({"op": "reshape", "shape": list(c["target_shape"])})
        elif cn == "Activation":
            ops.append({"op": "act", "act": c["activation"]})
        elif cn in ("Dropout", "Flatten"):
            ops.append({"op": "flatten"} if cn == "Flatten" else {"op": "noop"})
        else:
            raise NotImplementedError(f"layer type {cn} not supported yet")
    walk(cfg)
    return ops

def _norm_key(name):
    parts = name.split("/")
    return parts[-2] + "/" + parts[-1].replace(":0", "")

def load_model(path):
    """Returns (weights dict 'layer/kernel'->float32 array, ops list, note)."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        W = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("__")}
        ops = json.loads(str(z["__ops__"])) if "__ops__" in z.files else DLSS_FALLBACK_OPS
        return W, ops, "npz"
    W, ops, note = {}, None, "h5 (no config: fallback ops)"
    try:
        import h5py
        f = h5py.File(path, "r")
        root = f["model_weights"] if "model_weights" in f else f
        def visit(name, node):
            if isinstance(node, h5py.Dataset):
                W[_norm_key(name)] = node[()].astype(np.float32)
        root.visititems(visit)
    except ImportError:                       # pure-python fallback (pip install pyfive==0.4.0)
        import pyfive
        f = pyfive.File(path)
        root = f["model_weights"] if "model_weights" in f.keys() else f
        def walk(g, p=""):
            for k in g.keys():
                o = g[k]
                if hasattr(o, "keys"): walk(o, p + "/" + k)
                else: W[_norm_key(p + "/" + k)] = np.asarray(o[()], np.float32)
        walk(root)
    if "model_config" in f.attrs:
        cfg = f.attrs["model_config"]
        cfg = cfg.decode("utf-8") if isinstance(cfg, bytes) else cfg
        ops, note = parse_keras_config(json.loads(cfg)), "h5 + embedded model_config"
    return W, (ops or DLSS_FALLBACK_OPS), note

def strip_clip_equivalent_tail(ops):
    """A LeakyReLU (alpha>=0) after the last conv keeps positives and keeps negatives
    negative, so clip(lrelu(y), 0, 1) == clip(y, 0, 1). Dropping it makes the final
    conv the linear output layer, where folds are exact on the clipped image."""
    tail = []
    while ops and ops[-1]["op"] == "lrelu" and ops[-1]["alpha"] >= 0:
        tail.insert(0, ops[-1]); ops = ops[:-1]
    return ops, tail

def export_npz(h5_path):
    W, ops, note = load_model(h5_path)
    out = os.path.splitext(h5_path)[0] + ".npz"
    np.savez_compressed(out, __ops__=np.array(json.dumps(ops)), **W)
    print(f"[+] wrote {out}  ({note}, {sum(v.size for v in W.values()):,} params)")
    print("    ops:", " -> ".join(o.get("name", o["op"]) + (f"(a={o['alpha']})" if o["op"] == "lrelu" else "") for o in ops))

def weights_hash(W):
    h = hashlib.sha256()
    for k in sorted(W):
        h.update(k.encode()); h.update(np.ascontiguousarray(W[k], np.float32).tobytes())
    return h.hexdigest()[:16]

# ---------------------------------------------------------------------------
# 2. Forward pass with activation caching (patched models resume from cache)
# ---------------------------------------------------------------------------
def conv2d_same(x, K, b):
    kh, kw, ci, co = K.shape
    assert kh % 2 and kw % 2, "odd kernels only"
    H, Wd, _ = x.shape
    xp = np.pad(x, ((kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
    out = np.zeros((H, Wd, co), np.float32)
    for i in range(kh):
        for j in range(kw):
            out += xp[i:i + H, j:j + Wd, :] @ K[i, j]
    return out + b

def _act(x, a):
    if a == "linear": return x
    if a == "relu": return np.maximum(x, 0)
    if a == "tanh": return np.tanh(x)
    if a == "sigmoid": return 1 / (1 + np.exp(-x))
    raise NotImplementedError(a)

def apply_op(x, op, W):
    t = op["op"]
    if t == "conv":
        return _act(conv2d_same(x, W[op["name"] + "/kernel"], W[op["name"] + "/bias"]), op["act"])
    if t == "lrelu": return np.where(x >= 0, x, op["alpha"] * x).astype(np.float32)
    if t == "up": return np.repeat(np.repeat(x, op["size"][0], 0), op["size"][1], 1)
    if t == "dense": return _act(x @ W[op["name"] + "/kernel"] + W[op["name"] + "/bias"], op["act"])
    if t == "reshape": return x.reshape(op["shape"])
    if t == "act": return _act(x, op["act"])
    if t in ("noop",): return x
    if t == "flatten": return x.reshape(-1)
    raise NotImplementedError(t)

def run(x, W, ops, cache=None, base_W=None):
    """Forward pass returning the pre-clip output.
    If cache (list of op inputs from a base run) and base_W are given, resume
    from the first op whose weights differ — patches to late layers are cheap."""
    start = 0
    if cache is not None and base_W is not None:
        start = len(ops)
        for i, op in enumerate(ops):
            n = op.get("name")
            if n and any(not np.array_equal(W[n + s], base_W[n + s]) for s in ("/kernel", "/bias")):
                start = i; break
        if start == len(ops):
            return cache[-1]
        x = cache[start]
    h = x.astype(np.float32)
    for op in ops[start:]:
        h = apply_op(h, op, W)
    return h

def run_cached(x, W, ops):
    cache, h = [], x.astype(np.float32)
    for op in ops:
        cache.append(h); h = apply_op(h, op, W)
    cache.append(h)  # final pre-clip output
    return cache

# ---------------------------------------------------------------------------
# 3. .weightpatch: text-domain edits that re-apply from JSON alone
# ---------------------------------------------------------------------------
def _assert_last_linear(ops, layer):
    last = ops[-1]
    if not (last["op"] == "conv" and last["name"] == layer and last["act"] == "linear"):
        raise ValueError(f"{layer} must be the final, linear layer for an exact fold")

def fold_spatial(K, S):
    """Compose output kernel K (kh,kw,ci,co) with a 3x3 per-channel filter S applied
    after it. Exact result is (kh+2)x(kw+2); project back to kh x kw by dropping the
    outer ring and putting its mass on the centre tap (DC response preserved)."""
    kh, kw = K.shape[:2]
    big = np.zeros((kh + 2, kw + 2) + K.shape[2:], np.float32)
    for vi in range(3):
        for vj in range(3):
            big[vi:vi + kh, vj:vj + kw] += S[vi, vj] * K
    small = big[1:-1, 1:-1].copy()
    small[kh // 2, kw // 2] += big.sum((0, 1)) - small.sum((0, 1))
    return small, big

def unsharp_kernel(lam):
    g = np.outer([1, 2, 1], [1, 2, 1]).astype(np.float32) / 16
    d = np.zeros((3, 3), np.float32); d[1, 1] = 1
    return d + lam * (d - g)          # sums to 1 for any lam; lam<0 blurs

def apply_weightpatch(W, patch, ops):
    W = {k: v.copy() for k, v in W.items()}
    for op in patch["ops"]:
        L, t = op["layer"], op["type"]
        kk, bk = L + "/kernel", L + "/bias"
        if t == "fold_output_affine":         # y' = M y + t      (exact)
            _assert_last_linear(ops, L)
            M, tt = np.array(op["M"], np.float32), np.array(op["t"], np.float32)
            W[kk] = np.einsum("hwio,po->hwip", W[kk], M).astype(np.float32)
            W[bk] = (M @ W[bk] + tt).astype(np.float32)
        elif t == "fold_output_spatial":      # y' = S * y        (exact up to 5x5 projection)
            _assert_last_linear(ops, L)
            W[kk] = fold_spatial(W[kk], np.array(op["S"], np.float32))[0]
        elif t == "set_bias":
            W[bk] = np.array(op["value"], np.float32)
        elif t == "scale_ac":                 # v1-style high-frequency gain
            K = W[kk]; dc = K.mean((0, 1), keepdims=True)
            W[kk] = (dc + op["gain"] * (K - dc)).astype(np.float32)
        else:
            raise ValueError(t)
    return W

def make_patch(pid, desc, base_hash, ops, **meta):
    return {"format": "weightpatch/v0", "id": pid, "base": base_hash,
            "description": desc, "ops": ops, **meta}

def merge_patches(pid, *patches):
    bases = {p["base"] for p in patches}
    assert len(bases) == 1, "patches target different base weights"
    return make_patch(pid, " + ".join(p["id"] for p in patches), bases.pop(),
                      [o for p in patches for o in p["ops"]], merged_from=[p["id"] for p in patches])

def changed_params(W0, W1):
    return int(sum(np.sum(W0[k] != W1[k]) for k in W0))

# ---------------------------------------------------------------------------
# 4. Test data: held-out splits
# ---------------------------------------------------------------------------
_R = getattr(Image, "Resampling", Image)
RESAMPLE = {"bicubic": _R.BICUBIC, "bilinear": _R.BILINEAR, "lanczos": _R.LANCZOS, "nearest": _R.NEAREST}

def square(img_u8, size):
    h, w = img_u8.shape[:2]; s = min(h, w)
    im = Image.fromarray(img_u8[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s, :3])
    return np.asarray(im.resize((size, size), Image.BICUBIC))

def pair_from_hr(hr_u8, lr_filter):
    lr = np.asarray(Image.fromarray(hr_u8).resize((128, 128), RESAMPLE[lr_filter]))
    return lr.astype(np.float32) / 255, hr_u8.astype(np.float32) / 255

def make_sprite(seed, size=256):
    rng = np.random.default_rng(seed)
    bg = (0, 0, 0) if rng.random() < 0.7 else tuple(int(v) for v in rng.integers(0, 40, 3))
    im = Image.new("RGB", (size, size), bg); d = ImageDraw.Draw(im)
    for _ in range(rng.integers(4, 9)):
        x0, y0 = rng.integers(0, size - 60, 2); w, h = rng.integers(30, 140, 2)
        box = [int(x0), int(y0), int(min(size - 1, x0 + w)), int(min(size - 1, y0 + h))]
        fill = tuple(int(v) for v in rng.integers(0, 256, 3)); lw = int(rng.integers(2, 7))
        kind = rng.integers(0, 3)
        if kind == 0: d.ellipse(box, fill=fill, outline=(0, 0, 0), width=lw)
        elif kind == 1: d.rectangle(box, fill=fill, outline=(0, 0, 0), width=lw)
        else:
            pts = [tuple(int(v) for v in rng.integers(0, size, 2)) for _ in range(3)]
            d.polygon(pts, fill=fill, outline=(0, 0, 0))
    for _ in range(rng.integers(2, 5)):
        p = [int(v) for v in rng.integers(0, size, 4)]
        d.line(p, fill=(0, 0, 0) if rng.random() < .6 else (255, 255, 255), width=int(rng.integers(2, 5)))
    try:
        font = ImageFont.load_default(size=26)
        d.text((10, size - 40), "LV%02d HP" % (seed % 100), fill=(255, 255, 255), font=font)
    except TypeError:
        pass
    return np.asarray(im)

def legacy_suites():
    """The exact v1 Suite A / Suite B (bilinear LR), so numbers can be matched to the v1 run."""
    Y, X = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
    a = np.stack([0.15 + 0.70 * X, 0.20 + 0.60 * Y, 0.30 + 0.50 * (1.0 - 0.5 * (X + Y))], -1)
    a[((X - 0.5) ** 2 + (Y - 0.5) ** 2) < 0.08] = [0.92, 0.78, 0.68]
    a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    img_b = Image.new("RGB", (256, 256), (0, 0, 0)); d = ImageDraw.Draw(img_b)
    d.rectangle([30, 30, 226, 226], fill=(24, 28, 44), outline=(255, 220, 60), width=4)
    d.ellipse([64, 64, 192, 192], fill=(240, 90, 70), outline=(0, 0, 0), width=6)
    d.ellipse([92, 100, 116, 124], fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    d.ellipse([140, 100, 164, 124], fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    d.line([88, 156, 168, 156], fill=(0, 0, 0), width=6)
    b8 = np.asarray(img_b)
    lr = lambda u8: np.asarray(Image.fromarray(u8).resize((128, 128), Image.BILINEAR)) / 255.0
    return {"v1_suiteA_gradient": (lr(a8).astype(np.float32), a8 / 255.0),
            "v1_suiteB_smiley": (lr(b8).astype(np.float32), b8 / 255.0)}

def build_suites(lr_filter, user_dir=None):
    S = {"natural_calib": {}, "natural_test": {}, "sprite_calib": {}, "sprite_test": {}, "legacy": legacy_suites()}
    try:
        import skimage.data as sd
        for split, names in [("natural_calib", ["rocket", "immunohistochemistry", "retina"]),
                             ("natural_test", ["astronaut", "chelsea", "coffee", "hubble_deep_field"])]:
            for n in names:
                try: S[split][n] = pair_from_hr(square(getattr(sd, n)(), 256), lr_filter)
                except Exception as e: print(f"   (skip {n}: {type(e).__name__})")
    except ImportError:
        print("   (scikit-image missing: natural photo suites skipped)")
    for s in range(6):
        S["sprite_calib"][f"sprite_{s}"] = pair_from_hr(make_sprite(s), lr_filter)
        S["sprite_test"][f"sprite_{100 + s}"] = pair_from_hr(make_sprite(100 + s), lr_filter)
    if user_dir:
        S["user"] = {}
        files = sorted(f for f in os.listdir(user_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for f in files[:24]:
            im = np.asarray(Image.open(os.path.join(user_dir, f)).convert("RGB"))
            S["user"][f] = pair_from_hr(square(im, 256), lr_filter)
    return S

# Which suites gate what
REGRESSION = ["natural_test", "user", ("legacy", "v1_suiteA_gradient")]
EDGE = ["sprite_test", ("legacy", "v1_suiteB_smiley")]

# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
def psnr(a, b):
    return float(10 * np.log10(1.0 / (np.mean((a - b) ** 2) + 1e-12)))

def metrics(pred, gt, lr, lr_filter):
    p = np.clip(pred, 0, 1)
    m = {"psnr": psnr(p, gt),
         "ssim": float(_ssim(gt, p, channel_axis=-1, data_range=1.0)) if _ssim else float("nan"),
         "color_bias": [float(v) for v in (p - gt).mean((0, 1)) * 255]}
    blk = gt.max(-1) < 0.02
    m["black_floor"] = float(p[blk].mean() * 255) if blk.mean() > 0.01 else float("nan")
    band = np.zeros(gt.shape[:2], bool); band[:8] = band[-8:] = True; band[:, :8] = band[:, -8:] = True
    m["border_psnr"] = psnr(p[band], gt[band]); m["interior_psnr"] = psnr(p[~band], gt[~band])
    down = np.asarray(Image.fromarray((p * 255).round().astype(np.uint8)).resize(lr.shape[:2][::-1], RESAMPLE[lr_filter])) / 255.0
    m["lr_consistency"] = psnr(down, lr)
    return m

def baselines(lr):
    u8 = Image.fromarray((lr * 255).round().astype(np.uint8))
    return {"nearest": np.asarray(u8.resize((256, 256), Image.NEAREST)) / 255.0,
            "bicubic": np.asarray(u8.resize((256, 256), Image.BICUBIC)) / 255.0}

# ---------------------------------------------------------------------------
# 6. Fitting patches on the calibration split (forward passes only, no gradients)
# ---------------------------------------------------------------------------
def fit_output_affine(calib, caches, n_per_img=20000, ridge=1e-3, seed=0, dark_weight=1.0):
    """Weighted ridge least squares y_target ≈ M y + t on calibration pixels.
    dark_weight > 1 up-weights pixels whose ground truth is (near) black — the
    edge case — trading some overall fidelity for a lower black floor."""
    rng = np.random.default_rng(seed); Ys, Ts = [], []
    for key, (lr, hr) in calib.items():
        y = caches[key][-1].reshape(-1, 3); t = hr.reshape(-1, 3)
        idx = rng.choice(len(y), min(n_per_img, len(y)), replace=False)
        Ys.append(y[idx]); Ts.append(t[idx])
    Y, T = np.concatenate(Ys), np.concatenate(Ts)
    w = np.where(T.max(1) < 0.02, dark_weight, 1.0)[:, None]
    X = np.hstack([Y, np.ones((len(Y), 1), np.float32)]).astype(np.float64)
    A0 = np.vstack([np.eye(3), np.zeros((1, 3))])            # ridge pulls toward "no change"
    lam = ridge * w.sum()
    A = np.linalg.solve(X.T @ (w * X) + lam * np.eye(4), X.T @ (w * T) + lam * A0)
    return A[:3].T, A[3], len(X)                              # M (3x3, out x in), t (3,)

def calib_scores(calib, caches, M, t):
    ps, bf = [], []
    for k, (lr, hr) in calib.items():
        p = np.clip(caches[k][-1] @ M.T + t, 0, 1); ps.append(psnr(p, hr))
        blk = hr.max(-1) < 0.02
        if blk.mean() > 0.01: bf.append(p[blk].mean() * 255)
    return float(np.mean(ps)), float(np.mean(bf))

def sweep_sharpen(calib, caches, W, last, lams):
    K, b = W[last + "/kernel"], W[last + "/bias"]; rows = []
    for lam in lams:
        K5 = fold_spatial(K, unsharp_kernel(lam))[0]
        scores = [psnr(np.clip(conv2d_same(caches[k][-2], K5, b), 0, 1), hr) for k, (lr, hr) in calib.items()]
        rows.append((lam, float(np.mean(scores))))
    return rows

# ---------------------------------------------------------------------------
# 7. Evaluation, gate, report
# ---------------------------------------------------------------------------
def iter_group(S, group):
    for g in group:
        if isinstance(g, tuple):
            if g[0] in S and g[1] in S[g[0]]: yield g[1], S[g[0]][g[1]], g[0]
        elif g in S:
            for k, v in S[g].items(): yield k, v, g

def evaluate(models, S, caches, base_W, ops, lr_filter):
    """models: name -> weights dict or 'nearest'/'bicubic'. Returns per-image metrics and outputs."""
    R, outs = {}, {}
    for name, W in models.items():
        R[name] = {}
        for grp in (REGRESSION, EDGE):
            for key, (lr, hr), suite in iter_group(S, grp):
                pred = baselines(lr)[W] if isinstance(W, str) else run(lr, W, ops, caches[key], base_W)
                R[name][key] = {"suite": suite, **metrics(pred, hr, lr, lr_filter)}
                outs[(name, key)] = np.clip(pred, 0, 1)
    return R, outs

def summarize(R, name, keys):
    rows = [R[name][k] for k in keys]
    f = lambda m: float(np.nanmean([r[m] for r in rows])) if rows else float("nan")
    return {m: f(m) for m in ("psnr", "ssim", "black_floor", "border_psnr", "lr_consistency")}

def gate(R, name, reg_keys, edge_keys):
    b_reg, p_reg = summarize(R, "base_model", reg_keys), summarize(R, name, reg_keys)
    b_edg, p_edg = summarize(R, "base_model", edge_keys), summarize(R, name, edge_keys)
    d_psnr, d_ssim = p_reg["psnr"] - b_reg["psnr"], p_reg["ssim"] - b_reg["ssim"]
    reg_ok = d_psnr >= -0.10 and (np.isnan(d_ssim) or d_ssim >= -0.005)
    gain_ok, floor_ok = p_edg["psnr"] > b_edg["psnr"], p_edg["black_floor"] < b_edg["black_floor"]
    verdict = ("FAIL (regression)" if not reg_ok else "FAIL (no edge gain)" if not gain_ok
               else "FAIL (black floor rises)" if not floor_ok else "PASS")
    return {"reg_dPSNR": d_psnr, "reg_dSSIM": d_ssim, "edge_dPSNR": p_edg["psnr"] - b_edg["psnr"],
            "edge_black_floor": (b_edg["black_floor"], p_edg["black_floor"]), "verdict": verdict}

def save_figure(outs, S, cols, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [("legacy", "v1_suiteB_smiley"), ("sprite_test", "sprite_101"), ("natural_test", "astronaut"),
            ("legacy", "v1_suiteA_gradient")]
    rows = [r for r in rows if r[0] in S and r[1] in S[r[0]]]
    if "user" in S and S["user"]: rows.insert(0, ("user", next(iter(S["user"]))))
    fig, ax = plt.subplots(len(rows), len(cols) + 1, figsize=(2.6 * (len(cols) + 1), 2.7 * len(rows)))
    ax = np.atleast_2d(ax)
    for i, (suite, key) in enumerate(rows):
        hr = S[suite][key][1]
        for j, c in enumerate(cols):
            p = outs[(c, key)]
            ax[i, j].imshow(p); ax[i, j].set_title(f"{c}\n{psnr(p, hr):.2f} dB", fontsize=8)
        ax[i, -1].imshow(hr); ax[i, -1].set_title(f"ground truth\n{key}", fontsize=8)
    for a in ax.flat: a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)

def experiment(W, ops, args, outdir, planted=None):
    os.makedirs(outdir, exist_ok=True); t0 = time.time()
    full_ops = ops
    ops, tail = strip_clip_equivalent_tail(ops)
    if tail:
        x = np.random.default_rng(0).random((32, 32, 3), dtype=np.float32)
        d = np.abs(np.clip(run(x, W, full_ops), 0, 1) - np.clip(run(x, W, ops), 0, 1)).max()
        print(f"[*] dropped final LeakyReLU (identical after clipping to [0,1]; max diff {d:.1e})")
        assert d == 0
    if ops[-1]["op"] != "conv":
        raise ValueError("last op must be a conv layer for output-layer folds")
    last = ops[-1]["name"]; base_hash = weights_hash(W)
    print("[*] ops:", " -> ".join(o.get("name", o["op"]) + (f"(a={o['alpha']})" if o["op"] == "lrelu" else "") for o in ops))
    print(f"[*] base weights hash {base_hash}; building suites (LR filter: {args.lr_filter})")
    S = build_suites(args.lr_filter, args.images)

    print("[*] base forward passes (cached per layer)")
    caches = {k: run_cached(lr, W, ops) for sname, d in S.items() for k, (lr, hr) in d.items()}
    calib = {**S["natural_calib"], **S["sprite_calib"]}

    # --- v1 patches, as text, plus ablations -------------------------------
    v1_bias = {"type": "set_bias", "layer": last, "value": [0.0, 0.0, 0.0]}
    v1_ac = [{"type": "scale_ac", "layer": "conv2d_8", "gain": 1.18}, {"type": "scale_ac", "layer": last, "gain": 1.25}]
    patches = {
        "v1_hotfix": make_patch("v1_hotfix", "zero output bias + AC gain (as run in v1)", base_hash, [v1_bias] + v1_ac),
        "v1_bias_only": make_patch("v1_bias_only", "ablation: zero output bias only", base_hash, [v1_bias]),
        "v1_ac_only": make_patch("v1_ac_only", "ablation: AC gain only", base_hash, v1_ac),
    }
    if "conv2d_8/kernel" not in W:  # non-DLSS model: drop the layer-specific v1 entries
        patches = {k: v for k, v in patches.items() if k == "v1_bias_only"}

    # --- Patch A: exact color/black-level fold, fit on calibration split -----
    M, t, n = fit_output_affine(calib, caches)
    patches["color_fold"] = make_patch(
        "color_fold", "affine color/black-level correction folded into the final linear conv",
        base_hash, [{"type": "fold_output_affine", "layer": last, "M": M.round(6).tolist(), "t": t.round(6).tolist()}],
        fit={"split": "natural_calib+sprite_calib", "pixels": n, "method": "ridge least squares toward identity"})

    # --- Patch A2: black-anchored color fold (edge-case hotfix) --------------
    # Sweep the weight on near-black calibration pixels; take the smallest weight
    # that halves the calibration black floor relative to the base model.
    base_bf = calib_scores(calib, caches, np.eye(3), np.zeros(3))[1]
    pareto = []
    for dw in [1, 4, 16, 64, 256, 1024]:
        Mw, tw, _ = fit_output_affine(calib, caches, dark_weight=dw)
        pareto.append((dw, *calib_scores(calib, caches, Mw, tw), Mw, tw))
    ok = [r for r in pareto if r[2] <= 0.5 * base_bf]
    dw, cps, cbf, Mb, tb = (ok[0] if ok else min(pareto, key=lambda r: r[2]))
    patches["color_fold_black"] = make_patch(
        "color_fold_black", f"affine color fold with near-black pixels weighted x{dw}",
        base_hash, [{"type": "fold_output_affine", "layer": last, "M": Mb.round(6).tolist(), "t": tb.round(6).tolist()}],
        fit={"split": "calib", "dark_weight": dw, "selection": "smallest weight halving calib black floor",
             "pareto": [{"dark_weight": r[0], "calib_psnr": round(r[1], 3), "calib_black_floor": round(r[2], 2)} for r in pareto]})

    # --- Patch B: sharpen/blur folded into final conv, strength chosen on calib
    lams = [-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    sweep = sweep_sharpen(calib, caches, W, last, lams)
    best_lam = max(sweep, key=lambda r: r[1])[0]
    if best_lam in (lams[0], lams[-1]):
        print(f"   [!] sharpen sweep peaked at the grid edge (lambda={best_lam}); widen the grid")
    patches["sharpen_fold"] = make_patch(
        "sharpen_fold", f"unsharp (lambda={best_lam}) folded into final conv, 5x5 projection",
        base_hash, [{"type": "fold_output_spatial", "layer": last, "S": unsharp_kernel(best_lam).round(6).tolist()}],
        fit={"split": "calib", "sweep": [{"lambda": l, "calib_psnr": round(s, 3)} for l, s in sweep]})

    # --- Merge (two independent branches -> one patch) ------------------------
    patches["merged"] = merge_patches("merged", patches["color_fold"], patches["sharpen_fold"])

    # --- Exactness / composability unit tests ---------------------------------
    checks = {}
    Wc = apply_weightpatch(W, patches["color_fold"], ops)
    k0 = next(iter(calib)); y0 = caches[k0][-1]
    checks["affine_fold_max_err"] = float(np.abs(run(S_lr(S, k0), Wc, ops, caches[k0], W) - (y0 @ M.T + t)).max())
    Wm1 = apply_weightpatch(Wc, patches["sharpen_fold"], ops)
    Wm2 = apply_weightpatch(apply_weightpatch(W, patches["sharpen_fold"], ops), patches["color_fold"], ops)
    checks["merge_order_max_diff"] = float(max(np.abs(Wm1[k] - Wm2[k]).max() for k in W))
    Kexact = fold_spatial(W[last + "/kernel"], unsharp_kernel(best_lam))[1]
    xin = caches[k0][-2]
    exact = conv2d_same(xin, Kexact, W[last + "/bias"]); proj = conv2d_same(xin, fold_spatial(W[last + "/kernel"], unsharp_kernel(best_lam))[0], W[last + "/bias"])
    checks["sharpen_projection_psnr_vs_exact7x7"] = psnr(np.clip(proj, 0, 1), np.clip(exact, 0, 1)) if best_lam != 0 else float("inf")
    rt = json.loads(json.dumps(patches["merged"]))
    checks["json_roundtrip_identical"] = all(np.array_equal(apply_weightpatch(W, rt, ops)[k], Wm1[k]) for k in W)

    # --- Evaluate everything -----------------------------------------------
    models = {"nearest": "nearest", "bicubic": "bicubic", "base_model": W}
    models.update({pid: apply_weightpatch(W, p, ops) for pid, p in patches.items()})
    print(f"[*] evaluating {len(models)} variants")
    R, outs = evaluate(models, S, caches, W, ops, args.lr_filter)
    reg_keys = [k for k, _, _ in iter_group(S, REGRESSION)]; edge_keys = [k for k, _, _ in iter_group(S, EDGE)]

    # --- Write artifacts ---------------------------------------------------
    pdir = os.path.join(outdir, "patches"); os.makedirs(pdir, exist_ok=True)
    for pid, p in patches.items():
        p["changed_params"] = changed_params(W, models[pid])
        with open(os.path.join(pdir, pid + ".weightpatch.json"), "w") as f: json.dump(p, f, indent=1)
    for pid in ("color_fold", "merged"):
        np.savez_compressed(os.path.join(outdir, f"generator_{pid}.npz"), __ops__=np.array(json.dumps(ops)), **models[pid])

    L = [f"# Experiment 2b results", "", f"base weights `{base_hash}`, LR filter `{args.lr_filter}`, "
         f"{len(reg_keys)} regression images, {len(edge_keys)} edge-case images, runtime {time.time() - t0:.0f}s", ""]
    if planted: L += [f"**SELF-TEST on a synthetic stand-in with a planted defect — not the real model.** {planted}", ""]
    L += ["## Summary (means over held-out images)", "",
          "| variant | reg PSNR | reg SSIM | edge PSNR | edge black floor (RGB) | border PSNR (reg) | LR-consistency (reg) | params changed | gate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in models:
        r, e = summarize(R, name, reg_keys), summarize(R, name, edge_keys)
        g = gate(R, name, reg_keys, edge_keys)["verdict"] if name in patches else ("reference" if name == "base_model" else "baseline")
        pc = patches[name]["changed_params"] if name in patches else ""
        L.append(f"| {name} | {r['psnr']:.2f} | {r['ssim']:.4f} | {e['psnr']:.2f} | {e['black_floor']:.2f} | {r['border_psnr']:.2f} | {r['lr_consistency']:.2f} | {pc} | {g} |")
    L += ["", "## Gate details", ""]
    for pid in patches:
        g = gate(R, pid, reg_keys, edge_keys)
        L.append(f"- **{pid}**: regression ΔPSNR {g['reg_dPSNR']:+.2f} dB, ΔSSIM {g['reg_dSSIM']:+.4f}; "
                 f"edge ΔPSNR {g['edge_dPSNR']:+.2f} dB; black floor {g['edge_black_floor'][0]:.2f} → {g['edge_black_floor'][1]:.2f} → **{g['verdict']}**")
    L += ["", "## Fitted values", "", f"- color_fold M (rows = output R,G,B): {np.round(M, 4).tolist()}",
          f"- color_fold t (×255): {np.round(t * 255, 2).tolist()}",
          f"- black-anchor sweep (dark weight → calib PSNR / calib black floor; base floor {base_bf:.2f}): "
          + ", ".join(f"×{r[0]}→{r[1]:.2f} dB/{r[2]:.2f}" for r in pareto) + f"; chosen ×{dw}",
          f"- sharpen sweep (λ → calib PSNR): " + ", ".join(f"{l:+.2f}→{s:.2f}" for l, s in sweep) + f"; chosen λ = {best_lam}",
          "", "## Harness checks", ""] + [f"- {k}: {v}" for k, v in checks.items()]
    L += ["", "## Per-image", "", "| variant | image | suite | PSNR | SSIM | black floor | color bias R,G,B |", "|---|---|---|---|---|---|---|"]
    for name in ("bicubic", "base_model", "v1_hotfix", "color_fold", "color_fold_black"):
        if name not in R: continue
        for k, m in R[name].items():
            L.append(f"| {name} | {k} | {m['suite']} | {m['psnr']:.2f} | {m['ssim']:.4f} | {m['black_floor']:.2f} | "
                     + ", ".join(f"{v:+.1f}" for v in m["color_bias"]) + " |")
    with open(os.path.join(outdir, "exp2b_results.md"), "w") as f: f.write("\n".join(L) + "\n")
    with open(os.path.join(outdir, "exp2b_results.json"), "w") as f:
        json.dump({"base": base_hash, "checks": checks, "per_image": R, "sweep": sweep}, f, indent=1)
    cols = [c for c in ("bicubic", "base_model", "v1_hotfix", "color_fold", "color_fold_black") if c in models]
    save_figure(outs, S, cols, os.path.join(outdir, "exp2b_comparison.png"))
    print("\n".join(L[:L.index("## Per-image")]))
    print(f"\n[+] wrote {outdir}/exp2b_results.md, exp2b_results.json, exp2b_comparison.png, patches/*.weightpatch.json")
    return R, checks, M, t

def S_lr(S, key):
    for d in S.values():
        if key in d: return d[key][0]

# ---------------------------------------------------------------------------
# 8. Self-test: a stand-in generator with a known, planted defect
# ---------------------------------------------------------------------------
def planted_standin(seed=0):
    rng = np.random.default_rng(seed); W = {}
    def noise(shape, s): return (rng.standard_normal(shape) * s).astype(np.float32)
    K = noise((5, 5, 3, 64), 0.002)
    for c in range(3): K[2, 2, c, c] = 1.0
    W["conv2d_5/kernel"], W["conv2d_5/bias"] = K, np.zeros(64, np.float32)
    for n in ("conv2d_6", "conv2d_7"):
        K = noise((5, 5, 64, 64), 0.0005)
        for c in range(3): K[2, 2, c, c] = 1.0
        W[n + "/kernel"], W[n + "/bias"] = K, np.zeros(64, np.float32)
    K = noise((5, 5, 64, 64), 0.0005); g = np.outer([1, 2, 1], [1, 2, 1]) / 16
    for c in range(3): K[1:4, 1:4, c, c] = g
    W["conv2d_8/kernel"], W["conv2d_8/bias"] = K, np.zeros(64, np.float32)
    M_bug = np.array([[0.85, 0.10, 0.0], [0.05, 0.80, 0.10], [0.0, 0.10, 0.90]], np.float32)
    K = np.zeros((5, 5, 64, 3), np.float32); K[2, 2, :3, :] = M_bug.T
    W["conv2d_9/kernel"], W["conv2d_9/bias"] = K, np.array([0.09, 0.046, 0.034], np.float32)
    return W, M_bug

def selftest(args):
    W, M_bug = planted_standin()
    note = ("Planted: output = M_bug·rgb + [0.090, 0.046, 0.034] after a blurred nearest upsample. "
            "A correct harness should fit roughly the inverse affine and raise PSNR.")
    R, checks, M, t = experiment(W, DLSS_FALLBACK_OPS, args, args.out, planted=note)
    Minv = np.linalg.inv(M_bug); tinv = -Minv @ W["conv2d_9/bias"]
    print(f"\n[selftest] fitted M vs true inverse: max |diff| = {np.abs(M - Minv).max():.3f}; "
          f"t vs true: max |diff| = {np.abs(t - tinv).max() * 255:.2f} RGB")
    assert checks["affine_fold_max_err"] < 1e-4, checks
    assert checks["merge_order_max_diff"] < 1e-6, checks
    assert checks["json_roundtrip_identical"], checks
    print("[selftest] fold exactness, merge commutativity and JSON round-trip: OK")

def verify_keras(path, W, ops):
    """Optional: compare this numpy forward pass to Keras on one input."""
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"[verify-keras] skipped ({type(e).__name__}: {str(e)[:120]})"); return
    x = np.random.default_rng(0).random((1, 128, 128, 3), dtype=np.float32)
    ref = np.asarray(m.predict(x, verbose=0))[0]; ours = run(x[0], W, ops)
    print(f"[verify-keras] max |numpy - keras| = {np.abs(ref - ours).max():.2e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights"); ap.add_argument("--images", help="folder of extra in-domain images (e.g. anime/pokemon)")
    ap.add_argument("--export-npz"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-keras", action="store_true")
    ap.add_argument("--lr-filter", default="bicubic", choices=list(RESAMPLE))
    ap.add_argument("--out", default="results/exp2b_out")
    a = ap.parse_args()
    if a.export_npz: export_npz(a.export_npz)
    elif a.selftest: selftest(a)
    elif a.weights:
        W, ops, note = load_model(a.weights); print(f"[*] loaded {a.weights} ({note})")
        if a.verify_keras: verify_keras(a.weights, W, ops)
        experiment(W, ops, a, a.out)
    else: ap.print_help()
