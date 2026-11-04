"""
Forward passes for the 2026 models (TinySR-26, CatGAN-26) from Keras-style weight dicts
(`layer/kernel` in HWIO, `layer/bias`), so the weightpatch tools apply to them unchanged.

Two backends over the same dict: NumPy (reference, CPU) and PyTorch functional (fast, GPU).
Both are checked against the trained nn.Module at export time.
"""
import numpy as np

import keras_np as kn

# ---------------------------------------------------------------------------
# Shared NumPy ops (images are (n, H, W, C) float32)
# ---------------------------------------------------------------------------
def lrelu(x, a=0.2): return np.where(x >= 0, x, a * x).astype(np.float32)

def pixel_shuffle(x, s):
    """Same channel order as torch.nn.functional.pixel_shuffle: channel c*s*s + i*s + j -> (c, row i, col j)."""
    n, H, W, C = x.shape; c = C // (s * s)
    return x.reshape(n, H, W, c, s, s).transpose(0, 1, 4, 2, 5, 3).reshape(n, H * s, W * s, c)

def cubic_matrix(n_in, s, A=-0.75):
    """Rows interpolate one output coordinate, exactly as torch bicubic (align_corners=False, edge clamp)."""
    U = np.zeros((n_in * s, n_in), np.float64)
    def w(t):
        t = abs(t)
        return ((A + 2) * t - (A + 3)) * t * t + 1 if t <= 1 else (((t - 5) * t + 8) * t - 4) * A if t < 2 else 0.0
    for o in range(n_in * s):
        src = (o + 0.5) / s - 0.5; i0 = int(np.floor(src)); t = src - i0
        for k in range(-1, 3): U[o, min(max(i0 + k, 0), n_in - 1)] += w(k - t)
    return U

def bicubic_up(x, s):
    Uh, Uw = cubic_matrix(x.shape[1], s), cubic_matrix(x.shape[2], s)
    return np.einsum("oh,nhwc,pw->nopc", Uh, x.astype(np.float64), Uw).astype(np.float32)

# ---------------------------------------------------------------------------
# TinySR-26: head -> residual blocks -> tail (+head) -> pixel-shuffle up, plus a global bicubic skip
# ---------------------------------------------------------------------------
def tinysr_blocks(W): return sorted({int(k[4:k.index("_")]) for k in W if k.startswith("body")})

def tinysr_np(x, W, s):
    """x: (n, h, w, 3) in [0, 1]. Returns (n, h*s, w*s, 3), unclipped."""
    h = kn.conv2d(x, W["head/kernel"], W["head/bias"]); r = h
    for i in tinysr_blocks(W):
        t = np.maximum(kn.conv2d(r, W[f"body{i}_a/kernel"], W[f"body{i}_a/bias"]), 0)
        r = r + kn.conv2d(t, W[f"body{i}_b/kernel"], W[f"body{i}_b/bias"])
    r = kn.conv2d(r, W["tail/kernel"], W["tail/bias"]) + h
    return pixel_shuffle(kn.conv2d(r, W["up/kernel"], W["up/bias"]), s) + bicubic_up(x, s)

# ---------------------------------------------------------------------------
# CatGAN-26, exported in the legacy Keras layout (BatchNorm folded, dense columns in Reshape((4,4,C))
# order, output folded to [0,1]): dense -> (4,4,C) -> lrelu -> [up2, conv, lrelu, conv, lrelu] x B -> conv
# ---------------------------------------------------------------------------
def catgan_blocks(W): return sorted({int(k[1:k.index("c")]) for k in W if k.startswith("b") and "/kernel" in k})

def catgan_ops(W, a=0.2):
    """The same network as an op list for the legacy tools (exp3 generate, exp4, wpgit)."""
    ops = [{"op": "dense", "name": "dense", "act": "linear"}, {"op": "reshape", "shape": [4, 4, W["dense/kernel"].shape[1] // 16]},
           {"op": "lrelu", "alpha": a}]
    for i in catgan_blocks(W):
        ops.append({"op": "up", "size": [2, 2]})
        for j in (1, 2): ops += [{"op": "conv", "name": f"b{i}c{j}", "act": "linear"}, {"op": "lrelu", "alpha": a}]
    # trailing no-op: exp4's run() reads the output conv at len(ops) - 2 (the 2018 models end in an activation)
    return ops + [{"op": "conv", "name": "out", "act": "linear"}, {"op": "noop"}]

def catgan_np(z, W, a=0.2):
    """z: (n, 128). Returns images (n, 128, 128, 3) in [0, 1] space, unclipped."""
    x = lrelu((z @ W["dense/kernel"] + W["dense/bias"]).reshape(len(z), 4, 4, -1), a)
    for i in catgan_blocks(W):
        x = x.repeat(2, 1).repeat(2, 2)
        for j in (1, 2): x = lrelu(kn.conv2d(x, W[f"b{i}c{j}/kernel"], W[f"b{i}c{j}/bias"]), a)
    return kn.conv2d(x, W["out/kernel"], W["out/bias"])

# ---------------------------------------------------------------------------
# PyTorch functional versions of the same dicts (imported lazily so NumPy-only users need no torch)
# ---------------------------------------------------------------------------
def to_torch(W, device="cuda"):
    import torch
    T = {}
    for k, v in W.items():
        if not hasattr(v, "ndim") or v.ndim == 0: continue
        t = torch.from_numpy(np.ascontiguousarray(v)).float()
        T[k] = (t.permute(3, 2, 0, 1).contiguous() if t.ndim == 4 else t).to(device)       # HWIO -> OIHW
    return T

def legacy_torch(x, T, ops):
    """GPU version of the legacy conv / lrelu / up op lists (TinySR-18); odd kernels, 'same' padding."""
    import torch.nn.functional as F
    for op in ops:
        if op["op"] == "conv": x = F.conv2d(x, T[op["name"] + "/kernel"], T[op["name"] + "/bias"], padding=T[op["name"] + "/kernel"].shape[-1] // 2)
        elif op["op"] == "lrelu": x = F.leaky_relu(x, op["alpha"])
        elif op["op"] == "up": x = F.interpolate(x, scale_factor=tuple(op["size"]), mode="nearest")
        else: raise NotImplementedError(op["op"])
    return x

def tinysr_torch(x, T, s):
    import torch.nn.functional as F
    c = lambda h, n: F.conv2d(h, T[n + "/kernel"], T[n + "/bias"], padding=T[n + "/kernel"].shape[-1] // 2)
    h = c(x, "head"); r = h
    for i in sorted({int(k[4:k.index("_")]) for k in T if k.startswith("body")}):
        r = r + c(F.relu(c(r, f"body{i}_a")), f"body{i}_b")
    r = c(r, "tail") + h
    return F.pixel_shuffle(c(r, "up"), s) + F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)

def catgan_torch(z, T, a=0.2):
    import torch.nn.functional as F
    x = F.leaky_relu((z @ T["dense/kernel"] + T["dense/bias"]).view(len(z), 4, 4, -1).permute(0, 3, 1, 2), a)
    for i in sorted({int(k[1:k.index("c")]) for k in T if k.startswith("b") and "/kernel" in k}):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        for j in (1, 2): x = F.leaky_relu(F.conv2d(x, T[f"b{i}c{j}/kernel"], T[f"b{i}c{j}/bias"], padding=1), a)
    return F.conv2d(x, T["out/kernel"], T["out/bias"], padding=1)
