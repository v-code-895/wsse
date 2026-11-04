"""
keras_np: run Keras 2.x .h5 / tfjs models in numpy, no TensorFlow needed.

Covers the layer types used across these repos: Conv2D (any kernel size, 'same'
or 'valid', stride 1), BatchNormalization (inference), MaxPooling2D, LeakyReLU,
UpSampling2D (nearest), Dense, Flatten, Reshape, Dropout, Activation.

    W, layers = load("model.h5")          # W: {"layer/param": array}, layers: list of dicts
    y = forward(x, W, layers)             # x: (n, ...) float32
    y, taps = forward(x, W, layers, tap=True)   # taps[i] = output of layer i
"""
import json, os
import numpy as np

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_config(cfg):
    layers = []
    def walk(l):
        cn, c = l["class_name"], l["config"]
        if cn == "Sequential":
            for s in (c["layers"] if isinstance(c, dict) else c): walk(s)
        elif cn in ("Model", "Functional"):
            for s in c["layers"]: walk(s)
        elif cn == "InputLayer":
            pass
        else:
            d = {"type": cn, "name": c.get("name")}
            for k in ("filters", "kernel_size", "strides", "padding", "activation", "units", "pool_size",
                      "alpha", "epsilon", "size", "target_shape", "rate", "use_bias", "interpolation"):
                if k in c: d[k] = c[k]
            layers.append(d)
    walk(cfg)
    return layers

def _key(name):
    p = name.split("/")
    return p[-2] + "/" + p[-1].replace(":0", "")

def load(path):
    if path.endswith(".json"):
        j = json.load(open(path)); topo = j["modelTopology"]
        layers = parse_config(topo.get("model_config", topo)); W = {}; d = os.path.dirname(path)
        for m in j["weightsManifest"]:
            buf = b"".join(open(os.path.join(d, p), "rb").read() for p in m["paths"]); off = 0
            for w in m["weights"]:
                n = int(np.prod(w["shape"])) * 4
                W[w["name"]] = np.frombuffer(buf[off:off + n], np.float32).reshape(w["shape"]).copy(); off += n
        return W, layers
    import h5py
    W = {}
    with h5py.File(path, "r") as f:
        cfg = f.attrs["model_config"]; cfg = cfg.decode() if isinstance(cfg, bytes) else cfg
        root = f["model_weights"] if "model_weights" in f else f
        root.visititems(lambda n, o: W.__setitem__(_key(n), o[()].astype(np.float32)) if hasattr(o, "shape") else None)
    return W, parse_config(json.loads(cfg))

def nparams(W): return int(sum(v.size for v in W.values()))

# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
def conv2d(x, K, b=None, padding="same"):
    """x (n,H,W,C), K (kh,kw,ci,co). 'same' pads like TensorFlow: the extra
    pixel of an even kernel goes on the bottom/right."""
    kh, kw = K.shape[:2]
    if padding == "same":
        ph, pw = kh - 1, kw - 1
        x = np.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    n, H, Wd, _ = x.shape; Ho, Wo = H - kh + 1, Wd - kw + 1
    out = np.zeros((n, Ho, Wo, K.shape[3]), np.float32)
    for i in range(kh):
        for j in range(kw):
            out += x[:, i:i + Ho, j:j + Wo, :] @ K[i, j]
    return out if b is None else out + b

ACT = {"linear": lambda x: x, "relu": lambda x: np.maximum(x, 0), "tanh": np.tanh,
       "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
       "softmax": lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True),
       "selu": lambda x: 1.0507009873554805 * np.where(x > 0, x, 1.6732632423543772 * (np.exp(np.minimum(x, 0)) - 1)),
       "elu": lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)}

def apply(x, L, W):
    t, n = L["type"], L["name"]
    if t == "Conv2D":
        assert list(L.get("strides", [1, 1])) == [1, 1], "stride-1 only"
        return ACT[L.get("activation", "linear")](conv2d(x, W[n + "/kernel"], W.get(n + "/bias"), L.get("padding", "valid")))
    if t == "BatchNormalization":
        g = W.get(n + "/gamma", 1.0); be = W.get(n + "/beta", 0.0)
        return (x - W[n + "/moving_mean"]) / np.sqrt(W[n + "/moving_variance"] + L.get("epsilon", 1e-3)) * g + be
    if t == "MaxPooling2D":
        p = L.get("pool_size", [2, 2]); s, _ = p
        N, H, Wd, C = x.shape; H2, W2 = H // s, Wd // s
        return x[:, :H2 * s, :W2 * s].reshape(N, H2, s, W2, s, C).max((2, 4))
    if t == "LeakyReLU": return np.where(x >= 0, x, np.float32(L.get("alpha", 0.3)) * x)
    if t == "UpSampling2D": return x.repeat(L["size"][0], 1).repeat(L["size"][1], 2)
    if t == "Dense": return ACT[L.get("activation", "linear")](x @ W[n + "/kernel"] + W.get(n + "/bias", 0))
    if t == "Flatten": return x.reshape(len(x), -1)
    if t == "Reshape": return x.reshape((len(x),) + tuple(L["target_shape"]))
    if t == "Dropout": return x
    if t == "Activation": return ACT[L["activation"]](x)
    raise NotImplementedError(t)

def forward(x, W, layers, tap=False, start=0, stop=None, chunk=None):
    if chunk and len(x) > chunk:
        parts = [forward(x[i:i + chunk], W, layers, tap, start, stop) for i in range(0, len(x), chunk)]
        if not tap: return np.concatenate(parts)
        return np.concatenate([p[0] for p in parts]), {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
    h, taps = x.astype(np.float32), {}
    for i, L in enumerate(layers[start:stop], start):
        h = apply(h, L, W)
        if tap: taps[i] = h
    return (h, taps) if tap else h

def summary(layers):
    return " -> ".join(L["name"] + ({"Conv2D": f"[{L.get('kernel_size')},{L.get('padding')},{L.get('activation')}]",
                                      "Dense": f"[{L.get('units')},{L.get('activation')}]"}.get(L["type"], "")) for L in layers)
