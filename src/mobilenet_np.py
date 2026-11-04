"""
mobilenet_np: MobileNetV2 (alpha 1.0, 224 px, ImageNet) in numpy, from the Keras
2.1.6 weights file (layer names mobl{i}_conv_{i}_expand / _depthwise / _project).

    W = load()                       # {"layer/param": array}
    p = predict(images_uint8)        # softmax over 1000 classes
    logits, feats = forward(x, W, return_features=True)   # feats: 1280-d pooled features
"""
import json, os
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "data", "mobilenet_v2", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5")
LABELS = os.path.join(HERE, "..", "data", "mobilenet_v2", "imagenet_class_index.json")

# (block id, expansion, output channels, stride)
BLOCKS = [(0, 1, 16, 1), (1, 6, 24, 2), (2, 6, 24, 1), (3, 6, 32, 2), (4, 6, 32, 1), (5, 6, 32, 1),
          (6, 6, 64, 2), (7, 6, 64, 1), (8, 6, 64, 1), (9, 6, 64, 1), (10, 6, 96, 1), (11, 6, 96, 1),
          (12, 6, 96, 1), (13, 6, 160, 2), (14, 6, 160, 1), (15, 6, 160, 1), (16, 6, 320, 1)]
EPS = 1e-3

def load(path=WEIGHTS):
    W = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: W.__setitem__("/".join(n.split("/")[-2:]).replace(":0", ""), o[()].astype(np.float32))
                     if hasattr(o, "shape") else None)
    return W

def labels():
    j = json.load(open(LABELS)); return [j[str(i)][1] for i in range(1000)]

def names(i):
    """Layer names of block i: (expand, depthwise, project) convs and their BNs."""
    return (f"mobl{i}_conv_{i}_expand", f"mobl{i}_conv_{i}_depthwise", f"mobl{i}_conv_{i}_project",
            f"bn{i}_conv_{i}_bn_expand", f"bn{i}_conv_{i}_bn_depthwise", f"bn{i}_conv_{i}_bn_project")

# ---------------------------------------------------------------------------
def pad_same(x, k, s, value=0.0):
    """TensorFlow 'same' padding (extra pixel after, for even totals)."""
    H = x.shape[1]; out = -(-H // s); tot = max((out - 1) * s + k - H, 0)
    return np.pad(x, ((0, 0), (tot // 2, tot - tot // 2), (tot // 2, tot - tot // 2), (0, 0)), constant_values=value), out

def conv3x3(x, K, s, pad_value=0.0):
    xp, o = pad_same(x, 3, s, pad_value); out = np.zeros((x.shape[0], o, o, K.shape[3]), np.float32)
    for i in range(3):
        for j in range(3): out += xp[:, i:i + s * o:s, j:j + s * o:s] @ K[i, j]
    return out

def depthwise(x, K, s):
    xp, o = pad_same(x, 3, s); out = np.zeros((x.shape[0], o, o, x.shape[3]), np.float32)
    for i in range(3):
        for j in range(3): out += xp[:, i:i + s * o:s, j:j + s * o:s] * K[i, j, :, 0]
    return out

def bn(x, W, n):
    if n + "/gamma" not in W: return x                  # already folded
    return (x - W[n + "/moving_mean"]) / np.sqrt(W[n + "/moving_variance"] + EPS) * W[n + "/gamma"] + W[n + "/beta"]

def add_bias(x, W, n): return x + W[n + "/bias"] if n + "/bias" in W else x
relu6 = lambda x: np.clip(x, 0, 6)

def forward(x, W, return_features=False, taps=None):
    """x: (n,224,224,3) preprocessed to [-1,1]. taps: optional dict to collect named activations.
    W may carry "Conv1/pad_value" (a scalar) when an input conversion was folded into Conv1."""
    pv = float(W.get("Conv1/pad_value", 0.0))
    h = relu6(bn(add_bias(conv3x3(x.astype(np.float32), W["Conv1/kernel"], 2, pv), W, "Conv1"), W, "bn_Conv1"))
    if taps is not None: taps["Conv1_raw"] = conv3x3(x.astype(np.float32), W["Conv1/kernel"], 2)
    cin = 32
    for i, t, c, s in BLOCKS:
        e, d, p, be, bd, bp = names(i); inp = h
        if t != 1: h = relu6(bn(add_bias(h @ W[e + "/kernel"][0, 0], W, e), W, be))
        h = relu6(bn(add_bias(depthwise(h, W[d + "/depthwise_kernel"], s), W, d), W, bd))
        h = bn(add_bias(h @ W[p + "/kernel"][0, 0], W, p), W, bp)
        if s == 1 and cin == c: h = h + inp
        if taps is not None: taps[f"block_{i}"] = h
        cin = c
    h = relu6(bn(add_bias(h @ W["Conv_1/kernel"][0, 0], W, "Conv_1"), W, "Conv_1_bn"))
    feats = h.mean((1, 2))                               # 7x7 average pool + flatten
    logits = feats @ W["Logits/kernel"] + W["Logits/bias"]
    return (logits, feats) if return_features else logits

def preprocess(img_u8):
    from PIL import Image
    a = np.asarray(img_u8)[..., :3]; h, w = a.shape[:2]; s = min(h, w)
    a = a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    a = np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC)).astype(np.float32)
    return a / 127.5 - 1                                  # keras mobilenet_v2 preprocess_input ("tf" mode)

def softmax(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def predict(imgs_u8, W=None, chunk=8):
    W = W or load(); x = np.stack([preprocess(i) for i in imgs_u8])
    return softmax(np.concatenate([forward(x[k:k + chunk], W) for k in range(0, len(x), chunk)]))
