"""
Circuits for trained networks: exact activation identities, and channel freeing by regression.

A trained network with (Leaky)ReLU activations can host new computation if we can (a) find room
for it and (b) express it with the layer types the network already has. This module is the
instruction set for both.

Room:
  * free_channels: regress k channels of a layer onto the others, fold that regression into the
    next layer and zero the freed slots (approximate: costs the unexplained variance).
  * width growth: a new channel whose outgoing weights are zero changes nothing (exact).

Exact identities for LeakyReLU with slope a (ReLU is a = 0), one hidden layer each:
  carry      u >= 0:      lrelu(u) = u                                     1 channel
  pass       any u:       (lrelu(u) - lrelu(-u)) / (1 + a) = u              2 channels
  relu       any u:       (lrelu(u) + a lrelu(-u)) / (1 - a^2) = relu(u)     2 channels (a < 1)
  abs        any u:       (lrelu(u) + lrelu(-u)) / (1 - a) = |u|             2 channels
  min, max   min(x, y) = x - relu(x - y);  max(x, y) = y + relu(x - y)       4 channels
  clamp      lo + relu(x - lo) - relu(x - hi)                                4 channels
  step       relu(k (u - t)) - relu(k (u - t) - 1): exactly 0 / 1 outside [t, t + 1/k]
  and, or    relu(b1 + b2 - 1);  b1 + b2 - relu(b1 + b2 - 1)   (binary b)
  gate       relu(p - t - M (1 - g)): p - t where binary g = 1 (and p > t), 0 where g = 0
  mux        y + relu(x - y - M (1 - g)) - relu(y - x - M (1 - g)) = g ? x : y   (|x - y| <= M)
"""
import numpy as np

# ---------------------------------------------------------------------------------------------
# 1. The identities as hidden-layer recipes: (pre-activation rows, readout) over named inputs.
#    Each recipe is compiled into a real 1-hidden-layer network by compile_recipe() and tested.
# ---------------------------------------------------------------------------------------------
def lrelu(u, a): return np.where(u >= 0, u, a * u)

def recipe(name, a, M=8.0, k=64.0, t=0.0, lo=-0.5, hi=0.5):
    """Returns (inputs, hidden pre-activations as (weights over inputs, bias), readout weights, target fn)."""
    g, p = 1.0 / (1 - a * a), 1.0 / (1 + a)
    R = {
        "carry": (["u"], [([1], 0)], [1], lambda u: u),                        # inputs must be >= 0
        "pass":  (["u"], [([1], 0), ([-1], 0)], [p, -p], lambda u: u),
        "relu":  (["u"], [([1], 0), ([-1], 0)], [g, a * g], lambda u: np.maximum(u, 0)),
        "abs":   (["u"], [([1], 0), ([-1], 0)], [1 / (1 - a), 1 / (1 - a)], np.abs),
        "min":   (["x", "y"], [([1, 0], 0), ([-1, 0], 0), ([1, -1], 0), ([-1, 1], 0)], [p, -p, -g, -a * g], np.minimum),
        "max":   (["x", "y"], [([0, 1], 0), ([0, -1], 0), ([1, -1], 0), ([-1, 1], 0)], [p, -p, g, a * g], np.maximum),
        "clamp": (["x"], [([1], -lo), ([-1], lo), ([1], -hi), ([-1], hi)], [g, a * g, -g, -a * g], lambda x: np.clip(x, lo, hi), lo),
        "step":  (["u"], [([k], -k * t), ([-k], k * t), ([k], -k * t - 1), ([-k], k * t + 1)], [g, a * g, -g, -a * g],
                  lambda u: (u >= t + 1 / k).astype(float)),                  # tested outside the band
        "and":   (["b1", "b2"], [([1, 1], -1), ([-1, -1], 1)], [g, a * g], lambda b1, b2: b1 * b2),
        "or":    (["b1", "b2"], [([1, 1], 0), ([-1, -1], 0), ([1, 1], -1), ([-1, -1], 1)], [p, -p, -g, -a * g],
                  lambda b1, b2: np.maximum(b1, b2)),
        "gate":  (["p", "g"], [([1, M], -t - M), ([-1, -M], t + M)], [g, a * g], lambda p_, g_: g_ * np.maximum(p_ - t, 0)),
        "mux":   (["g", "x", "y"], [([0, 0, 1], 0), ([0, 0, -1], 0), ([M, 1, -1], -M), ([-M, -1, 1], M), ([M, -1, 1], -M), ([-M, 1, -1], M)],
                  [p, -p, g, a * g, -g, -a * g], lambda g_, x, y: np.where(g_ > 0.5, x, y)),
    }
    r = R[name]
    return r if len(r) == 5 else r + (0.0,)

def compile_recipe(name, a, **kw):
    """-> (W1 (n_in, n_hidden), b1, w2 (n_hidden,), c, target fn, input names): f(X) = lrelu(X W1 + b1) w2 + c."""
    ins, rows, w2, fn, c = recipe(name, a, **kw)
    W1 = np.array([r[0] for r in rows], np.float64).T; b1 = np.array([r[1] for r in rows], np.float64)
    return W1, b1, np.array(w2, np.float64), c, fn, ins

def sample_inputs(name, n, rng, M=8.0, k=64.0, t=0.0):
    if name == "carry": return {"u": rng.uniform(0, 5, n)}
    if name in ("and", "or"): return {"b1": rng.integers(0, 2, n).astype(float), "b2": rng.integers(0, 2, n).astype(float)}
    if name == "gate": return {"p": rng.uniform(-M / 2, M / 2, n), "g": rng.integers(0, 2, n).astype(float)}
    if name == "mux": return {"g": rng.integers(0, 2, n).astype(float), "x": rng.uniform(-M / 2, M / 2, n), "y": rng.uniform(-M / 2, M / 2, n)}
    if name == "step":
        u = rng.normal(t, 1, n); band = (u > t) & (u < t + 1 / k); return {"u": u[~band]}
    if name in ("min", "max"): return {"x": rng.normal(0, 3, n), "y": rng.normal(0, 3, n)}
    return {"u" if name not in ("clamp",) else "x": rng.normal(0, 3, n)}

# ---------------------------------------------------------------------------------------------
# 2. Room: free k channels of a producer layer by regressing them onto the rest and folding the
#    regression into the consumer. Activations are the producer's outputs *after* its activation
#    (nearest upsampling in between commutes with a per-pixel linear map, so it is allowed).
# ---------------------------------------------------------------------------------------------
def gram(acts_batches):
    """Sum of [X 1]^T [X 1] over batches of (N, C) activations, in float64."""
    G = None
    for X in acts_batches:
        Xb = np.c_[np.asarray(X, np.float64), np.ones(len(X))]
        G = Xb.T @ Xb if G is None else G + Xb.T @ Xb
    return G

def choose_free(G, out_energy, k, ridge=1e-6, protect=()):
    """Greedy: repeatedly free the channel with the smallest (unexplained variance x outgoing energy),
    regressing on the channels still kept. Returns (drop, keep, B) with B (|keep|+1, k) mapping
    [kept activations, 1] -> freed activations. `protect` channels are never freed."""
    C = G.shape[0] - 1; N = G[-1, -1]; keep = list(range(C)); drop = []; cost_log = []
    lam = ridge * np.trace(G[:C, :C]) / C
    for _ in range(k):
        idx = keep + [C]; Gk = G[np.ix_(idx, idx)] + lam * np.diag([1.0] * len(keep) + [0.0])
        P = np.linalg.inv(Gk); resid = 1.0 / np.diag(P)[:-1] / N          # mean squared residual of each kept channel
        cost = resid * out_energy[keep]
        for c in protect:
            if c in keep: cost[keep.index(c)] = np.inf
        j = int(np.argmin(cost)); cost_log.append(float(cost[j])); drop.append(keep.pop(j))
    idx = keep + [C]; Gk = G[np.ix_(idx, idx)] + lam * np.diag([1.0] * len(keep) + [0.0])
    B = np.linalg.solve(Gk, G[np.ix_(idx, drop)])
    return drop, keep, B, cost_log

def fold_free(W, producer, consumer, drop, keep, B, dense_cells=None):
    """Zero the freed channels in `producer` and fold their regression into `consumer` (HWIO kernel).
    dense_cells: for a dense producer reshaped to (h, w, C), the number of spatial cells h*w."""
    Wp = {k_: v.copy() for k_, v in W.items()}
    K = Wp[consumer + "/kernel"].astype(np.float64); bias = Wp[consumer + "/bias"].astype(np.float64)
    for j, c in enumerate(drop):
        Kc = K[:, :, c, :].copy()
        K[:, :, keep, :] += Kc[:, :, None, :] * B[:-1, j][None, None, :, None]
        bias += B[-1, j] * Kc.sum((0, 1))                                    # exact away from zero-padded borders
        K[:, :, c, :] = 0
        if dense_cells is None:
            Wp[producer + "/kernel"][..., c] = 0; Wp[producer + "/bias"][c] = 0
        else:
            C = Wp[producer + "/bias"].shape[0] // dense_cells; cols = c + C * np.arange(dense_cells)
            Wp[producer + "/kernel"][:, cols] = 0; Wp[producer + "/bias"][cols] = 0
    Wp[consumer + "/kernel"] = K.astype(np.float32); Wp[consumer + "/bias"] = bias.astype(np.float32)
    return Wp

def out_energy(W, consumer):
    """Energy of each input channel's outgoing weights in the consumer kernel (HWIO)."""
    return (W[consumer + "/kernel"].astype(np.float64) ** 2).sum((0, 1, 3))
