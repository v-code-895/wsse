"""
Experiment 4: recolour the Anime DCGAN's eyes with a hand-built weight circuit.

Exp 3 showed latent edits cannot do this cleanly: pushing z toward "red eyes"
washes the eyes out to skin tone and changes the whole face. A 1x1 colour fold
on the last layer reaches the target eye colour but tints everything, because
a linear edit has no way to switch on only at the eyes.

So we build a gate out of the network's own nonlinearity, without adding layers:
  1. prune the two most redundant conv2d_9 channels; their contribution is
     regressed onto the remaining channels and folded into conv2d_10
  2. reuse the two freed slots for an eye detector p (a 3x3x64 linear probe on
     conv2d_8's output, fit to a pixel mask). With LeakyReLU slope a,
        lrelu(u) + a*lrelu(-u) = (1 - a^2) relu(u)
     so slots  [p - theta, -(p - theta)]  give an exact ReLU gate
  3. conv2d_10 writes a colour vector v (fit by least squares) through the gate

Every edit is saved as a sparse-diff .weightpatch (tensor, flat indices, deltas)
so it re-applies from text, and pruning and gating are separate patches.

  python src/exp4_anime_eye_circuit.py --weights legacy/Anime-DCGAN/generator50000.h5
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2b_tinysr18_weightpatch import load_model, weights_hash, make_patch, merge_patches, changed_params
from exp3_anime_weightpatch import apply_op_b, save_grid

SWAP_RG = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], np.float32)   # teal -> magenta, keeps shading

# ---------------------------------------------------------------------------
# 1. Forward pass with the tensors this experiment needs
# ---------------------------------------------------------------------------
def run(z, W, ops, chunk=100, keep=False):
    """Returns clipped image; with keep=True also conv2d_8 / conv2d_9 activations
    (post-LeakyReLU) and the pre-activation output of the last conv."""
    imgs, h8s, h9s, pres = [], [], [], []
    convs = [i for i, o in enumerate(ops) if o["op"] == "conv"]
    i8, i9 = convs[-3] + 1, convs[-2] + 1          # index of the lrelu after each
    for s in range(0, len(z), chunk):
        h = z[s:s + chunk].astype(np.float32)
        for i, op in enumerate(ops):
            h = apply_op_b(h, op, W)
            if keep and i == i8: h8s.append(h)
            if keep and i == i9: h9s.append(h)
            if keep and i == len(ops) - 2: pres.append(h)
        imgs.append(np.clip(h, 0, 1))
    img = np.concatenate(imgs)
    return (img, np.concatenate(h8s), np.concatenate(h9s), np.concatenate(pres)) if keep else img

def eye_mask(x):
    """Teal iris pixels in the eye band. Defined on base images only; edited
    images are scored at the same pixels."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    m = (g - r > 0.25) & (b - r > 0.15) & (np.abs(g - b) < 0.25)
    m[:, :22] = m[:, 50:] = False; m[:, :, :6] = m[:, :, 58:] = False
    return m

def patches3x3(h, idx):
    """3x3 neighbourhoods of h (n,H,W,C) at flat pixel indices idx -> (len(idx), 9C)."""
    n, H, Wd, C = h.shape
    hp = np.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)))
    im, rem = np.divmod(idx, H * Wd); y, x = np.divmod(rem, Wd)
    return np.stack([hp[im, y + i, x + j] for i in range(3) for j in range(3)], 1).reshape(len(idx), -1)

# ---------------------------------------------------------------------------
# 2. Sparse-diff weightpatch (any edit becomes a text diff of the weights)
# ---------------------------------------------------------------------------
def diff_ops(W0, W1, tol=0.0):
    ops = []
    for k in sorted(W0):
        d = (W1[k].astype(np.float64) - W0[k].astype(np.float64)).ravel()
        idx = np.nonzero(np.abs(d) > tol)[0]
        if len(idx):
            ops.append({"type": "sparse_add", "tensor": k, "shape": list(W0[k].shape),
                        "idx": idx.tolist(), "val": [float(np.float32(v)) for v in d[idx]]})
    return ops

def apply_weightpatch(W, patch, ops=None):
    W = {k: v.copy() for k, v in W.items()}
    for op in patch["ops"]:
        if op["type"] != "sparse_add": raise ValueError(op["type"])
        t = W[op["tensor"]]; assert list(t.shape) == op["shape"]
        flat = t.reshape(-1); flat[np.array(op["idx"], np.int64)] += np.array(op["val"], np.float32)
    return W

# ---------------------------------------------------------------------------
# 3. Edits
# ---------------------------------------------------------------------------
def prune_fold(W, h9, n_prune, n_px, rng, ridge=1e-3, l9="conv2d_9", l10="conv2d_10"):
    """Pick the conv2d_9 channels whose removal costs least, regress each onto the
    kept channels (1x1, per pixel) and fold that into conv2d_10. Freed slots are
    zeroed (kernel, bias, outgoing weights)."""
    A = h9.reshape(-1, h9.shape[-1]); A = A[rng.choice(len(A), n_px, replace=False)].astype(np.float64)
    K10 = W[l10 + "/kernel"]; C = A.shape[1]
    cost = np.zeros(C)
    for c in range(C):                           # unexplained variance x outgoing weight energy
        X = np.c_[np.delete(A, c, 1), np.ones(len(A))]
        beta = np.linalg.lstsq(X, A[:, c], rcond=None)[0]
        cost[c] = np.mean((A[:, c] - X @ beta) ** 2) * np.sum(K10[:, :, c, :] ** 2)
    drop = list(np.argsort(cost)[:n_prune]); keep = [c for c in range(C) if c not in drop]
    X = np.c_[A[:, keep], np.ones(len(A))]
    B = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ A[:, drop])   # (|keep|+1, n_prune)
    Wp = {k: v.copy() for k, v in W.items()}; K = Wp[l10 + "/kernel"].astype(np.float64)
    for j, c in enumerate(drop):
        Kc = K[:, :, c, :].copy()
        K[:, :, keep, :] += Kc[:, :, None, :] * B[:-1, j][None, None, :, None]
        # constant term: K_c * beta0 is a constant field except at the zero-padded border
        Wp[l10 + "/bias"] = (Wp[l10 + "/bias"] + B[-1, j] * Kc.sum((0, 1))).astype(np.float32)
        K[:, :, c, :] = 0
        Wp[l9 + "/kernel"][..., c] = 0; Wp[l9 + "/bias"][c] = 0
    Wp[l10 + "/kernel"] = K.astype(np.float32)
    return Wp, drop, cost

def fit_gate(h8, mask, n_neg, rng, q, ridge=1.0):
    """Linear eye detector on 3x3x64 neighbourhoods of conv2d_8 output.
    theta = q-quantile of the probe on non-eye pixels, so the gate is ~0 there."""
    flatm = mask.ravel(); pos = np.nonzero(flatm)[0]; neg = np.nonzero(~flatm)[0]
    neg = rng.choice(neg, n_neg, replace=False)
    X = np.r_[patches3x3(h8, pos), patches3x3(h8, neg)].astype(np.float64)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sw = np.r_[np.full(len(pos), len(neg) / len(pos)), np.ones(len(neg))]   # balance classes
    Xb = np.c_[X, np.ones(len(X))]
    w = np.linalg.solve((Xb * sw[:, None]).T @ Xb + ridge * np.eye(Xb.shape[1]), (Xb * sw[:, None]).T @ y)
    p = Xb @ w; theta = np.quantile(p[len(pos):], q)
    return w[:-1].reshape(3, 3, -1), w[-1], theta, pos, neg

def install_gate(W, slots, kern, bias, theta, v, alpha, l9="conv2d_9", l10="conv2d_10"):
    """slot0 = lrelu(p - theta), slot1 = lrelu(theta - p); conv2d_10 centre tap
    reads  v/(1-a^2) * (slot0 + a*slot1) = v * relu(p - theta)."""
    Wp = {k: v_.copy() for k, v_ in W.items()}
    s0, s1 = slots; g = 1.0 / (1 - alpha ** 2)
    Wp[l9 + "/kernel"][..., s0] = kern; Wp[l9 + "/bias"][s0] = bias - theta
    Wp[l9 + "/kernel"][..., s1] = -kern; Wp[l9 + "/bias"][s1] = theta - bias
    Wp[l10 + "/kernel"][:, :, [s0, s1], :] = 0
    Wp[l10 + "/kernel"][1, 1, s0, :] = g * v
    Wp[l10 + "/kernel"][1, 1, s1, :] = g * alpha * v
    return Wp

def fold_1x1_colour(W, h9, pre, mask, n_neg, rng, w_eye, l10="conv2d_10", target=None):
    """Exp-4 baseline: least-squares 1x1 colour fold over all 64 channels (no gate)."""
    flatm = mask.reshape(-1); A = h9.reshape(-1, h9.shape[-1]); Y = pre.reshape(-1, 3)
    pos = np.nonzero(flatm)[0]; neg = rng.choice(np.nonzero(~flatm)[0], n_neg, replace=False)
    Xe, Xn = np.c_[A[pos], np.ones(len(pos))], np.c_[A[neg], np.ones(len(neg))]
    T = np.clip(Y[pos], 0, 1) @ (SWAP_RG if target is None else target).T - Y[pos]; sw = w_eye * len(neg) / len(pos)
    V = np.linalg.solve(sw * Xe.T @ Xe + Xn.T @ Xn + 1e-3 * np.eye(Xe.shape[1]), sw * Xe.T @ T)
    Wp = {k: v.copy() for k, v in W.items()}
    Wp[l10 + "/kernel"][1, 1] += V[:-1].astype(np.float32); Wp[l10 + "/bias"] += V[-1].astype(np.float32)
    return Wp

# ---------------------------------------------------------------------------
# 4. Metrics (held-out)
# ---------------------------------------------------------------------------
def hue(c): return np.degrees(np.arctan2(np.sqrt(3) * (c[..., 1] - c[..., 2]), 2 * c[..., 0] - c[..., 1] - c[..., 2])) % 360

def score(x1, x0, m0, target=None):
    ok = m0.sum((1, 2)) >= 8
    ec = np.array([x1[i][m0[i]].mean(0) for i in range(len(x1)) if ok[i]])
    tg = np.array([(x0[i][m0[i]] @ (SWAP_RG if target is None else target).T).mean(0) for i in range(len(x1)) if ok[i]])
    far = np.zeros_like(m0)            # pixels > 3px from any eye pixel: "rest of the face"
    for dy in range(-3, 4):
        for dx in range(-3, 4): far |= np.roll(np.roll(m0, dy, 1), dx, 2)
    far = ~far
    return {"eye_hue": float(np.median(hue(ec))), "eye_rgb": ec.mean(0).round(3).tolist(),
            "eye_dist_to_target": float(np.abs(ec - tg).mean()),
            "rest_mae": float(np.mean([np.abs(x1[i] - x0[i])[far[i]].mean() for i in range(len(x1))])),
            "rest_max_px_change_p99": float(np.mean([np.quantile(np.abs(x1[i] - x0[i])[far[i]].max(-1), 0.99) for i in range(len(x1))]))}

# ---------------------------------------------------------------------------
# 5. Experiment
# ---------------------------------------------------------------------------
def experiment(W, ops, a):
    os.makedirs(os.path.join(a.out, "patches"), exist_ok=True)
    alpha = next(o["alpha"] for o in ops if o["op"] == "lrelu")
    base = weights_hash(W); rng = np.random.default_rng(a.seed)
    z_fit = rng.standard_normal((a.n_fit, 64)); z_test = rng.standard_normal((a.n_test, 64))
    t0 = time.time()
    xf, h8, h9, pre = run(z_fit, W, ops, keep=True); mf = eye_mask(xf)
    x0 = run(z_test, W, ops); m0 = eye_mask(x0)
    print(f"[*] base {base}: fit {a.n_fit} / test {a.n_test} in {time.time()-t0:.0f}s; "
          f"teal eyes detected in {np.mean(m0.sum((1,2)) >= 8):.0%} of test images")
    R, models = {"base": score(x0, x0, m0)}, {}

    # (a) latent baseline: best linear "eye redness" direction (exp3-style edit)
    ec = np.array([xf[i][mf[i]].mean(0) if mf[i].sum() else xf[i, 28:40, 12:52].reshape(-1, 3).mean(0) for i in range(len(xf))])
    red = ec[:, 0] - ec[:, 1]
    d = np.linalg.lstsq(z_fit - z_fit.mean(0), red - red.mean(), rcond=None)[0]; d /= np.linalg.norm(d)
    for s in (2.0, 4.0):
        models[f"latent_redness_{s:g}"] = run(z_test + s * d, W, ops)

    # (b) 1x1 colour fold over all channels, no gate
    Wf = fold_1x1_colour(W, h9, pre, mf, a.n_neg, rng, w_eye=10)
    models["fold_1x1_nogate"] = run(z_test, Wf, ops)

    # (c) prune 2 channels, then gated circuit in the freed slots
    Wp, slots, cost = prune_fold(W, h9, 2, a.n_neg, rng)
    models["prune2_only"] = run(z_test, Wp, ops)
    kern, kb, theta, pos, neg = fit_gate(h8, mf, a.n_gate_neg, rng, a.q)
    # colour vector: least squares on eye pixels for target  SWAP(y) - y,  through relu(p - theta)
    feats = np.c_[patches3x3(h8, pos), np.ones(len(pos))] @ np.r_[kern.reshape(-1), kb]
    gate = np.maximum(feats - theta, 0)
    T = np.clip(pre.reshape(-1, 3)[pos], 0, 1) @ SWAP_RG.T - pre.reshape(-1, 3)[pos]
    v = (gate @ T) / (gate @ gate)
    Wg = install_gate(Wp, slots, kern, kb, theta, v.astype(np.float32), alpha)
    models["prune2_eye_gate"] = run(z_test, Wg, ops)

    # gate exactness: the two-slot construction equals v * relu(p - theta)
    u = np.linspace(-3, 3, 13); lr = lambda t: np.where(t >= 0, t, alpha * t)
    gate_err = float(np.abs((lr(u) + alpha * lr(-u)) / (1 - alpha ** 2) - np.maximum(u, 0)).max())

    for k, x1 in models.items(): R[k] = score(x1, x0, m0)

    # patches: prune and gate are separate sparse diffs; merged = base -> Wg
    P = {"prune2": make_patch("prune2", f"prune conv2d_9 channels {slots}, fold into conv2d_10", base, diff_ops(W, Wp)),
         "fold_1x1_nogate": make_patch("fold_1x1_nogate", "1x1 RG-swap colour fold, no gate", base, diff_ops(W, Wf))}
    gpatch = make_patch("eye_gate", f"eye detector + magenta writer in slots {slots} (requires prune2)",
                        weights_hash(Wp), diff_ops(Wp, Wg), requires="prune2")
    P["eye_gate"] = gpatch
    Wre = apply_weightpatch(apply_weightpatch(W, json.loads(json.dumps(P["prune2"]))), json.loads(json.dumps(gpatch)))
    checks = {"gate_identity_maxerr": gate_err,
              "json_roundtrip_maxabs": float(max(np.abs(Wre[k] - Wg[k]).max() for k in W)),
              "params_changed_prune2": changed_params(W, Wp), "params_changed_gate": changed_params(Wp, Wg),
              "params_changed_fold": changed_params(W, Wf), "pruned_slots": [int(s) for s in slots],
              "prune_cost_rank_of_slots": [int(np.argsort(np.argsort(cost))[s]) for s in slots],
              "theta_quantile": a.q, "colour_vector": v.round(4).tolist()}
    for pid, p in P.items():
        with open(os.path.join(a.out, "patches", pid + ".weightpatch.json"), "w") as f: json.dump(p, f)
    with open(os.path.join(a.out, "exp4_results.json"), "w") as f: json.dump({"base": base, "results": R, "checks": checks}, f, indent=1)

    order = ["latent_redness_2", "latent_redness_4", "fold_1x1_nogate", "prune2_only", "prune2_eye_gate"]
    save_grid([x0[:8]] + [models[k][:8] for k in order], ["base"] + order, os.path.join(a.out, "exp4_grid.png"))
    L = ["# Experiment 4: eye-colour edit (teal -> magenta)\n",
         f"Base `{base}`, fit n={a.n_fit}, held-out n={a.n_test}, seed {a.seed}. Eye pixels = teal iris mask on the base image; "
         "'rest' = pixels more than 3 px from any eye pixel.\n",
         "| edit | params changed | eye hue (base 181°, target ~300°) | eye dist to target | rest MAE | rest p99 px change |",
         "|---|---|---|---|---|---|"]
    pc = {"base": 0, "latent_redness_2": "0 (input edit)", "latent_redness_4": "0 (input edit)",
          "fold_1x1_nogate": checks["params_changed_fold"], "prune2_only": checks["params_changed_prune2"],
          "prune2_eye_gate": checks["params_changed_prune2"] + checks["params_changed_gate"]}
    for k in ["base"] + order:
        r = R[k]; L.append(f"| {k} | {pc[k]} | {r['eye_hue']:.0f}° | {r['eye_dist_to_target']:.3f} | {r['rest_mae']:.4f} | {r['rest_max_px_change_p99']:.3f} |")
    L += ["\n## Checks\n"] + [f"- {k}: {v_}" for k, v_ in checks.items()]
    open(os.path.join(a.out, "exp4_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[+] wrote {a.out}/exp4_results.md, exp4_results.json, exp4_grid.png, patches/*.weightpatch.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="legacy/Anime-DCGAN/generator50000.h5")
    ap.add_argument("--out", default="results/exp4_out")
    ap.add_argument("--n-fit", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--n-neg", type=int, default=150000)
    ap.add_argument("--n-gate-neg", type=int, default=50000)
    ap.add_argument("--q", type=float, default=0.95, help="gate threshold quantile on non-eye pixels")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    W, ops, note = load_model(a.weights)
    print(f"[*] loaded {a.weights} ({note})")
    experiment(W, ops, a)
