"""
Experiment 3: attribute edits on the Anime DCGAN as .weightpatch files.

The generator's first layer is linear (dense_2: z -> 1024), so any affine edit
of the latent  z' = z P + t  folds exactly into that layer:
    K' = P K,   b' = b + t K
Two patch types, both stored as text (a 64-float direction + one scalar):
  * latent_shift  : z' = z + alpha*d            -> bias-only edit, patches commute
  * latent_clamp  : z' = z - (z.d)d + t*d       -> kernel+bias edit, order matters
                                                   unless the directions are orthogonal

Attributes are measured from pixels (no labels, no dataset), directions are fit
by ridge regression on one set of latents and every number is reported on a
held-out set.

  python src/exp3_anime_weightpatch.py --weights legacy/Anime-DCGAN/generator50000.h5
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2b_tinysr18_weightpatch import (load_model, weights_hash, make_patch,
                                    merge_patches, changed_params)

# ---------------------------------------------------------------------------
# 1. Batched forward pass driven by the ops parsed from the .h5 config
# ---------------------------------------------------------------------------
def conv2d_same_b(x, K, b):
    kh, kw, ci, co = K.shape
    n, H, Wd, _ = x.shape
    xp = np.pad(x, ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)))
    out = np.zeros((n, H, Wd, co), np.float32)
    for i in range(kh):
        for j in range(kw):
            out += xp[:, i:i + H, j:j + Wd, :] @ K[i, j]
    return out + b

def apply_op_b(x, op, W):
    t = op["op"]
    if t == "dense": return x @ W[op["name"] + "/kernel"] + W[op["name"] + "/bias"]
    if t == "reshape": return x.reshape((-1,) + tuple(op["shape"]))
    if t == "up": return x.repeat(op["size"][0], 1).repeat(op["size"][1], 2)
    if t == "conv": return conv2d_same_b(x, W[op["name"] + "/kernel"], W[op["name"] + "/bias"])
    if t == "lrelu": return np.where(x >= 0, x, np.float32(op["alpha"]) * x)
    if t == "noop": return x
    raise NotImplementedError(t)

def generate(z, W, ops, chunk=100):
    out = []
    for s in range(0, len(z), chunk):
        h = z[s:s + chunk].astype(np.float32)
        for op in ops:
            h = apply_op_b(h, op, W)
        out.append(h)
    return np.concatenate(out)          # pre-clip, ~[0,1] (notebooks display x*256)

# ---------------------------------------------------------------------------
# 2. Pixel-defined attributes (64x64 face crops; regions eyeballed from samples)
# ---------------------------------------------------------------------------
def lum(x): return x @ np.array([0.299, 0.587, 0.114], np.float32)

ATTRS = {
    "hair_warmth":    lambda x: (x[:, :16, :, 0] - x[:, :16, :, 2]).mean((1, 2)),
    "hair_lightness": lambda x: lum(x[:, :16]).mean((1, 2)),
    "background":     lambda x: lum(np.concatenate([x[:, :, :6], x[:, :, -6:]], 2)).mean((1, 2)),
    "face_lightness": lambda x: lum(x[:, 40:56, 20:44]).mean((1, 2)),
    "eye_teal":       lambda x: ((x[:, 28:40, 12:52, 1] + x[:, 28:40, 12:52, 2]) / 2
                                 - x[:, 28:40, 12:52, 0]).mean((1, 2)),
    "saturation":     lambda x: (x.max(-1) - x.min(-1)).mean((1, 2)),
}

def attributes(img):
    x = np.clip(img, 0, 1)
    return np.stack([f(x) for f in ATTRS.values()], 1)      # (n, n_attr)

# ---------------------------------------------------------------------------
# 3. Latent-affine weightpatch ops (exact folds into the first dense layer)
# ---------------------------------------------------------------------------
def _assert_first_dense(ops, layer):
    first = ops[0]
    if not (first["op"] == "dense" and first["name"] == layer):
        raise ValueError(f"{layer} must be the first (linear dense) layer for an exact fold")

def apply_weightpatch(W, patch, ops):
    W = {k: v.copy() for k, v in W.items()}
    for op in patch["ops"]:
        L = op["layer"]; kk, bk = L + "/kernel", L + "/bias"
        _assert_first_dense(ops, L)
        d = np.array(op["direction"], np.float64); d /= np.linalg.norm(d)
        K, b = W[kk].astype(np.float64), W[bk].astype(np.float64)
        dK = d @ K
        if op["type"] == "latent_shift":          # z' = z + alpha d
            b = b + op["alpha"] * dK
        elif op["type"] == "latent_clamp":        # z' = z - (z.d) d + t d
            K = K - np.outer(d, dK)
            b = b + op["target"] * dK
        else:
            raise ValueError(op["type"])
        W[kk], W[bk] = K.astype(np.float32), b.astype(np.float32)
    return W

def latent_edit(z, patch):
    """Reference: the same edit applied to z (in latent order = reverse patch order)."""
    z = z.astype(np.float64)
    for op in reversed(patch["ops"]):
        d = np.array(op["direction"], np.float64); d /= np.linalg.norm(d)
        if op["type"] == "latent_shift": z = z + op["alpha"] * d
        else: z = z - np.outer(z @ d, d) + op["target"] * d
    return z

def shift_op(layer, d, alpha):
    return {"type": "latent_shift", "layer": layer, "direction": [round(float(v), 7) for v in d], "alpha": alpha}

def clamp_op(layer, d, target):
    return {"type": "latent_clamp", "layer": layer, "direction": [round(float(v), 7) for v in d], "target": target}

# ---------------------------------------------------------------------------
# 4. Direction fitting
# ---------------------------------------------------------------------------
def fit_directions(z, A, ridge=1.0):
    Zc, Ac = z - z.mean(0), A - A.mean(0)
    B = np.linalg.solve(Zc.T @ Zc + ridge * np.eye(z.shape[1]), Zc.T @ Ac)   # (64, n_attr)
    return B / np.linalg.norm(B, axis=0)

def r2(z, A, D):
    """Held-out R^2 of each attribute from its 1-D projection z.d (linear refit)."""
    out = []
    for j in range(A.shape[1]):
        p = z @ D[:, j]; out.append(np.corrcoef(p, A[:, j])[0, 1] ** 2)
    return np.array(out)

def orthogonalize(D):
    """d_i with the other directions projected out (reduces off-target leakage)."""
    Do = D.copy()
    for i in range(D.shape[1]):
        O = np.delete(D, i, 1); Q, _ = np.linalg.qr(O)
        v = D[:, i] - Q @ (Q.T @ D[:, i]); Do[:, i] = v / np.linalg.norm(v)
    return Do

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
def save_grid(rows, labels, path, scale=2):
    n = rows[0].shape[0]; s = 64 * scale; lw = 150
    canvas = Image.new("RGB", (lw + n * s, len(rows) * s), "white")
    dr = ImageDraw.Draw(canvas)
    for r, (imgs, lab) in enumerate(zip(rows, labels)):
        for c in range(n):
            im = Image.fromarray((np.clip(imgs[c], 0, 1) * 255).astype(np.uint8)).resize((s, s), Image.NEAREST)
            canvas.paste(im, (lw + c * s, r * s))
        dr.text((6, r * s + s // 2 - 6), lab, fill="black")
    canvas.save(path)

# ---------------------------------------------------------------------------
# 6. Experiment
# ---------------------------------------------------------------------------
def experiment(W, ops, a):
    os.makedirs(os.path.join(a.out, "patches"), exist_ok=True)
    layer = ops[0]["name"]; base = weights_hash(W); names = list(ATTRS)
    latent = W[layer + "/kernel"].shape[0]
    rng = np.random.default_rng(a.seed)
    z_fit = rng.standard_normal((a.n_fit, latent)); z_test = rng.standard_normal((a.n_test, latent))
    t0 = time.time()
    A_fit = attributes(generate(z_fit, W, ops)); img_test = generate(z_test, W, ops)
    A_test = attributes(img_test); sd = A_test.std(0)
    print(f"[*] base {base}: generated {a.n_fit}+{a.n_test} samples in {time.time()-t0:.0f}s")

    D = fit_directions(z_fit, A_fit); Do = orthogonalize(D)
    R2, R2o = r2(z_test, A_test, D), r2(z_test, A_test, Do)
    cos = D.T @ D
    checks, rep = {}, {"base": base, "attributes": names, "sd": sd.tolist(),
                       "r2_heldout": dict(zip(names, R2.round(4).tolist())),
                       "r2_heldout_orth": dict(zip(names, R2o.round(4).tolist())),
                       "direction_cosine": cos.round(3).tolist()}

    def delta(patch):
        Wp = apply_weightpatch(W, patch, ops); img = generate(z_test, Wp, ops)
        return Wp, img, (attributes(img) - A_test).mean(0) / sd

    patches, outs = {}, {}
    # shift patches: raw and orthogonalised directions -> entanglement matrices
    E, Eo = np.zeros((len(names),) * 2), np.zeros((len(names),) * 2)
    for i, n in enumerate(names):
        for Dm, M, tag in ((D, E, ""), (Do, Eo, "_orth")):
            pid = f"shift_{n}{tag}"
            p = make_patch(pid, f"shift latent along {n}{tag} direction by {a.alpha}", base,
                           [shift_op(layer, Dm[:, i], a.alpha)])
            Wp, img, dl = delta(p); M[i] = dl
            patches[pid] = p; outs[pid] = img
            rep.setdefault("changed_params", {})[pid] = changed_params(W, Wp)
    rep["entanglement_shift"] = E.round(3).tolist(); rep["entanglement_shift_orth"] = Eo.round(3).tolist()

    # exactness: weight fold == latent edit (float64 reference)
    p = patches["shift_hair_warmth"]
    ref = generate(latent_edit(z_test[:32], p), W, ops); fold = outs["shift_hair_warmth"][:32]
    checks["shift_fold_vs_latent_maxabs"] = float(np.abs(ref - fold).max())

    # clamp patches: set the attribute coordinate to +target for every z
    clamp = {}
    for i, n in enumerate(names):
        pid = f"clamp_{n}"
        p = make_patch(pid, f"clamp latent {n} coordinate to {a.target}", base, [clamp_op(layer, D[:, i], a.target)])
        Wp, img, dl = delta(p); A_c = attributes(img)
        clamp[n] = {"shift_sd": round(float(dl[i]), 3),
                    "var_ratio": round(float(A_c[:, i].var() / A_test[:, i].var()), 3),
                    "changed_params": changed_params(W, Wp)}
        patches[pid] = p; outs[pid] = img
    rep["clamp"] = clamp
    p = patches["clamp_hair_warmth"]
    ref = generate(latent_edit(z_test[:32], p), W, ops)
    checks["clamp_fold_vs_latent_maxabs"] = float(np.abs(ref - outs["clamp_hair_warmth"][:32]).max())

    # composition: shift patches commute; clamp patches commute only if orthogonal
    i, j = names.index("hair_warmth"), names.index("background")
    sA, sB = patches["shift_hair_warmth"], patches["shift_background"]
    m1, m2 = merge_patches("merge_shift_AB", sA, sB), merge_patches("merge_shift_BA", sB, sA)
    W1, W2 = apply_weightpatch(W, m1, ops), apply_weightpatch(W, m2, ops)
    checks["shift_merge_order_maxabs_weights"] = float(max(np.abs(W1[k] - W2[k]).max() for k in W))
    img_m = generate(z_test, W1, ops); dl_m = (attributes(img_m) - A_test).mean(0) / sd
    rep["shift_merge"] = {"pair": [names[i], names[j]], "merged_delta_sd": dl_m.round(3).tolist(),
                          "sum_of_individual_sd": (E[i] + E[j]).round(3).tolist(),
                          "additivity_error_sd": float(np.abs(dl_m - E[i] - E[j]).max().round(3))}
    patches["merge_shift_AB"] = m1; outs["merge_shift_AB"] = img_m

    order = []
    for (u, v) in [(x, y) for x in range(len(names)) for y in range(x + 1, len(names))]:
        cA, cB = patches[f"clamp_{names[u]}"], patches[f"clamp_{names[v]}"]
        Wab = apply_weightpatch(W, merge_patches("ab", cA, cB), ops)
        Wba = apply_weightpatch(W, merge_patches("ba", cB, cA), ops)
        ia, ib = generate(z_test[:64], Wab, ops), generate(z_test[:64], Wba, ops)
        order.append({"pair": [names[u], names[v]], "cos": round(float(cos[u, v]), 3),
                      "weight_maxabs": float(max(np.abs(Wab[k] - Wba[k]).max() for k in W)),
                      "output_mae": float(np.abs(np.clip(ia, 0, 1) - np.clip(ib, 0, 1)).mean())})
    rep["clamp_order"] = order
    oc = np.array([[abs(o["cos"]), o["output_mae"]] for o in order])
    checks["clamp_order_conflict_vs_abs_cos_corr"] = float(np.corrcoef(oc.T)[0, 1])

    rt = json.loads(json.dumps(patches["merge_shift_AB"]))
    checks["json_roundtrip_identical"] = all(np.array_equal(apply_weightpatch(W, rt, ops)[k], W1[k]) for k in W)
    rep["checks"] = checks

    # write artefacts
    for pid, p in patches.items():
        with open(os.path.join(a.out, "patches", pid + ".weightpatch.json"), "w") as f: json.dump(p, f, indent=1)
    with open(os.path.join(a.out, "exp3_results.json"), "w") as f: json.dump(rep, f, indent=1)
    k = slice(0, 8)
    rows = [img_test[k]] + [outs[f"shift_{n}"][k] for n in names] + \
           [outs["merge_shift_AB"][k], outs["clamp_hair_warmth"][k], outs["clamp_background"][k]]
    labels = ["base"] + [f"shift {n} +{a.alpha}" for n in names] + \
             ["merge hair_warmth+background", f"clamp hair_warmth={a.target}", f"clamp background={a.target}"]
    save_grid(rows, labels, os.path.join(a.out, "exp3_grid.png"))
    write_md(rep, names, a, os.path.join(a.out, "exp3_results.md"))
    for kk, v in checks.items(): print(f"    check {kk}: {v}")
    print(f"[+] wrote {a.out}/exp3_results.md, exp3_results.json, exp3_grid.png, patches/*.weightpatch.json")

def _matrix_md(M, names):
    s = "| patch \\ measured | " + " | ".join(names) + " |\n|---|" + "---|" * len(names) + "\n"
    for i, n in enumerate(names):
        s += f"| {n} | " + " | ".join(("**%+.2f**" if i == j else "%+.2f") % M[i][j] for j in range(len(names))) + " |\n"
    return s

def write_md(r, names, a, path):
    E, Eo = np.array(r["entanglement_shift"]), np.array(r["entanglement_shift_orth"])
    off = lambda M: float(np.abs(M - np.diag(np.diag(M))).sum(1).mean())
    L = [f"# Experiment 3: Anime DCGAN attribute weightpatches\n",
         f"Base weights `{r['base']}`, fit n={a.n_fit}, held-out n={a.n_test}, seed {a.seed}. "
         f"Shifts are alpha={a.alpha} latent units; effects in held-out attribute SDs.\n",
         "## Linear predictability of each attribute from z (held-out R²)\n",
         "| attribute | R² (raw dir) | R² (orthogonalised dir) |\n|---|---|---|"]
    L += [f"| {n} | {r['r2_heldout'][n]:.3f} | {r['r2_heldout_orth'][n]:.3f} |" for n in names]
    L += ["\n## Entanglement: shift patch (row) -> attribute change (column), SD units\n", _matrix_md(E, names),
          "### Same, with orthogonalised directions\n", _matrix_md(Eo, names),
          f"Mean off-target |effect| per patch: raw {off(E):.2f} SD, orthogonalised {off(Eo):.2f} SD. "
          f"Mean on-target: raw {np.diag(E).mean():.2f}, orthogonalised {np.diag(Eo).mean():.2f}.\n",
          f"## Clamp patches (set coordinate to {a.target})\n", "| attribute | shift (SD) | variance ratio | params changed |\n|---|---|---|---|"]
    L += [f"| {n} | {c['shift_sd']:+.2f} | {c['var_ratio']:.2f} | {c['changed_params']} |" for n, c in r["clamp"].items()]
    m = r["shift_merge"]
    L += [f"\n## Merging\n", f"Shift merge {m['pair'][0]} + {m['pair'][1]}: additivity error {m['additivity_error_sd']} SD "
          f"(merged vs sum of individual effects; generator is nonlinear after layer 1).\n",
          "Clamp merge order conflicts (A∘B vs B∘A):\n", "| pair | cos(dA,dB) | max weight diff | output MAE |\n|---|---|---|---|"]
    L += [f"| {o['pair'][0]} / {o['pair'][1]} | {o['cos']:+.3f} | {o['weight_maxabs']:.2e} | {o['output_mae']:.4f} |"
          for o in sorted(r["clamp_order"], key=lambda o: -abs(o["cos"]))]
    L += ["\n## Checks\n"] + [f"- {k}: {v}" for k, v in r["checks"].items()]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="legacy/Anime-DCGAN/generator50000.h5")
    ap.add_argument("--out", default="results/exp3_out")
    ap.add_argument("--n-fit", type=int, default=1200)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    W, ops, note = load_model(a.weights)
    print(f"[*] loaded {a.weights} ({note})")
    experiment(W, ops, a)
