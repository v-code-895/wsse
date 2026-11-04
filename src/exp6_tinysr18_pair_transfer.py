"""
Experiment 6: transfer between two independently trained DLSS generators.

  A = generator.h5   5x5 kernels,  64 channels, 2 convs before the 2x upsample (317K params)
  B = model.json     3x3 kernels, 128 channels, 3 convs before the 2x upsample (597K params)

Same task, same author, different architecture and training run. Three questions:
  1. Representation similarity: linear CKA between every hidden layer of A and B
     (control: same features with B's pixels shuffled).
  2. Does the measured colour error transfer? Fit the exp2b output-affine fix on
     each model, then apply A's fix to B.
  3. Architecture transfer by stitching: run the front of one model, map its
     features with a 1x1 linear map (least squares, forward passes only) and fold
     that map exactly into the next conv of the other model. The hybrid is an
     ordinary single network with no extra layers.

  python src/exp6_tinysr18_pair_transfer.py
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import exp2b_tinysr18_weightpatch as e2
from exp2b_tinysr18_weightpatch import (load_model, parse_keras_config, apply_op, build_suites,
                                    psnr, weights_hash, baselines)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_tfjs(path):
    j = json.load(open(path))
    topo = j["modelTopology"]; ops = parse_keras_config(topo.get("model_config", topo))
    W, d = {}, os.path.dirname(path)
    for m in j["weightsManifest"]:
        buf = b"".join(open(os.path.join(d, p), "rb").read() for p in m["paths"]); off = 0
        for w in m["weights"]:
            assert w.get("dtype", "float32") == "float32"
            n = int(np.prod(w["shape"])) * 4
            W[w["name"]] = np.frombuffer(buf[off:off + n], np.float32).reshape(w["shape"]).copy(); off += n
    return W, ops

def strip_tail(ops):
    """A trailing LeakyReLU after the output conv is a no-op once outputs are clipped to [0,1]."""
    return ops[:-1] if ops[-1]["op"] == "lrelu" and ops[-2]["op"] == "conv" else ops

def stages(ops):
    """Indices i such that ops[:i] ends in a hidden lrelu: the feature taps."""
    return [i + 1 for i, o in enumerate(ops) if o["op"] == "lrelu" and i + 1 < len(ops)]

def run_from(x, W, ops, start=0, stop=None):
    h = x.astype(np.float32)
    for op in ops[start:stop]: h = apply_op(h, op, W)
    return h

def taps(x, W, ops):
    out, h = {}, x.astype(np.float32); T = set(stages(ops))
    for i, op in enumerate(ops):
        h = apply_op(h, op, W)
        if i + 1 in T: out[i + 1] = h
    return out, h

def label(ops, i):
    conv = [o["name"] for o in ops[:i] if o["op"] == "conv"][-1]
    res = "256" if any(o["op"] == "up" for o in ops[:i]) else "128"
    return f"{conv}@{res}"

# ---------------------------------------------------------------------------
# 2. Linear CKA
# ---------------------------------------------------------------------------
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xy = np.linalg.norm(X.T @ Y) ** 2
    return float(xy / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))

# ---------------------------------------------------------------------------
# 3. Folds
# ---------------------------------------------------------------------------
def fit_affine(F, T, ridge=1e-3):
    """Least squares T ~ F @ S + s0 (F: n x ci, T: n x co)."""
    X = np.c_[F, np.ones(len(F))].astype(np.float64)
    A = np.linalg.solve(X.T @ X + ridge * len(X) * np.eye(X.shape[1]), X.T @ T.astype(np.float64))
    return A[:-1], A[-1]

def fold_input_affine(K, b, S, s0):
    """conv(K, b) applied to (S^T-mapped features + s0) == conv(K', b') on the raw
    features (exact away from the zero-padded border, where s0 is not added)."""
    K2 = np.einsum("ij,hwjo->hwio", S, K)
    b2 = b + np.einsum("j,hwjo->o", s0, K)
    return K2.astype(np.float32), b2.astype(np.float32)

def fold_output_affine(K, b, M, t):
    return np.einsum("hwio,po->hwip", K, M).astype(np.float32), (M @ b + t).astype(np.float32)

def stitch(Wf, opsf, cut_f, Wb, opsb, cut_b, S, s0, tag):
    """Hybrid = front ops of model f up to cut_f, then model b from cut_b, with the
    1x1 map folded into b's first conv after the cut. Any 'up' between cut_b and
    that conv commutes with a 1x1 map, so it is kept."""
    j = next(i for i in range(cut_b, len(opsb)) if opsb[i]["op"] == "conv")
    name = opsb[j]["name"]
    K2, b2 = fold_input_affine(Wb[name + "/kernel"], Wb[name + "/bias"], S, s0)
    W = {f"{tag}F/{k}": v for k, v in Wf.items()}
    W.update({f"{tag}B/{k}": v for k, v in Wb.items()})
    W[f"{tag}B/{name}/kernel"], W[f"{tag}B/{name}/bias"] = K2, b2
    ren = lambda o, p: dict(o, name=f"{tag}{p}/{o['name']}") if "name" in o else o
    ops = [ren(o, "F") for o in opsf[:cut_f]] + [ren(o, "B") for o in opsb[cut_b:]]
    # drop the unused tail of the front model / head of the back model from W
    used = {o["name"] for o in ops if "name" in o}
    W = {k: v for k, v in W.items() if k.rsplit("/", 1)[0] in used}
    return W, ops

def nparams(W): return int(sum(v.size for v in W.values()))

# ---------------------------------------------------------------------------
# 4. Experiment
# ---------------------------------------------------------------------------
def experiment(a):
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    WA, opsA, _ = load_model(a.a); opsA = strip_tail(opsA)
    WB, opsB = load_tfjs(a.b); opsB = strip_tail(opsB)
    S = build_suites(a.lr_filter)
    calib = {**S["natural_calib"], **S["sprite_calib"]}
    test = {**S["natural_test"], **S["sprite_test"], **S["legacy"]}
    groups = {"natural": list(S["natural_test"]), "sprite": list(S["sprite_test"]), "legacy": list(S["legacy"])}
    print(f"[*] A {weights_hash(WA)} ({nparams(WA):,} params)  B {weights_hash(WB)} ({nparams(WB):,} params); "
          f"{len(calib)} calib / {len(test)} test images")

    TA, TB, YA, YB = {}, {}, {}, {}
    for k, (lr, hr) in {**calib, **test}.items():
        TA[k], YA[k] = taps(lr, WA, opsA); TB[k], YB[k] = taps(lr, WB, opsB)
    print(f"[*] forward passes done in {time.time()-t0:.0f}s")

    rng = np.random.default_rng(a.seed)
    def sample(Tmap, keys, i, n, res_idx):
        rows = []
        for k in keys:
            f = Tmap[k][i].reshape(-1, Tmap[k][i].shape[-1]); rows.append(f[res_idx[k][f.shape[0]]])
        return np.concatenate(rows)
    idx = {k: {r: rng.choice(r, a.px, replace=False) for r in (128 * 128, 256 * 256)} for k in {**calib, **test}}

    # --- 1. CKA on test pixels (same pixel positions in both models) ---
    sA, sB = stages(opsA), stages(opsB); tkeys = list(test)
    C = np.zeros((len(sA), len(sB))); C_ctrl = np.zeros_like(C)
    for i, ia in enumerate(sA):
        XA = sample(TA, tkeys, ia, a.px, idx)
        for j, jb in enumerate(sB):
            if TA[tkeys[0]][ia].shape[:2] != TB[tkeys[0]][jb].shape[:2]:
                C[i, j] = C_ctrl[i, j] = np.nan; continue
            XB = sample(TB, tkeys, jb, a.px, idx)
            C[i, j] = cka(XA, XB); C_ctrl[i, j] = cka(XA, XB[rng.permutation(len(XB))])
    rep = {"cka": {"rows_A": [label(opsA, i) for i in sA], "cols_B": [label(opsB, j) for j in sB],
                   "values": np.round(C, 3).tolist(), "shuffled_control": np.round(C_ctrl, 3).tolist()}}

    # --- 2. colour fix transfer ---
    def fit_colour(Y):
        F = np.concatenate([Y[k].reshape(-1, 3)[idx[k][256 * 256]] for k in calib])
        T = np.concatenate([calib[k][1].reshape(-1, 3)[idx[k][256 * 256]] for k in calib])
        S_, s0 = fit_affine(F, T); return S_.T, s0          # M (out x in), t
    MA, tA = fit_colour(YA); MB, tB = fit_colour(YB)
    lastA, lastB = [o["name"] for o in opsA if o["op"] == "conv"][-1], [o["name"] for o in opsB if o["op"] == "conv"][-1]
    def recolour(Y, M, t): return {k: Y[k] @ M.T + t for k in Y}
    models = {"A": YA, "B": YB, "A + own colour fix": recolour(YA, MA, tA), "B + own colour fix": recolour(YB, MB, tB),
              "B + A's colour fix": recolour(YB, MA, tA), "A + B's colour fix": recolour(YA, MB, tB)}
    rep["colour_fix"] = {"M_A": np.round(MA, 3).tolist(), "t_A_rgb255": np.round(tA * 255, 1).tolist(),
                         "M_B": np.round(MB, 3).tolist(), "t_B_rgb255": np.round(tB * 255, 1).tolist(),
                         "fold_exactness_maxabs": None}
    # exactness of the colour fold into the last conv (one image)
    k0 = tkeys[0]; Wc = dict(WB); Wc[lastB + "/kernel"], Wc[lastB + "/bias"] = fold_output_affine(WB[lastB + "/kernel"], WB[lastB + "/bias"], MA, tA)
    rep["colour_fix"]["fold_exactness_maxabs"] = float(np.abs(run_from(test[k0][0], Wc, opsB) - (YB[k0] @ MA.T + tA)).max())

    # --- 3. stitching ---
    ckeys = list(calib)
    def fit_stitch(Tf, i_f, Tb, i_b):
        return fit_affine(sample(Tf, ckeys, i_f, a.px, idx), sample(Tb, ckeys, i_b, a.px, idx))
    stitches = []
    resA = {i: TA[k0][i].shape[0] for i in sA}; resB = {j: TB[k0][j].shape[0] for j in sB}
    for i in sA:
        for j in sB:
            if resA[i] == resB[j]: stitches.append(("A", i, "B", j)); stitches.append(("B", j, "A", i))
    hyb = {}; srows = []
    for f, i_f, b, i_b in stitches:
        Tf, Tb = (TA, TB) if f == "A" else (TB, TA)
        Wf, opsf, Wb, opsb = (WA, opsA, WB, opsB) if f == "A" else (WB, opsB, WA, opsA)
        if not any(o["op"] == "conv" for o in opsb[i_b:]): continue
        S_, s0 = fit_stitch(Tf, i_f, Tb, i_b)
        Wh, opsh = stitch(Wf, opsf, i_f, Wb, opsb, i_b, S_, s0, tag=f"{f}{b}")
        name = f"{f}[..{label(opsf, i_f)}] -> {b}[{label(opsb, i_b)}..]"
        out = {k: run_from(test[k][0], Wh, opsh) for k in tkeys}
        hyb[name] = out
        # feature-map fit quality on held-out pixels (R^2 of the 1x1 map)
        Xf, Xb = sample(Tf, tkeys, i_f, a.px, idx), sample(Tb, tkeys, i_b, a.px, idx)
        pred = Xf @ S_ + s0; r2 = 1 - ((Xb - pred) ** 2).sum() / ((Xb - Xb.mean(0)) ** 2).sum()
        srows.append({"name": name, "front": f, "back": b, "params": nparams(Wh), "map_r2_heldout": float(r2)})
    models.update(hyb)

    # --- evaluation ---
    def ev(Y):
        out = {}
        for g, keys in groups.items():
            out[g] = float(np.mean([psnr(np.clip(Y[k], 0, 1), test[k][1]) for k in keys]))
        return out
    res = {n: ev(Y) for n, Y in models.items()}
    res["bicubic"] = ev({k: baselines(test[k][0])["bicubic"] for k in tkeys})
    agree = {}
    for r in srows:
        parentB = YB if r["back"] == "B" else YA
        agree[r["name"]] = float(np.mean([psnr(np.clip(hyb[r["name"]][k], 0, 1), np.clip(parentB[k], 0, 1)) for k in tkeys]))
        r["psnr_vs_back_parent"] = agree[r["name"]]; r.update({f"psnr_{g}": v for g, v in res[r["name"]].items()})
    rep["psnr"] = res; rep["stitches"] = srows
    rep["parents_agree_psnr"] = float(np.mean([psnr(np.clip(YA[k], 0, 1), np.clip(YB[k], 0, 1)) for k in tkeys]))
    json.dump(rep, open(os.path.join(a.out, "exp6_results.json"), "w"), indent=1)
    write_md(rep, a, os.path.join(a.out, "exp6_results.md"))
    save_figure(models, test, os.path.join(a.out, "exp6_grid.png"),
                ["A", "B", "B + A's colour fix", "B + own colour fix"] + [r["name"] for r in sorted(srows, key=lambda r: -r["psnr_vs_back_parent"])[:2]])
    print(open(os.path.join(a.out, "exp6_results.md"), encoding="utf-8").read())

def save_figure(models, test, path, cols):
    from PIL import Image, ImageDraw
    keys = [k for k in ("astronaut", "sprite_101", "v1_suiteA_gradient") if k in test]
    s = 192; lw = 0; canvas = Image.new("RGB", (s * (len(cols) + 1), (s + 28) * len(keys)), "white"); d = ImageDraw.Draw(canvas)
    for r, k in enumerate(keys):
        for c, n in enumerate(cols + ["ground truth"]):
            im = test[k][1] if n == "ground truth" else np.clip(models[n][k], 0, 1)
            canvas.paste(Image.fromarray((im * 255).astype(np.uint8)).resize((s, s)), (c * s, r * (s + 28) + 28))
            short = n if len(n) < 30 else n[:28] + ".."
            d.text((c * s + 4, r * (s + 28) + 4), f"{short}\n{'' if n == 'ground truth' else '%.2f dB' % psnr(im, test[k][1])}", fill="black")
    canvas.save(path)

def write_md(r, a, path):
    c = r["cka"]
    L = ["# Experiment 6: DLSS pair, A = generator.h5 (5x5/64) vs B = model.json (3x3/128)\n",
         f"LR filter `{a.lr_filter}`, {a.px} pixels per image per layer, seed {a.seed}. PSNR in dB on held-out test images.\n",
         "## 1. Linear CKA between hidden layers (test pixels; shuffled-pixel control in brackets)\n",
         "| A \\ B | " + " | ".join(c["cols_B"]) + " |", "|---|" + "---|" * len(c["cols_B"])]
    for i, row in enumerate(c["values"]):
        L.append(f"| {c['rows_A'][i]} | " + " | ".join("-" if v != v else f"{v:.2f} ({c['shuffled_control'][i][j]:.2f})" for j, v in enumerate(row)) + " |")
    cf = r["colour_fix"]
    L += ["\n## 2. Colour error and fix transfer\n",
          f"Fitted output correction  y' = M y + t  (t in RGB/255):\n",
          f"- A: M = {cf['M_A']}, t = {cf['t_A_rgb255']}", f"- B: M = {cf['M_B']}, t = {cf['t_B_rgb255']}",
          f"- folding the fix into the last conv is exact to {cf['fold_exactness_maxabs']:.1e}\n",
          "| model | natural | sprite | legacy |", "|---|---|---|---|"]
    for n in ["bicubic", "A", "B", "A + own colour fix", "B + own colour fix", "B + A's colour fix", "A + B's colour fix"]:
        p = r["psnr"][n]; L.append(f"| {n} | {p['natural']:.2f} | {p['sprite']:.2f} | {p['legacy']:.2f} |")
    L += ["\n## 3. Stitching (front of one model -> 1x1 map folded into the other's next conv)\n",
          f"Parents agree with each other at {r['parents_agree_psnr']:.2f} dB (A output vs B output).\n",
          "| hybrid | params | map R² (held-out) | PSNR vs back parent | natural | sprite | legacy |", "|---|---|---|---|---|---|---|"]
    for s in sorted(r["stitches"], key=lambda s: -s["psnr_vs_back_parent"]):
        L.append(f"| {s['name']} | {s['params']:,} | {s['map_r2_heldout']:.3f} | {s['psnr_vs_back_parent']:.2f} | "
                 f"{s['psnr_natural']:.2f} | {s['psnr_sprite']:.2f} | {s['psnr_legacy']:.2f} |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="legacy/TinySR-18/generator.h5")
    ap.add_argument("--b", default="legacy/TinySR-18/model.json")
    ap.add_argument("--out", default="results/exp6_out")
    ap.add_argument("--lr-filter", default="nearest", choices=["nearest", "bicubic", "bilinear", "lanczos"])
    ap.add_argument("--px", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    experiment(ap.parse_args())
