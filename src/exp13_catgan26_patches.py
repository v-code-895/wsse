"""
Experiment 13: the 2018 weight-space tools, unchanged, on CatGAN-26.

The patch operators and fitting functions are imported from exp3 (latent shifts and clamps,
direction fitting, orthogonalisation) and exp4 (channel pruning by regression, gated circuit,
1x1 colour fold, sparse-diff patches); only the test domain is new (cat_domain.py).
  1. latent directions for five cat attributes: raw vs orthogonalised shifts, clamps, merges
  2. blue eyes: free two channels of the last hidden conv and wire a gated iris circuit into them,
     vs an ungated 1x1 colour fold and a latent shift
  3. exact symmetry and truncation patches (mirror, z -> 0.7 z folded into the dense layer)
  4. MobileNetV2 (ImageNet) as an oracle: does every patched generator still draw cats?
  5. figures, optionally super-sampled 4x by TinySR-26

  <venv python> src/exp13_catgan26_patches.py [--weights results/exp11_out/catgan26.npz]
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_domain as cd, modern_np as mnp, mobilenet_np as mn
import exp3_anime_weightpatch as e3, exp4_anime_eye_circuit as e4, exp7d_anime_styles as st
from exp2b_tinysr18_weightpatch import weights_hash, make_patch, merge_patches, changed_params

OUT = "results/exp13_out"; CAT_CLASSES = [281, 282, 283, 284, 285]

def u8(x): return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)

def cat_rate(imgs, Wm, chunk=16):
    top = np.concatenate([mn.forward(np.stack([mn.preprocess(a) for a in u8(imgs[s:s + chunk])]), Wm).argmax(1) for s in range(0, len(imgs), chunk)])
    return float(np.isin(top, CAT_CLASSES).mean())

def grid(rows, labels, path, scale=1, lw=170):
    from PIL import ImageDraw
    n, s = rows[0].shape[0], rows[0].shape[1] * scale
    canvas = Image.new("RGB", (lw + n * s, len(rows) * s), "white"); dr = ImageDraw.Draw(canvas)
    for r, (imgs, lab) in enumerate(zip(rows, labels)):
        for c in range(n): canvas.paste(Image.fromarray(u8(imgs[c])).resize((s, s), Image.LANCZOS), (lw + c * s, r * s))
        dr.text((6, r * s + s // 2 - 6), lab, fill="black")
    canvas.save(path)

def upscale4(imgs):
    """TinySR-26 x4 (128 -> 512) when its weights exist, else Lanczos."""
    try:
        import torch
        T = mnp.to_torch(dict(np.load("results/exp10_out/tinysr26_x4.npz")))
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(np.clip(imgs, 0, 1), np.float32)).permute(0, 3, 1, 2).cuda()
            return mnp.tinysr_torch(x, T, 4).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    except Exception as e:
        print(f"(TinySR-26 x4 unavailable: {type(e).__name__}; using Lanczos)")
        return np.stack([np.asarray(Image.fromarray(u8(a)).resize((512, 512), Image.LANCZOS), np.float32) / 255 for a in imgs])

def orthogonalize_except_duplicates(D, thr=0.9):
    """Like exp3's orthogonalize, but each direction keeps any near-duplicate (|cos| > thr): projecting
    out a near-copy of yourself erases the attribute (cats: fur warmth vs saturation, cos 0.95)."""
    Dn, C = D.copy(), D.T @ D
    for i in range(D.shape[1]):
        others = [j for j in range(D.shape[1]) if j != i and abs(C[i, j]) <= thr]
        Q, _ = np.linalg.qr(D[:, others]); v = D[:, i] - Q @ (Q.T @ D[:, i]); Dn[:, i] = v / np.linalg.norm(v)
    return Dn

HUE_BAND = (200, 250)                            # the blue-eyes test in the cat git repo (exp5 --profile cat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/exp11_out/catgan26.npz"); ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-fit", type=int, default=1200); ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-eye", type=int, default=150); ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--q", type=float, default=None, help="gate threshold quantile (default: the iris-area rule)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(os.path.join(a.out, "patches"), exist_ok=True)
    W = {k: v.astype(np.float32) for k, v in np.load(a.weights).items()}; ops = mnp.catgan_ops(W)
    base = weights_hash(W); layer = ops[0]["name"]; names = list(cd.ATTRS); run = cd.run
    rng = np.random.default_rng(a.seed); zdim = W[layer + "/kernel"].shape[0]
    z_fit, z_test = rng.standard_normal((a.n_fit, zdim)), rng.standard_normal((a.n_test, zdim))
    t0 = time.time(); A_fit = cd.attributes(run(z_fit, W, ops)); x_test = run(z_test, W, ops); A_test = cd.attributes(x_test); sd = A_test.std(0)
    print(f"[*] CatGAN-26 base {base}: {a.n_fit}+{a.n_test} samples in {time.time() - t0:.1f}s", flush=True)
    R = {"base": base, "attributes": names, "checks": {}}; P = {}; outs = {}

    # 1. latent directions --------------------------------------------------------------------
    D = e3.fit_directions(z_fit, A_fit); Do = e3.orthogonalize(D); Dn = orthogonalize_except_duplicates(D)
    R["r2_heldout"] = dict(zip(names, e3.r2(z_test, A_test, D).round(3).tolist()))
    R["r2_heldout_orth"] = dict(zip(names, e3.r2(z_test, A_test, Do).round(3).tolist()))
    R["r2_heldout_orthnd"] = dict(zip(names, e3.r2(z_test, A_test, Dn).round(3).tolist()))
    def apply(p): Wp = e3.apply_weightpatch(W, p, ops); x = run(z_test, Wp, ops); return Wp, x, (cd.attributes(x) - A_test).mean(0) / sd
    E, Eo, En = (np.zeros((len(names),) * 2) for _ in range(3))
    for i, n in enumerate(names):
        for Dm, M, tag in ((D, E, ""), (Do, Eo, "_orth"), (Dn, En, "_orthnd")):
            pid = f"shift_{n}{tag}"; p = make_patch(pid, f"shift latent along {n}{tag} by {a.alpha}", base, [e3.shift_op(layer, Dm[:, i], a.alpha)])
            _, outs[pid], M[i] = apply(p); P[pid] = p
    R["entanglement_raw"], R["entanglement_orth"], R["entanglement_orthnd"] = E.round(3).tolist(), Eo.round(3).tolist(), En.round(3).tolist()
    off = lambda M: float(np.mean([np.abs(np.delete(M[i], i)).sum() for i in range(len(M))]))
    R["mean_offtarget_sd"] = {"raw": off(E), "orth": off(Eo), "orthnd": off(En)}
    R["mean_ontarget_sd"] = {"raw": float(np.diag(E).mean()), "orth": float(np.diag(Eo).mean()), "orthnd": float(np.diag(En).mean())}
    ref = run(e3.latent_edit(z_test[:32], P["shift_fur_warmth"]), W, ops)
    R["checks"]["shift_fold_vs_latent_edit_maxabs"] = float(np.abs(ref - outs["shift_fur_warmth"][:32]).max())
    R["clamp"] = {}
    for i, n in enumerate(names):
        pid = f"clamp_{n}"; p = make_patch(pid, f"clamp latent {n} coordinate to {a.target}", base, [e3.clamp_op(layer, D[:, i], a.target)])
        _, outs[pid], dl = apply(p); A_c = cd.attributes(outs[pid]); P[pid] = p
        R["clamp"][n] = {"shift_sd": float(dl[i]), "var_ratio": float(A_c[:, i].var() / A_test[:, i].var())}
    cos = D.T @ D; R["direction_cosine"] = cos.round(3).tolist(); R["clamp_order"] = []
    for u in range(len(names)):
        for v in range(u + 1, len(names)):
            Wab = e3.apply_weightpatch(W, merge_patches("ab", P[f"clamp_{names[u]}"], P[f"clamp_{names[v]}"]), ops)
            Wba = e3.apply_weightpatch(W, merge_patches("ba", P[f"clamp_{names[v]}"], P[f"clamp_{names[u]}"]), ops)
            R["clamp_order"].append({"pair": [names[u], names[v]], "cos": float(cos[u, v]),
                                     "weight_maxabs": float(max(np.abs(Wab[k] - Wba[k]).max() for k in W)),
                                     "output_mae": float(np.abs(run(z_test[:64], Wab, ops) - run(z_test[:64], Wba, ops)).mean())})
    print("[1] latent patches done", flush=True)

    # 2. blue eyes ------------------------------------------------------------------------------
    convs = [o["name"] for o in ops if o["op"] == "conv"]; l9, l10 = convs[-2], convs[-1]
    alpha = next(o["alpha"] for o in ops if o["op"] == "lrelu")
    xf, h8, h9, pre = run(z_fit[:a.n_eye], W, ops, keep=True); mf = cd.eye_mask(xf)
    x0 = x_test[:a.n_eye]; m0 = cd.eye_mask(x0); EYE = {"base": cd.score(x0, x0, m0)}; eye_out = {"base": x0}
    R["eye_mask_share_test"] = float(np.mean(m0.sum((1, 2)) >= 8))
    ec = np.array([xf[i][mf[i]].mean(0) if mf[i].sum() else xf[i][cd.EYES].mean(0) for i in range(len(xf))])
    u = (ec @ cd.TARGET.T - ec).mean(0); u /= np.linalg.norm(u); s = ec @ u
    d = np.linalg.lstsq(z_fit[:a.n_eye] - z_fit[:a.n_eye].mean(0), s - s.mean(), rcond=None)[0]; d /= np.linalg.norm(d)
    for k in (2.0, 4.0): eye_out[f"latent towards blue, shift {k:g}"] = run(z_test[:a.n_eye] + k * d, W, ops)
    Wf = e4.fold_1x1_colour(W, h9, pre, mf, 150000, rng, w_eye=10, l10=l10, target=cd.TARGET); eye_out["ungated 1x1 colour fold"] = run(z_test[:a.n_eye], Wf, ops)
    Wp, slots, cost = e4.prune_fold(W, h9, 2, 150000, rng, l9=l9, l10=l10); eye_out["prune2"] = run(z_test[:a.n_eye], Wp, ops)
    # gate threshold, chosen the CI way on the *fitting* images: the strictest quantile (least leakage) whose
    # result already meets the patch's own spec (eye hue inside HUE_BAND); exp4 used a fixed 0.95
    iris = mf.sum((1, 2)).mean()
    kern, kb, _, pos, neg = e4.fit_gate(h8, mf, 50000, rng, 0.95)
    probe = lambda idx: np.c_[e4.patches3x3(h8, idx), np.ones(len(idx))] @ np.r_[kern.reshape(-1), kb]
    pp, pn = probe(pos), probe(neg); T = np.clip(pre.reshape(-1, 3)[pos], 0, 1) @ cd.TARGET.T - pre.reshape(-1, 3)[pos]
    def gated(th):
        g = np.maximum(pp - th, 0); v_ = (g @ T) / (g @ g)
        return e4.install_gate(Wp, slots, kern, kb, th, v_.astype(np.float32), alpha, l9=l9, l10=l10), v_
    cands = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995]; fit_hue = {}; R["eye_gate_q_sweep"] = {}
    for qq in cands:
        Wq, _ = gated(np.quantile(pn, qq))
        fit_hue[qq] = cd.score(run(z_fit[:a.n_eye], Wq, ops), xf, mf)["eye_hue"]
        R["eye_gate_q_sweep"][qq] = {**cd.score(run(z_test[:a.n_eye], Wq, ops), x0, m0), "fit_eye_hue": fit_hue[qq]}
    ok = [qq for qq in cands if HUE_BAND[0] <= fit_hue[qq] <= HUE_BAND[1]]
    q = a.q or (max(ok) if ok else min(cands, key=lambda qq: abs(fit_hue[qq] - sum(HUE_BAND) / 2)))
    theta = float(np.quantile(pn, q)); Wg, v = gated(theta); eye_out["prune2 + eye gate"] = run(z_test[:a.n_eye], Wg, ops)
    for k, x1 in eye_out.items():
        if k != "base": EYE[k] = cd.score(x1, x0, m0)
    pc = {"base": 0, "latent towards blue, shift 2": "0 (input edit)", "latent towards blue, shift 4": "0 (input edit)",
          "ungated 1x1 colour fold": changed_params(W, Wf), "prune2": changed_params(W, Wp), "prune2 + eye gate": changed_params(W, Wg)}
    R["eye"] = {k: {**v_, "params_changed": pc[k]} for k, v_ in EYE.items()}
    R["checks"].update({"pruned_slots": [int(c) for c in slots], "gate_layers": [l9, l10], "colour_vector": v.round(4).tolist(), "theta_quantile": q,
                        "mean_iris_pixels": float(iris)})
    P["prune2"] = make_patch("prune2", f"prune {l9} channels {[int(c) for c in slots]}, fold into {l10}", base, e4.diff_ops(W, Wp))
    P["eye_gate"] = make_patch("eye_gate", f"iris detector + blue writer in {l9} slots {[int(c) for c in slots]} (requires prune2)",
                               weights_hash(Wp), e4.diff_ops(Wp, Wg), requires="prune2")
    Wre = e4.apply_weightpatch(e4.apply_weightpatch(W, json.loads(json.dumps(P["prune2"]))), json.loads(json.dumps(P["eye_gate"])))
    R["checks"]["eye_patches_json_roundtrip_maxabs"] = float(max(np.abs(Wre[k] - Wg[k]).max() for k in W))
    print("[2] eye circuit done", flush=True)

    # 3. exact symmetry and truncation ------------------------------------------------------------
    Wm_ = st.mirror(W, ops); xm = run(z_test[:32], Wm_, ops)
    R["checks"]["mirror_vs_flipped_maxabs"] = float(np.abs(xm - x_test[:32, :, ::-1]).max())
    Wt = {k: v.copy() for k, v in W.items()}; Wt[layer + "/kernel"] = Wt[layer + "/kernel"] * np.float32(0.7)
    xt = run(z_test, Wt, ops); R["checks"]["truncation_fold_vs_scaled_z_maxabs"] = float(np.abs(xt[:32] - run(0.7 * z_test[:32], W, ops)).max())
    print("[3] symmetry and truncation done", flush=True)

    # 4. MobileNetV2 oracle -------------------------------------------------------------------------
    Wmob = mn.load(); n_o = 64
    variants = {"base": x_test, "mirror (exact)": run(z_test[:n_o], Wm_, ops), "truncation 0.7 (exact)": xt}
    variants.update({f"shift {n} (orth, keep near-duplicates)": outs[f"shift_{n}_orthnd"] for n in names})
    variants.update({f"clamp {n}": outs[f"clamp_{n}"] for n in ("fur_lightness", "background")})
    variants.update({k: eye_out[k] for k in ("ungated 1x1 colour fold", "prune2 + eye gate")})
    R["oracle_cat_rate"] = {k: cat_rate(x[:n_o], Wmob) for k, x in variants.items()}
    print("[4] oracle:", json.dumps(R["oracle_cat_rate"]), flush=True)

    # 5. artefacts ------------------------------------------------------------------------------------
    for pid, p in P.items(): json.dump(p, open(os.path.join(a.out, "patches", pid + ".weightpatch.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "catgan26_with_ops.npz"), __ops__=json.dumps(ops), **W)          # base for the exp5 cat repo
    json.dump(R, open(os.path.join(a.out, "exp13_results.json"), "w"), indent=1, default=float)
    k8 = slice(0, 8)
    grid([x_test[k8]] + [outs[f"shift_{n}_orthnd"][k8] for n in names] + [outs["clamp_background"][k8], xt[k8]],
         ["base"] + [f"shift {n} (orth) +{a.alpha:g}" for n in names] + [f"clamp background={a.target:g}", "truncation 0.7"],
         os.path.join(a.out, "fig_cat_latent_patches.png"))
    order = ["base", "latent towards blue, shift 2", "latent towards blue, shift 4", "ungated 1x1 colour fold", "prune2", "prune2 + eye gate"]
    grid([eye_out[k][k8] for k in order], order, os.path.join(a.out, "fig_cat_blue_eyes.png"))
    hero = upscale4(np.concatenate([x0[:4], eye_out["prune2 + eye gate"][:4]]))
    Image.fromarray(np.concatenate([np.concatenate(list(u8(hero[:4])), 1), np.concatenate(list(u8(hero[4:])), 1)], 0)).save(os.path.join(a.out, "fig_cat_blue_eyes_512.png"))
    write_md(R, names, a)

def write_md(R, names, a):
    L = ["# Experiment 13: 2018 weight-space tools on CatGAN-26\n",
         f"Base `{R['base']}`. Latent fits n={a.n_fit}, held-out n={a.n_test}; shifts α={a.alpha:g}, clamps to {a.target:g}. Effects in held-out SD.\n",
         "## Latent directions\n", "| attribute | held-out R² (raw) | (orthogonalised) | clamp: shift (SD) | clamp: variance ratio |", "|---|---|---|---|---|"]
    L[-2:] = ["| attribute | held-out R² (raw) | (orthogonalised) | (orth., keep near-duplicates) | clamp: shift (SD) | clamp: variance ratio |", "|---|---|---|---|---|---|"]
    L += [f"| {n} | {R['r2_heldout'][n]:.3f} | {R['r2_heldout_orth'][n]:.3f} | {R['r2_heldout_orthnd'][n]:.3f} | {R['clamp'][n]['shift_sd']:+.2f} | {R['clamp'][n]['var_ratio']:.2f} |" for n in names]
    for key, title in (("entanglement_raw", "raw directions"), ("entanglement_orth", "orthogonalised against all others (exp3)"),
                       ("entanglement_orthnd", "orthogonalised against all but near-duplicates (|cos| > 0.9)")):
        L += [f"\nShift patch (row) -> measured change (column), {title}:\n", "| patch \\ measured | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        L += [f"| {n} | " + " | ".join((f"**{x:+.2f}**" if i == j else f"{x:+.2f}") for j, x in enumerate(row)) + " |" for i, (n, row) in enumerate(zip(names, R[key]))]
    mo, mt = R["mean_offtarget_sd"], R["mean_ontarget_sd"]
    L.append(f"\nMean off-target effect per patch: raw {mo['raw']:.2f} SD, orthogonalised {mo['orth']:.2f}, keep-near-duplicates {mo['orthnd']:.2f}; "
             f"on-target: raw {mt['raw']:.2f}, orthogonalised {mt['orth']:.2f}, keep-near-duplicates {mt['orthnd']:.2f}.\n")
    L += ["Clamp merge order (A∘B vs B∘A):\n", "| pair | cos | max weight diff | output MAE |", "|---|---|---|---|"]
    L += [f"| {o['pair'][0]} / {o['pair'][1]} | {o['cos']:+.3f} | {o['weight_maxabs']:.2e} | {o['output_mae']:.4f} |" for o in R["clamp_order"]]
    L += ["\n## Blue eyes (iris pixels from the base image; 'rest' = pixels more than 3 px from any iris pixel)\n",
          f"Iris detected in {R['eye_mask_share_test']:.0%} of held-out cats.\n",
          "| edit | params changed | eye hue | dist. to target | rest MAE | rest p99 px change |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['params_changed']} | {v['eye_hue']:.0f}° | {v['eye_dist_to_target']:.3f} | {v['rest_mae']:.4f} | {v['rest_max_px_change_p99']:.3f} |" for k, v in R["eye"].items()]
    L += [f"\nGate threshold sweep. The shipped quantile ({R['checks']['theta_quantile']}) is the strictest whose eye hue on the *fitting* "
          f"images lands in {HUE_BAND[0]}–{HUE_BAND[1]}°; the other columns are held-out.\n",
          "| quantile | fit eye hue | held-out eye hue | dist. to target | rest MAE | rest p99 px change |", "|---|---|---|---|---|---|"]
    L += [f"| {q} | {v['fit_eye_hue']:.0f}° | {v['eye_hue']:.0f}° | {v['eye_dist_to_target']:.3f} | {v['rest_mae']:.4f} | {v['rest_max_px_change_p99']:.3f} |" for q, v in R["eye_gate_q_sweep"].items()]
    L += ["\n## MobileNetV2 oracle (share of 64 samples whose top-1 ImageNet class is a domestic cat)\n", "| generator | cat rate |", "|---|---|"]
    L += [f"| {k} | {v:.0%} |" for k, v in R["oracle_cat_rate"].items()]
    L += ["\n## Checks\n"] + [f"- {k}: {v}" for k, v in R["checks"].items()]
    open(os.path.join(a.out, "exp13_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
