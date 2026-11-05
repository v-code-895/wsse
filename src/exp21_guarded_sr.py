"""
Experiment 21: a repair no output-layer patch can express, by rewiring the network (TinySR-18A).

§7.2: TinySR-18A lifts pure black (+8.9 RGB on the edge suite). Every output-layer patch failed
the regression gate: the tone error is nonlinear, and anchoring black destroys mid-tones. Here the
patch changes the topology instead, using only the layer types TinySR-18A already has:
  1. a reference path: three carrier channels per hidden layer carry the low-resolution image
     (copied by conv5 and conv6, upsampled by the network's own nearest upsampling, smoothed by a
     5x5 binomial kernel in conv7). Images are >= 0 and the kernels are >= 0, so `carry` passes
     the reference R through every LeakyReLU exactly;
  2. a guard: conv9 also emits [f, -f, u, -u] with u = f - (1 + k) R - delta, and one new 1x1 conv
     computes f - relu(u) = min(f, (1 + k) R + delta) with the `pass` and `relu` identities.
For every input, every output pixel satisfies out <= (1 + k) R + delta: where the input
neighbourhood is black, the output is at most delta. That holds by construction, not on data.
k and delta are chosen on the calibration split; the test suites and the gate are exp2b's.

Variants: guard alone; the 12-number colour fold of §7.2 plus the guard; and the same with the
carriers placed in channels freed by regression instead of new ones (no wider layers).
Baseline: fine-tune all weights of TinySR-18A with Adam on the same calibration images.

  <venv python> src/exp21_guarded_sr.py
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits as cc
import exp2b_tinysr18_weightpatch as e2

OUT = "results/exp21_out"; A = 0.2; LR_FILTER = "nearest"          # as in §7.2 (exp2b --lr-filter nearest)
BINOM = np.outer(*[np.array([1, 4, 6, 4, 1], np.float64) / 16] * 2)

def torch_run(x, W, ops):
    import torch, modern_np as mnp
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    T = mnp.to_torch(W)
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x[None], np.float32)).permute(0, 3, 1, 2).cuda()
        return mnp.legacy_torch(t, T, ops)[0].permute(1, 2, 0).cpu().numpy()

def reference(lr):
    """R: nearest 2x upsampling, then the 5x5 binomial, zero-padded: what the carriers compute."""
    up = np.repeat(np.repeat(lr.astype(np.float64), 2, 0), 2, 1)
    return e2.conv2d_same(up.astype(np.float32), np.repeat(BINOM[:, :, None, None], 1, 2).astype(np.float32) * np.eye(3)[None, None], np.zeros(3, np.float32))

# ---- topology edits -----------------------------------------------------------------------------
HIDDEN = ["conv2d_5", "conv2d_6", "conv2d_7", "conv2d_8"]; LAST = "conv2d_9"

def wire_carriers(W, slots):
    """slots[layer] = 3 output channels of that hidden conv that carry R (new or freed)."""
    W = {k: v.copy() for k, v in W.items()}; prev = None
    for name in HIDDEN:
        K = W[name + "/kernel"]; s = slots[name]; K[..., s] = 0; W[name + "/bias"][s] = 0
        for c in range(3):
            if name == "conv2d_5": K[2, 2, c, s[c]] = 1.0                      # copy the image
            elif name == "conv2d_7": K[:, :, prev[c], s[c]] = BINOM            # smooth after nearest upsampling
            else: K[2, 2, prev[c], s[c]] = 1.0                                 # copy
        prev = s
    return W

def grow(W):
    """Add three zero channels to every hidden conv (zero outgoing weights: the model is unchanged)."""
    W = {k: v.copy() for k, v in W.items()}; slots = {}
    for name, nxt in zip(HIDDEN, HIDDEN[1:] + [LAST]):
        K = W[name + "/kernel"]; C = K.shape[3]
        W[name + "/kernel"] = np.concatenate([K, np.zeros(K.shape[:3] + (3,), np.float32)], 3)
        W[name + "/bias"] = np.r_[W[name + "/bias"], np.zeros(3, np.float32)]
        Kn = W[nxt + "/kernel"]; W[nxt + "/kernel"] = np.concatenate([Kn, np.zeros(Kn.shape[:2] + (3, Kn.shape[3]), np.float32)], 2)
        slots[name] = [C, C + 1, C + 2]
    return W, slots

def free(W, ops, calib):
    """Free three channels of every hidden conv by regression (refit after each layer), no new channels."""
    slots = {}
    for name, nxt in zip(HIDDEN, HIDDEN[1:] + [LAST]):
        stop = next(i for i, o in enumerate(ops) if o.get("name") == name) + 2           # through its lrelu
        G = cc.gram(torch_run(lr, W, ops[:stop]).reshape(-1, W[name + "/bias"].shape[0]) for lr, _ in calib.values())
        drop, keep, B, _ = cc.choose_free(G, cc.out_energy(W, nxt), 3)
        W = cc.fold_free(W, name, nxt, drop, keep, B); slots[name] = sorted(drop)
    return W, slots

def guard_head(W, ops, ref_slots, k, delta, Mc=None, tc=None):
    """conv9 -> [f, -f, u, -u] (12 ch), lrelu, new 1x1 conv: f - relu(u). Optional colour fold f = Mc f + tc."""
    W = {k_: v.copy() for k_, v in W.items()}
    K, b = W[LAST + "/kernel"].astype(np.float64), W[LAST + "/bias"].astype(np.float64)
    if Mc is not None: K = K @ Mc.T; b = Mc @ b + tc
    U = K.copy(); ub = b - delta
    for c in range(3): U[2, 2, ref_slots[c], c] -= (1 + k)
    W[LAST + "/kernel"] = np.concatenate([K, -K, U, -U], 3).astype(np.float32)
    W[LAST + "/bias"] = np.r_[b, -b, ub, -ub].astype(np.float32)
    p, g = 1 / (1 + A), 1 / (1 - A * A); R = np.zeros((1, 1, 12, 3))
    for c in range(3): R[0, 0, c, c], R[0, 0, 3 + c, c], R[0, 0, 6 + c, c], R[0, 0, 9 + c, c] = p, -p, -g, -A * g
    W["guard/kernel"], W["guard/bias"] = R.astype(np.float32), np.zeros(3, np.float32)
    return W, ops + [{"op": "lrelu", "alpha": A}, {"op": "conv", "name": "guard", "act": "linear"}]

def dark_guard_head(W, ops, ref_slots, delta, t, Kd, Mc=None, tc=None):
    """Guard only where the reference is dark: out = min(f, R + delta + Kd relu(R - t)).
    conv9 -> [f, -f, v, -v, R - t, -(R - t)] with v = f - R - delta (18 ch), lrelu,
    1x1 -> [f, -f, u, -u] with u = v - Kd relu(R - t) (12 ch), lrelu, 1x1 -> f - relu(u)."""
    W = {k_: v_.copy() for k_, v_ in W.items()}
    K, b = W[LAST + "/kernel"].astype(np.float64), W[LAST + "/bias"].astype(np.float64)
    if Mc is not None: K = K @ Mc.T; b = Mc @ b + tc
    V = K.copy(); Rt = np.zeros_like(K)
    for c in range(3): V[2, 2, ref_slots[c], c] -= 1.0; Rt[2, 2, ref_slots[c], c] = 1.0
    W[LAST + "/kernel"] = np.concatenate([K, -K, V, -V, Rt, -Rt], 3).astype(np.float32)
    W[LAST + "/bias"] = np.r_[b, -b, b - delta, -(b - delta), np.full(3, -t), np.full(3, t)].astype(np.float32)
    p, g = 1 / (1 + A), 1 / (1 - A * A); G1 = np.zeros((1, 1, 18, 12)); G2 = np.zeros((1, 1, 12, 3))
    for c in range(3):
        G1[0, 0, c, c], G1[0, 0, 3 + c, c] = p, -p; G1[0, 0, c, 3 + c], G1[0, 0, 3 + c, 3 + c] = -p, p                 # f, -f
        G1[0, 0, 6 + c, 6 + c], G1[0, 0, 9 + c, 6 + c] = p, -p                                                          # u = v ...
        G1[0, 0, 12 + c, 6 + c], G1[0, 0, 15 + c, 6 + c] = -Kd * g, -Kd * A * g                                       # ... - Kd relu(R - t)
        G1[0, 0, :, 9 + c] = -G1[0, 0, :, 6 + c]                                                                         # -u
        G2[0, 0, c, c], G2[0, 0, 3 + c, c], G2[0, 0, 6 + c, c], G2[0, 0, 9 + c, c] = p, -p, -g, -A * g                 # f - relu(u)
    W["dguard1/kernel"], W["dguard1/bias"] = G1.astype(np.float32), np.zeros(12, np.float32)
    W["dguard2/kernel"], W["dguard2/bias"] = G2.astype(np.float32), np.zeros(3, np.float32)
    return W, ops + [{"op": "lrelu", "alpha": A}, {"op": "conv", "name": "dguard1", "act": "linear"},
                     {"op": "lrelu", "alpha": A}, {"op": "conv", "name": "dguard2", "act": "linear"}]

# ---- evaluation -----------------------------------------------------------------------------------
def evaluate(models, S, lr_filter):
    R, outs = {}, {}
    for name, (W, ops) in models.items():
        R[name] = {}
        for grp in (e2.REGRESSION, e2.EDGE):
            for key, (lr, hr), suite in e2.iter_group(S, grp):
                pred = torch_run(lr, W, ops); R[name][key] = {"suite": suite, **e2.metrics(pred, hr, lr, lr_filter)}; outs[(name, key)] = pred
    return R, outs

def finetune(W, ops, calib, steps, lr_rate=1e-4, dark_weight=1.0, seed=0, every=250):
    """Adam on all weights, L1 on the calibration pairs; returns checkpoints {step: (weights, seconds)}."""
    import torch, torch.nn.functional as F, modern_np as mnp
    torch.manual_seed(seed); T = {k: v.clone().requires_grad_(True) for k, v in mnp.to_torch(W).items()}
    opt = torch.optim.Adam(T.values(), lr=lr_rate)
    X = [torch.from_numpy(np.ascontiguousarray(lr[None], np.float32)).permute(0, 3, 1, 2).cuda() for lr, _ in calib.values()]
    Y = [torch.from_numpy(np.ascontiguousarray(hr[None], np.float32)).permute(0, 3, 1, 2).cuda() for _, hr in calib.values()]
    ck = {}; t0 = time.time()
    for step in range(1, steps + 1):
        i = (step - 1) % len(X); y = Y[i]; p = mnp.legacy_torch(X[i], T, ops).clamp(0, 1)
        w = torch.where(y.max(1, keepdim=True).values < 0.02, dark_weight, 1.0)
        loss = (w * (p - y).abs()).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step % every == 0:
            torch.cuda.synchronize()
            ck[step] = ({k: (v.detach().permute(2, 3, 1, 0) if v.ndim == 4 else v.detach()).cpu().numpy().copy() for k, v in T.items()}, time.time() - t0)
    return ck

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True); t_all = time.time()
    W0, ops_full, _ = e2.load_model("legacy/TinySR-18/generator.h5"); ops, tail = e2.strip_clip_equivalent_tail(ops_full)
    S = e2.build_suites(LR_FILTER, None); calib = {**S["natural_calib"], **S["sprite_calib"]}
    reg = [k for k, _, _ in e2.iter_group(S, e2.REGRESSION)]; edg = [k for k, _, _ in e2.iter_group(S, e2.EDGE)]
    Rj = {"calib_images": list(calib), "regression_images": reg, "edge_images": edg}

    # base outputs on calibration: f, and the reference R
    t0 = time.time()
    f_cal = {k: torch_run(lr, W0, ops) for k, (lr, _) in calib.items()}; R_cal = {k: reference(lr) for k, (lr, _) in calib.items()}
    caches = {k: [f_cal[k]] for k in calib}
    Mc, tc, _ = e2.fit_output_affine(calib, caches)
    def cal_score(fold, k, d):
        ps, bf = [], []
        for key, (lr, hr) in calib.items():
            f = f_cal[key] @ Mc.T + tc if fold else f_cal[key]
            p = np.clip(np.minimum(f, (1 + k) * R_cal[key] + d) if k is not None else f, 0, 1); ps.append(e2.psnr(p, hr))
            blk = hr.max(-1) < 0.02
            if blk.mean() > 0.01: bf.append(p[blk].mean() * 255)
        return float(np.mean(ps)), float(np.mean(bf))
    grid = [(k, d / 255) for k in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0) for d in (1, 2, 4, 8, 16)]
    choice = {}
    for fold in (False, True):
        p_ref, bf_ref = cal_score(fold, None, 0); rows = [(k, d) + cal_score(fold, k, d) for k, d in grid]
        ok = [r for r in rows if r[2] >= p_ref - 0.05]; best = min(ok, key=lambda r: r[3])
        choice[fold] = {"k": best[0], "delta_rgb": best[1] * 255, "calib_psnr": best[2], "calib_black_floor": best[3],
                        "unguarded_calib_psnr": p_ref, "unguarded_calib_black_floor": bf_ref, "grid": [list(map(float, r)) for r in rows]}
    Rj["choice"] = {("fold+guard" if f else "guard"): v for f, v in choice.items()}
    def cal_dark(d, t, Kd):
        ps, bf = [], []
        for key, (lr, hr) in calib.items():
            f = f_cal[key] @ Mc.T + tc; R_ = R_cal[key]
            p = np.clip(np.minimum(f, R_ + d + Kd * np.maximum(R_ - t, 0)), 0, 1); ps.append(e2.psnr(p, hr))
            blk = hr.max(-1) < 0.02
            if blk.mean() > 0.01: bf.append(p[blk].mean() * 255)
        return float(np.mean(ps)), float(np.mean(bf))
    p_ref = choice[True]["unguarded_calib_psnr"]
    rows = [(d / 255, t, Kd) + cal_dark(d / 255, t, Kd) for d in (1, 2, 4) for t in (0.02, 0.05, 0.1, 0.2) for Kd in (2, 8, 32)]
    ok = [r for r in rows if r[3] >= p_ref - 0.05]; best = min(ok, key=lambda r: r[4])
    Rj["choice"]["fold+dark guard"] = {"delta_rgb": best[0] * 255, "t": best[1], "K": best[2], "calib_psnr": best[3], "calib_black_floor": best[4],
                                       "grid": [list(map(float, r)) for r in rows]}
    fit_seconds = time.time() - t0
    print(f"[1] guard parameters on calibration ({fit_seconds:.1f}s): " + json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if kk != 'grid'} for k, v in Rj['choice'].items()}), flush=True)

    # build the networks
    Wg, slots_g = grow(W0); Wg = wire_carriers(Wg, slots_g)
    t1 = time.time(); Wf, slots_f = free(W0, ops, calib); Wf = wire_carriers(Wf, slots_f); free_seconds = time.time() - t1
    models = {"base_model": (W0, ops)}
    Wc = {k: v.copy() for k, v in W0.items()}; Wc[LAST + "/kernel"] = (W0[LAST + "/kernel"].astype(np.float64) @ Mc.T).astype(np.float32)
    Wc[LAST + "/bias"] = (Mc @ W0[LAST + "/bias"] + tc).astype(np.float32); models["color_fold"] = (Wc, ops)
    g, fg = choice[False], choice[True]
    models["guard"] = guard_head(Wg, ops, slots_g[HIDDEN[-1]], g["k"], g["delta_rgb"] / 255)
    models["fold+guard"] = guard_head(Wg, ops, slots_g[HIDDEN[-1]], fg["k"], fg["delta_rgb"] / 255, Mc, tc)
    models["fold+guard (freed channels)"] = guard_head(Wf, ops, slots_f[HIDDEN[-1]], fg["k"], fg["delta_rgb"] / 255, Mc, tc)
    dg = Rj["choice"]["fold+dark guard"]
    models["fold+dark guard"] = dark_guard_head(Wg, ops, slots_g[HIDDEN[-1]], dg["delta_rgb"] / 255, dg["t"], dg["K"], Mc, tc)
    models["fold+dark guard (freed channels)"] = dark_guard_head(Wf, ops, slots_f[HIDDEN[-1]], dg["delta_rgb"] / 255, dg["t"], dg["K"], Mc, tc)

    # exactness: the rewired network computes min(f, (1+k) R + delta), and the guarantee holds on noise too
    rng = np.random.default_rng(0); probes = [rng.random((64, 64, 3), dtype=np.float32) for _ in range(4)] + [lr for lr, _ in list(calib.values())[:3]]
    ex = {}
    for name in ("guard", "fold+guard"):
        Wm, om = models[name]; c = choice[name == "fold+guard"]; err = viol = 0.0
        for x in probes:
            f = torch_run(x, W0, ops); f = f @ Mc.T + tc if name == "fold+guard" else f
            cap = (1 + c["k"]) * reference(x) + c["delta_rgb"] / 255; out = torch_run(x, Wm, om)
            err = max(err, float(np.abs(out - np.minimum(f, cap)).max())); viol = max(viol, float((out - cap).max()))
        ex[name] = {"max_err_vs_formula": err, "max_violation_of_bound": viol}
    Wm, om = models["fold+dark guard"]; err = viol = 0.0
    for x in probes:
        f = torch_run(x, W0, ops) @ Mc.T + tc; R_ = reference(x); cap = R_ + dg["delta_rgb"] / 255 + dg["K"] * np.maximum(R_ - dg["t"], 0)
        out = torch_run(x, Wm, om); err = max(err, float(np.abs(out - np.minimum(f, cap)).max())); viol = max(viol, float((out - cap).max()))
    ex["fold+dark guard"] = {"max_err_vs_formula": err, "max_violation_of_bound": viol}
    Wn, on = guard_head(Wg, ops, slots_g[HIDDEN[-1]], 0.0, 10.0)          # guard disabled (cap = R + 10 > any output); a much larger M loses float32 precision
    ex["grown_network_without_guard_vs_base"] = float(max(np.abs(torch_run(x, Wn, on) - torch_run(x, W0, ops)).max() for x in probes))
    Rj["exactness"] = ex; print("[2] exactness:", json.dumps(ex), flush=True)

    # gradient baseline: fine-tune on the same calibration images
    t2 = time.time(); ck = finetune(W0, ops, calib, 4000, dark_weight=8.0); ft_seconds = time.time() - t2
    for step, (Wt, sec) in ck.items(): models[f"fine-tune {step} steps"] = (Wt, ops)

    R, outs = evaluate(models, S, LR_FILTER); Rj["variants"] = {}
    for name in models:
        b, e = e2.summarize(R, name, reg), e2.summarize(R, name, edg)
        v = {"reg_psnr": b["psnr"], "reg_ssim": b["ssim"], "edge_psnr": e["psnr"], "edge_black_floor": e["black_floor"],
             "params": int(sum(x.size for x in models[name][0].values()))}
        if name != "base_model": v["gate"] = e2.gate(R, name, reg, edg); v["verdict"] = v["gate"]["verdict"]
        if name.startswith("fine-tune"): v["train_seconds"] = ck[int(name.split()[1])][1]
        Rj["variants"][name] = v
        print(f"   {name:30s} reg {v['reg_psnr']:.2f} dB  edge {v['edge_psnr']:.2f} dB  black floor {v['edge_black_floor']:.2f}  "
              f"{v.get('verdict', 'reference')}", flush=True)
    Rj["seconds"] = {"guard_fit_on_calibration": fit_seconds, "free_channels": free_seconds, "fine_tune_4000_steps": ft_seconds, "total": time.time() - t_all}
    json.dump(Rj, open(os.path.join(OUT, "exp21_results.json"), "w"), indent=1, default=float)
    e2.save_figure(outs, S, ["base_model", "color_fold", "fold+guard", "fold+dark guard", "fine-tune 4000 steps"], os.path.join(OUT, "fig_guarded_sr.png"))
    print(f"done in {time.time() - t_all:.0f}s", flush=True)

if __name__ == "__main__":
    main()
