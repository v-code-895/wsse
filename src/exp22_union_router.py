"""
Experiment 22: resolving the §6 merge conflict by growing the network instead of averaging it.

exp14/exp16: two TinySR-26 fine-tunes (bicubic, noisy) each meet their own claim; no weighted merge,
cherry-pick or repair fine-tune meets the merge's specification, and only joint retraining does
(+0.03 / +3.19 / +0.08 dB). Within a fixed architecture the two specialisations compete.

Here the merged model keeps all three networks (base, bicubic branch, noisy branch) side by side
(block-diagonal union: each keeps its exact function) and adds a router that decides, per image,
which one answers:
  * router features, two versions:
    - input statistics: mean |response| of four fixed filters (horizontal and vertical differences,
      a Laplacian, a 2x2 checkerboard) at full and half resolution on the LR input;
    - self-consistency: degrade each expert's output again (nearest and bicubic 2x down) and take
      the mean |difference| from the LR input; the expert consistent with the input wins;
    both are convolutions, |u| = relu(u) + relu(-u) and one global average pool;
  * a linear classifier fitted in closed form (ridge regression to one-hot labels) on the 120
    fitting images, then argmax;
  * the experts' outputs are combined by the exact `mux` identity, y + relu(x - y - M(1 - g)) -
    relu(y - x - M(1 - g)), checked against plain selection.
With input statistics a runtime could run only the selected expert; the self-consistency router
needs every expert's output, so the union costs three networks' compute.

  <venv python> src/exp22_union_router.py
"""
import json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp10_tinysr26 as e10, exp14_tinysr26_branches as e14, exp16_merge_resolution as e16

OUT = "results/exp22_out"; DEV = "cuda"; CONDS = ("bicubic LR", "noisy bicubic LR", "nearest LR")
EXPERT_FOR = {"bicubic LR": "branch: bicubic", "noisy bicubic LR": "branch: noisy", "nearest LR": "base"}
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def lr_inputs(imgs, cond):
    """Exactly exp14.psnr_on's inputs (same PIL resize, same noise generator sequence)."""
    kind = "nearest" if cond == "nearest LR" else "bicubic"; noise = e14.NOISE if cond == "noisy bicubic LR" else 0.0
    g = torch.Generator(device=DEV).manual_seed(0); out = []
    for im in imgs:
        lr = np.asarray(im.resize((im.width // 2, im.height // 2), {"bicubic": Image.BICUBIC, "nearest": Image.NEAREST}[kind]), np.float32) / 255
        x = torch.from_numpy(lr).permute(2, 0, 1)[None].to(DEV)
        if noise: x = ((x + noise * torch.randn(x.shape, generator=g, device=DEV)).clamp(0, 1) * 255).round() / 255
        out.append(x)
    return out

FILTERS = torch.tensor([[[0, 0, 0], [0, 1, -1], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, -1], [0, -1, 1]]], dtype=torch.float32)

def features(x):
    """Mean |filter response| (grey input), at full and half resolution: convs, |.| as relu pairs, one global pool."""
    gy = x.mean(1, keepdim=True); f = []
    for h in (gy, F.avg_pool2d(gy, 2)):
        r = F.conv2d(h, FILTERS[:, None].to(x.device), padding=1)[..., 1:-1, 1:-1]
        f.append((F.relu(r) + F.relu(-r)).mean((2, 3)))
    return torch.cat(f, 1)[0]

@torch.no_grad()
def run_all(models, xs):
    return {name: [m(x).clamp(0, 1) for x in xs] for name, m in models.items()}

def mux(x, y, g, M=4.0):
    """g (0/1) ? x : y, with the relu identity (exact for |x - y| <= M)."""
    return y + F.relu(x - y - M * (1 - g)) - F.relu(y - x - M * (1 - g))

def consistency(ys, x):
    """Self-check features: degrade each expert's output again (nearest = sample the block centre, as PIL;
    bicubic = antialiased bicubic) and measure the mean |difference| from the LR input. Strided convs,
    |.| as relu pairs and one global pool."""
    f = []
    for y in ys:
        f.append((y[:, :, 1::2, 1::2] - x).abs().mean())
        f.append((F.interpolate(y, scale_factor=0.5, mode="bicubic", antialias=True, align_corners=False) - x).abs().mean())
    return torch.stack(f).cpu().numpy()

ROUTERS = ("input statistics", "self-consistency")

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True); t_all = time.time()
    W0 = {k: v.astype(np.float32) for k, v in np.load("results/exp10_out/tinysr26_x2.npz").items()}
    db, dn = (dict(np.load(f"results/exp14_out/delta_{n}.npz")) for n in ("bicubic", "noisy"))
    Ws = {"base": W0, "branch: bicubic": {k: W0[k] + db[k] for k in W0}, "branch: noisy": {k: W0[k] + dn[k] for k in W0}}
    models = {n: e14.from_dict(W).eval() for n, W in Ws.items()}; experts = list(models)
    fit, test = e16.fit_images(), e14.test_images(); R = {"fit_images": len(fit), "test_images": len(test), "routers": {}}
    feat = {"input statistics": lambda x, ys: features(x).cpu().numpy(), "self-consistency": lambda x, ys: consistency(ys, x)}

    # 1. routers: closed-form fits on the fitting images ------------------------------------------
    t0 = time.time(); X = {r: [] for r in ROUTERS}; Y = []
    with torch.no_grad():
        for cond in CONDS:
            xs = lr_inputs(fit, cond); outs = run_all(models, xs)
            for i, x in enumerate(xs):
                ys = [outs[n][i] for n in experts]
                for r in ROUTERS: X[r].append(feat[r](x, ys))
                Y.append(experts.index(EXPERT_FOR[cond]))
    Y = np.array(Y); fitted = {}
    for r in ROUTERS:
        Xr = np.array(X[r], np.float64); mu, sd = Xr.mean(0), Xr.std(0)
        Z = np.c_[(Xr - mu) / sd, np.ones(len(Xr))]; Wr = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ np.eye(3)[Y])
        fitted[r] = (mu, sd, Wr)
        R["routers"][r] = {"fit_accuracy": float(np.mean(np.argmax(Z @ Wr, 1) == Y)), "params": int(Wr.size + 2 * len(mu)), "features": int(len(mu))}
    R["fit_seconds_incl_expert_passes"] = time.time() - t0
    print(f"[1] routers fitted in {R['fit_seconds_incl_expert_passes']:.1f}s: " + json.dumps(R["routers"]), flush=True)
    route = lambda r, x, ys: int(np.argmax(np.r_[(feat[r](x, ys) - fitted[r][0]) / fitted[r][1], 1.0] @ fitted[r][2]))

    # 2. test: every expert, the union under each router, and an oracle router -------------------------
    names = experts + [f"union ({r})" for r in ROUTERS] + ["union (oracle)"]
    R["tests"] = {}; conf = {r: np.zeros((3, 3), int) for r in ROUTERS}; mux_err = 0.0
    psnr = lambda y, hr: float(10 * np.log10(1 / np.mean((y - hr) ** 2)))
    with torch.no_grad():
        for cond in CONDS:
            xs = lr_inputs(test, cond); outs = run_all(models, xs); row = {n: [] for n in names}
            for i, (im, x) in enumerate(zip(test, xs)):
                hr = np.asarray(im, np.float32) / 255; ys = [outs[n][i] for n in experts]
                for n, y in zip(experts, ys): row[n].append(psnr(y[0].permute(1, 2, 0).cpu().numpy(), hr))
                for r in ROUTERS:
                    k = route(r, x, ys); conf[r][CONDS.index(cond), k] += 1
                    g = [torch.tensor(float(k == j), device=DEV) for j in range(3)]
                    y = mux(ys[0], mux(ys[1], ys[2], g[1]), g[0])                                 # g0 ? e0 : (g1 ? e1 : e2)
                    mux_err = max(mux_err, float((y - ys[k]).abs().max()))
                    row[f"union ({r})"].append(psnr(y[0].permute(1, 2, 0).cpu().numpy(), hr))
                row["union (oracle)"].append(row[EXPERT_FOR[cond]][-1])
            R["tests"][cond] = {n: float(np.mean(v)) for n, v in row.items()}
            print(f"[2] {cond}: " + ", ".join(f"{n} {v:.2f}" for n, v in R["tests"][cond].items()), flush=True)
    for r in ROUTERS: R["routers"][r]["confusion_test"] = {"rows (input)": list(CONDS), "cols (expert)": experts, "counts": conf[r].tolist()}
    R["mux_max_err_vs_selection"] = mux_err
    base = {c: R["tests"][c]["base"] for c in CONDS}
    R["delta_vs_base"] = {n: {c: R["tests"][c][n] - base[c] for c in CONDS} for n in names}
    R["meets_spec"] = {n: bool(d["bicubic LR"] > 0 and d["noisy bicubic LR"] > 0 and min(d.values()) >= -0.1) for n, d in R["delta_vs_base"].items()}
    n1 = int(sum(v.size for v in W0.values())); R["params"] = {"one network": n1, "union (3 experts)": 3 * n1}
    ex16 = json.load(open("results/exp16_out/exp16_results.json"))
    R["exp16_reference"] = {"joint_retrain": ex16["test"]["(c) joint fine-tune from base (3k steps, no branches)"], "timing": ex16["resolution"]}
    R["seconds_total"] = time.time() - t_all
    json.dump(R, open(os.path.join(OUT, "exp22_results.json"), "w"), indent=1)
    for n in names: print(f"[3] {n:28s} " + "  ".join(f"{c} {v:+.2f}" for c, v in R["delta_vs_base"][n].items()) + f"  meets spec: {R['meets_spec'][n]}", flush=True)
    print("mux error vs plain selection:", mux_err, "| confusion:", {r: conf[r].tolist() for r in ROUTERS}, flush=True)

if __name__ == "__main__":
    main()
