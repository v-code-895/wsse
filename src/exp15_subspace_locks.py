"""
Experiment 15: subspace locks. The 2018 lock is a rank-1 latent clamp (exp3): it fixes z.d for one
fitted direction d. On CatGAN-26 the locked attributes are only partly linear in z (held-out R^2
0.32-0.66), so the rank-1 locks leave 68-71% of the variance in context and fail their spec
(variance ratio <= 0.3, exp5 --profile cat).

A rank-k lock clamps a k-dimensional latent subspace instead: k latent_clamp ops with orthonormal
directions (no new operator; orthonormal clamps commute). The first direction is the 2018 one, with
its target; each further direction is found greedily as the top eigenvector of the gradient outer
product E[g g^T] of the attribute, taken at already-clamped latents with the chosen directions
projected out (an active subspace [Constantine et al. 2014], refitted after each clamp). Extra
directions are clamped to 0, the latent mean. The rank is the smallest k whose variance ratio on a
selection set is <= 0.2 (the test bound is 0.3); the tests in exp5 use different latents.

Reported: variance ratio vs rank, collateral on the other attributes, a random-subspace control of
the same rank, MobileNetV2 Frechet distance / feature diversity / cat rate, and fold exactness.

  <venv python> src/exp15_subspace_locks.py [--weights results/exp13_out/catgan26_with_ops.npz] [--src results/exp13_out/patches] [--out results/exp15_out]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cat_domain as cd, modern_np as mnp, mobilenet_np as mn
import exp3_anime_weightpatch as e3
from exp2b_tinysr18_weightpatch import load_model, weights_hash, make_patch
from exp12_modern_eval import frechet, mobilenet, u8, CAT_CLASSES
from exp13_catgan26_patches import grid

DEV = "cuda"; LOCKS = ("fur_lightness", "background"); SELECT = 0.2; AFHQ_VAL = "data/afhq/data/afhq/val/cat"
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

_LUM = torch.tensor([0.299, 0.587, 0.114], device=DEV)
_FACE, _BACK = torch.from_numpy(cd.FACE).to(DEV), torch.from_numpy(cd.BACK).to(DEV)
ATTR_T = {"fur_lightness": lambda x: (x[:, _FACE] @ _LUM).mean(1), "background": lambda x: (x[:, _BACK] @ _LUM).mean(1)}

def clamp(z, U, t):
    """z with its coordinates in span(U) set to t (U orthonormal, d x k): sequential latent_clamp ops."""
    return z - (z @ U - np.asarray(t)) @ U.T

def grad_attr(z, name, T, bs=64):
    out = []
    for s in range(0, len(z), bs):
        zt = torch.tensor(z[s:s + bs], dtype=torch.float32, device=DEV, requires_grad=True)
        x = mnp.catgan_torch(zt, T).clamp(0, 1).permute(0, 2, 3, 1)
        g, = torch.autograd.grad(ATTR_T[name](x).sum(), zt); out.append(g.cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def lock_patch(pid, name, U, t, base, layer):
    ops = [e3.clamp_op(layer, U[:, i], float(round(t[i], 6))) for i in range(U.shape[1])]
    return make_patch(pid, f"lock {name}: clamp a rank-{U.shape[1]} latent subspace (2018 direction + "
                           f"{U.shape[1] - 1} active-subspace directions at the latent mean)", base, ops)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/exp13_out/catgan26_with_ops.npz"); ap.add_argument("--src", default="results/exp13_out/patches")
    ap.add_argument("--out", default="results/exp15_out"); ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--n-grad", type=int, default=1024); ap.add_argument("--n-sel", type=int, default=1024)
    ap.add_argument("--n-fd", type=int, default=300); ap.add_argument("--no-fd", action="store_true")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(os.path.join(a.out, "patches"), exist_ok=True)
    W, ops, _ = load_model(a.weights); base = weights_hash(W); layer = ops[0]["name"]; T = mnp.to_torch(W, DEV)
    names = list(cd.ATTRS); zdim = W[layer + "/kernel"].shape[0]
    z_g = np.random.default_rng(1).standard_normal((a.n_grad, zdim)); z_s = np.random.default_rng(2).standard_normal((a.n_sel, zdim))
    A0 = cd.attributes(cd.run(z_s, W, ops)); var0, sd0 = A0.var(0), A0.std(0)
    x_chk = mnp.catgan_torch(torch.tensor(z_s[:16], dtype=torch.float32, device=DEV), T).clamp(0, 1).permute(0, 2, 3, 1)
    R = {"base": base, "select_bound": SELECT, "test_bound": 0.3, "locks": {},
         "checks": {"torch_attr_vs_numpy_maxabs": max(float(np.abs(ATTR_T[n](x_chk).detach().cpu().numpy() - A0[:16, names.index(n)]).max()) for n in LOCKS)}}
    P, Z = {}, {}
    for name in LOCKS:
        j = names.index(name); p1 = json.load(open(os.path.join(a.src, f"clamp_{name}.weightpatch.json"), encoding="utf-8"))
        d = np.array(p1["ops"][0]["direction"], np.float64); d /= np.linalg.norm(d)
        U, t, curve, t0 = d[:, None], [p1["ops"][0]["target"]], [], time.time()
        for k in range(1, a.kmax + 1):
            A = cd.attributes(cd.run(clamp(z_s, U, t), W, ops)); r = A.var(0) / var0
            row = {"k": k, "var_ratio": float(r[j]), "shift_sd": float((A[:, j] - A0[:, j]).mean() / sd0[j]),
                   "others_var_ratio": {n: float(r[i]) for i, n in enumerate(names) if i != j}}
            if k == a.kmax: curve.append(row); break
            G = grad_attr(clamp(z_g, U, t), name, T); G -= (G @ U) @ U.T
            w, V = np.linalg.eigh(G.T @ G / len(G))
            row["resid_grad_energy"] = float(w.sum()); row["next_eig_share"] = float(w[-1] / w.sum())
            curve.append(row); U = np.c_[U, V[:, -1]]; t.append(0.0)
            if k % 5 == 0: print(f"  {name} k={k}: var ratio {row['var_ratio']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        ks = next((c["k"] for c in curve if c["var_ratio"] <= SELECT), a.kmax)
        Uk, tk = U[:, :ks], t[:ks]
        # random-subspace control: the 2018 direction plus ks-1 random orthonormal directions
        Q, _ = np.linalg.qr(np.c_[d, np.random.default_rng(3).standard_normal((zdim, ks - 1))])
        Q[:, 0] *= np.sign(Q[:, 0] @ d)
        A_rand = cd.attributes(cd.run(clamp(z_s, Q, tk), W, ops))
        P[f"rank1_{name}"] = p1; P[f"rankk_{name}"] = lock_patch(f"lock_{name}", name, Uk, tk, base, layer)
        Z[f"rank1_{name}"] = (U[:, :1], t[:1]); Z[f"rankk_{name}"] = (Uk, tk); Z[f"random_{name}"] = (Q, tk)
        R["locks"][name] = {"rank": ks, "curve": curve, "random_control_var_ratio": float(A_rand[:, j].var() / var0[j]),
                            "random_control_others": {n: float(A_rand[:, i].var() / var0[i]) for i, n in enumerate(names) if i != j},
                            "seconds": time.time() - t0}
        json.dump(P[f"rankk_{name}"], open(os.path.join(a.out, "patches", f"lock_{name}.weightpatch.json"), "w", encoding="utf-8"), indent=1)
        print(f"[{name}] rank {ks}: var ratio {curve[ks - 1]['var_ratio']:.3f} (rank 1: {curve[0]['var_ratio']:.3f}, "
              f"random rank {ks}: {R['locks'][name]['random_control_var_ratio']:.3f})", flush=True)

    # exactness: the folded patch vs editing z, and the JSON round trip
    for name in LOCKS:
        p = json.load(open(os.path.join(a.out, "patches", f"lock_{name}.weightpatch.json"), encoding="utf-8"))
        Wp = e3.apply_weightpatch(W, p, ops); zz = z_s[:32]
        R["checks"][f"{name}_fold_vs_latent_edit_maxabs"] = float(np.abs(cd.run(zz, Wp, ops) - cd.run(e3.latent_edit(zz, p), W, ops)).max())
        U, t = Z[f"rankk_{name}"]
        R["checks"][f"{name}_latent_edit_vs_projection_maxabs"] = float(np.abs(e3.latent_edit(zz, p) - clamp(zz, U, t)).max())
    # the two rank-k locks together: principal angles between the subspaces, and the joint lock
    U1, t1 = Z["rankk_fur_lightness"]; U2, t2 = Z["rankk_background"]
    s = np.linalg.svd(U1.T @ U2, compute_uv=False); R["subspace_cosines"] = s.round(3).tolist()
    D = np.c_[U1, U2]; c = D @ np.linalg.lstsq(D.T @ D, np.r_[t1, t2], rcond=None)[0]
    Qj, _ = np.linalg.qr(D); A = cd.attributes(cd.run(clamp(z_s, Qj, Qj.T @ c), W, ops)); r = A.var(0) / var0
    R["joint_lock_var_ratio"] = {n: float(r[i]) for i, n in enumerate(names)}

    # MobileNetV2: cat rate, Frechet distance to real AFHQ val cats, feature diversity
    if not a.no_fd:
        Wm = mn.load(); n = a.n_fd; zf = np.random.default_rng(4).standard_normal((n, zdim))
        val = np.stack([np.asarray(Image.open(os.path.join(AFHQ_VAL, f)).convert("RGB").resize((128, 128), Image.LANCZOS))
                        for f in sorted(os.listdir(AFHQ_VAL))[:n]])
        _, Fv = mobilenet(val, Wm); tr_v = float(np.trace(np.cov(Fv, rowvar=False))); R["fd"] = {}
        for key in ["base"] + [f"{kind}_{nm}" for nm in LOCKS for kind in ("rank1", "rankk", "random")]:
            zz = zf if key == "base" else clamp(zf, *Z[key]); t0 = time.time()
            Lg, Fe = mobilenet(u8(cd.run(zz, W, ops)), Wm)
            R["fd"][key] = {"frechet_vs_val": frechet(Fe, Fv), "feature_diversity_vs_val": float(np.trace(np.cov(Fe, rowvar=False)) / tr_v),
                            "domestic_cat_rate": float(np.isin(Lg.argmax(1), CAT_CLASSES).mean())}
            print(key, R["fd"][key], f"{time.time() - t0:.0f}s", flush=True)

    # figure: the same latents under rank-1 and rank-k locks
    zz = np.random.default_rng(5).standard_normal((8, zdim)); rows, labels = [cd.run(zz, W, ops)], ["base"]
    for name in LOCKS:
        for kind in ("rank1", "rankk"):
            rows.append(cd.run(clamp(zz, *Z[f"{kind}_{name}"]), W, ops))
            labels.append(f"lock {name}, rank {Z[f'{kind}_{name}'][0].shape[1]}")
    grid(rows, labels, os.path.join(a.out, "fig_cat_subspace_locks.png"), lw=200)

    json.dump(R, open(os.path.join(a.out, "exp15_results.json"), "w"), indent=1)
    L = ["# Experiment 15: subspace locks on CatGAN-26\n",
         f"Base `{base}`. Rank chosen on {a.n_sel} selection latents as the smallest k with variance ratio <= {SELECT} "
         f"(test bound 0.3); gradients from {a.n_grad} other latents.\n",
         "| lock | rank-1 var ratio | chosen rank k | rank-k var ratio | random rank-k control | other attributes (rank-k), var ratio |", "|---|---|---|---|---|---|"]
    for name, v in R["locks"].items():
        c1, ck = v["curve"][0], v["curve"][v["rank"] - 1]
        L.append(f"| {name} | {c1['var_ratio']:.3f} | {v['rank']} | {ck['var_ratio']:.3f} | {v['random_control_var_ratio']:.3f} | "
                 + ", ".join(f"{n} {x:.2f}" for n, x in ck["others_var_ratio"].items()) + " |")
    L += ["", "Variance ratio by rank: " + "; ".join(f"{n}: " + ", ".join(f"k={c['k']} {c['var_ratio']:.2f}" for c in v["curve"] if c["k"] in (1, 2, 3, 5, 8, 12, 16, 24, 32, 40)) for n, v in R["locks"].items()),
          f"\nSubspace cosines (principal angles) between the two rank-k locks: {R['subspace_cosines']}. "
          "Joint lock (span of both, targets kept): " + ", ".join(f"{n} {x:.2f}" for n, x in R["joint_lock_var_ratio"].items()) + "\n"]
    if "fd" in R:
        L += [f"MobileNetV2 on {a.n_fd} samples per set (Frechet distance to {a.n_fd} AFHQ val cats; diversity = trace of the feature covariance relative to the real cats):\n",
              "| generator | Frechet distance | feature diversity | domestic-cat rate |", "|---|---|---|---|"]
        L += [f"| {k} | {v['frechet_vs_val']:.1f} | {v['feature_diversity_vs_val']:.3f} | {v['domestic_cat_rate']:.1%} |" for k, v in R["fd"].items()]
    L += ["", "Checks: " + ", ".join(f"{k} {v:.1e}" for k, v in R["checks"].items())]
    open(os.path.join(a.out, "exp15_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
