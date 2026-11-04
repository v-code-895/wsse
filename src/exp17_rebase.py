"""
Experiment 17: rebasing a weight repo onto an upstream update.

Upstream: CatGAN-26 trained further, continuing exp11 from step 30,000 with the same settings,
exported at +500, +1,000, +2,000 and +4,000 steps. Downstream: the final cat repo of
exp5 --profile cat_subspace (prune2, the gated eye circuit and two subspace locks, every test passing
on the original base).

  1. drift: relative weight change, output change and per-attribute agreement on the same latents
  2. replay: `wpgit rebase` re-pins the repo to each new base (marking sparse patches whose context
     moved), then the same in-context tests run unchanged
  3. regenerate: on the +4,000 base, re-run the patch-fitting scripts (exp13, exp15) and the git story
     (exp5 --profile cat_subspace), i.e. rebuild the patches from their sources instead of porting them

  <venv python> src/exp17_rebase.py [--steps 4000]      (training is skipped when the exports exist)
"""
import argparse, copy, json, os, shutil, subprocess, sys, time
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exp11_catgan26 as e11, modern_np as mnp, cat_domain as cd, wpgit
from exp2b_tinysr18_weightpatch import load_model, weights_hash

OUT = "results/exp17_out"; DEV = "cuda"; SAVE_AT = (500, 1000, 2000, 4000); REPO = "results/exp5_cat_subspace_out/weightrepo"

def export(G_ema, path):
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    g = copy.deepcopy(G_ema).float().to(memory_format=torch.contiguous_format); e11.recalibrate_bn(g, DEV, batches=50)
    W = e11.fold_export(g); np.savez(path + ".npz", **W); np.savez(path + "_with_ops.npz", __ops__=json.dumps(mnp.catgan_ops(W)), **W)
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True

def continue_training(extra, save_at, lr=2e-4, B=32, seed=1):
    """exp11's training loop, resumed from its final checkpoint (G, D, EMA, both Adam states)."""
    torch.manual_seed(seed); torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    data = torch.from_numpy(e11.load_cats()).permute(0, 3, 1, 2).contiguous().to(DEV)
    G, D = e11.Generator().to(DEV, memory_format=torch.channels_last), e11.Discriminator().to(DEV, memory_format=torch.channels_last)
    G_ema = copy.deepcopy(G).eval()
    oG = torch.optim.Adam(G.parameters(), lr, betas=(0.0, 0.99)); oD = torch.optim.Adam(D.parameters(), lr, betas=(0.0, 0.99))
    c = torch.load(os.path.join(e11.OUT, "catgan26.pt"), map_location=DEV)
    for m, k in ((G, "G"), (D, "D"), (G_ema, "G_ema"), (oG, "oG"), (oD, "oD")): m.load_state_dict(c[k])
    step0 = step = c["step"]; print(f"resumed exp11 at step {step0}", flush=True)
    def real(n):
        x = data[torch.randint(0, len(data), (n,), device=DEV)].float() / 127.5 - 1
        return torch.where(torch.rand(n, 1, 1, 1, device=DEV) < 0.5, x.flip(3), x).contiguous(memory_format=torch.channels_last)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16); t0 = time.time()
    while step < step0 + extra:
        with torch.no_grad(), amp(): fake = G(torch.randn(B, e11.ZDIM, device=DEV))
        with amp(): dr, df = D(e11.diffaug(real(B))).float(), D(e11.diffaug(fake)).float()
        lD = F.softplus(-dr).mean() + F.softplus(df).mean()
        if step % e11.R1_EVERY == 0:
            xr = e11.diffaug(real(B // 2)).detach().requires_grad_(True)
            g, = torch.autograd.grad(D(xr).sum(), xr, create_graph=True)
            lD = lD + e11.R1_GAMMA / 2 * g.pow(2).sum((1, 2, 3)).mean() * e11.R1_EVERY
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
        with amp(): lG = F.softplus(-D(e11.diffaug(G(torch.randn(B, e11.ZDIM, device=DEV)))).float()).mean()
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step(); step += 1
        with torch.no_grad():
            beta = min(0.999, (1 + step) / (10 + step))
            for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 1 - beta)
        if (step - step0) in save_at:
            export(G_ema, os.path.join(OUT, f"catgan26_s{step}")); print(f"exported step {step} ({time.time() - t0:.0f}s)", flush=True)
    return step0

def drift(W0, ops, W1):
    flat = lambda W: np.concatenate([W[k].ravel() for k in sorted(W)])
    z = np.random.default_rng(wpgit.TEST_SEED).standard_normal((wpgit.TEST_N, W0[ops[0]["name"] + "/kernel"].shape[0]))
    x0, x1 = cd.run(z, W0, ops), cd.run(z, W1, ops); A0, A1 = cd.attributes(x0), cd.attributes(x1)
    return {"rel_weight_change": float(np.linalg.norm(flat(W1) - flat(W0)) / np.linalg.norm(flat(W0))),
            "output_mae": float(np.abs(x1 - x0).mean()),
            "attr_corr": {n: float(np.corrcoef(A0[:, i], A1[:, i])[0, 1]) for i, n in enumerate(cd.ATTRS)}}

def tests(repo):
    W0, ops, P = wpgit.load_repo(repo); rows, ok, h = wpgit.run_tests(W0, ops, P)
    return {"hash": h, "all_pass": bool(ok), "rows": [{"patch": r[0], "test": r[1], "value": r[2], "bound": r[3], "pass": bool(r[4])} for r in rows]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=max(SAVE_AT)); ap.add_argument("--skip-regen", action="store_true")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True)
    step0 = torch.load(os.path.join(e11.OUT, "catgan26.pt"), map_location="cpu")["step"]
    save_at = [s for s in SAVE_AT if s <= a.steps]
    if not all(os.path.exists(os.path.join(OUT, f"catgan26_s{step0 + s}_with_ops.npz")) for s in save_at):
        t0 = time.time(); continue_training(a.steps, save_at); print(f"training done in {time.time() - t0:.0f}s", flush=True)
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    base = json.load(open(os.path.join(REPO, "base.json"))); W0, ops, _ = load_model(os.path.join(REPO, base["weights"]))
    R = {"base": base["hash"], "repo": REPO, "original": tests(REPO), "upstream": {}}
    print("original base:", "ALL PASS" if R["original"]["all_pass"] else "FAIL", flush=True)
    for s in save_at:
        new = os.path.abspath(os.path.join(OUT, f"catgan26_s{step0 + s}_with_ops.npz")); W1, _, _ = load_model(new)
        d = os.path.join(OUT, f"replay_s{step0 + s}")
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(os.path.join(d, "patches")); shutil.copy(os.path.join(REPO, "base.json"), d)
        base_d = json.load(open(os.path.join(d, "base.json"))); base_d["weights"] = os.path.relpath(os.path.join(REPO, base["weights"]), d).replace("\\", "/")
        json.dump(base_d, open(os.path.join(d, "base.json"), "w"), indent=1)
        for f in os.listdir(os.path.join(REPO, "patches")): shutil.copy(os.path.join(REPO, "patches", f), os.path.join(d, "patches"))
        h_old, h_new, rows = wpgit.rebase(d, new)
        R["upstream"][f"+{s}"] = {"step": step0 + s, "hash": h_new, "drift": drift(W0, ops, W1),
                                  "rebase": [{"patch": r[0], "ops": r[1], "context_drift": r[2], "stale": r[3]} for r in rows], "replay": tests(d)}
        u = R["upstream"][f"+{s}"]
        print(f"+{s}: weights {u['drift']['rel_weight_change']:.4f}, output MAE {u['drift']['output_mae']:.4f}; replay "
              f"{'ALL PASS' if u['replay']['all_pass'] else 'FAIL'}: " + ", ".join(f"{r['patch']}/{r['test']} {r['value']:.3f}" for r in u["replay"]["rows"]), flush=True)
    json.dump(R, open(os.path.join(OUT, "exp17_results.json"), "w"), indent=1)

    if not a.skip_regen:                     # rebuild every patch from its source on the furthest base
        s = save_at[-1]; g = os.path.join(OUT, f"regen_s{step0 + s}"); py = sys.executable; t0 = time.time()
        for cmd in ([py, os.path.join(HERE, "exp13_catgan26_patches.py"), "--weights", os.path.join(OUT, f"catgan26_s{step0 + s}.npz"), "--out", os.path.join(g, "exp13")],
                    [py, os.path.join(HERE, "exp15_subspace_locks.py"), "--weights", os.path.join(g, "exp13", "catgan26_with_ops.npz"), "--src", os.path.join(g, "exp13", "patches"),
                     "--out", os.path.join(g, "exp15"), "--kmax", "16", "--no-fd"],
                    [py, os.path.join(HERE, "exp5_git_demo.py"), "--profile", "cat_subspace", "--weights", os.path.join(g, "exp13", "catgan26_with_ops.npz"),
                     "--src", os.path.join(g, "exp13", "patches"), "--src-lock", os.path.join(g, "exp15", "patches"), "--out", os.path.join(g, "exp5")]):
            print("$", os.path.basename(cmd[1]), " ".join(cmd[2:]), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            open(os.path.join(OUT, f"regen_{os.path.basename(cmd[1])[:5].rstrip('_')}.log"), "w", encoding="utf-8").write(r.stdout + r.stderr)
            if r.returncode != 0 and not os.path.basename(cmd[1]).startswith("exp5_"): raise SystemExit(f"{cmd[1]} failed:\n{r.stderr[-2000:]}")
        R["regenerated"] = {"step": step0 + s, "seconds": time.time() - t0, "tests": tests(os.path.join(g, "exp5", "weightrepo")),
                            "exp15": json.load(open(os.path.join(g, "exp15", "exp15_results.json")))["locks"]}
        R["regenerated"]["exp15"] = {k: {"rank": v["rank"], "var_ratio": v["curve"][v["rank"] - 1]["var_ratio"]} for k, v in R["regenerated"]["exp15"].items()}
        print("regenerated:", "ALL PASS" if R["regenerated"]["tests"]["all_pass"] else "FAIL", flush=True)
    json.dump(R, open(os.path.join(OUT, "exp17_results.json"), "w"), indent=1)

    keys = [(r["patch"], r["test"]) for r in R["original"]["rows"]]
    val = lambda T: {(r["patch"], r["test"]): r for r in T["rows"]}
    cols = [("original", R["original"])] + [(f"replay +{k[1:]}", u["replay"]) for k, u in R["upstream"].items()]
    if "regenerated" in R: cols.append((f"regenerated +{save_at[-1]}", R["regenerated"]["tests"]))
    L = ["# Experiment 17: rebasing the cat repo onto an upstream update\n",
         f"Downstream repo: `{REPO}` (base `{R['base']}`). Upstream: CatGAN-26 trained further from step {step0}.\n",
         "| upstream | relative weight change | output MAE | attribute correlation with the original (same latents) | stale sparse patches |", "|---|---|---|---|---|"]
    for k, u in R["upstream"].items():
        L.append(f"| {k} steps | {u['drift']['rel_weight_change']:.4f} | {u['drift']['output_mae']:.4f} | "
                 + ", ".join(f"{n} {v:.2f}" for n, v in u["drift"]["attr_corr"].items()) + " | "
                 + (", ".join(f"{r['patch']} ({r['context_drift']:.2f})" for r in u["rebase"] if r["stale"]) or "none") + " |")
    L += ["", "In-context tests (`wpgit test`), value and result:", "", "| patch | test | bound | " + " | ".join(c[0] for c in cols) + " |", "|---" * (len(cols) + 3) + "|"]
    for pk in keys:
        cells = []
        for _, T in cols:
            r = val(T).get(pk) or next((x for x in T["rows"] if x["test"] == pk[1] and pk[0].split("_")[-1] in x["patch"]), None)
            cells.append("-" if r is None else f"{r['value']:.3f} {'PASS' if r['pass'] else 'FAIL'}")
        L.append(f"| {pk[0]} | {pk[1]} | {val(R['original'])[pk]['bound']} | " + " | ".join(cells) + " |")
    if "regenerated" in R:
        L += ["", f"Regenerated on +{save_at[-1]} in {R['regenerated']['seconds'] / 60:.0f} min: subspace lock ranks "
              + ", ".join(f"{k} {v['rank']} (var ratio {v['var_ratio']:.2f})" for k, v in R["regenerated"]["exp15"].items()) + "."]
    open(os.path.join(OUT, "exp17_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
