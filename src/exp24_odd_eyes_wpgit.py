"""
Experiment 24: a topology patch under version control.

The odd-eye circuit of exp20 grows channels, wires a coordinate carrier through ten layers and installs
an AND gate. Here it is written as a .weightpatch (grow ops, then sparse_add ops that wire the new
channels) and put into the cat repository of exp5/exp15 in place of the fixed-architecture blue-eye
circuit (prune2 + eye_gate), next to the rank-2 latent locks on fur lightness and background.
wpgit then builds it, tests it in the context of the locks and merge-checks it against them.

The patch's own tests are its specification from §4.2: at least 30% of left irises end in the blue
band (200-250 deg), no right iris does, the right half of every image is unchanged against the build
without the patch, and the rest of the image barely moves.

Last, a synthetic check of the rebase rule for grown channels: an "upstream update" that changes only
the output layer leaves the patch's context clean; one that changes a layer the gate reads from
marks it stale.

  <venv python> src/exp24_odd_eyes_wpgit.py
"""
import json, os, shutil, stat, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wpgit, exp20_odd_eyes as e20
from exp2b_tinysr18_weightpatch import weights_hash

OUT = "results/exp24_out"; REPO = os.path.join(OUT, "weightrepo"); BUNDLE = "results/exp5_cat_subspace_out/weightrepo.bundle"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="wpgit demo", GIT_AUTHOR_EMAIL="wpgit-demo@example.invalid",
               GIT_COMMITTER_NAME="wpgit demo", GIT_COMMITTER_EMAIL="wpgit-demo@example.invalid")
LOG = []; R = {}
TESTS = [{"type": "eye_band_share", "side": "left", "lo": 200, "hi": 250, "min": 0.3},
         {"type": "eye_band_share", "side": "right", "lo": 200, "hi": 250, "max": 0.0},
         {"type": "half_max_change", "half": "right", "max": 1e-5},
         {"type": "rest_mae", "max": 0.015}]

def sh(*cmd, ok=(0,), show=True, cwd=None):
    p = subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=GIT_ENV)
    out = (p.stdout + p.stderr).strip()
    if p.returncode not in ok: raise SystemExit(f"{' '.join(cmd)} failed ({p.returncode}):\n{out}")
    if show:
        shown = " ".join(cmd).replace(sys.executable, "python").replace(os.path.join(HERE, "wpgit.py"), "wpgit.py")
        LOG.append(f"```\n$ {shown}\n{out}\n```"); print(f"$ {shown}\n{out}", flush=True)
    return out, p.returncode
def git(*a, **k): return sh("git", *a, **k)
def wp(cmd, *extra, ok=(0, 1)): return sh(sys.executable, os.path.join(HERE, "wpgit.py"), cmd, ".", *extra, ok=ok)
def note(s): LOG.append(s); print(s, flush=True)

def make_patch(W_repo_base, W_fit, pid, desc):
    """Fit the left-eye circuit on W_fit (exp20, grown room) and express it as ops on the repo base:
    the grow ops exp20 used, then the sparse differences from the grown, unwired network."""
    ctx = e20.run_room("grow", write=False, W0=W_fit); Wm = ctx["nets"]["left"]; L = ctx["L"]
    names = [l[1] for l in L if l[0] in ("dense", "conv")]
    grows = [{"type": "grow", "producer": p, "consumer": names[names.index(p) + 1], "k": 1} for p in ctx["R"]["chain"]]
    grows.append({"type": "grow", "producer": "b5c2", "consumer": "out", "k": 2})
    Wg = W_fit
    for g in grows: Wg = wpgit.grow(Wg, g["producer"], g["consumer"], g["k"])
    sparse = []
    for k in sorted(Wm):
        assert Wm[k].shape == Wg[k].shape, (k, Wm[k].shape, Wg[k].shape)
        d = (Wm[k].astype(np.float64) - Wg[k].astype(np.float64)).reshape(-1); idx = np.nonzero(d)[0]
        if len(idx): sparse.append({"type": "sparse_add", "tensor": k, "shape": list(Wm[k].shape), "idx": idx.tolist(), "val": d[idx].astype(np.float32).tolist()})
    patch = {"format": "weightpatch/v0", "id": pid, "base": weights_hash(W_repo_base), "description": desc, "ops": grows + sparse, "tests": TESTS}
    built = wpgit.apply_patch(W_fit, patch, None)
    err = max(float(np.abs(built[k] - Wm[k]).max()) for k in Wm)
    stats = {"grow_ops": len(grows), "sparse_ops": len(sparse), "entries": int(sum(len(o["idx"]) for o in sparse)),
             "params_added": int(sum(v.size for v in Wm.values()) - sum(v.size for v in W_fit.values())),
             "rebuild_max_err_vs_exp20": err, "exp20_left_share": ctx["R"]["gates"]["left"]["share_left_in_band"],
             "exp20_right_share": ctx["R"]["gates"]["left"]["share_right_in_band"], "quantile": ctx["R"]["gates"]["left"]["quantile"]}
    return patch, stats

def write_patch(name, patch):
    with open(os.path.join(REPO, "patches", name), "w", encoding="utf-8") as f: json.dump(patch, f, indent=1)
    git("add", "patches/" + name, show=False)

def main():
    sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    if os.path.exists(REPO): shutil.rmtree(REPO, onerror=lambda f, p, e: (os.chmod(p, stat.S_IWRITE), f(p)))
    sh("git", "clone", "-q", os.path.abspath(BUNDLE), os.path.abspath(REPO), cwd=OUT, show=False)
    git("log", "--oneline", "-1")
    W0, ops, P = wpgit.load_repo(REPO)
    note(f"Base {weights_hash(W0)}; patches at HEAD: {', '.join(sorted(P))}.")

    # 1. the blue-eye circuit of 2018 out, the grown-channel left-eye circuit in ----------------------
    git("checkout", "-q", "-b", "odd-eyes")
    git("rm", "-q", "patches/30_prune2.weightpatch.json", "patches/31_eye_gate.weightpatch.json", show=False)
    patch, st = make_patch(W0, W0, "left_eye_blue", "left iris blue: coordinate wire (1 grown channel in each of 10 layers) AND an iris gate (2 grown b5c2 channels); fitted on the base model")
    R["patch_fitted_on_base"] = st; note(f"Patch: {json.dumps(st)}")
    write_patch("32_left_eye_blue.weightpatch.json", patch)
    git("commit", "-q", "-m", "Replace the blue-eye circuit with a grown-channel left-eye circuit" + TRAILER, show=False); git("log", "--oneline", "-1")
    out, rc = wp("test"); R["test_fitted_on_base"] = {"pass": rc == 0, "table": out}
    out, rc = wp("check"); R["check_fitted_on_base"] = {"conflicts": rc != 0, "table": out}

    # 2. if the in-context test fails, regenerate the gate in context (on the build with the locks) ---
    if not R["test_fitted_on_base"]["pass"]:
        W0, ops, P = wpgit.load_repo(REPO)
        ctx_ids = [i for i in wpgit.order(P) if i != "left_eye_blue"]
        W_ctx = wpgit.build(W0, ops, P, ctx_ids)
        note(f"Regenerating the gate on the build with {', '.join(ctx_ids)}.")
        patch, st = make_patch(W0, W_ctx, "left_eye_blue", "left iris blue: coordinate wire (1 grown channel in each of 10 layers) AND an iris gate (2 grown b5c2 channels); gate fitted in context, on the build with the latent locks")
        R["patch_fitted_in_context"] = st; note(f"Patch: {json.dumps(st)}")
        write_patch("32_left_eye_blue.weightpatch.json", patch)
        git("commit", "-q", "-m", "Regenerate the left-eye gate in context (fitted on the locked build)" + TRAILER, show=False); git("log", "--oneline", "-1")
        out, rc = wp("test"); R["test_fitted_in_context"] = {"pass": rc == 0, "table": out}
        out, rc = wp("check"); R["check_fitted_in_context"] = {"conflicts": rc != 0, "table": out}

    # 3. synthetic upstream updates: does rebase flag the gate when the features it reads move? ------
    W0, ops, P = wpgit.load_repo(REPO); R["rebase_synthetic"] = {}
    for name, layer in (("output layer only", "out"), ("a layer the gate reads (b3c1)", "b3c1")):
        Wn = {k: v.copy() for k, v in W0.items()}; Wn[layer + "/kernel"] = Wn[layer + "/kernel"] * np.float32(1.001)
        path = os.path.abspath(os.path.join(OUT, f"synthetic_{layer}.npz")); np.savez(path, **Wn, __ops__=json.dumps(ops))
        tmp = os.path.join(OUT, f"rebase_{layer}"); shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(REPO, tmp, ignore=shutil.ignore_patterns(".git", "build"))
        base = json.load(open(os.path.join(tmp, "base.json"))); base["weights"] = os.path.relpath(os.path.join(REPO, base["weights"]), tmp).replace("\\", "/")
        json.dump(base, open(os.path.join(tmp, "base.json"), "w"), indent=1)
        _, _, rows = wpgit.rebase(tmp, path)
        R["rebase_synthetic"][name] = {pid: {"drift": d, "stale": s} for pid, n, d, s in rows}
        note(f"Synthetic update, {name}: " + ", ".join(f"{pid} {'STALE' if s else 'clean'}" for pid, n, d, s in rows))
        shutil.rmtree(tmp, ignore_errors=True); os.remove(path)

    # 4. ship the repository as a bundle, like exp5 ------------------------------------------------
    git("bundle", "create", "../weightrepo.bundle", "--all", show=False)
    R["seconds_total"] = time.time() - t0
    json.dump(R, open(os.path.join(OUT, "exp24_results.json"), "w"), indent=1)
    open(os.path.join(OUT, "transcript.md"), "w", encoding="utf-8").write(
        "# Experiment 24: the odd-eye circuit under wpgit\n\n" + "\n\n".join(LOG) + "\n")
    print(f"done in {R['seconds_total']:.0f}s")

if __name__ == "__main__":
    main()
