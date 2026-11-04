"""
wpgit: build, test and merge-check a directory of .weightpatch files under git.

A weight repo is a git repository holding text only:
    base.json                      {"weights": <path to .h5>, "hash": <weights_hash>}
    patches/*.weightpatch.json     edits, each with optional "requires" and "tests"

Commands (run from anywhere, pass the repo path):
    python src/wpgit.py build <repo>   apply all patches -> <repo>/build/model.npz, print hash
    python src/wpgit.py test  <repo>   run every patch's tests *in context*: the full build
                                   vs the full build without that patch
    python src/wpgit.py check <repo>   pairwise merge check: does A-then-B equal B-then-A?
    python src/wpgit.py rebase <repo> <weights>
                                   move the repo onto new base weights (an upstream update):
                                   re-pin base.json and every patch, and mark patches whose
                                   context changed (sparse ops whose target entries moved)

base.json may name a test domain module ("domain": "cat_domain") providing ATTRS, attributes,
run, eye_mask and score; without it the 2018 Anime-DCGAN tests (exp3/exp4) are used.

Patches are applied in dependency order ("requires"), ties broken by file name.
Two patches that touch disjoint tensors commute exactly; otherwise the check
measures it. A textually clean git merge can still be a semantic conflict.
"""
import argparse, glob, importlib, itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2b_tinysr18_weightpatch import load_model, weights_hash
import exp3_anime_weightpatch as e3
import exp4_anime_eye_circuit as e4

TEST_SEED, TEST_N = 123, 150

class AnimeTests:
    """Default test domain: the Anime-DCGAN attributes and teal-eye mask from exp3/exp4."""
    ATTRS = e3.ATTRS; attributes = staticmethod(e3.attributes); run = staticmethod(e4.run)
    eye_mask = staticmethod(e4.eye_mask); score = staticmethod(e4.score)

DOMAIN = AnimeTests

# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
def touches(patch):
    t = set()
    for op in patch["ops"]:
        if op["type"] == "latent_shift": t.add(op["layer"] + "/bias")
        elif op["type"] == "latent_clamp": t |= {op["layer"] + "/kernel", op["layer"] + "/bias"}
        elif op["type"] == "sparse_add": t.add(op["tensor"])
        else: raise ValueError(op["type"])
    return t

def apply_patch(W, patch, ops):
    for op in patch["ops"]:
        one = {"ops": [op]}
        W = e4.apply_weightpatch(W, one) if op["type"] == "sparse_add" else e3.apply_weightpatch(W, one, ops)
    return W

def load_repo(repo):
    global DOMAIN
    base = json.load(open(os.path.join(repo, "base.json")))
    DOMAIN = importlib.import_module(base["domain"]) if base.get("domain") else AnimeTests
    W, ops, _ = load_model(os.path.join(repo, base["weights"]))
    if weights_hash(W) != base["hash"]:
        raise SystemExit(f"base weights hash {weights_hash(W)} != base.json {base['hash']}")
    P = {}
    for f in sorted(glob.glob(os.path.join(repo, "patches", "*.weightpatch.json"))):
        p = json.load(open(f, encoding="utf-8"))
        if p["base"] != base["hash"]:
            raise SystemExit(f"{os.path.basename(f)} targets base {p['base']}, repo base is {base['hash']}")
        P[p["id"]] = p
    return W, ops, P

def order(P, ids=None):
    ids = sorted(P) if ids is None else list(ids)
    out, seen = [], set()
    def visit(i, stack=()):
        if i in seen: return
        if i in stack: raise SystemExit(f"dependency cycle at {i}")
        req = P[i].get("requires")
        if req:
            if req not in P: raise SystemExit(f"{i} requires missing patch {req}")
            visit(req, stack + (i,))
        seen.add(i); out.append(i)
    for i in ids: visit(i)
    return out

def build(W, ops, P, ids):
    for i in ids: W = apply_patch(W, P[i], ops)
    return W

def closure(P, i):
    """Patch ids that would be removed with i (i and everything requiring it)."""
    out = {i}; grew = True
    while grew:
        grew = False
        for j, p in P.items():
            if p.get("requires") in out and j not in out: out.add(j); grew = True
    return out

# ---------------------------------------------------------------------------
# Tests (each compares the full build against the build without that patch)
# ---------------------------------------------------------------------------
def run_tests(W0, ops, P):
    z = np.random.default_rng(TEST_SEED).standard_normal((TEST_N, W0[ops[0]["name"] + "/kernel"].shape[0]))
    full_ids = order(P); Wfull = build(W0, ops, P, full_ids)
    x_base = DOMAIN.run(z, W0, ops); x_full = DOMAIN.run(z, Wfull, ops)
    A_base = DOMAIN.attributes(x_base); sd = A_base.std(0); names = list(DOMAIN.ATTRS); A_full = DOMAIN.attributes(x_full)
    rows, ok_all = [], True
    for pid in full_ids:
        tests = P[pid].get("tests", [])
        if not tests: continue
        drop = closure(P, pid)
        x_wo = DOMAIN.run(z, build(W0, ops, P, [i for i in full_ids if i not in drop]), ops)
        A_wo = DOMAIN.attributes(x_wo)
        eye = DOMAIN.score(x_full, x_wo, DOMAIN.eye_mask(x_wo)) if any(t["type"] in ("eye_hue", "rest_mae") for t in tests) else None
        for t in tests:
            k = t["type"]
            if k == "attr_delta":
                j = names.index(t["attr"]); v = float((A_full[:, j] - A_wo[:, j]).mean() / sd[j]); label = f"Δ{t['attr']} (SD)"
            elif k == "attr_var_ratio":          # "vs": "base" for absolute claims ("X is locked")
                ref = A_base if t.get("vs") == "base" else A_wo
                j = names.index(t["attr"]); v = float(A_full[:, j].var() / ref[:, j].var())
                label = f"var ratio {t['attr']} vs {t.get('vs', 'without patch')}"
            elif k == "eye_hue": v, label = eye["eye_hue"], "eye hue (deg)"
            elif k == "rest_mae": v, label = eye["rest_mae"], "rest-of-image MAE"
            else: raise ValueError(k)
            ok = (t.get("min", -np.inf) <= v <= t.get("max", np.inf)); ok_all &= ok
            bound = " ".join(f"{b}={t[b]}" for b in ("min", "max") if b in t)
            rows.append((pid, label, v, bound, ok))
    return rows, ok_all, weights_hash(Wfull)

# ---------------------------------------------------------------------------
# Pairwise merge check
# ---------------------------------------------------------------------------
def check(W0, ops, P, tol=1e-5):
    z = np.random.default_rng(TEST_SEED).standard_normal((32, W0[ops[0]["name"] + "/kernel"].shape[0]))
    res, conflicts = [], 0
    for a, b in itertools.combinations(sorted(P), 2):
        if P[a].get("requires") == b or P[b].get("requires") == a: continue   # ordered by dependency
        ta = set().union(*(touches(P[i]) for i in order(P, [a])))
        tb = set().union(*(touches(P[i]) for i in order(P, [b])))
        shared = sorted(ta & tb)
        ab = build(W0, ops, P, order(P, [a, b])); ba = build(W0, ops, P, order(P, [b, a]))
        dw = float(max(np.abs(ab[k] - ba[k]).max() for k in W0))
        if not shared: verdict = "commutes (disjoint tensors)"
        elif dw < tol: verdict = "commutes (numerically)"
        else:
            mae = float(np.abs(DOMAIN.run(z, ab, ops) - DOMAIN.run(z, ba, ops)).mean())
            verdict = f"CONFLICT: order changes output, MAE {mae:.4f}"; conflicts += 1
        res.append((a, b, ",".join(shared) or "-", dw, verdict))
    return res, conflicts

# ---------------------------------------------------------------------------
# Rebase onto new base weights
# ---------------------------------------------------------------------------
def rebase(repo, weights, tol=1e-6):
    """Re-pin the repo to new base weights. Latent ops are defined on the latent (the model's input
    interface), so they apply to any base with the same interface. A sparse_add stores a difference
    from the old values of its entries; like a text diff whose context lines changed, it is marked
    stale when those entries moved by more than tol (relative), so it no longer means what it did
    (prune2 no longer zeroes its channels). Tests decide whether any patch still meets its spec."""
    bpath = os.path.join(repo, "base.json"); base = json.load(open(bpath))
    W_old, ops_old, _ = load_model(os.path.join(repo, base["weights"])); W_new, ops_new, _ = load_model(weights)
    if json.dumps(ops_old) != json.dumps(ops_new) or any(W_old[k].shape != W_new.get(k, np.empty(0)).shape for k in W_old):
        raise SystemExit("new base has a different architecture: patches cannot be rebased, regenerate them")
    h_old, h_new = base["hash"], weights_hash(W_new); rows = []
    for f in sorted(glob.glob(os.path.join(repo, "patches", "*.weightpatch.json"))):
        p = json.load(open(f, encoding="utf-8")); drift = {}
        for op in p["ops"]:
            if op["type"] != "sparse_add": continue
            idx = np.array(op["idx"], np.int64); a = W_old[op["tensor"]].reshape(-1)[idx]; b = W_new[op["tensor"]].reshape(-1)[idx]
            drift[op["tensor"]] = max(drift.get(op["tensor"], 0.0), float(np.abs(b - a).max() / (np.abs(a).max() + 1e-12)))
        stale = {k: round(v, 6) for k, v in drift.items() if v > tol}
        p["base"] = h_new; p.setdefault("rebased_from", []).append(h_old); p.pop("stale_context", None)
        if stale: p["stale_context"] = stale
        with open(f, "w", encoding="utf-8") as fh: json.dump(p, fh, indent=1)
        rows.append((p["id"], len(p["ops"]), max(drift.values()) if drift else None, bool(stale)))
    base["weights"] = os.path.relpath(weights, repo).replace("\\", "/"); base["hash"] = h_new
    json.dump(base, open(bpath, "w"), indent=1)
    return h_old, h_new, rows

# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "test", "check", "rebase"]); ap.add_argument("repo")
    ap.add_argument("weights", nargs="?", help="rebase: the new base weights")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")         # Windows consoles default to cp1252
    if a.cmd == "rebase":
        if not a.weights: ap.error("rebase needs the new base weights")
        h_old, h_new, rows = rebase(a.repo, a.weights)
        print(f"rebased {len(rows)} patches from {h_old} onto {h_new}")
        print("| patch | ops | context drift (sparse ops) | status |\n|---|---|---|---|")
        for pid, n, d, stale in rows:
            print(f"| {pid} | {n} | {'-' if d is None else f'{d:.2e}'} | {'STALE CONTEXT: regenerate' if stale else 'clean'} |")
        print("now run `wpgit test` to see which patches still meet their tests")
        return 1 if any(r[3] for r in rows) else 0
    W0, ops, P = load_repo(a.repo)
    if a.cmd == "build":
        ids = order(P); W = build(W0, ops, P, ids)
        os.makedirs(os.path.join(a.repo, "build"), exist_ok=True)
        np.savez_compressed(os.path.join(a.repo, "build", "model.npz"), **W)
        print(f"applied {len(ids)} patches: {' -> '.join(ids) or '(none)'}")
        print(f"model hash {weights_hash(W)}")
        return 0
    if a.cmd == "test":
        rows, ok, h = run_tests(W0, ops, P)
        print(f"model hash {h}")
        print("| patch | test | value | bound | result |\n|---|---|---|---|---|")
        for pid, label, v, bound, r in rows: print(f"| {pid} | {label} | {v:.3f} | {bound} | {'PASS' if r else 'FAIL'} |")
        print("ALL TESTS PASS" if ok else "TESTS FAILED")
        return 0 if ok else 1
    if a.cmd == "check":
        res, n = check(W0, ops, P)
        print("| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |\n|---|---|---|---|---|")
        for r in res: print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1e} | {r[4]} |")
        print(f"{n} semantic conflict(s)")
        return 1 if n else 0

if __name__ == "__main__":
    sys.exit(main())
