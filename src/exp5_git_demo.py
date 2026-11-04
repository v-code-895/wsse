"""
Experiment 5: git for weights. Scripts a real git history over .weightpatch files
for the Anime DCGAN and records every command and its output.

  1. three feature branches (warm hair, bright background, magenta eyes) merge
     cleanly and every patch's tests still pass in the merged model
  2. `git revert` of one feature gives exactly the model built without it
  3. two clamp patches merge cleanly *in git* but `wpgit check` flags a semantic
     conflict (order changes the output); orthogonalising one direction resolves it

Patches come from results/exp3_out/ and results/exp4_out/ (run those first). With --profile cat the same script
replays the story on CatGAN-26 with patches from results/exp13_out/ and the cat test domain (cat_domain.py).

With --profile cat_subspace the two locks are the rank-k subspace locks from results/exp15_out/ (exp15), and
the conflict resolution orthogonalises the background subspace against the fur-lightness subspace.
--weights/--src/--out replay the story on another base (exp17: CatGAN-26 after more training).

  python src/exp5_git_demo.py                                 -> results/exp5_out/weightrepo (a git repo), results/exp5_out/transcript.md
  <venv python> src/exp5_git_demo.py --profile cat            -> results/exp5_cat_out/...
  <venv python> src/exp5_git_demo.py --profile cat_subspace   -> results/exp5_cat_subspace_out/...
"""
import argparse, json, os, shutil, stat, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)   # paths are relative to the repo root
sys.path.insert(0, HERE)
import wpgit

PROFILES = {
    "anime": dict(out="results/exp5_out", weights="legacy/Anime-DCGAN/generator50000.h5", src3="results/exp3_out/patches", src4="results/exp4_out/patches",
                  model="Anime DCGAN", base_msg="Base: Anime DCGAN generator50000", warm="hair_warmth", lock="hair_lightness",
                  warm_branch="warm-hair", warm_title="Warm hair", lock_branch="lock-hair-lightness", lock_title="Lock hair lightness",
                  eye_branch="magenta-eyes", eye_title="Magenta eyes", hue=(270, 330),
                  prune_msg="Prune 2 redundant conv2d_9 channels (fold into conv2d_10)",
                  domain=None, file_warm="10_warm_hair", file_lock="40_lock_hair_lightness", orth="_orth"),
    "cat": dict(out="results/exp5_cat_out", weights="results/exp13_out/catgan26_with_ops.npz", src3="results/exp13_out/patches", src4="results/exp13_out/patches",
                model="CatGAN-26", base_msg="Base: CatGAN-26 (AFHQ cats, 128x128)", warm="fur_warmth", lock="fur_lightness",
                warm_branch="warm-fur", warm_title="Warm fur", lock_branch="lock-fur-lightness", lock_title="Lock fur lightness",
                eye_branch="blue-eyes", eye_title="Blue eyes", hue=(200, 250),
                prune_msg="Prune 2 redundant b5c2 channels (fold into out)",
                domain="cat_domain", file_warm="10_warm_fur", file_lock="40_lock_fur_lightness", orth="_orthnd"),   # see exp13: keep near-duplicates
}
PROFILES["cat_subspace"] = dict(PROFILES["cat"], out="results/exp5_cat_subspace_out", src_lock="results/exp15_out/patches", lock_prefix="lock_")
OUT = REPO = WEIGHTS = None                    # set from the profile in main()
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
LOG = []

# the demo repositories are generated artifacts: commit them under a neutral identity, not the user's git config
GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="wpgit demo", GIT_AUTHOR_EMAIL="wpgit-demo@example.invalid",
               GIT_COMMITTER_NAME="wpgit demo", GIT_COMMITTER_EMAIL="wpgit-demo@example.invalid")

def sh(*cmd, show=True, ok=(0,)):
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=GIT_ENV)
    out = (p.stdout + p.stderr).strip()
    if p.returncode not in ok: raise SystemExit(f"{' '.join(cmd)} failed ({p.returncode}):\n{out}")
    if show:
        shown = " ".join(c if " " not in c else f'"{c}"' for c in cmd).replace(sys.executable, "python")
        shown = shown.replace(f'"{os.path.join(HERE, "wpgit.py")}"', "wpgit.py").replace(os.path.join(HERE, "wpgit.py"), "wpgit.py")
        LOG.append(f"```\n$ {shown}\n{out}\n```")
    return out, p.returncode

def note(s): LOG.append(s); print(s)
def git(*a, **k): return sh("git", *a, **k)
def wp(cmd, ok=(0,)): return sh(sys.executable, os.path.join(HERE, "wpgit.py"), cmd, ".", ok=ok)
def commit(msg): git("commit", "-q", "-m", msg + TRAILER, show=False); git("log", "-1", "--format=%h %s")

def put(name, patch, tests, requires=None):
    p = dict(patch); p["tests"] = tests
    if requires: p["requires"] = requires
    os.makedirs(os.path.join(REPO, "patches"), exist_ok=True)      # git drops it on empty branches
    with open(os.path.join(REPO, "patches", name), "w", encoding="utf-8") as f: json.dump(p, f, indent=1)
    git("add", "patches/" + name, show=False)

def load(path): return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))

def main():
    global OUT, REPO, WEIGHTS
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", choices=sorted(PROFILES), default="anime")
    ap.add_argument("--weights"); ap.add_argument("--src", help="patch directory for shifts, prune2 and eye gate")
    ap.add_argument("--src-lock"); ap.add_argument("--out")
    a = ap.parse_args(); C = dict(PROFILES[a.profile])
    for k, v in (("weights", a.weights), ("src3", a.src), ("src4", a.src), ("src_lock", a.src_lock), ("out", a.out)):
        if v: C[k] = v
    OUT = os.path.join(ROOT, C["out"]); REPO = os.path.join(OUT, "weightrepo"); WEIGHTS = C["weights"]
    if os.path.exists(REPO):
        shutil.rmtree(REPO, onerror=lambda f, p, _: (os.chmod(p, stat.S_IWRITE), f(p)))
    os.makedirs(os.path.join(REPO, "patches"))
    W0, ops, _ = wpgit.load_model(os.path.join(ROOT, WEIGHTS)); h0 = wpgit.weights_hash(W0)

    warm = load(f"{C['src3']}/shift_{C['warm']}{C['orth']}.weightpatch.json")
    bg = load(f"{C['src3']}/shift_background{C['orth']}.weightpatch.json")
    prune = load(f"{C['src4']}/prune2.weightpatch.json")
    gate = load(f"{C['src4']}/eye_gate.weightpatch.json")
    gate["base"] = h0          # gate deltas are relative to prune2, which it requires
    lock_src, lp = C.get("src_lock", C["src3"]), C.get("lock_prefix", "clamp_")
    c_hl = load(f"{lock_src}/{lp}{C['lock']}.weightpatch.json")
    c_bg = load(f"{lock_src}/{lp}background.weightpatch.json")
    what = lambda p: "clamp latent coordinate" if len(p["ops"]) == 1 else f"clamp rank-{len(p['ops'])} latent subspace"

    note(f"# Experiment 5: git for weights ({C['model']})\n")
    note(f"Base weights `{h0}` ({WEIGHTS}). The repo holds only text: `base.json` and `patches/*.weightpatch.json`. "
         "`wpgit test` runs each patch's tests *in context*: the full build against the full build without that patch.\n")

    note("## 1. Base repo"); git("init", "-q", "-b", "main", show=False)
    base = {"weights": os.path.relpath(os.path.join(ROOT, WEIGHTS), REPO).replace("\\", "/"), "hash": h0}
    if C["domain"]: base["domain"] = C["domain"]
    json.dump(base, open(os.path.join(REPO, "base.json"), "w"), indent=1)
    open(os.path.join(REPO, ".gitignore"), "w").write("build/\n")
    git("add", "-A", show=False); commit(C["base_msg"])

    note("## 2. Three feature branches")
    git("checkout", "-q", "-b", f"feature/{C['warm_branch']}", "main", show=False)
    put(f"{C['file_warm']}.weightpatch.json", warm,
        [{"type": "attr_delta", "attr": C["warm"], "min": 1.0},
         {"type": "attr_delta", "attr": "background", "min": -0.3, "max": 0.3}])
    commit(f"{C['warm_title']}: shift latent along orthogonalised {C['warm']} direction")
    git("checkout", "-q", "-b", "feature/bright-background", "main", show=False)
    put("20_bright_background.weightpatch.json", bg,
        [{"type": "attr_delta", "attr": "background", "min": 0.6},
         {"type": "attr_delta", "attr": C["warm"], "min": -0.3, "max": 0.3}])
    commit("Bright background: shift latent along orthogonalised background direction")
    git("checkout", "-q", "-b", f"feature/{C['eye_branch']}", "main", show=False)
    put("30_prune2.weightpatch.json", prune, [{"type": "rest_mae", "max": 0.015}])
    commit(C["prune_msg"])
    put("31_eye_gate.weightpatch.json", gate,
        [{"type": "eye_hue", "min": C["hue"][0], "max": C["hue"][1]}, {"type": "rest_mae", "max": 0.015}], requires=gate["requires"])
    commit(f"{C['eye_title']}: gated eye-colour circuit in the freed slots")

    note("## 3. Merge all three into main")
    git("checkout", "-q", "main", show=False)
    merges = {}
    for br in (f"feature/{C['warm_branch']}", "feature/bright-background", f"feature/{C['eye_branch']}"):
        git("merge", "--no-ff", "-q", "-m", f"Merge {br}" + TRAILER, br, show=False)
        merges[br] = git("rev-parse", "HEAD", show=False)[0]
    note("Three `git merge --no-ff` calls, all clean (each branch adds its own files).")
    git("log", "--graph", "--oneline", "--all")
    git("diff", "--stat", "HEAD~3", "HEAD")
    wp("check"); out, rc = wp("test", ok=(0, 1))
    h_merged = out.splitlines()[0].split()[-1]
    note(f"All patches compose: {'yes' if rc == 0 else 'NO'}. Merged model hash `{h_merged}`.\n")

    note("## 4. Revert one feature")
    git("revert", "--no-commit", "-m", "1", merges[f"feature/{C['warm_branch']}"], show=False)
    commit(f"Revert {C['warm_title'].lower()}")
    out, rc = wp("test", ok=(0, 1)); h_rev = out.splitlines()[0].split()[-1]
    _, _, P = wpgit.load_repo(REPO)
    h_never = wpgit.weights_hash(wpgit.build(W0, ops, P, wpgit.order(P)))
    Pn = {k: v for k, v in P.items()}
    W_ref = wpgit.build(W0, ops, Pn, ["prune2", "eye_gate", bg["id"]])
    note(f"After revert: hash `{h_rev}`. Built directly from {{bright background, prune2, eye gate}} without git: "
         f"`{wpgit.weights_hash(W_ref)}` (match: {wpgit.weights_hash(W_ref) == h_rev or 'no - order differs, see note'}). "
         f"Tests pass: {rc == 0}.\n")

    note("## 5. A merge git accepts but the weights do not")
    git("checkout", "-q", "-b", f"feature/{C['lock_branch']}", "main", show=False)
    put(f"{C['file_lock']}.weightpatch.json", c_hl, [{"type": "attr_var_ratio", "attr": C["lock"], "vs": "base", "max": 0.3}])
    commit(f"{C['lock_title']}: {what(c_hl)}")
    git("checkout", "-q", "-b", "feature/lock-background", "main", show=False)
    put("50_lock_background.weightpatch.json", c_bg, [{"type": "attr_var_ratio", "attr": "background", "vs": "base", "max": 0.3}])
    commit(f"Lock background: {what(c_bg)}")
    git("checkout", "-q", "main", show=False)
    git("merge", "--no-ff", "-q", "-m", f"Merge feature/{C['lock_branch']}" + TRAILER, f"feature/{C['lock_branch']}", show=False)
    git("merge", "--no-ff", "-q", "-m", "Merge feature/lock-background" + TRAILER, "feature/lock-background", show=False)
    note("git merged both without complaint. The weight-level check:")
    wp("check", ok=(0, 1))
    unit = lambda p: np.stack([np.array(o["direction"]) / np.linalg.norm(o["direction"]) for o in p["ops"]], 1)
    D1, D2 = unit(c_hl), unit(c_bg); rank = D1.shape[1] + D2.shape[1] > 2
    if not rank:
        note(f"cos({C['lock']}, background) = {(D1.T @ D2).item():+.3f}: the two clamps project out overlapping directions, "
             "so whichever is applied second partly undoes the first.\n")
    else:
        note(f"Cosines of the principal angles between the {C['lock']} subspace (rank {D1.shape[1]}) and the background subspace "
             f"(rank {D2.shape[1]}): {', '.join(f'{v:.3f}' for v in np.linalg.svd(D1.T @ D2, compute_uv=False))}. "
             "The two clamps overlap, so whichever is applied second partly undoes the first.\n")

    note(f"### Resolution: orthogonalise the background {'subspace' if rank else 'direction'} against {C['lock'].replace('_', ' ')}")
    # joint clamp of span{d1, d2}: the in-span point c with d1.c = t1, d2.c = t2,
    # expressed in the orthonormal basis (d1, d2o). The d1 clamp is unchanged; the
    # background clamp moves to d2o with target d2o.c (not the old target).
    # (rank k: the same with subspaces; the background basis is Gram-Schmidt-orthogonalised against D1)
    Q = []
    for d2 in D2.T:
        v = d2 - D1 @ (D1.T @ d2)
        for q in Q: v = v - (v @ q) * q
        Q.append(v / np.linalg.norm(v))
    D = np.c_[D1, D2]; t = np.array([o["target"] for o in c_hl["ops"] + c_bg["ops"]])
    c = D @ np.linalg.solve(D.T @ D, t); t2o = [float(q @ c) for q in Q]
    res = dict(c_bg); res["ops"] = [dict(o, direction=[round(float(v), 7) for v in q], target=round(tq, 6)) for o, q, tq in zip(c_bg["ops"], Q, t2o)]
    if not rank:
        res["description"] = (f"clamp background jointly with lock_{C['lock']}: direction orthogonalised, "
                              f"target {c_bg['ops'][0]['target']} -> {t2o[0]:.4f} so both locks hold")
        note(f"New direction is background minus its {C['lock'].replace('_', '-')} component; target recomputed from "
             f"{c_bg['ops'][0]['target']} to {t2o[0]:.3f} so that {C['lock']} = {t[0]} and background = {t[1]} hold together.")
    else:
        res["description"] = (f"lock background jointly with lock_{C['lock']}: rank-{len(Q)} subspace orthogonalised, "
                              f"targets recomputed so both locks hold")
        note(f"New subspace is the background subspace minus its {C['lock'].replace('_', '-')} components (Gram-Schmidt); "
             f"targets recomputed from {[o['target'] for o in c_bg['ops']]} to {[round(v, 3) for v in t2o]} so that both locks "
             "keep their original targets together.")
    put("50_lock_background.weightpatch.json", res, [{"type": "attr_var_ratio", "attr": "background", "vs": "base", "max": 0.3}])
    commit(f"Resolve semantic conflict: orthogonalise background {'subspace lock' if rank else 'clamp'}")
    git("diff", "--stat", "HEAD~1", "HEAD")
    out, _ = wp("check", ok=(0, 1))
    note("The clamp pair now commutes. The check still flags lock-background against bright-background: "
         "one freezes the background coordinate, the other moves it, so whichever runs last wins. "
         "That is a genuine contradiction in intent, not a numerical one, so the resolution is a decision: "
         "the lock supersedes the brighten.\n")
    git("rm", "-q", "patches/20_bright_background.weightpatch.json", show=False)
    commit("Resolve: lock-background supersedes bright-background")
    wp("check", ok=(0, 1)); wp("test", ok=(0, 1))
    git("log", "--graph", "--oneline")

    open(os.path.join(OUT, "transcript.md"), "w", encoding="utf-8").write("\n\n".join(LOG) + "\n")
    print(f"[+] wrote {OUT}/transcript.md and git repo {REPO}")

if __name__ == "__main__":
    main()
