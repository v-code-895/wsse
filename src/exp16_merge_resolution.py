"""
Experiment 16: resolving the fine-tune merge conflict of exp14.

exp14 merged two TinySR-26 x2 fine-tunes (d_bicubic, d_noisy) by adding their weight deltas. Each
branch met its own claim, but both merges broke the nearest-LR regression test. The spec for an
acceptable merge here: both claims still hold (gain > 0 on bicubic LR and on noisy LR) and no test
regresses by more than 0.1 dB. Three ways to meet it, all chosen on a fitting split (the first 40
AFHQ val images per class) and reported on exp14's test split (the last 60 per class):

  (a) weighted merge   base + a d_bicubic + b d_noisy, (a, b) on a 5 x 5 grid (task arithmetic)
  (b) blame and cherry-pick   which layers of a delta cause the regression? Apply one layer group of
      the delta at a time, then drop groups greedily (worst first) until the regression is within
      0.1 dB, like reverting single hunks of a commit
  (c) resolution commit   fine-tune the merged model for 1,000 steps on all three conditions and store
      that as one more delta; vs training jointly from the base for 3,000 steps (no branches)

  <venv python> src/exp16_merge_resolution.py
"""
import copy, itertools, json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp10_tinysr26 as e10, exp14_tinysr26_branches as e14
from exp2b_tinysr18_weightpatch import weights_hash

OUT = "results/exp16_out"; DEV = "cuda"; TOL = 0.1; GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TESTS = (("bicubic LR", "bicubic", 0.0), ("noisy bicubic LR", "bicubic", e14.NOISE), ("nearest LR", "nearest", 0.0))
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def fit_images(per_class=40):
    out = []
    for c in e14.CLASSES:
        d = os.path.join(e10.ROOT, "val", c); out += [Image.open(os.path.join(d, f)).convert("RGB") for f in sorted(os.listdir(d))[:per_class]]
    return out

def scores(W, imgs, only=None):
    return {name: e14.psnr_on(W, imgs, kind, noise) for name, kind, noise in TESTS if only is None or name in only}

def deltas(s, base): return {k: s[k] - base[k] for k in s}

def meets_spec(d):
    return d["bicubic LR"] > 0 and d["noisy bicubic LR"] > 0 and min(d.values()) >= -TOL

def group(k): return k.split("/")[0].split("_")[0]          # head, body0..body5, tail, up

def finetune_mix(W0, steps, lr=1e-4, seed=3):
    """Each batch: a third clean bicubic LR, a third bicubic LR with noise up to 8/255, a third clean nearest LR."""
    torch.manual_seed(seed); m = e14.from_dict(W0); ema = copy.deepcopy(m).eval()
    opt = torch.optim.Adam(m.parameters(), lr, betas=(0.9, 0.99))
    dl = torch.utils.data.DataLoader(e10.Patches(e10.files("train", e14.CLASSES), 96), batch_size=8, shuffle=True,
                                     num_workers=4, drop_last=True, persistent_workers=True)
    step = 0
    while step < steps:
        for b in dl:
            hr = b.view(-1, 3, 96, 96).to(DEV).float() / 255; h1, h2, h3 = hr.chunk(3)
            noisy = e10.degrade(h2, 2, "bicubic"); noisy = ((noisy + torch.rand(len(noisy), 1, 1, 1, device=DEV) * e14.NOISE * torch.randn_like(noisy)).clamp(0, 1) * 255).round() / 255
            lr_ = torch.cat([e10.degrade(h1, 2, "bicubic"), noisy, e10.degrade(h3, 2, "nearest")])
            loss = F.l1_loss(m(lr_), torch.cat([h1, h2, h3])); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), m.parameters()): pe.lerp_(p, 0.01)
            if step >= steps: break
    return e10.export(ema)

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True)
    W0 = {k: v.astype(np.float32) for k, v in np.load("results/exp10_out/tinysr26_x2.npz").items()}
    db, dn = (dict(np.load(f"results/exp14_out/delta_{n}.npz")) for n in ("bicubic", "noisy"))
    fit, test = fit_images(), e14.test_images(); t0 = time.time()
    add = lambda *terms: {k: W0[k] + sum(c * d[k] for c, d in terms) for k in W0}
    R = {"base": weights_hash(W0), "tol_db": TOL, "fit_images": len(fit), "test_images": len(test)}
    s0 = scores(W0, fit); R["fit_base"] = s0

    # (a) weighted merges -------------------------------------------------------------------------
    R["grid"] = []
    for a_, b_ in itertools.product(GRID, GRID):
        d = deltas(scores(add((a_, db), (b_, dn)), fit), s0); R["grid"].append({"a": a_, "b": b_, "fit_delta": d, "meets_spec": meets_spec(d)})
    ok = [g for g in R["grid"] if g["meets_spec"]]
    R["grid_choice"] = max(ok, key=lambda g: g["fit_delta"]["bicubic LR"] + g["fit_delta"]["noisy bicubic LR"]) if ok else None
    print(f"[a] grid done ({time.time() - t0:.0f}s): {len(ok)} of {len(R['grid'])} weightings meet the spec on the fitting split", flush=True)

    # (b) blame and cherry-pick ---------------------------------------------------------------------
    present = {group(k) for k in W0}; groups = [g for g in ["head"] + [f"body{i}" for i in range(len(present))] + ["tail", "up"] if g in present]
    R["blame"], R["cherry_pick"] = {}, {}
    for name, d in (("bicubic", db), ("noisy", dn)):
        only = {g: deltas(scores({k: W0[k] + (d[k] if group(k) == g else 0) for k in W0}, fit, ["nearest LR"]), s0)["nearest LR"] for g in groups}
        R["blame"][name] = only
        keep = list(groups); dropped = []
        cur = deltas(scores({k: W0[k] + d[k] for k in W0}, fit), s0)
        while cur["nearest LR"] < -TOL and keep:
            worst = min(keep, key=lambda g: only[g]); keep.remove(worst); dropped.append(worst)
            cur = deltas(scores({k: W0[k] + (d[k] if group(k) in keep else 0) for k in W0}, fit), s0)
        R["cherry_pick"][name] = {"dropped": dropped, "fit_delta": cur}
        print(f"[b] {name}: nearest-LR blame {', '.join(f'{g} {v:+.2f}' for g, v in only.items())}; dropped {dropped} -> {cur}", flush=True)
    pick = lambda name, d: {k: (d[k] if group(k) not in R["cherry_pick"][name]["dropped"] else 0 * d[k]) for k in d}
    db_p, dn_p = pick("bicubic", db), pick("noisy", dn)

    # (c) resolution commit vs joint training ---------------------------------------------------------
    W_merge = add((1, db), (1, dn)); t1 = time.time()
    W_res = finetune_mix(W_merge, 1000); W_joint = finetune_mix(W0, 3000, seed=4)
    d_res = {k: W_res[k] - W_merge[k] for k in W0}; np.savez(os.path.join(OUT, "delta_resolution.npz"), **d_res)
    flat = lambda d: np.concatenate([d[k].ravel() for k in sorted(d)])
    R["resolution"] = {"steps": 1000, "rel_norm": float(np.linalg.norm(flat(d_res)) / np.linalg.norm(flat(W0))), "joint_steps": 3000, "seconds": time.time() - t1}
    print(f"[c] fine-tunes done in {time.time() - t1:.0f}s", flush=True)

    # test split ---------------------------------------------------------------------------------------
    models = {"base": W0, "branch: bicubic": add((1, db)), "branch: noisy": add((1, dn)),
              "merge: sum (exp14)": W_merge, "merge: average (exp14)": add((0.5, db), (0.5, dn))}
    if R["grid_choice"]: models[f"(a) weighted merge a={R['grid_choice']['a']}, b={R['grid_choice']['b']}"] = add((R["grid_choice"]["a"], db), (R["grid_choice"]["b"], dn))
    models["(b) cherry-picked bicubic"] = add((1, db_p)); models["(b) cherry-picked noisy"] = add((1, dn_p))
    models["(b) merge of the two cherry-picks"] = add((1, db_p), (1, dn_p))
    models["(c) sum merge + resolution commit (1k steps)"] = W_res; models["(c) joint fine-tune from base (3k steps, no branches)"] = W_joint
    st = scores(W0, test); R["test"] = {}
    for name, W in models.items():
        s = st if name == "base" else scores(W, test); d = deltas(s, st)
        R["test"][name] = {"psnr": s, "delta": d, "meets_spec": None if name == "base" else meets_spec(d)}
        print(name, {k: round(v, 2) for k, v in d.items()}, flush=True)
    json.dump(R, open(os.path.join(OUT, "exp16_results.json"), "w"), indent=1)

    cols = [t[0] for t in TESTS]
    L = ["# Experiment 16: resolving the fine-tune merge conflict (TinySR-26 x2)\n",
         f"Spec: gain > 0 on bicubic LR and on noisy LR, and no test below base by more than {TOL} dB. Choices made on "
         f"{len(fit)} fitting images (first 40 AFHQ val images per class); results on exp14's {len(test)} test images (last 60 per class). "
         "PSNR in dB, 2x (256 -> 512); Δ vs base in brackets.\n",
         "| model | " + " | ".join(cols) + " | meets spec |", "|---" * (len(cols) + 2) + "|"]
    for name, v in R["test"].items():
        L.append(f"| {name} | " + " | ".join(f"{v['psnr'][c]:.2f}" + ("" if name == "base" else f" ({v['delta'][c]:+.2f})") for c in cols)
                 + f" | {'-' if v['meets_spec'] is None else ('yes' if v['meets_spec'] else 'no')} |")
    L += ["", f"(a) Weighted merges meeting the spec on the fitting split: {sum(g['meets_spec'] for g in R['grid'])} of {len(R['grid'])}"
          + (f"; best a={R['grid_choice']['a']}, b={R['grid_choice']['b']}." if R["grid_choice"] else "."),
          "", "Fitting-split Δ (dB) for each weighting, as bicubic / noisy / nearest:", "",
          "| a \\ b | " + " | ".join(str(b) for b in GRID) + " |", "|---" * (len(GRID) + 1) + "|"]
    for a_ in GRID:
        cells = [next(g for g in R["grid"] if g["a"] == a_ and g["b"] == b_) for b_ in GRID]
        L.append(f"| {a_} | " + " | ".join("/".join(f"{g['fit_delta'][c]:+.2f}" for c in cols) + (" ✓" if g["meets_spec"] else "") for g in cells) + " |")
    L += ["", "(b) Nearest-LR change on the fitting split when only one layer group of the delta is applied (dB):", ""]
    for name, v in R["blame"].items():
        L.append(f"- d_{name}: " + ", ".join(f"{g} {x:+.2f}" for g, x in v.items()) + f"; cherry-pick drops {R['cherry_pick'][name]['dropped']}")
    L += ["", f"(c) Resolution commit: 1,000 steps from the sum merge on a mix of clean bicubic, noisy bicubic and clean nearest LR; "
          f"stored as a delta of relative size {R['resolution']['rel_norm']:.4f}. Joint fine-tune: the same mix from the base for 3,000 steps."]
    open(os.path.join(OUT, "exp16_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
