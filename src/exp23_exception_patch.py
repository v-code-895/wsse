"""
Experiment 23: fixing a classifier's mistakes with a compiled exception, not a fine-tune.

Spec: an AFHQ cat photo should get a domestic-cat class (ImageNet 281-285) as MobileNetV2's top-1;
dogs and wild animals (lions, tigers, leopards, foxes, wolves...) must keep their predictions.

The patch adds one hidden unit beside MobileNetV2's linear classifier (a new layer branch):
    logits' = logits + beta * relu(w . phi + c - theta) * 1_cats
phi is the 1280-d pooled feature. w, c: a ridge probe "cat photo vs not" on the fitting images;
theta: the smallest threshold that fires on no fitting non-cat; beta: the smallest boost that
fixes the fitting errors where the unit fires. Where the unit is off (w . phi + c <= theta), the
logits are exactly the base model's: the patch is local by construction, and the network can
report, per input, whether it fired.

Baselines on the same fitting images, Adam for 300 steps:
  * the head (1280 x 1000 + bias) fine-tuned with preservation: -log P(cat classes) on cats + KL(base || new)
    on dogs and wild;
  * the head fine-tuned on the errors only (the quick fix);
  * the same one-unit branch as the patch (1,281 input weights and 5 output weights), trained by Adam
    instead of fitted in closed form, with each of the two objectives above and two learning rates.
    This separates the structure (a new gated unit beside the classifier) from the fitting rule
    (a ridge probe with its threshold at the largest fitting non-cat score).

  <venv python> src/exp23_exception_patch.py
"""
import json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobilenet_np as mn

OUT = "results/exp23_out"; ROOT = "data/afhq/data/afhq"; CATS = np.arange(281, 286); DOGS = np.arange(151, 269)
N_FIT = 600

def image_list():
    rng = np.random.default_rng(0); L = {}
    for c in ("cat", "dog", "wild"):
        tr = sorted(os.listdir(os.path.join(ROOT, "train", c))); va = sorted(os.listdir(os.path.join(ROOT, "val", c)))
        L[("fit", c)] = [os.path.join(ROOT, "train", c, f) for f in rng.choice(tr, N_FIT, replace=False)]
        L[("val", c)] = [os.path.join(ROOT, "val", c, f) for f in va]
    return L

def extract(paths, W, chunk=16):
    lg, ft = [], []
    for s in range(0, len(paths), chunk):
        x = np.stack([mn.preprocess(np.asarray(Image.open(p).convert("RGB"))) for p in paths[s:s + chunk]])
        l, f = mn.forward(x, W, return_features=True); lg.append(l); ft.append(f)
    return np.concatenate(lg), np.concatenate(ft)

def is_cat(logits): return np.isin(logits.argmax(1), CATS)

def softmax(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def kl(p, q): return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), -1)))

def head_finetune(F_, L0, Wh, bh, cat_idx, keep_idx, steps, lr, preserve, seed=0):
    import torch
    torch.manual_seed(seed); dev = "cuda"
    W = torch.tensor(Wh, device=dev, requires_grad=True); b = torch.tensor(bh, device=dev, requires_grad=True)
    X = torch.tensor(F_, device=dev); P0 = torch.softmax(torch.tensor(L0, device=dev), 1)
    ci, ki = torch.tensor(cat_idx, device=dev), torch.tensor(keep_idx, device=dev); cats = torch.tensor(CATS, device=dev)
    opt = torch.optim.Adam([W, b], lr=lr); t0 = time.time()
    for _ in range(steps):
        z = X @ W + b; logp = torch.log_softmax(z, 1)
        loss = -torch.logsumexp(logp[ci][:, cats], 1).mean()
        if preserve and len(ki): loss = loss + (P0[ki] * (torch.log(P0[ki] + 1e-12) - logp[ki])).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy(), time.time() - t0

def unit_finetune(F_, L0, cat_idx, keep_idx, steps, lr, preserve, seed=0):
    """The patch's structure, trained by gradient descent: logits' = logits + v * relu(w . [phi, 1]) on the
    cat classes. w starts small and random, v at zero (so the base model is unchanged at step 0)."""
    import torch
    torch.manual_seed(seed); dev = "cuda"
    X = torch.tensor(np.c_[F_, np.ones(len(F_))], dtype=torch.float32, device=dev); L = torch.tensor(L0, dtype=torch.float32, device=dev)
    w = (0.01 * torch.randn(X.shape[1], device=dev)).requires_grad_(); v = torch.zeros(len(CATS), device=dev, requires_grad=True)
    P0 = torch.softmax(L, 1); ci, ki = torch.tensor(cat_idx, device=dev), torch.tensor(keep_idx, device=dev); cats = torch.tensor(CATS, device=dev)
    opt = torch.optim.Adam([w, v], lr=lr); t0 = time.time()
    for _ in range(steps):
        z = L.clone(); z[:, cats] = z[:, cats] + torch.relu(X @ w)[:, None] * v[None]; logp = torch.log_softmax(z, 1)
        loss = -torch.logsumexp(logp[ci][:, cats], 1).mean()
        if preserve and len(ki): loss = loss + (P0[ki] * (torch.log(P0[ki] + 1e-12) - logp[ki])).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return w.detach().cpu().numpy().astype(np.float64), v.detach().cpu().numpy().astype(np.float64), time.time() - t0

def main(write=True):
    """write=False returns the image lists, logits and the patch without writing (for make_figures_v2.py)."""
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True); t_all = time.time()
    W = mn.load(); Wh, bh = W["Logits/kernel"].astype(np.float64), W["Logits/bias"].astype(np.float64)
    cache = os.path.join(OUT, "features.npz"); L = image_list()
    if os.path.exists(cache):
        z = np.load(cache); D = {tuple(k.split("_")): (z[k + "_logits"], z[k + "_feats"]) for k in ("fit_cat", "fit_dog", "fit_wild", "val_cat", "val_dog", "val_wild")}
    else:
        D = {}; t0 = time.time()
        for k, paths in L.items():
            D[k] = extract(paths, W); print(f"[*] {k}: {len(paths)} images ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(cache, **{f"{a}_{b}_logits": v[0] for (a, b), v in D.items()}, **{f"{a}_{b}_feats": v[1] for (a, b), v in D.items()})
    lab = mn.labels(); R = {"n_fit_per_class": N_FIT, "n_val_per_class": len(L[("val", "cat")])}

    def stack(split): return (np.concatenate([D[(split, c)][1] for c in ("cat", "dog", "wild")]).astype(np.float64),
                              np.concatenate([D[(split, c)][0] for c in ("cat", "dog", "wild")]).astype(np.float64),
                              np.repeat([0, 1, 2], [len(D[(split, c)][0]) for c in ("cat", "dog", "wild")]))
    Ff, Lf, yf = stack("fit"); Fv, Lv, yv = stack("val")
    base = {s: {"cat_acc": float(is_cat(Lx[y == 0]).mean()), "dog_acc": float(np.isin(Lx[y == 1].argmax(1), DOGS).mean()),
                "wild_called_cat": float(is_cat(Lx[y == 2]).mean())} for s, Lx, y in (("fit", Lf, yf), ("val", Lv, yv))}
    errs = Lv[yv == 0][~is_cat(Lv[yv == 0])].argmax(1)
    R["base"] = base; R["val_cat_errors_top_classes"] = {lab[c]: int(n) for c, n in zip(*np.unique(errs, return_counts=True))}
    print("[1] base:", json.dumps(base), flush=True)

    # --- the exception patch: closed form -------------------------------------------------------
    t0 = time.time()
    X = np.c_[Ff, np.ones(len(Ff))]; t = (yf == 0).astype(float); sw = np.where(yf == 0, 2.0, 1.0)
    w = np.linalg.solve((X * sw[:, None]).T @ X + 1.0 * np.eye(X.shape[1]), (X * sw[:, None]).T @ t)
    p_f = X @ w; theta = float(p_f[yf != 0].max())                                    # fires on no fitting non-cat
    h_f = np.maximum(p_f - theta, 0); bad = (yf == 0) & ~is_cat(Lf)
    need = Lf[:, np.setdiff1d(np.arange(1000), CATS)].max(1) - Lf[:, CATS].max(1) + 0.5
    fixable = bad & (h_f > 0); beta = float(np.max(need[fixable] / h_f[fixable])) if fixable.any() else 0.0
    fit_seconds = time.time() - t0
    def patched(F_, L_):
        h = np.maximum(np.c_[F_, np.ones(len(F_))] @ w - theta, 0); Lp = L_.copy(); Lp[:, CATS] += beta * h[:, None]; return Lp, h > 0
    R["patch"] = {"theta": theta, "beta": beta, "fit_errors": int(bad.sum()), "fit_errors_where_unit_fires": int(fixable.sum()),
                  "new_params": int(w.size + 1 + len(CATS)), "fit_seconds": fit_seconds}

    # --- baselines: head fine-tunes on the same images --------------------------------------------
    cat_idx = np.nonzero(yf == 0)[0]; keep_idx = np.nonzero(yf != 0)[0]
    Wp, bp, s_pres = head_finetune(Ff.astype(np.float32), Lf.astype(np.float32), Wh.astype(np.float32), bh.astype(np.float32), cat_idx, keep_idx, 300, 1e-3, True)
    We, be, s_err = head_finetune(Ff.astype(np.float32), Lf.astype(np.float32), Wh.astype(np.float32), bh.astype(np.float32), np.nonzero(bad)[0], [], 300, 1e-3, False)
    units = {}
    for obj, ci_, ki_, pres in (("errors only", np.nonzero(bad)[0], [], False), ("with preservation", cat_idx, keep_idx, True)):
        for lr in (1e-3, 1e-2):
            wu, vu, su = unit_finetune(Ff, Lf, ci_, ki_, 300, lr, pres)
            units[f"one unit trained by Adam, {obj}, lr {lr:g}"] = (wu, vu, su)
    def unit(wu, vu):
        def f(F_, L_):
            h = np.maximum(np.c_[F_, np.ones(len(F_))] @ wu, 0); Lp = L_.copy(); Lp[:, CATS] += h[:, None] * vu[None]; return Lp, h > 0
        return f
    R["unit_adam_fit_nonfiring"] = {}
    for name, (wu, vu, su) in units.items():
        hf = np.maximum(np.c_[Ff, np.ones(len(Ff))] @ wu, 0)
        R["unit_adam_fit_nonfiring"][name] = {"fires_on_fitting_noncats": int((hf[yf != 0] > 0).sum()), "of": int((yf != 0).sum()),
                                             "fires_on_fitting_cats": int((hf[yf == 0] > 0).sum())}
    variants = {"base": lambda F_, L_: (L_, np.zeros(len(L_), bool)), "exception unit": patched,
                "head fine-tune, with preservation": lambda F_, L_: (F_ @ Wp + bp, None),
                "head fine-tune, errors only": lambda F_, L_: (F_ @ We + be, None)}
    variants.update({name: unit(wu, vu) for name, (wu, vu, su) in units.items()})
    seconds = {"exception unit": fit_seconds, "head fine-tune, with preservation": s_pres, "head fine-tune, errors only": s_err}
    seconds.update({name: su for name, (wu, vu, su) in units.items()})

    # --- evaluation on the validation images ----------------------------------------------------
    P0 = softmax(Lv); R["val"] = {}
    for name, f in variants.items():
        Lp, fired = f(Fv, Lv); Pp = softmax(Lp); top0, top1 = Lv.argmax(1), Lp.argmax(1); r = {}
        r["cat_acc"] = float(is_cat(Lp[yv == 0]).mean())
        was_bad = (yv == 0) & ~is_cat(Lv); r["cat_errors_fixed"] = f"{int(is_cat(Lp[was_bad]).sum())}/{int(was_bad.sum())}"
        r["cat_correct_broken"] = int(((yv == 0) & is_cat(Lv) & ~is_cat(Lp)).sum())
        for c, nm in ((1, "dog"), (2, "wild")):
            m = yv == c
            r[f"{nm}_top1_changed"] = int((top0[m] != top1[m]).sum()); r[f"{nm}_called_cat"] = int(is_cat(Lp[m]).sum())
            r[f"{nm}_kl_vs_base"] = kl(P0[m], Pp[m]); r[f"{nm}_max_logit_change"] = float(np.abs(Lp[m] - Lv[m]).max())
        if fired is not None:
            r["unit_fired"] = {nm: int(fired[yv == c].sum()) for c, nm in ((0, "cat"), (1, "dog"), (2, "wild"))}
            r["max_logit_change_where_unit_off"] = float(np.abs(Lp[~fired] - Lv[~fired]).max()) if (~fired).any() else 0.0
        if name in seconds: r["seconds"] = seconds[name]
        R["val"][name] = r; print(f"[2] {name}: {json.dumps(r)}", flush=True)
    R["seconds_total"] = time.time() - t_all
    if not write: return {"R": R, "L": L, "Lv": Lv, "yv": yv, "Fv": Fv, "patched": patched, "labels": lab}
    json.dump(R, open(os.path.join(OUT, "exp23_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
