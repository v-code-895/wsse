"""
Experiment 8: weight-space programming on a public classifier (MobileNetV2, ImageNet).

  A. Forensics      recover ImageNet's mean and std RGB from the first BatchNorm's running stats;
                    read the label hierarchy (dogs cluster) from the classifier weights
  B. Refactoring    fold all 52 BatchNorms (exact); low-rank the 1.28M-param classifier
  C. Bug fixes      common deployment bugs (raw 0-255 input, BGR channel order) fixed by
                    folding the conversion into Conv1, no retraining
  D. New class      add an "anime face" class (class 1000) in closed form from faces the Anime
                    DCGAN generates; test on held-out faces, a real anime frame and negatives

No labelled ImageNet data is used. "Agreement" means agreement with the original model.

  python src/exp8_mobilenet.py
"""
import json, os, sys
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import skimage.data as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobilenet_np as mn

OUT = "results/exp8_out"
IMAGENET_MEAN = np.array([123.68, 116.78, 103.94])        # the standard published values (RGB, 0-255)
IMAGENET_STD = np.array([58.395, 57.12, 57.375])

# ---------------------------------------------------------------------------
# Test images (no ImageNet data on disk: scikit-image samples + the repos' own images)
# ---------------------------------------------------------------------------
def base_images():
    I = {}
    for n in ["astronaut", "coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry", "retina", "colorwheel"]:
        I[n] = getattr(sd, n)()[..., :3]
    I["motorcycle_left"], I["motorcycle_right"] = sd.stereo_motorcycle()[:2]
    for n in ["camera", "coins", "moon", "page", "clock", "brick", "grass", "gravel"]:
        g = getattr(sd, n)(); I[n] = np.repeat(g[..., None], 3, -1)
    for g in ["44.gif", "63.gif"]:
        im = Image.open("legacy/Recycle-Bot/" + g); im.seek(0); I["recycle_" + g[:2]] = np.asarray(im.convert("RGB"))
    return I

def crops(a, n_extra=4, seed=0):
    """Centre square plus random 70-90% crops (half of them flipped)."""
    rng = np.random.default_rng(seed); h, w = a.shape[:2]; s = min(h, w); out = [a]
    for k in range(n_extra):
        c = int(s * rng.uniform(0.7, 0.9)); y, x = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
        v = a[y:y + c, x:x + c]; out.append(v[:, ::-1] if k % 2 else v)
    return out

def feats_logits(imgs, W, pre=mn.preprocess, chunk=8):
    x = np.stack([pre(i) for i in imgs]); L, F = [], []
    for k in range(0, len(x), chunk):
        l, f = mn.forward(x[k:k + chunk], W, return_features=True); L.append(l); F.append(f)
    return np.concatenate(L), np.concatenate(F)

def kl(p, q): return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), -1)))

# ---------------------------------------------------------------------------
# A. Forensics
# ---------------------------------------------------------------------------
def recover_mean(W):
    """Conv1 has no bias, so BN1's running mean = Conv1 applied to the mean (preprocessed) pixel."""
    K = W["Conv1/kernel"].astype(np.float64); A = K.sum((0, 1)).T            # (32 filters, 3 colours)
    m = W["bn_Conv1/moving_mean"].astype(np.float64)
    mu, *_ = np.linalg.lstsq(A, m, rcond=None)
    r2 = 1 - ((m - A @ mu) ** 2).sum() / ((m - m.mean()) ** 2).sum()
    return (mu + 1) * 127.5, float(r2)                                      # back to 0-255 RGB

def patch_cov(C, rho):
    """27x27 covariance of a 3x3x3 patch: colour covariance C (3x3) times spatial rho^dist."""
    idx = [(i, j) for i in range(3) for j in range(3)]
    S = np.array([[rho ** np.hypot(a[0] - b[0], a[1] - b[1]) for b in idx] for a in idx])
    return np.kron(S, C)                                                    # order (i, j, colour)

def recover_std(W, v=None):
    """BN1 running variance = k^T Sigma k per filter. Fit Sigma = (rho^dist) x C by least squares.
    v: per-filter output variances (default: the stored running variance)."""
    K = W["Conv1/kernel"].astype(np.float64).reshape(27, 32)               # rows ordered (i, j, colour)
    v = W["bn_Conv1/moving_variance"].astype(np.float64) if v is None else v
    def unpack(p):
        L = np.array([[p[0], 0, 0], [p[1], p[2], 0], [p[3], p[4], p[5]]]); return L @ L.T, 1 / (1 + np.exp(-p[6]))
    def resid(p):
        C, rho = unpack(p); S = patch_cov(C, rho)
        return (np.einsum("pk,pq,qk->k", K, S, K) - v) / v
    best = None
    for rho0 in (0.0, 1.5, 3.0):
        r = least_squares(resid, np.r_[0.5, 0.4, 0.3, 0.4, 0.2, 0.3, rho0], max_nfev=4000)
        if best is None or r.cost < best.cost: best = r
    C, rho = unpack(best.x)
    std = np.sqrt(np.diag(C)) * 127.5
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    fit_r2 = 1 - (best.fun ** 2).sum() / (((v - v.mean()) / v) ** 2).sum()
    return std, float(rho), corr, float(fit_r2)

def label_hierarchy(W, lab):
    Wl = W["Logits/kernel"].T; Wn = Wl / np.linalg.norm(Wl, axis=1, keepdims=True); S = Wn @ Wn.T
    np.fill_diagonal(S, -1); nn = np.argsort(-S, 1)[:, :5]
    dogs = np.arange(151, 269)                                               # ImageNet indices 151-268 are dog breeds
    hit = np.isin(nn[dogs], dogs).mean()
    chance = (len(dogs) - 1) / 999
    examples = {lab[i]: [lab[j] for j in nn[i]] for i in [281, 504, 817, 948]}   # tabby, coffee_mug, sports_car, Granny_Smith
    return float(hit), float(chance), examples

# ---------------------------------------------------------------------------
# B. Refactoring
# ---------------------------------------------------------------------------
def fold_all_bn(W):
    """Every conv here is bias-free and followed by BN: conv' = conv*s, bias' = beta - mean*s."""
    W = dict(W); pairs = [("Conv1", "bn_Conv1"), ("Conv_1", "Conv_1_bn")]
    for i, *_ in mn.BLOCKS:
        e, d, p, be, bd, bp = mn.names(i); pairs += [(d, bd), (p, bp)] + ([(e, be)] if i else [])
    for c, b in pairs:
        s = W[b + "/gamma"] / np.sqrt(W[b + "/moving_variance"] + mn.EPS)
        key = c + ("/depthwise_kernel" if c.endswith("depthwise") else "/kernel")
        W[key] = (W[key] * (s[None, None, :, None] if c.endswith("depthwise") else s)).astype(np.float32)
        W[c + "/bias"] = (W[b + "/beta"] - W[b + "/moving_mean"] * s).astype(np.float32)
        for k in ("gamma", "beta", "moving_mean", "moving_variance"): del W[f"{b}/{k}"]
    return W

# ---------------------------------------------------------------------------
# C. Bug fixes by folding input conversions into Conv1
# ---------------------------------------------------------------------------
def accept_raw_uint8(W, exact_padding=False):
    """Model expects x/127.5 - 1. Fold it: Conv1(x/127.5 - 1) = Conv1(x)/127.5 - sum(K). Exact
    except on the bottom/right padding row (where the original pads -1-space zeros)."""
    W = dict(W); K = W["Conv1/kernel"]
    W["Conv1/kernel"] = (K / 127.5).astype(np.float32); W["Conv1/bias"] = (-K.sum((0, 1, 2))).astype(np.float32)
    if exact_padding:          # a zero in [-1,1] space is 127.5 in raw space: pad with that instead
        W["Conv1/pad_value"] = np.float32(127.5)
    return W

def accept_bgr(W):
    W = dict(W); W["Conv1/kernel"] = W["Conv1/kernel"][:, :, ::-1].copy(); return W

def pre_raw(a): return (mn.preprocess(a) + 1) * 127.5
def pre_bgr(a): return mn.preprocess(a)[..., ::-1].copy()
def pre_unit(a): return (mn.preprocess(a) + 1) / 2                         # a common mistake: [0,1] input

# ---------------------------------------------------------------------------
# D. New class from generated data, closed form
# ---------------------------------------------------------------------------
def anime_faces(n, seed):
    from exp2b_tinysr18_weightpatch import load_model
    from exp3_anime_weightpatch import generate
    Wa, oa, _ = load_model("legacy/Anime-DCGAN/generator50000.h5")
    f = np.clip(generate(np.random.default_rng(seed).standard_normal((n, 64)), Wa, oa), 0, 1)
    return [(x * 255).astype(np.uint8) for x in f]

def readme_panel(row):
    """'Original Image' panels of the Deep-Color README figure: 0 painting, 1 Lisa Simpson,
    2 anime girl (a real anime frame), 3 skateboard photo."""
    im = np.asarray(Image.open("legacy/Deep-Color/combined.png").convert("RGB"))
    y0 = [20, 306, 594, 883][row]; return im[y0:y0 + 242, 34:276]

def real_anime_crop(): return readme_panel(2)

def class_direction(Fpos, Fneg, method):
    """A linear score s(f) = f.w + w0 that is high for the new class.
    fisher : LDA with a shrunk covariance (1280-d from ~80 samples, so heavy shrinkage)
    cosine : 'imprinting' - direction from the negatives' mean to the positives' mean"""
    mp, mq = Fpos.mean(0), Fneg.mean(0)
    if method == "fisher":
        X = np.r_[Fpos - mp, Fneg - mq]; S = X.T @ X / len(X)
        S = 0.5 * S + 0.5 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        w = np.linalg.solve(S, mp - mq)
    else:
        w = mp - mq
    w = w / np.linalg.norm(w); return w, -float(w @ (mp + mq) / 2)

def add_class(W, w, w0, Fpos, Fneg, sharp=20.0, level=12.0):
    """New logit = A*(s(f) - tau) + level, with tau halfway between the classes on the fit set and A
    large, so the new class wins roughly when s(f) > tau, whatever the original logits are."""
    sp, sq = Fpos @ w + w0, Fneg @ w + w0
    tau = (np.median(sp) + np.median(sq)) / 2; A = sharp / max(np.std(np.r_[sp, sq]), 1e-6)
    W = dict(W)
    W["Logits/kernel"] = np.concatenate([W["Logits/kernel"], (A * w)[:, None]], 1).astype(np.float32)
    W["Logits/bias"] = np.r_[W["Logits/bias"], A * (w0 - tau) + level].astype(np.float32)
    return W, float(sp.min() - sq.max())

# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True); sys.stdout.reconfigure(encoding="utf-8")
    W = mn.load(); lab = mn.labels(); R = {}

    # A ------------------------------------------------------------------
    mean, r2m = recover_mean(W); std, rho, corr, r2s = recover_std(W)
    hit, chance, ex = label_hierarchy(W, lab)
    R["A"] = {"recovered_mean": mean.round(2).tolist(), "published_mean": IMAGENET_MEAN.tolist(), "mean_fit_r2": r2m,
              "recovered_std": std.round(2).tolist(), "published_std": IMAGENET_STD.tolist(), "std_fit_r2": r2s,
              "recovered_spatial_corr_1px": rho, "recovered_colour_corr": corr.round(3).tolist(),
              "dog_nn_rate": hit, "dog_chance": chance, "neighbours": ex}

    # test inputs
    I = base_images(); names = list(I)
    # method check: run the same std recovery on our own images, where the true pixel std is known
    xs = np.stack([mn.preprocess(I[n]) for n in names if not n.startswith("recycle")])
    co = mn.conv3x3(xs, W["Conv1/kernel"], 2).reshape(-1, 32).astype(np.float64)
    std_own, *_ = recover_std(W, co.var(0))
    R["A"]["method_check_own_images"] = {"recovered_std": std_own.round(2).tolist(),
                                         "true_std": ((xs.reshape(-1, 3).std(0)) * 127.5).round(2).tolist()}
    test = [(n, c) for n in names for c in crops(I[n])]
    imgs = [c for _, c in test]
    L0, F0 = feats_logits(imgs, W); P0 = mn.softmax(L0); top0 = P0.argmax(1)

    # B ------------------------------------------------------------------
    Wf = fold_all_bn(W); Lf = feats_logits(imgs[::5], Wf)[0]
    n_before, n_after = sum(v.size for v in W.values()), sum(v.size for v in Wf.values())
    R["B_fold"] = {"params_before": n_before, "params_after": n_after, "bn_layers_removed": 52,
                   "max_logit_diff": float(np.abs(Lf - L0[::5]).max())}
    K = W["Logits/kernel"].astype(np.float64); U, s, Vt = np.linalg.svd(K, full_matrices=False)
    rows = []
    for r in (32, 64, 128, 256, 384, 512, 768):
        Kr = (U[:, :r] * s[:r]) @ Vt[:r]; Lr = F0 @ Kr + W["Logits/bias"]; Pr = mn.softmax(Lr)
        rows.append({"rank": r, "params": r * (K.shape[0] + K.shape[1]), "saved": K.size - r * (K.shape[0] + K.shape[1]),
                     "top1_agree": float((Pr.argmax(1) == top0).mean()), "top5_contains_top1": float(np.mean([top0[i] in np.argsort(-Pr[i])[:5] for i in range(len(Pr))])),
                     "kl": kl(P0, Pr)})
    R["B_lowrank"] = {"classifier_params": int(K.size), "share_of_model": K.size / n_before, "rows": rows,
                      "energy_99_rank": int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.99) + 1)}

    # C ------------------------------------------------------------------
    sub = imgs[::5]; t0 = top0[::5]
    agree = lambda Wm, pre: float((feats_logits(sub, Wm, pre)[0].argmax(1) == t0).mean())
    Wraw, Wbgr = accept_raw_uint8(W), accept_bgr(W); Wraw_x = accept_raw_uint8(W, exact_padding=True)
    R["C"] = {"n_images": len(sub),
              "raw 0-255 fed to original": agree(W, pre_raw), "raw 0-255 fed to folded model": agree(Wraw, pre_raw),
              "BGR fed to original": agree(W, pre_bgr), "BGR fed to channel-swapped model": agree(Wbgr, pre_bgr),
              "[0,1] fed to original": agree(W, pre_unit),
              "raw fold max logit diff": float(np.abs(feats_logits(sub, Wraw, pre_raw)[0] - L0[::5]).max()),
              "raw fold + pad 127.5 max logit diff": float(np.abs(feats_logits(sub, Wraw_x, pre_raw)[0] - L0[::5]).max()),
              "bgr fold max logit diff": float(np.abs(feats_logits(sub, Wbgr, pre_bgr)[0] - L0[::5]).max())}

    # D ------------------------------------------------------------------
    pos_fit, pos_test = anime_faces(24, seed=100), anime_faces(24, seed=200)
    neg_names = sorted(set(names)); rng = np.random.default_rng(0); rng.shuffle(neg_names)
    neg_fit_n, neg_test_n = set(neg_names[:10]), set(neg_names[10:])
    nf = [i for i, (n, _) in enumerate(test) if n in neg_fit_n]; nt = [i for i, (n, _) in enumerate(test) if n in neg_test_n]
    Lp, Fp = feats_logits(pos_fit, W); Lpt, Fpt = feats_logits(pos_test, W)
    real = crops(real_anime_crop(), 4, seed=5); Lr_, Fr_ = feats_logits(real, W)
    # more negatives for fitting: 10 crops of each fit image plus 6 synthetic sprites (exp2b)
    from exp2b_tinysr18_weightpatch import make_sprite
    extra = [c for n in sorted(neg_fit_n) for c in crops(I[n], 9, seed=11)[1:]] + [make_sprite(s) for s in range(6)]
    Fneg = np.r_[F0[nf], feats_logits(extra, W)[1]]
    hard = [c for r in (0, 1, 3) for c in crops(readme_panel(r), 4, seed=9)]       # painting, cartoon, skateboard
    Lh, Fh = feats_logits(hard, W)
    D = {"what the original model called the generated faces": {lab[k]: int(v) for k, v in zip(*np.unique(Lpt.argmax(1), return_counts=True))},
         "what the original model called the real anime frame": [lab[k] for k in Lr_.argmax(1)], "methods": {}}
    for method in ("fisher", "cosine"):
        w, w0 = class_direction(Fp, Fneg, method)
        W2, gap = add_class(W, w, w0, Fp, Fneg)
        pred = lambda F: (F @ W2["Logits/kernel"] + W2["Logits/bias"]).argmax(1)
        D["methods"][method] = {"fit_gap": gap, "n_neg_fit": len(Fneg),
            "held-out generated faces -> new class": float((pred(Fpt) == 1000).mean()),
            "real anime frame (5 crops) -> new class": float((pred(Fr_) == 1000).mean()),
            "held-out negatives -> new class (false positive)": float((pred(F0[nt]) == 1000).mean()),
            "held-out negatives: original top-1 unchanged": float((pred(F0[nt]) == top0[nt]).mean()),
            "hard negatives (cartoon, painting, photo) -> new class": float((pred(Fh) == 1000).mean()),
            "Lisa Simpson cartoon (5 crops) -> new class": float((pred(Fh[5:10]) == 1000).mean())}
    R["D"] = D

    json.dump(R, open(os.path.join(OUT, "exp8_results.json"), "w"), indent=1, default=float)
    write_md(R, len(test), len(names))
    print(open(os.path.join(OUT, "exp8_results.md"), encoding="utf-8").read())

def write_md(R, n_test, n_img):
    A = R["A"]; L = ["# Experiment 8: MobileNetV2 (ImageNet), weight-space programming on a public classifier\n",
        f"Test inputs: {n_img} images × 5 crops = {n_test} (scikit-image samples, Recycle-Bot frames). "
        "No ImageNet labels; 'agreement' = same top-1 as the original model.\n",
        "## A. Forensics: what the weights say about the training data\n",
        "| | R | G | B |", "|---|---|---|---|",
        "| mean recovered from `bn_Conv1` running mean | " + " | ".join(f"{v:.1f}" for v in A["recovered_mean"]) + " |",
        "| published ImageNet mean | " + " | ".join(f"{v:.2f}" for v in A["published_mean"]) + " |",
        "| std recovered from `bn_Conv1` running variance | " + " | ".join(f"{v:.1f}" for v in A["recovered_std"]) + " |",
        "| published ImageNet std | " + " | ".join(f"{v:.2f}" for v in A["published_std"]) + " |",
        f"\nFits: mean R² {A['mean_fit_r2']:.3f} over 32 filters (3 unknowns); variance model (colour covariance × spatial "
        f"correlation) R² {A['std_fit_r2']:.3f}, recovered neighbour-pixel correlation {A['recovered_spatial_corr_1px']:.2f}, "
        f"colour correlations R–G {A['recovered_colour_corr'][0][1]:.2f}, R–B {A['recovered_colour_corr'][0][2]:.2f}, "
        f"G–B {A['recovered_colour_corr'][1][2]:.2f}.\n",
        f"Method check on our own test photos (true std known): recovered "
        f"{A['method_check_own_images']['recovered_std']} vs true {A['method_check_own_images']['true_std']}.\n",
        f"Label hierarchy from the classifier weights: for the 118 dog-breed classes, {A['dog_nn_rate']:.0%} of each one's "
        f"5 nearest class vectors (cosine) are also dogs (chance {A['dog_chance']:.0%}).\n"]
    L += [f"- {k}: nearest classes {', '.join(v)}" for k, v in A["neighbours"].items()]
    B = R["B_fold"]; Lr = R["B_lowrank"]
    L += ["\n## B. Exact refactoring and compression\n",
          f"- Folding all 52 BatchNorms: {B['params_before']:,} → {B['params_after']:,} parameters, max logit difference "
          f"{B['max_logit_diff']:.1e}.",
          f"- The 1000-way classifier is {Lr['classifier_params']:,} parameters ({Lr['share_of_model']:.0%} of the model); "
          f"99% of its energy is in the top {Lr['energy_99_rank']} of 1000 singular directions.\n",
          "| classifier rank | params saved | top-1 agreement | original top-1 in new top-5 | KL(orig‖new) |", "|---|---|---|---|---|"]
    L += [f"| {r['rank']} | {r['saved']:,} | {r['top1_agree']:.1%} | {r['top5_contains_top1']:.1%} | {r['kl']:.4f} |" for r in Lr["rows"]]
    C = R["C"]
    L += ["\n## C. Deployment bugs fixed by folding into Conv1 (top-1 agreement with correct usage)\n",
          "| input | agreement |", "|---|---|"] + [f"| {k} | {v:.0%} |" for k, v in C.items() if k != "n_images" and "diff" not in k]
    L += [f"\nMax logit difference vs. correct preprocessing: raw fold {C['raw fold max logit diff']:.2e} (the "
          "bottom/right padding row: a zero in [-1,1] space is 127.5 in raw space); raw fold with padding value 127.5 "
          f"{C['raw fold + pad 127.5 max logit diff']:.1e}; BGR fold {C['bgr fold max logit diff']:.1e}.\n"]
    D = R["D"]; meth = D["methods"]
    L += ["## D. A new class ('anime face', index 1000) in closed form\n",
          f"Fit on 24 faces generated by the Anime DCGAN vs {meth['fisher']['n_neg_fit']} negative crops (10 images + 6 "
          "synthetic sprites); appended as column 1000 of the classifier. Test: 24 new generated faces, a real anime "
          "frame never seen, and 12 held-out negative images × 5 crops.\n",
          "| test | Fisher (shrunk LDA) | cosine to class mean |", "|---|---|---|"]
    for k in meth["fisher"]:
        if k in ("fit_gap", "n_neg_fit"): continue
        L.append(f"| {k} | {meth['fisher'][k]:.0%} | {meth['cosine'][k]:.0%} |")
    L += [f"\nWhat the original model called the 24 held-out generated faces: {D['what the original model called the generated faces']}; "
          f"the real anime frame: {D['what the original model called the real anime frame']}."]
    open(os.path.join(OUT, "exp8_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
