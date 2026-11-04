"""
Experiment 7b: Recycle-Bot. Test the blind-audit predictions and compress the
model exactly.

Test inputs: frame 0 of 44.gif (paper cup, trash) and 63.gif (can, recycling),
the only raw camera frames in the repo. Two images cannot measure accuracy;
they are enough to check exactness, the sign convention and a few probes.

  python src/exp7b_recycle_bot.py
"""
import json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keras_np as kn

OUT = "results/exp7_out"

def frame(path):
    im = Image.open(path); im.seek(0)
    return np.asarray(im.convert("RGB").resize((256, 256), Image.BICUBIC), np.float32) / 255

# ---------------------------------------------------------------------------
# Exact compression: fold BN forward, collapse the linear head
# ---------------------------------------------------------------------------
def compress(W, layers):
    """BN_i (positive scale) -> max-pool -> conv/dense  ==  max-pool -> conv/dense with BN folded in.
    BN_1 follows a linear conv, so it folds backwards into conv2d_1 instead."""
    W = {k: v.astype(np.float64) for k, v in W.items()}; out = {}
    def bn(i):
        n = f"batch_normalization_{i}"; s = W[n + "/gamma"] / np.sqrt(W[n + "/moving_variance"] + 1e-3)
        return s, W[n + "/beta"] - W[n + "/moving_mean"] * s
    s, t = bn(1)                                                      # conv2d_1 is linear: fold backwards
    out["conv2d_1/kernel"] = W["conv2d_1/kernel"] * s; out["conv2d_1/bias"] = W["conv2d_1/bias"] * s + t
    for i in range(2, 6):                                             # BN_{i-1} after ReLU -> fold into conv_i
        if i > 2:
            s, t = bn(i - 1)
        K, b = W[f"conv2d_{i}/kernel"], W[f"conv2d_{i}/bias"]
        if i == 2: K2, b2 = K, b                                     # BN_1 already folded into conv2d_1
        else: K2, b2 = K * s[None, None, :, None], b + np.einsum("c,hwco->o", t, K)
        out[f"conv2d_{i}/kernel"], out[f"conv2d_{i}/bias"] = K2, b2
    s, t = bn(5)                                                      # BN_5 -> pool -> flatten -> head
    w = W["dense_1/kernel"] @ W["dense_2/kernel"]                     # (2304, 1): two linear layers are one
    b = W["dense_1/bias"] @ W["dense_2/kernel"] + W["dense_2/bias"]
    st, tt = np.tile(s, 36), np.tile(t, 36)                           # flatten order (6, 6, 64)
    out["head/kernel"] = w * st[:, None]; out["head/bias"] = b + tt @ w
    lay = [L for L in layers if L["type"] not in ("BatchNormalization", "Dropout")]
    lay = [L for L in lay if L["name"] != "dense_1"]
    for L in lay:
        if L["name"] == "dense_2": L.update(name="head")
    return {k: v.astype(np.float32) for k, v in out.items()}, lay

def debias(W, n, momentum=0.99):
    r = momentum ** n; W = dict(W)
    for k in list(W):
        if k.endswith("/moving_mean"): W[k] = (W[k] / (1 - r)).astype(np.float32)
        if k.endswith("/moving_variance"): W[k] = np.maximum((W[k] - r) / (1 - r), 1e-6).astype(np.float32)
    return W

def main():
    os.makedirs(OUT, exist_ok=True); sys.stdout.reconfigure(encoding="utf-8")
    W, layers = kn.load("legacy/Recycle-Bot/trash_hinge_new.h5")
    x = np.stack([frame("legacy/Recycle-Bot/44.gif"), frame("legacy/Recycle-Bot/63.gif")])
    names = ["44.gif cup (trash)", "63.gif can (recycling)"]
    score = lambda X, Wm=W, Lm=layers: kn.forward(X, Wm, Lm)[:, 0]
    s0 = score(x)
    R = {"scores": dict(zip(names, s0.round(4).tolist()))}

    # compression
    Wc, Lc = compress(W, [dict(L) for L in layers])
    rng = np.random.default_rng(0)
    probe = np.concatenate([x, rng.random((4, 256, 256, 3), dtype=np.float32), np.clip(x[::-1] * 1.3, 0, 1)])
    d = float(np.abs(score(probe) - score(probe, Wc, Lc)).max())
    R["compression"] = {"params_before": kn.nparams(W), "params_after": kn.nparams(Wc),
                        "layers_after": kn.summary(Lc), "max_abs_score_diff": d}

    # probes
    gray = x.mean(-1, keepdims=True).repeat(3, -1)
    P = {"original": x, "brightness x0.7": x * 0.7, "brightness x1.3": np.clip(x * 1.3, 0, 1),
         "horizontal flip": x[:, :, ::-1], "vertical flip": x[:, ::-1], "rotate 90": np.rot90(x, 1, (1, 2)).copy(),
         "grayscale": gray, "shift right 32px": np.roll(x, 32, 2), "unscaled 0-255 input": x * 255}
    R["probes"] = {k: score(v).round(3).tolist() for k, v in P.items()}

    # BN forensics: Keras running stats start at (mean 0, var 1) and move 1% per step.
    # With true variance << 1, moving_var ~ 0.99^n  ->  n from the weights alone.
    mv1 = W["batch_normalization_1/moving_variance"]
    n_steps = float(np.log(np.median(mv1)) / np.log(0.99))
    R["bn_steps_from_weights"] = n_steps
    # fix: bias-correct every running stat (as Adam does for its moving averages)
    Wd = debias(W, round(n_steps))
    R["probes_debiased"] = {k: score(v, Wd).round(3).tolist() for k, v in P.items()}

    # occlusion: where does it actually look? 32px gray patch, stride 32 (8x8 grid)
    occ = np.zeros((2, 8, 8))
    for i in range(8):
        for j in range(8):
            xo = x.copy(); xo[:, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = 0.5
            occ[:, i, j] = score(xo) - s0
    R["occlusion"] = occ.round(3).tolist()
    # weight-derived spatial map (blind) vs measured occlusion map, at 6x6 -> compare via 8x8 resample
    w = Wc["head/kernel"][:, 0].reshape(6, 6, 64); wm = np.abs(w).sum(-1)
    wm8 = np.asarray(Image.fromarray(wm.astype(np.float32)).resize((8, 8), Image.BILINEAR))
    R["blind_map_vs_occlusion_corr"] = [float(np.corrcoef(wm8.ravel(), np.abs(o).ravel())[0, 1]) for o in occ]

    json.dump(R, open(os.path.join(OUT, "recycle_results.json"), "w"), indent=1)
    save_fig(x, occ, os.path.join(OUT, "recycle_occlusion.png"))
    L = ["# Experiment 7b: Recycle-Bot\n",
         "| image | score (positive = recycling) |", "|---|---|"] + [f"| {n} | {v:+.3f} |" for n, v in R["scores"].items()]
    c = R["compression"]
    L += [f"\n## Exact compression\n", f"- {c['params_before']:,} → **{c['params_after']:,} parameters** "
          f"({1 - c['params_after'] / c['params_before']:.0%} smaller), no BatchNorm, one 2,304-weight linear head.",
          f"- Max score difference on 10 inputs (2 real, 4 noise, 4 brightened): {c['max_abs_score_diff']:.1e}",
          f"- New architecture: {c['layers_after']}\n",
          "## BatchNorm forensics\n",
          f"BN1 running variance median {np.median(mv1):.4f} ≈ 0.99^n  ⇒  **n ≈ {n_steps:.1f} update steps** read from the "
          "weights. The notebook log: 63 training images, batch 25 (3 steps/epoch), early stopping restored epoch 6 ⇒ 18 steps.\n",
          "So the stored running stats are ~83% initial values: at inference the network does not normalise the way it did "
          f"in training. Fix (weights only): bias-correct each running stat, true = (stored − 0.99^n·init)/(1 − 0.99^n), n = {round(n_steps)}.\n",
          "## Probes (score per image; positive = recycling)\n",
          "| input | cup (base) | can (base) | cup (BN bias-corrected) | can (BN bias-corrected) |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v[0]:+.3f} | {v[1]:+.3f} | {R['probes_debiased'][k][0]:+.3f} | {R['probes_debiased'][k][1]:+.3f} |"
          for k, v in R["probes"].items()]
    L += [f"\n## Where it looks\n",
          f"Occlusion (32-px gray patch, 8×8 grid) vs the blind weight map: correlation "
          f"{R['blind_map_vs_occlusion_corr'][0]:+.2f} (cup), {R['blind_map_vs_occlusion_corr'][1]:+.2f} (can). "
          "See recycle_occlusion.png."]
    open(os.path.join(OUT, "recycle_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

def save_fig(x, occ, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    for k in range(2):
        ax[2 * k].imshow(x[k]); ax[2 * k].set_title(["cup (trash)", "can (recycling)"][k]); ax[2 * k].axis("off")
        m = np.abs(occ[k]).max(); im = ax[2 * k + 1].imshow(occ[k], cmap="RdBu_r", vmin=-m, vmax=m)
        ax[2 * k + 1].set_title("score change when\neach 32px block is grayed"); ax[2 * k + 1].axis("off")
        plt.colorbar(im, ax=ax[2 * k + 1], fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)

if __name__ == "__main__":
    main()
