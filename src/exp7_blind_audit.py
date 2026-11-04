"""
Experiment 7a: blind audit. What can be said about each model from its weights
and config alone, before running a single image through it?

Writes results/exp7_out/blind_audit.md. Predictions made here are tested by exp7b-d,
which run the models. Nothing in this script calls forward().

  python src/exp7_blind_audit.py
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keras_np as kn

OUT = "results/exp7_out"

def receptive_field(layers):
    r, j = 1, 1
    for L in layers:
        if L["type"] == "Conv2D": r += (L["kernel_size"][0] - 1) * j
        elif L["type"] == "MaxPooling2D": r += (L["pool_size"][0] - 1) * j; j *= L["pool_size"][0]
        elif L["type"] == "UpSampling2D": j /= L["size"][0]
    return r

def bn_scale(W, n, eps=1e-3):
    return W[n + "/gamma"] / np.sqrt(W[n + "/moving_variance"] + eps)

def recycle(L):
    W, layers = kn.load("legacy/Recycle-Bot/trash_hinge_new.h5"); tot = kn.nparams(W)
    d1, d2 = W["dense_1/kernel"], W["dense_2/kernel"]
    w = (d1 @ d2)[:, 0]; b = float(W["dense_1/bias"] @ d2[:, 0] + W["dense_2/bias"][0])
    acts = {l["name"]: l.get("activation") for l in layers}
    gam = np.concatenate([W[f"batch_normalization_{i}/gamma"] for i in range(1, 6)])
    bet = np.concatenate([W[f"batch_normalization_{i}/beta"] for i in range(1, 6)])
    sc = np.concatenate([bn_scale(W, f"batch_normalization_{i}") for i in range(1, 6)])
    M = w.reshape(6, 6, 64)                         # Flatten order of (6,6,64)
    spatial = np.abs(M).sum(-1); spatial /= spatial.sum()
    sv = np.linalg.svd(d1, compute_uv=False)
    K1 = W["conv2d_1/kernel"] * bn_scale(W, "batch_normalization_1")    # BN1 folded
    col = K1.sum((0, 1))                             # (3, 64): net colour response of each filter
    lum_like = int(np.sum(np.all(np.sign(col) == np.sign(col[:1]), 0)))
    L += ["## Recycle-Bot (`trash_hinge_new.h5`): trash vs recycling classifier\n",
          f"Architecture: {kn.summary(layers)}\n",
          f"- **{tot:,} parameters, {d1.size / tot:.0%} of them in `dense_1`**, which feeds a 6×6×64 map into 256 units.",
          f"- `dense_1` is **linear** ({acts['dense_1']}) and so is `dense_2` ({acts['dense_2']}); the Dropout between them does "
          "nothing at inference. Two stacked linear layers are one linear layer: the whole head is a single 2,304-weight "
          f"vector plus a bias. Prediction: folding it removes {d1.size + d2.size + 256 + 1 - 2305:,} parameters "
          f"({(d1.size + d2.size + 256 + 1 - 2305) / tot:.0%} of the model) with identical outputs.",
          f"- `dense_1`'s 256 units are not even used as 256 directions: only the 1-D projection through `dense_2` matters "
          f"(its singular values run {sv[0]:.2f} … {sv[-1]:.2f}, irrelevant to the output).",
          f"- All 320 BatchNorm scales are positive (min γ/σ = {sc.min():.3f}). Positive per-channel scaling commutes with "
          "ReLU-free max-pooling, and every conv uses 'valid' padding, so **every BN folds exactly** into the next conv/dense. "
          "Prediction: a BN-free model with identical outputs.",
          f"- γ ranges {gam.min():.3f}–{gam.max():.3f} and β {bet.min():+.3f}–{bet.max():+.3f}: the BN affine parameters barely "
          "moved from their initial (1, 0). Either training was short or the learning rate on them tiny.",
          f"- Receptive field of each of the 6×6 output cells: {receptive_field(layers)} px of the 256-px image, so the "
          "classifier sees the whole object, and `conv2d_1` is linear (no activation), so layer 1 + BN + max-pool is just a "
          "learned colour/edge pooling.",
          f"- `conv2d_1` colour: {lum_like}/64 filters respond with the same sign to R, G and B (brightness/edge detectors); "
          f"{64 - lum_like} are colour-opponent.",
          "- **Where it looks** (share of |weight| per cell of the 6×6 grid the head reads; rows top→bottom):\n",
          "```", "\n".join(" ".join(f"{v:5.1%}" for v in row) for row in spatial), "```",
          f"  Centre 4×4 holds {spatial[1:5, 1:5].sum():.0%} of the weight (it would hold 44% if uniform); the outer ring "
          f"{1 - spatial[1:5, 1:5].sum():.0%}. Left column {spatial[:, 0].sum():.0%}, right column {spatial[:, -1].sum():.0%}.",
          "- From the training code: label +1 = folder `r` (recycling), −1 = `t` (trash), inputs /255, hinge loss, "
          "50/50 train/test split. **Prediction:** a positive score means recyclable.",
          "- The deployed script (`linux CNN.py`) is for a different model: it loads `trash.h5`, which expects a 2-unit softmax "
          "and un-scaled 0–255 input, and still names the classes 'Dog' and 'Cat'. This hinge checkpoint has a "
          "1-unit output and would always land in class 0 under that script's argmax.\n"]

def deep_color(L):
    W, layers = kn.load("legacy/Deep-Color/generator2800.h5")
    mv = {i: W[f"batch_normalization_{i}/moving_variance"] for i in range(4, 9)}
    k5 = W["conv2d_5/kernel"]; k10, b10 = W["conv2d_10/kernel"], W["conv2d_10/bias"]
    L += ["## Deep-Color (`generator2800.h5`): grayscale → colour\n",
          f"Architecture: {kn.summary(layers)}\n",
          f"- {kn.nparams(W):,} parameters, 5 conv+BN+LeakyReLU blocks at full 128×128 resolution, no downsampling, no skip "
          "connection from input to output.",
          f"- **Receptive field: {receptive_field(layers)} px.** Every output pixel's colour is decided by a "
          f"{receptive_field(layers)}×{receptive_field(layers)} gray patch. Prediction: it cannot know whether a region is "
          "sky, skin or grass. It can only map local brightness/texture to colour, so it behaves like a tone-mapping palette, "
          "not a semantic colorizer.",
          f"- **Input scale, read from BatchNorm:** the first BN's running variance has median {np.median(mv[4]):.1e} "
          f"(std ≈ {np.sqrt(np.median(mv[4])):.1e}). `conv2d_5` weights have std {k5.std():.3f} over 9 taps, so the inputs "
          "it saw had magnitude ~1e-3, not ~0.5. That matches the code: skimage's `rgb2gray` already returns 0–1, and the "
          "code divides by 255 again, so **the model was trained on gray values in [0, 0.0039]**. Prediction: feeding normal "
          "0–1 gray images breaks it; the fix is an exact fold of 1/255 into `conv2d_5`.",
          f"- BN running variances climb from {np.median(mv[4]):.0e} to {np.median(mv[8]):.0e} through the blocks: the "
          "network re-amplifies the tiny input signal by roughly 10^4 in amplitude.",
          f"- **Output conv is 2×2 with 'same' padding.** TensorFlow pads even kernels on the bottom/right only, so each "
          "output pixel is computed from pixels (i, i+1): the colour is shifted half a pixel up-left relative to the input.",
          f"- Output bias = {np.round(b10, 3).tolist()} (RGB): where features are near zero the image defaults to that colour.",
          "- From the training code: the generator's only loss is fooling a discriminator that sees the colour image "
          "**without** the gray input. Nothing ties the colours, or even the brightness, to the input. Prediction: "
          "output brightness drifts from the input, and a fixed colour-space fix (keep the input's luminance, take only "
          "chroma from the model) can only help.\n"]

def cat_vs_pca(L):
    Wc, lc = kn.load("legacy/Cat-GAN/generator580.h5"); Wp, lp = kn.load("legacy/PCA-GAN/model.h5")
    rows, rs = [], []
    for k in sorted(Wc):
        if Wc[k].ndim < 2: continue
        a, b = Wc[k].ravel(), Wp[k].ravel(); r = np.corrcoef(a, b)[0, 1]; rs.append(r)
        rows.append(f"| {k} | {r:+.3f} | {np.linalg.norm(a - b) / np.linalg.norm(b):.2f} |")
    # channel-permutation check on conv2d_8: best match of each Cat filter among PCA-GAN filters
    A = Wc["conv2d_8/kernel"].reshape(-1, 64); B = Wp["conv2d_8/kernel"].reshape(-1, 64)
    A = A / np.linalg.norm(A, axis=0); B = B / np.linalg.norm(B, axis=0); Cm = np.abs(A.T @ B)
    sv_c = np.linalg.svd(Wc["dense_2/kernel"], compute_uv=False); sv_p = np.linalg.svd(Wp["dense_2/kernel"], compute_uv=False)
    er = lambda s: s.sum() ** 2 / (s ** 2).sum()
    L += ["## Cat-GAN `generator580.h5` vs PCA-GAN `model.h5`\n",
          f"Same architecture ({kn.summary(lc)}), same 587,715 parameters. Cat-GAN's notebook saves `generator<epoch>.h5` "
          "every 10 epochs from a PCA-input GAN on the same cat dataset, so these could be two checkpoints of one run.\n",
          "| tensor | correlation | relative distance |", "|---|---|---|"] + rows + [
          f"\n- Same-index filters in `conv2d_8`: mean |cos| {np.mean(np.diag(Cm)):.3f}; best match anywhere: mean {Cm.max(1).mean():.3f}. "
          "Random 1,600-d vectors give |cos| ≈ 0.02.",
          f"- Effective rank of `dense_2`: Cat-GAN {er(sv_c):.0f}/512, PCA-GAN {er(sv_p):.0f}/512.",
          ("- **Reading:** kernel correlations average {:+.2f} and same-index filters don't match, so these are **separate "
           "training runs**, not checkpoints of one run. Interpolating between them would mix unrelated coordinate systems; "
           "merging them first needs the channels aligned (permutation matching on activations, which needs PCA inputs)."
           if np.mean(rs) < 0.2 and np.mean(np.diag(Cm)) < 0.2 else
           "- **Reading:** kernel correlations average {:+.2f} with same-index filters matching, so these look like "
           "**checkpoints of one run**: interpolation and 'training direction' arithmetic between them are meaningful.").format(np.mean(rs)),
          "- Both take 512 PCA coefficients of the Kaggle cat-faces dataset as input. Neither repo ships the PCA basis or "
          "dataset (`pca_dataset.npy` lived on the original PC), so their outputs can't be sampled faithfully without "
          "re-downloading that dataset.\n"]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    L = ["# Experiment 7a: blind audit (weights and config only, no forward passes)\n",
         "Each bullet marked **Prediction** is tested by running the model in exp7b–d.\n"]
    recycle(L); deep_color(L); cat_vs_pca(L)
    open(os.path.join(OUT, "blind_audit.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
