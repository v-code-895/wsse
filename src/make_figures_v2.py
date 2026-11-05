"""
Figures added in paper v2.0.1. Each one re-runs the experiment that it shows (without writing that
experiment's result files) and draws from the patched networks directly.

  hero        Figure 1: odd-eyed cats, and where the left-eye patch changes the image (exp20, grow)
  identities  the instruction set, compiled into one-hidden-layer LeakyReLU networks (exp18, circuits.py)
  room        freeing a quarter of a layer with and without the regression fold (exp19)
  union       the merge conflict resolved by a self-checking router (exp22)
  exception   every cat MobileNetV2 gets wrong, before and after the exception unit (exp23)

  <venv python> src/make_figures_v2.py [hero identities room union exception]

Cat images from CatGAN-26 (128 px) are shown 2x upscaled by TinySR-26 (exp10); change maps are
computed on the 128-px outputs themselves.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "paper/figures"; DPI = 200
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
INK, MUTED, BLUE, ORANGE, RED = "#1f2328", "#57606a", "#0969da", "#d1611a", "#cf222e"

def u8(x): return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)

_T26 = None
def sr2(x):
    """(N, 128, 128, 3) in [0, 1] -> (N, 256, 256, 3) by TinySR-26 x2."""
    global _T26
    import torch, modern_np as mnp
    if _T26 is None: _T26 = mnp.to_torch(dict(np.load("results/exp10_out/tinysr26_x2.npz")), "cuda")
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(np.clip(x, 0, 1), np.float32)).permute(0, 3, 1, 2).cuda()
        return mnp.tinysr_torch(t, _T26, 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

def show(ax, img, **kw):
    ax.imshow(img, interpolation="lanczos" if img.ndim == 3 else "nearest", **kw); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def sci(v):
    if v == 0: return "0"
    m, e = f"{v:.1e}".split("e"); return f"{m}×10$^{{{int(e)}}}$"

def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.04); plt.close(fig); print("wrote", p, flush=True)

# ---------------------------------------------------------------------------------------------
def fig_hero(seed=1):
    """Three cats drawn at random (seed 1 in the paper) from the 30 where both gates fire with the least change
    outside the eyes; seed=None takes the three cleanest."""
    import exp20_odd_eyes as e20
    ctx = e20.run_room("grow", write=False); imgs, nets, L, z, eye = ctx["imgs"], ctx["nets"], ctx["L"], ctx["z_test"], ctx["eye"]
    lo, hi = e20.HUE_BAND
    hl = e20.per_image_hue(imgs["left"], eye["left"]); hr = e20.per_image_hue(imgs["right"], eye["right"])
    eyes = eye["left"] | eye["right"]
    far = ~np.any([np.roll(np.roll(eyes, dy, 1), dx, 2) for dy in range(-3, 4) for dx in range(-3, 4)], 0)
    leak = np.array([np.abs(imgs["both"][i] - imgs["base"][i])[far[i]].mean() for i in range(len(z))])
    ok = [i for i in range(len(z)) if hl[i] is not None and hr[i] is not None and lo <= hl[i] <= hi and lo <= hr[i] <= hi]
    clean = sorted(ok, key=lambda i: leak[i])
    pick = clean[:3] if seed is None else [int(i) for i in np.random.default_rng(seed).choice(clean[:30], 3, replace=False)]
    print(f"[hero] {len(ok)} of {len(z)} cats have both irises in band; showing {pick} (leak {leak[pick].round(3)})", flush=True)
    a64 = e20.forward(z[pick], nets["left"], L, double=True); b64 = e20.forward(z[pick], nets["room (no gate)"], L, double=True)
    rows = {k: sr2(imgs[k][pick]) for k in ("base", "left", "right", "both")}
    cols = [("base", "Base CatGAN-26"), ("left", "Left-eye patch"), ("right", "Right-eye patch"), ("both", "Both eyes")]
    fig, axs = plt.subplots(len(pick), 5, figsize=(10.6, 2.18 * len(pick)), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for r, i in enumerate(pick):
        for c, (k, title) in enumerate(cols):
            show(axs[r, c], rows[k][r])
            if r == 0: axs[r, c].set_title(title, color=BLUE if k != "base" else INK)
        d = np.abs(a64[r] - b64[r]).max(-1); ax = axs[r, 4]
        show(ax, np.repeat(np.repeat(d, 2, 0), 2, 1), cmap="magma", vmin=0, vmax=0.35)
        ax.axvline(127.5, color="white", lw=1.0, ls=(0, (3, 2)))
        ax.text(192, 246, f"right half:\nmax {sci(d[:, 64:].max())}", color="white", ha="center", va="bottom", fontsize=7.5)
        ax.text(64, 246, f"left half:\nmax {d[:, :64].max():.2f}", color="white", ha="center", va="bottom", fontsize=7.5)
        if r == 0: ax.set_title("Change made by the\nleft-eye patch (0–1 scale)", color=INK, fontsize=8.5)
    save(fig, "fig_odd_eyes_hero.png")
    return pick

# ---------------------------------------------------------------------------------------------
def fig_identities(a=0.2):
    import circuits as cc
    def net(name, X, **kw):
        W1, b1, w2, c, fn, ins = cc.compile_recipe(name, a, **kw)
        H = cc.lrelu(X @ W1 + b1, a); return H @ w2 + c, H * w2, fn
    fig, axs = plt.subplots(2, 4, figsize=(8.6, 4.7), gridspec_kw={"wspace": 0.3, "hspace": 0.62})
    u = np.linspace(-1, 1, 401)[:, None]; dots = slice(None, None, 20)
    def style(ax, title, note):
        ax.set_title(title, loc="left", fontsize=9); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.0, -0.17, note, transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    def panel(ax, x, y, target, title, note, parts=None):
        if parts is not None:
            for j in range(parts.shape[1]): ax.plot(x, parts[:, j], color=MUTED, lw=0.8, ls="--", alpha=0.8)
        ax.plot(x, target, color="#d0d7de", lw=5, solid_capstyle="round", label="target")
        ax.plot(x[dots], y[dots], "o", color=BLUE, ms=3, label="compiled network")
        style(ax, title, note)
    def err(y, t): return f"max error {sci(float(np.abs(y - t).max()))}"
    y, P, fn = net("pass", u); panel(axs[0, 0], u[:, 0], y, fn(u[:, 0]), "pass  (2 channels)", f"dashed: each channel's term\n{err(y, fn(u[:, 0]))}", P)
    y, P, fn = net("relu", u); panel(axs[0, 1], u[:, 0], y, fn(u[:, 0]), "relu  (2)", f"dashed: each channel's term\n{err(y, fn(u[:, 0]))}", P)
    y, P, fn = net("abs", u); panel(axs[0, 2], u[:, 0], y, fn(u[:, 0]), "abs  (2)", f"dashed: each channel's term\n{err(y, fn(u[:, 0]))}", P)
    y, _, fn = net("clamp", u); panel(axs[0, 3], u[:, 0], y, fn(u[:, 0]), "clamp  (4)", f"to [−0.5, 0.5]\n{err(y, fn(u[:, 0]))}")
    s = np.linspace(-0.06, 0.08, 401)[:, None]; y, _, _ = net("step", s); t_ = np.clip(64 * s[:, 0], 0, 1)
    panel(axs[1, 0], s[:, 0], y, t_, "step  (4)", f"0 below t = 0, 1 above 1/64\n{err(y, t_)}")
    X = np.c_[u, np.full_like(u, 0.3)]; y, _, fn = net("min", X); panel(axs[1, 1], u[:, 0], y, fn(X[:, 0], X[:, 1]), "min  (4)", f"min(x, 0.3)\n{err(y, fn(X[:, 0], X[:, 1]))}")
    ax = axs[1, 2]; p = np.linspace(-1, 1, 401); e = 0.0
    for gv, col, lab in ((1.0, BLUE, "g = 1"), (0.0, ORANGE, "g = 0")):
        X = np.c_[p, np.full_like(p, gv)]; y, _, fn = net("gate", X); t_ = fn(X[:, 0], X[:, 1]); e = max(e, float(np.abs(y - t_).max()))
        ax.plot(p, t_, color="#d0d7de", lw=5); ax.plot(p[dots], y[dots], "o", color=col, ms=3, label=lab)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handletextpad=0.1); style(ax, "gate  (2)", f"relu(p) if g = 1, else 0\nmax error {sci(e)}")
    ax = axs[1, 3]; n = np.arange(200); xs, ys_ = 0.6 * np.sin(n / 12), 0.4 * np.cos(n / 7) - 0.2; g = ((n // 50) % 2 == 0).astype(float)
    X = np.c_[g, xs, ys_]; y, _, fn = net("mux", X); t_ = fn(X[:, 0], X[:, 1], X[:, 2])
    for k in range(0, 200, 100): ax.axvspan(k, k + 50, color=BLUE, alpha=0.07, lw=0)
    ax.plot(n, xs, color=BLUE, lw=0.9, alpha=0.7, label="x"); ax.plot(n, ys_, color=ORANGE, lw=0.9, alpha=0.7, label="y")
    ax.plot(n[::4], y[::4], "o", color=INK, ms=2.4, label="output"); ax.set_ylim(-0.75, 1.15)
    ax.legend(fontsize=6.3, frameon=False, loc="upper center", ncol=3, handlelength=1.2, columnspacing=0.8, handletextpad=0.3)
    style(ax, "mux  (6)", f"x where g = 1 (shaded), else y\n{err(y, t_)}")
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    save(fig, "fig_identities.png")

# ---------------------------------------------------------------------------------------------
def fig_room(layer="b1c2", share=4, n_show=5):
    import json, circuits as cc, exp19_free_capacity as e19
    from exp18_circuit_primitives import catgan_layers, run_layers
    W = {k: v.astype(np.float32) for k, v in np.load("results/exp11_out/catgan26.npz").items()}; L = catgan_layers(W)
    rng = np.random.default_rng(0); z_fit, z_test = rng.standard_normal((512, 128)), rng.standard_normal((256, 128))
    cons = dict(e19.producer_consumer(L))[layer]
    G = cc.gram(e19.acts_after(z_fit, W, L, layer)); C = G.shape[0] - 1; k = C // share
    drop_all = cc.choose_free(G, cc.out_energy(W, cons), max(1, C // 2))[0]            # as exp19: one greedy order, then prefixes
    Wf, Wz = e19.fold_and_zero(W, layer, cons, G, drop_all[:k], None)
    x0, xf, xz = (run_layers(z_test, w, L) for w in (W, Wf, Wz))
    pf, pz = e19.psnr(xf, x0), e19.psnr(xz, x0)
    ref = json.load(open("results/exp19_out/exp19_results.json"))
    row = next(r for r in ref["layers"] if r["layer"] == layer)["k"][str(k)]
    print(f"[room] {layer} k={k}: fold {pf:.2f} / zero {pz:.2f} dB (exp19: {row['psnr_fold']:.2f} / {row['psnr_zero']:.2f})", flush=True)
    fig = plt.figure(figsize=(10.6, 4.6)); gs = fig.add_gridspec(3, n_show + 2, width_ratios=[2.3, 0.35] + [1] * n_show, wspace=0.03, hspace=0.05)
    ax = fig.add_subplot(gs[:, 0]); divs = (64, 32, 16, 8, 4, 2); xs = np.arange(len(divs))
    for key, col, lab in (("psnr_fold", BLUE, "freed with the fold"), ("psnr_zero", ORANGE, "zeroed, no fold")):
        V = np.array([[r["k"][str(max(1, r["channels"] // d))][key] for d in divs] for r in ref["layers"]])
        ax.fill_between(xs, V.min(0), V.max(0), color=col, alpha=0.15, lw=0); ax.plot(xs, np.median(V, 0), color=col, lw=2, marker="o", ms=4, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"1/{d}" for d in divs], fontsize=7.5); ax.set_xlabel("share of one layer's channels freed", fontsize=8)
    ax.set_ylabel("PSNR against the unedited generator (dB)", fontsize=8); ax.tick_params(labelsize=7.5); ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right"); ax.set_title("All 11 layers: median and range", loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    show_imgs = {"base": sr2(x0[:n_show]), "fold": sr2(xf[:n_show]), "zero": sr2(xz[:n_show])}
    labels = [("base", "Base", INK), ("zero", f"Zeroed\n{pz:.1f} dB", ORANGE), ("fold", f"Fold\n{pf:.1f} dB", BLUE)]
    for r, (key, lab, col) in enumerate(labels):
        for c in range(n_show):
            a = fig.add_subplot(gs[r, c + 2]); show(a, show_imgs[key][c])
            if c == 0: a.set_ylabel(lab, color=col, fontsize=8.5, fontweight="bold", rotation=0, ha="right", va="center", labelpad=6)
    fig.text(0.62, 0.95, f"{layer}: {k} of its {C} channels freed (1/{share})", ha="center", fontsize=9.5, fontweight="bold")
    save(fig, "fig_room.png")

# ---------------------------------------------------------------------------------------------
def fig_union(idx=None, crop=176):
    import torch, exp22_union_router as e22
    ctx = e22.main(fit_only=True); models, experts, route, test = ctx["models"], ctx["experts"], ctx["route"], ctx["test"]
    idx = 7 if idx is None else idx; im = test[idx]; hr = np.asarray(im, np.float32) / 255
    names = {"base": "Base TinySR-26", "branch: bicubic": "Bicubic branch", "branch: noisy": "Noisy branch"}
    rows = {"bicubic LR": "Bicubic input", "noisy bicubic LR": "Noisy input", "nearest LR": "Nearest input"}
    H, Wd = hr.shape[:2]; y0, x0 = (H - crop) // 2 - 40, (Wd - crop) // 2
    cr = lambda a: a[y0:y0 + crop, x0:x0 + crop]
    fig, axs = plt.subplots(3, 5, figsize=(10.6, 6.75), gridspec_kw={"wspace": 0.04, "hspace": 0.16})
    psnr = lambda y: float(10 * np.log10(1 / np.mean((y - hr) ** 2)))
    with torch.no_grad():
        for r, cond in enumerate(e22.CONDS):
            x = e22.lr_inputs([im], cond)[0]; ys = [models[n](x).clamp(0, 1) for n in experts]; k = route("self-consistency", x, ys)
            lr = x[0].permute(1, 2, 0).cpu().numpy(); show(axs[r, 0], cr(lr.repeat(2, 0).repeat(2, 1)))
            axs[r, 0].set_ylabel(rows[cond], fontsize=9, fontweight="bold", color=INK)
            if r == 0: axs[r, 0].set_title("Input (2× nearest)")
            pb = psnr(ys[0][0].permute(1, 2, 0).cpu().numpy())
            for j, n in enumerate(experts):
                y = ys[j][0].permute(1, 2, 0).cpu().numpy(); ax = axs[r, j + 1]; show(ax, cr(y)); p = psnr(y)
                if r == 0: ax.set_title(names[n])
                ax.text(0.03, 0.03, f"{p:.2f} dB" + ("" if j == 0 else f" ({p - pb:+.2f})"), transform=ax.transAxes, color="white", fontsize=7.5,
                        bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=1.5))
                if j == k:
                    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=BLUE, lw=4, clip_on=False))
                    ax.text(0.97, 0.97, "router's choice", transform=ax.transAxes, color="white", fontsize=7.5, ha="right", va="top",
                            bbox=dict(facecolor=BLUE, lw=0, pad=1.5))
            show(axs[r, 4], cr(hr))
            if r == 0: axs[r, 4].set_title("Ground truth")
            print(f"[union] {cond}: router picks {experts[k]}", flush=True)
    save(fig, "fig_union.png")

# ---------------------------------------------------------------------------------------------
def fig_exception():
    from PIL import Image
    import exp23_exception_patch as e23
    ctx = e23.main(write=False); L, Lv, yv, Fv, lab = ctx["L"], ctx["Lv"], ctx["yv"], ctx["Fv"], ctx["labels"]
    Lp, fired = ctx["patched"](Fv, Lv); name = lambda c: lab[c].split(",")[0].replace("_", " ")
    paths = L[("val", "cat")] + L[("val", "dog")] + L[("val", "wild")]
    bad = np.nonzero((yv == 0) & ~e23.is_cat(Lv))[0]
    changed = np.nonzero((yv != 0) & (Lv.argmax(1) != Lp.argmax(1)))[0]
    show_idx = list(bad) + list(changed); ncol = 6; nrow = int(np.ceil(len(show_idx) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(10.6, 2.02 * nrow), gridspec_kw={"wspace": 0.05, "hspace": 0.42})
    for a in axs.flat: a.axis("off")
    for a, i in zip(axs.flat, show_idx):
        img = np.asarray(Image.open(paths[i]).convert("RGB").resize((224, 224), Image.LANCZOS)); show(a, img); a.axis("on")
        b, p = name(Lv[i].argmax()), name(Lp[i].argmax()); fixed = e23.is_cat(Lp[i:i + 1])[0]
        if yv[i] != 0: col, txt = ORANGE, f"{b} → {p}\n(not a cat: changed)"
        elif fixed: col, txt = BLUE, f"{b} → {p}"
        else: col, txt = RED, f"{b} → {p}\n(unit did not fire)" if not fired[i] else f"{b} → {p}"
        a.set_title(txt, fontsize=7.3, color=col, fontweight="bold" if col != BLUE else "normal", pad=3)
        if col != BLUE: a.add_patch(Rectangle((0, 0), 1, 1, transform=a.transAxes, fill=False, ec=col, lw=3, clip_on=False))
    print(f"[exception] {len(bad)} cat errors, {int(e23.is_cat(Lp[bad]).sum())} fixed; {len(changed)} other prediction(s) changed", flush=True)
    save(fig, "fig_exception.png")

FIGS = {"hero": fig_hero, "identities": fig_identities, "room": fig_room, "union": fig_union, "exception": fig_exception}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True)
    for f in (sys.argv[1:] or list(FIGS)):                       # hero=SEED draws other cats for Figure 1
        t0 = time.time(); name, _, arg = f.partition("="); FIGS[name](*([int(arg)] if arg else [])); print(f"{f}: {time.time() - t0:.0f}s", flush=True)
