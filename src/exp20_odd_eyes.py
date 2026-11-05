"""
Experiment 20: compiling position and Boolean logic into a trained GAN (odd-eyed cats).

CatGAN-26's convolutions cannot tell left from right; only its dense layer, whose bias is set per
4x4 cell, can hold a spatial pattern. We give the network a coordinate wire and use it in logic:
  1. free one channel in each of the ten layers from the dense layer to the last hidden conv
     but one (circuits.choose_free / fold_free, refitting after each layer);
  2. wire a carrier through them: the dense layer's freed channel holds RIGHT = 1 on the right
     half of the 4x4 grid and 0 on the left (its bias; zero kernel), and each conv copies it with
     a centre-tap 1. RIGHT >= 0, so `carry` passes it through every LeakyReLU exactly;
  3. free two channels of the last hidden conv and install the exp4/exp13 iris gate there, with
     the carrier entering the gate's pre-activation: relu(p - t - M * RIGHT) fires only on the
     left half, relu(p - t - M * (1 - RIGHT)) only on the right (`gate` = AND).
The gate reads the carrier at the centre tap and the output conv reads the gate at the centre
tap, so the left-eye patch cannot change any pixel in the right half of any image, by
construction. We check that bit for bit.

  <venv python> src/exp20_odd_eyes.py
"""
import json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits as cc, cat_domain as cd, exp4_anime_eye_circuit as e4
from exp18_circuit_primitives import catgan_layers

OUT = "results/exp20_out"; HUE_BAND = (200, 250); A = 0.2

def forward(z, W, L, keep=None, chunk=50, double=False):
    """GPU forward (float32, or float64 for exactness checks). keep: producer whose post-activation output is also returned (NHWC)."""
    import torch, torch.nn.functional as F
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    dt = torch.float64 if double else torch.float32
    T = {k: (torch.from_numpy(v).permute(3, 2, 0, 1).contiguous() if v.ndim == 4 else torch.from_numpy(v)).to(dt).cuda() for k, v in W.items()}
    outs, kept = [], []
    with torch.no_grad():
        for s in range(0, len(z), chunk):
            x = torch.from_numpy(np.asarray(z[s:s + chunk], np.float64)).to(dt).cuda(); last = None
            for l in L:
                if l[0] == "dense": x = (x @ T["dense/kernel"] + T["dense/bias"]).view(len(x), 4, 4, -1).permute(0, 3, 1, 2); last = "dense"
                elif l[0] == "lrelu":
                    x = F.leaky_relu(x, A)
                    if keep is not None and last == keep: kept.append(x.permute(0, 2, 3, 1).cpu().numpy())
                elif l[0] == "up": x = F.interpolate(x, scale_factor=2, mode="nearest")
                else: x = F.conv2d(x, T[l[1] + "/kernel"], T[l[1] + "/bias"], padding=1); last = l[1]
            outs.append(x.permute(0, 2, 3, 1).cpu().numpy())
    out = np.concatenate(outs)
    return (out, np.concatenate(kept)) if keep is not None else out

def gram_of(z, W, L, producer, chunk=50):
    G = None
    for s in range(0, len(z), chunk):
        _, h = forward(z[s:s + chunk], W, L, keep=producer); X = h.reshape(-1, h.shape[-1]).astype(np.float64)
        Xb = np.c_[X, np.ones(len(X))]; G = Xb.T @ Xb if G is None else G + Xb.T @ Xb
    return G

def free(W, L, z, producer, consumer, k, cells=None):
    G = gram_of(z, W, L, producer); drop, keep, B, _ = cc.choose_free(G, cc.out_energy(W, consumer), k)
    return cc.fold_free(W, producer, consumer, drop, keep, B, dense_cells=cells), drop

def grow(W, producer, consumer, k, cells=None):
    """Add k channels to `producer` with zero weights and zero outgoing weights: the model is unchanged."""
    W = {k_: v.copy() for k_, v in W.items()}
    if cells is None:
        K = W[producer + "/kernel"]; C = K.shape[3]
        W[producer + "/kernel"] = np.concatenate([K, np.zeros(K.shape[:3] + (k,), np.float32)], 3)
        W[producer + "/bias"] = np.r_[W[producer + "/bias"], np.zeros(k, np.float32)]
    else:                                                          # dense reshaped to (cells, C): append per cell
        K = W[producer + "/kernel"]; C = K.shape[1] // cells
        W[producer + "/kernel"] = np.concatenate([K.reshape(-1, cells, C), np.zeros((K.shape[0], cells, k), np.float32)], 2).reshape(K.shape[0], -1)
        W[producer + "/bias"] = np.concatenate([W[producer + "/bias"].reshape(cells, C), np.zeros((cells, k), np.float32)], 1).reshape(-1)
    Kc = W[consumer + "/kernel"]; W[consumer + "/kernel"] = np.concatenate([Kc, np.zeros(Kc.shape[:2] + (k, Kc.shape[3]), np.float32)], 2)
    return W, list(range(C, C + k))

def hue_of(x, m):
    ok = m.sum((1, 2)) >= 8
    ec = np.array([x[i][m[i]].mean(0) for i in range(len(x)) if ok[i]])
    return float(np.median(e4.hue(ec))), int(ok.sum())

def per_image_hue(x, m):
    return [float(e4.hue(x[i][m[i]].mean(0)[None])[0]) if m[i].sum() >= 8 else None for i in range(len(x))]

def psnr(a, b):
    m = np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2)
    return float("inf") if m == 0 else float(10 * np.log10(1 / m))

def u8(x): return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True)
    for room in ("free", "grow"): run_room(room)

def run_room(room, write=True, W0=None):
    """Build and measure the patches for one kind of room. write=False skips the result files and
    returns the patched networks and images (for make_figures_v2.py and exp24)."""
    t_start = time.time(); print(f"=== room by {room} ===", flush=True)
    if W0 is None: W0 = {k: v.astype(np.float32) for k, v in np.load("results/exp11_out/catgan26.npz").items()}
    L = catgan_layers(W0)                                            # W0: another base, e.g. a build with latent locks (exp24)
    rng = np.random.default_rng(0); z_free, z_fit, z_test = (rng.standard_normal((n, 128)) for n in (256, 300, 300))
    names = [l[1] for l in L if l[0] in ("dense", "conv")]           # dense, b1c1 ... b5c2, out
    chain = names[:names.index("b5c1") + 1]                          # producers that carry the wire
    R = {"chain": chain, "room_mode": room}

    # 1. make room: one channel per layer along the chain ----------------------------------------
    W = W0; slots = []
    for prod in chain:
        cons = names[names.index(prod) + 1]
        W, drop = (free if room == "free" else lambda W_, L_, z_, p_, c_, k_, cells: grow(W_, p_, c_, k_, cells))(W, L, z_free, prod, cons, 1, cells=16 if prod == "dense" else None)
        slots.append(drop[0])
    x_base = forward(z_test, W0, L); x_room = forward(z_test, W, L)
    R["room"] = {"slots": dict(zip(chain, map(int, slots))), "psnr_vs_base": psnr(x_room, x_base)}
    print(f"[1] freed one channel in {len(chain)} layers: PSNR vs base {R['room']['psnr_vs_base']:.1f} dB ({time.time() - t_start:.0f}s)", flush=True)

    # 2. the coordinate wire -------------------------------------------------------------------
    C0 = W["dense/bias"].shape[0] // 16; right = np.array([[0, 0, 1, 1]] * 4, np.float32).reshape(-1)   # cell = h*4 + w
    W["dense/bias"][slots[0] + C0 * np.arange(16)] = right
    for prev, prod, s in zip(slots[:-1], chain[1:], slots[1:]):
        W[prod + "/kernel"][..., s] = 0; W[prod + "/kernel"][1, 1, prev, s] = 1.0; W[prod + "/bias"][s] = 0
    x_wire, h8 = forward(z_test[:8], W, L, keep="b5c1")
    want = np.zeros((128, 128), np.float32); want[:, 64:] = 1
    R["wire"] = {"carrier_max_err": float(np.abs(h8[..., slots[-1]] - want).max()),
                 "output_change_vs_room": float(np.abs(x_wire - x_room[:8]).max())}
    print(f"[2] wire: carrier error {R['wire']['carrier_max_err']:.1e}, output change {R['wire']['output_change_vs_room']:.1e}", flush=True)

    # 3. room for the gate, then the gates ------------------------------------------------------
    W, gslots = free(W, L, z_free, "b5c2", "out", 2) if room == "free" else grow(W, "b5c2", "out", 2)
    x_fit, h8 = forward(z_fit, W, L, keep="b5c1"); x_fit0 = forward(z_fit, W0, L); x_test0 = x_base
    m_fit, m_test = cd.eye_mask(np.clip(x_fit0, 0, 1)), cd.eye_mask(np.clip(x_test0, 0, 1))
    left = np.zeros((128, 128), bool); left[:, :64] = True
    h8f = h8.copy(); h8f[..., slots[-1]] = 0                          # the detector does not see the wire
    pre_out = x_fit
    kern, kb, _, pos, neg = e4.fit_gate(h8f, m_fit, 50000, rng, 0.95)
    probe = lambda idx: np.c_[e4.patches3x3(h8f, idx), np.ones(len(idx))] @ np.r_[kern.reshape(-1), kb]
    pp, pn = probe(pos), probe(neg); Tgt = np.clip(pre_out.reshape(-1, 3)[pos], 0, 1) @ cd.TARGET.T - pre_out.reshape(-1, 3)[pos]
    M = float(4 * (max(pp.max(), pn.max()) - min(pp.min(), pn.min())))
    def install(th, mode):
        g = np.maximum(pp - th, 0); v = ((g @ Tgt) / (g @ g)).astype(np.float32); k_ = kern.copy(); b_ = kb
        if mode == "left": k_[1, 1, slots[-1]] -= M                  # p - t - M * RIGHT
        if mode == "right": k_[1, 1, slots[-1]] += M; b_ = kb - M    # p - t - M * (1 - RIGHT)
        return e4.install_gate(W, gslots, k_, b_, th, v, A, l9="b5c2", l10="out")
    eye = {"left": m_test & left[None], "right": m_test & ~left[None]}; eyef = {"left": m_fit & left[None], "right": m_fit & ~left[None]}
    R["gates"] = {}; imgs = {"base": x_test0, "room (no gate)": forward(z_test, W, L)}; nets = {"room (no gate)": W}
    for mode in ("both", "left", "right"):
        fit_mask = m_fit if mode == "both" else eyef[mode]
        sweep = {}
        for q in (0.9, 0.95, 0.97, 0.98, 0.99):
            th = float(np.quantile(pn, q)); sweep[q] = hue_of(forward(z_fit, install(th, mode), L), fit_mask)[0]
        ok = [q for q in sweep if HUE_BAND[0] <= sweep[q] <= HUE_BAND[1]]; q = max(ok) if ok else 0.95
        Wm = install(float(np.quantile(pn, q)), mode); x = forward(z_test, Wm, L); imgs[mode] = x; nets[mode] = Wm
        far = ~np.any([np.roll(np.roll(m_test, dy, 1), dx, 2) for dy in range(-3, 4) for dx in range(-3, 4)], 0)
        r = {"quantile": q, "fit_hue_sweep": sweep,
             "left_eye_hue": hue_of(x, eye["left"]), "right_eye_hue": hue_of(x, eye["right"]),
             "rest_mae_vs_base": float(np.abs(np.clip(x, 0, 1) - np.clip(x_test0, 0, 1))[far].mean()),
             "max_change_right_half_vs_room": float(np.abs(x[:, :, 64:] - imgs["room (no gate)"][:, :, 64:]).max()),
             "max_change_left_half_vs_room": float(np.abs(x[:, :, :64] - imgs["room (no gate)"][:, :, :64]).max()),
             "weights_changed_or_added": int(sum(int((Wm[k] != W0[k]).sum()) if Wm[k].shape == W0[k].shape else Wm[k].size for k in W0)),
             "params": int(sum(v.size for v in Wm.values())), "params_base": int(sum(v.size for v in W0.values()))}
        hl, hb = per_image_hue(x, eye["left"]), per_image_hue(x_test0, eye["left"]); hr_, hbr = per_image_hue(x, eye["right"]), per_image_hue(x_test0, eye["right"])
        r["share_left_in_band"] = float(np.mean([HUE_BAND[0] <= h <= HUE_BAND[1] for h in hl if h is not None]))
        r["share_right_in_band"] = float(np.mean([HUE_BAND[0] <= h <= HUE_BAND[1] for h in hr_ if h is not None]))
        if mode == "left":                                           # float64: is the right half untouched?
            a64 = forward(z_test[:16], Wm, L, double=True); b64 = forward(z_test[:16], W, L, double=True)
            r["float64_max_change_right_half"] = float(np.abs(a64[:, :, 64:] - b64[:, :, 64:]).max())
            r["float64_max_change_left_half"] = float(np.abs(a64[:, :, :64] - b64[:, :, :64]).max())
        R["gates"][mode] = r; print(f"[3] {mode}: {json.dumps({k: v for k, v in r.items() if k != 'fit_hue_sweep'})}", flush=True)
    R["base_hues"] = {"left": hue_of(x_test0, eye["left"]), "right": hue_of(x_test0, eye["right"])}
    R["seconds_total"] = time.time() - t_start
    if not write: return {"R": R, "nets": nets, "imgs": imgs, "L": L, "z_test": z_test, "eye": eye, "W0": W0}
    json.dump(R, open(os.path.join(OUT, f"exp20_results_{room}.json"), "w"), indent=1)

    # 4. figure: the same latents through base / left-only / right-only / both ------------------
    fires = np.abs(imgs["left"] - imgs["room (no gate)"])[:, 56:74, 36:58].max((1, 2, 3)) > 0.08
    pick = list(np.nonzero(fires)[0][:6]) or list(range(6))
    rows = [("base", "base"), ("left", "left eye only"), ("right", "right eye only"), ("both", "both eyes")]
    s = 256; canvas = Image.new("RGB", (len(pick) * s, len(rows) * s), "white")
    for r_, (k, _) in enumerate(rows):
        for c, i in enumerate(pick): canvas.paste(Image.fromarray(u8(imgs[k][i])).resize((s, s), Image.LANCZOS), (c * s, r_ * s))
    canvas.save(os.path.join(OUT, f"fig_odd_eyes_{room}.png")); R["figure_latents"] = [int(i) for i in pick]
    json.dump(R, open(os.path.join(OUT, f"exp20_results_{room}.json"), "w"), indent=1)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
