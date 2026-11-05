"""
Experiment 18: an instruction set for rewriting trained networks, tested for exactness.

1. Every identity in circuits.py is compiled into a real one-hidden-layer network (LeakyReLU 0.2,
   as in CatGAN-26 and TinySR-18A, and ReLU, as in TinySR-26) and compared with the function it
   claims to compute on 100,000 random inputs, in float64 and float32.
2. Two topology edits on CatGAN-26 that must not change the model at all:
   - width growth: add 16 channels to every hidden conv, with arbitrary incoming weights and zero
     outgoing weights;
   - depth growth: insert a new conv + LeakyReLU layer after every hidden conv, wired as an
     identity with the two-channel `pass` identity (so the network gets 10 layers deeper).

  <venv python> src/exp18_circuit_primitives.py
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits as cc

OUT = "results/exp18_out"
NAMES = ["carry", "pass", "relu", "abs", "min", "max", "clamp", "step", "and", "or", "gate", "mux"]
CHANNELS = {n: cc.compile_recipe(n, 0.2)[0].shape[1] for n in NAMES}

def test_identities(n=100_000, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for name in NAMES:
        for a in (0.2, 0.0):
            W1, b1, w2, c, fn, ins = cc.compile_recipe(name, a)
            X = cc.sample_inputs(name, n, rng); Xm = np.stack([X[k] for k in ins], 1)
            want = fn(*[X[k] for k in ins])
            err64 = float(np.abs(cc.lrelu(Xm @ W1 + b1, a) @ w2 + c - want).max())
            f = np.float32
            got32 = cc.lrelu(Xm.astype(f) @ W1.astype(f) + b1.astype(f), f(a)) @ w2.astype(f) + f(c)
            err32 = float(np.abs(got32 - want).max())
            rows.append({"identity": name, "slope": a, "hidden_channels": int(W1.shape[1]), "max_err_f64": err64,
                         "max_err_f32": err32, "scale": float(np.abs(want).max())})
    return rows

# ---- CatGAN-26 as a layer list, so topology can change -----------------------------------------
def catgan_layers(W):
    L = [("dense", "dense"), ("lrelu",)]
    for i in sorted({int(k[1:k.index("c")]) for k in W if k.startswith("b") and "/kernel" in k}):
        L += [("up",), ("conv", f"b{i}c1"), ("lrelu",), ("conv", f"b{i}c2"), ("lrelu",)]
    return L + [("conv", "out")]

def run_layers(z, W, L, a=0.2, chunk=64):
    import torch, torch.nn.functional as F
    T = {k: (torch.from_numpy(v).permute(3, 2, 0, 1).contiguous() if v.ndim == 4 else torch.from_numpy(v)).double().cuda()
         for k, v in W.items()}
    out = []
    with torch.no_grad():
        for s in range(0, len(z), chunk):
            x = torch.from_numpy(z[s:s + chunk]).double().cuda()
            for l in L:
                if l[0] == "dense":
                    x = x @ T[l[1] + "/kernel"] + T[l[1] + "/bias"]; x = x.view(len(x), 4, 4, -1).permute(0, 3, 1, 2)
                elif l[0] == "lrelu": x = F.leaky_relu(x, a)
                elif l[0] == "up": x = F.interpolate(x, scale_factor=2, mode="nearest")
                else: x = F.conv2d(x, T[l[1] + "/kernel"], T[l[1] + "/bias"], padding=T[l[1] + "/kernel"].shape[-1] // 2)
            out.append(x.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)

def widen(W, L, extra=16, seed=0):
    """Add `extra` channels to every hidden conv: random incoming weights and bias, zero outgoing."""
    rng = np.random.default_rng(seed); Wn = {k: v.copy() for k, v in W.items()}
    convs = [l[1] for l in L if l[0] == "conv"]
    for prod, cons in zip(convs[:-1], convs[1:]):
        K = Wn[prod + "/kernel"]; Wn[prod + "/kernel"] = np.concatenate([K, rng.normal(0, 0.05, K.shape[:3] + (extra,)).astype(np.float32)], 3)
        Wn[prod + "/bias"] = np.r_[Wn[prod + "/bias"], rng.normal(0, 0.1, extra).astype(np.float32)]
        Kc = Wn[cons + "/kernel"]; Wn[cons + "/kernel"] = np.concatenate([Kc, np.zeros(Kc.shape[:2] + (extra, Kc.shape[3]), np.float32)], 2)
    return Wn

def deepen(W, L, a=0.2):
    """After every hidden conv + lrelu, insert conv(1x1 as 3x3 centre tap) + lrelu computing [h, -h], and fold
    the readout (h+ - h-)/(1+a) into the next conv. Exact identity for any sign of h."""
    Wn = {k: v.copy() for k, v in W.items()}; Ln = []; convs = [l[1] for l in L if l[0] == "conv"]
    for idx, l in enumerate(L):
        Ln.append(l)
        if l[0] == "lrelu" and idx > 1 and L[idx - 1][0] == "conv":
            prev = L[idx - 1][1]; nxt = convs[convs.index(prev) + 1]; C = Wn[prev + "/bias"].shape[0]; name = prev + "_id"
            K = np.zeros((3, 3, C, 2 * C), np.float32); K[1, 1, :, :C] = np.eye(C); K[1, 1, :, C:] = -np.eye(C)
            Wn[name + "/kernel"], Wn[name + "/bias"] = K, np.zeros(2 * C, np.float32)
            Kn = Wn[nxt + "/kernel"]; Wn[nxt + "/kernel"] = np.concatenate([Kn, -Kn], 2) / (1 + a)
            Ln += [("conv", name), ("lrelu",)]
    return Wn, Ln

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True); R = {}
    t0 = time.time(); R["identities"] = test_identities(); print(f"[1] identities in {time.time() - t0:.1f}s", flush=True)
    W = {k: v.astype(np.float32) for k, v in np.load("results/exp11_out/catgan26.npz").items()}; L = catgan_layers(W)
    z = np.random.default_rng(1).standard_normal((64, 128))
    x0 = run_layers(z, W, L)
    Ww = widen(W, L); xw = run_layers(z, Ww, L)
    Wd, Ld = deepen(W, L); xd = run_layers(z, Wd, Ld)
    n = lambda D: int(sum(v.size for v in D.values()))
    R["topology"] = {
        "widen": {"params_before": n(W), "params_after": n(Ww), "hidden_convs": 10, "channels_added_per_conv": 16,
                  "max_abs_output_change": float(np.abs(xw - x0).max())},
        "deepen": {"layers_before": sum(l[0] == "conv" for l in L) + 1, "layers_after": sum(l[0] == "conv" for l in Ld) + 1,
                   "params_after": n(Wd), "max_abs_output_change": float(np.abs(xd - x0).max())},
        "precision": "float64 forward on the GPU"}
    print(json.dumps(R["topology"], indent=1), flush=True)
    json.dump(R, open(os.path.join(OUT, "exp18_results.json"), "w"), indent=1)
    Lm = ["# Experiment 18: exact identities for rewriting trained networks", "",
          "Each identity is compiled into a one-hidden-layer network and compared with its target on 100,000 random inputs.",
          "`step` is tested outside its transition band [t, t + 1/k]; `gate` and `mux` within their big-M range.", "",
          "| identity | slope | hidden channels | max error (float64) | max error (float32) | max |target| |", "|---|---|---|---|---|---|"]
    Lm += [f"| {r['identity']} | {r['slope']} | {r['hidden_channels']} | {r['max_err_f64']:.1e} | {r['max_err_f32']:.1e} | {r['scale']:.2f} |" for r in R["identities"]]
    t = R["topology"]
    Lm += ["", "## Topology edits on CatGAN-26 that must change nothing (64 latents, float64)", "",
           f"- widen every hidden conv by 16 channels (random incoming weights, zero outgoing): {t['widen']['params_before']:,} → "
           f"{t['widen']['params_after']:,} parameters; max output change {t['widen']['max_abs_output_change']:.1e}",
           f"- insert an identity conv + LeakyReLU after every hidden conv (two-channel `pass`): {t['deepen']['layers_before']} → "
           f"{t['deepen']['layers_after']} weight layers; max output change {t['deepen']['max_abs_output_change']:.1e}"]
    open(os.path.join(OUT, "exp18_results.md"), "w", encoding="utf-8").write("\n".join(Lm) + "\n"); print("\n".join(Lm))

if __name__ == "__main__":
    main()
