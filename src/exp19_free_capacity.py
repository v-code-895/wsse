"""
Experiment 19: how much room is there inside a trained network?

For every hidden layer of CatGAN-26 (the dense layer and the ten convs), free k channels by
regressing them onto the layer's other channels and folding the regression into the next layer
(circuits.choose_free / fold_free). Fit on 512 latents, measure on 256 held-out latents: PSNR of
the patched generator's images against the base generator's, so infinity means unchanged. The
baseline frees the same channels by zeroing them without the fold (plain pruning).

  <venv python> src/exp19_free_capacity.py
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits as cc
from exp18_circuit_primitives import catgan_layers, run_layers

OUT = "results/exp19_out"

def producer_consumer(L):
    names = [l[1] for l in L if l[0] in ("dense", "conv")]
    return list(zip(names[:-1], names[1:]))

def acts_after(z, W, L, producer, a=0.2, chunk=32):
    """Yield (N*H*W, C) float64 batches of the producer's post-activation outputs."""
    import torch, torch.nn.functional as F
    T = {k: (torch.from_numpy(v).permute(3, 2, 0, 1).contiguous() if v.ndim == 4 else torch.from_numpy(v)).float().cuda() for k, v in W.items()}
    stop = next(i for i, l in enumerate(L) if len(l) > 1 and l[1] == producer) + 1      # the lrelu after the producer
    with torch.no_grad():
        for s in range(0, len(z), chunk):
            x = torch.from_numpy(z[s:s + chunk]).float().cuda()
            for l in L[:stop + 1]:
                if l[0] == "dense": x = (x @ T["dense/kernel"] + T["dense/bias"]).view(len(x), 4, 4, -1).permute(0, 3, 1, 2)
                elif l[0] == "lrelu": x = F.leaky_relu(x, a)
                elif l[0] == "up": x = F.interpolate(x, scale_factor=2, mode="nearest")
                else: x = F.conv2d(x, T[l[1] + "/kernel"], T[l[1] + "/bias"], padding=1)
            yield x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).double().cpu().numpy()

def fold_and_zero(W, producer, consumer, G, drop, cells=None):
    """Free `drop` by the ridge regression fold, and (the baseline) by zeroing without a fold."""
    C = G.shape[0] - 1; keep = [c for c in range(C) if c not in drop]; idx = keep + [C]
    lam = 1e-6 * np.trace(G[:C, :C]) / C
    B = np.linalg.solve(G[np.ix_(idx, idx)] + lam * np.diag([1.0] * len(keep) + [0.0]), G[np.ix_(idx, drop)])
    return (cc.fold_free(W, producer, consumer, drop, keep, B, dense_cells=cells),
            cc.fold_free(W, producer, consumer, drop, keep, np.zeros_like(B), dense_cells=cells))

def psnr(a, b):
    m = np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2)
    return float("inf") if m == 0 else float(10 * np.log10(1 / m))

def main():
    sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(OUT, exist_ok=True)
    W = {k: v.astype(np.float32) for k, v in np.load("results/exp11_out/catgan26.npz").items()}; L = catgan_layers(W)
    rng = np.random.default_rng(0); z_fit, z_test = rng.standard_normal((512, 128)), rng.standard_normal((256, 128))
    x0 = run_layers(z_test, W, L); R = {"layers": []}
    for prod, cons in producer_consumer(L):
        t0 = time.time()
        G = cc.gram(acts_after(z_fit, W, L, prod)); C = G.shape[0] - 1
        E = cc.out_energy(W, cons); ks = sorted({max(1, C // d) for d in (64, 32, 16, 8, 4, 2)})
        drop_all, _, _, _ = cc.choose_free(G, E, max(ks))
        cells = 16 if prod == "dense" else None; row = {"layer": prod, "channels": C, "k": {}}
        for k in ks:
            Wf, Wz = fold_and_zero(W, prod, cons, G, drop_all[:k], cells)
            row["k"][k] = {"psnr_fold": psnr(run_layers(z_test, Wf, L), x0), "psnr_zero": psnr(run_layers(z_test, Wz, L), x0)}
        R["layers"].append(row); json.dump(R, open(os.path.join(OUT, "exp19_results.json"), "w"), indent=1)   # save after every layer
        print(f"{prod:6s} C={C:3d} " + "  ".join(f"k={k}: {v['psnr_fold']:.1f}/{v['psnr_zero']:.1f}" for k, v in row["k"].items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(R, open(os.path.join(OUT, "exp19_results.json"), "w"), indent=1)
    fr = ["1/64", "1/32", "1/16", "1/8", "1/4", "1/2"]
    Lm = ["# Experiment 19: room inside CatGAN-26", "",
          "PSNR (dB) of the generator's images against the unedited generator's, on 256 held-out latents, after freeing k channels of one layer.",
          "Each cell: **regression fold** / zeroed without a fold. Fit on 512 other latents.", "",
          "| layer | channels | " + " | ".join(fr) + " |", "|---|---|" + "---|" * len(fr)]
    for row in R["layers"]:
        C = row["channels"]; cells = []
        for d in (64, 32, 16, 8, 4, 2):
            k = max(1, C // d); v = row["k"][k]; cells.append(f"k={k}: **{v['psnr_fold']:.1f}** / {v['psnr_zero']:.1f}")
        Lm.append(f"| {row['layer']} | {C} | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "exp19_results.md"), "w", encoding="utf-8").write("\n".join(Lm) + "\n"); print("\n".join(Lm))

if __name__ == "__main__":
    main()
