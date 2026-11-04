"""
Figure 1 of the paper: CatGAN-26 under version control. The same random latents (not selected) go
through successive builds of the cat weight repository: the base model, then each patch added in
dependency order, ending at the repository's HEAD build, where every test passes. Images are
upscaled 4x by TinySR-26 for display.

  git clone results/exp5_cat_subspace_out/weightrepo.bundle results/exp5_cat_subspace_out/weightrepo
  <venv python> src/make_figure1.py            -> results/figure1/fig1_cat_repository.png
"""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wpgit
from exp13_catgan26_patches import upscale4, u8

ROWS = [("CatGAN-26 base model", []),
        ("+ prune2, eye_gate: blue irises from a gated circuit in two freed channels", ["prune2", "eye_gate"]),
        ("+ lock_fur_lightness: rank-2 latent lock (all cats light-furred)", ["prune2", "eye_gate", "lock_fur_lightness"]),
        ("+ lock_background = HEAD: all four patches, all tests pass", ["prune2", "eye_gate", "lock_fur_lightness", "lock_background"])]

def font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try: return ImageFont.truetype(f, size)
        except OSError: pass
    return ImageFont.load_default()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="results/exp5_cat_subspace_out/weightrepo")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--out", default="results/figure1/fig1_cat_repository.png")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    W0, ops, P = wpgit.load_repo(a.repo)
    z = np.random.default_rng(a.seed).standard_normal((a.n, W0[ops[0]["name"] + "/kernel"].shape[0]))
    t, pad, head = a.tile, 6, 34; F = font(20)
    canvas = Image.new("RGB", (a.n * t + (a.n - 1) * pad, len(ROWS) * (t + head)), "white"); dr = ImageDraw.Draw(canvas)
    for r, (label, ids) in enumerate(ROWS):
        W = wpgit.build(W0, ops, P, wpgit.order(P, ids)) if ids else W0
        imgs = upscale4(wpgit.DOMAIN.run(z, W, ops))
        y = r * (t + head); dr.text((2, y + 7), label, fill="black", font=F)
        for c in range(a.n):
            canvas.paste(Image.fromarray(u8(imgs[c])).resize((t, t), Image.LANCZOS), (c * (t + pad), y + head))
    canvas.save(a.out); print("wrote", a.out, canvas.size)

if __name__ == "__main__":
    main()
