"""
Experiment 8b: one model's weights as a unit test for another model's edits.

The MobileNetV2 + 'anime face' class from exp8 scores every Anime DCGAN edit from
exp3/exp4/exp7d: does the edited generator still produce anime faces, and what
does ImageNet think each style looks like?

  python src/exp8b_oracle.py
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobilenet_np as mn
import exp8_mobilenet as e8
import exp7d_anime_styles as st
from exp2b_tinysr18_weightpatch import load_model
from exp3_anime_weightpatch import generate
import exp3_anime_weightpatch as e3, exp4_anime_eye_circuit as e4

OUT = "results/exp8_out"

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    W = mn.load(); lab = mn.labels() + ["ANIME_FACE (new)"]
    # rebuild the exp8 head (Fisher version) exactly as exp8 fits it
    I = e8.base_images(); names = list(I)
    neg = sorted(set(names)); np.random.default_rng(0).shuffle(neg); fit_n = set(neg[:10])
    from exp2b_tinysr18_weightpatch import make_sprite
    negs = [c for n in sorted(fit_n) for c in e8.crops(I[n])] + \
           [c for n in sorted(fit_n) for c in e8.crops(I[n], 9, seed=11)[1:]] + [make_sprite(s) for s in range(6)]
    Fp = e8.feats_logits(e8.anime_faces(24, seed=100), W)[1]; Fn = e8.feats_logits(negs, W)[1]
    w, w0 = e8.class_direction(Fp, Fn, "fisher"); W2, _ = e8.add_class(W, w, w0, Fp, Fn)

    Wa, oa, _ = load_model("legacy/Anime-DCGAN/generator50000.h5")
    z = np.random.default_rng(300).standard_normal((16, 64))
    to_u8 = lambda x: [(np.clip(i, 0, 1) * 255).astype(np.uint8) for i in x]
    V = {"base": generate(z, Wa, oa), "mirror (exact)": generate(z, st.mirror(Wa, oa), oa),
         "rotate 90 (exact)": generate(z, st.rotate90(Wa, oa), oa),
         "pencil sketch": generate(z, st.sketch(Wa, oa, 6.0), oa),
         "duotone": generate(z, st.colour_fold(Wa, oa, *st.duotone((30, 10, 70), (120, 255, 230))), oa),
         "sepia": generate(z, st.colour_fold(Wa, oa, st.SEPIA, np.zeros(3)), oa)}
    p3 = json.load(open("results/exp3_out/patches/shift_hair_warmth_orth.weightpatch.json"))
    V["warm hair (exp3 patch)"] = generate(z, e3.apply_weightpatch(Wa, p3, oa), oa)
    pr = json.load(open("results/exp4_out/patches/prune2.weightpatch.json")); pg = json.load(open("results/exp4_out/patches/eye_gate.weightpatch.json"))
    V["magenta eyes (exp4 patches)"] = generate(z, e4.apply_weightpatch(e4.apply_weightpatch(Wa, pr), pg), oa)
    V["latent x3 (off-distribution input)"] = generate(3 * z, Wa, oa)
    V["random weights (control)"] = generate(z, {k: np.random.default_rng(1).normal(0, v.std(), v.shape).astype(np.float32) for k, v in Wa.items()}, oa)

    rows = {}
    for k, x in V.items():
        L = e8.feats_logits(to_u8(x), W2)[0]; top = L.argmax(1)
        u, c = np.unique(top, return_counts=True); common = sorted(zip(c, u), reverse=True)[:3]
        rows[k] = {"anime_face_rate": float((top == 1000).mean()),
                   "most_common_label": ", ".join(f"{lab[i]} ({n})" for n, i in common)}
    json.dump(rows, open(os.path.join(OUT, "oracle.json"), "w"), indent=1)
    md = ["# Experiment 8b: MobileNetV2 + new 'anime face' class as a test oracle for generator edits\n",
          "16 faces per variant (same latents). The oracle was fit only on unedited generated faces.\n",
          "| generator variant | classified as anime face | top labels (count) |", "|---|---|---|"]
    md += [f"| {k} | {v['anime_face_rate']:.0%} | {v['most_common_label']} |" for k, v in rows.items()]
    open(os.path.join(OUT, "oracle.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
