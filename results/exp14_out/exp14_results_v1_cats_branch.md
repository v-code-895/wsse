# Experiment 14: git for fine-tunes (TinySR-26 x2)

Base `8cb6601c1b449ffc`. Two branches fine-tuned independently for 3,000 steps each; patches are weight deltas. cos(d_cats, d_noisy) = +0.080; relative size of each delta: cats 0.0525, noisy 0.1683. Merge order (A then B vs B then A): max weight difference 7.5e-09.

PSNR (dB) on held-out AFHQ, 2x (128 -> 256), 60 images per class; Δ vs base in brackets.

| model | cats, clean | all, noisy LR | dogs + wildlife, clean |
|---|---|---|---|
| base | 30.99 | 26.45 | 30.70 |
| branch: cats | 31.08 (+0.08) | 26.33 (-0.12) | 30.85 (+0.14) |
| branch: noisy | 31.88 (+0.88) | 29.75 (+3.31) | 31.80 (+1.09) |
| merge: base + d_cats + d_noisy | 31.88 (+0.89) | 29.76 (+3.31) | 31.79 (+1.08) |
| merge: base + (d_cats + d_noisy) / 2 | 32.01 (+1.02) | 27.91 (+1.46) | 31.93 (+1.23) |
