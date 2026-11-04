# Experiment 14: git for fine-tunes (TinySR-26 x2)

Base `8cb6601c1b449ffc`. Two branches fine-tuned independently for 3,000 steps each; patches are weight deltas. cos(d_bicubic, d_noisy) = +0.088; relative size of each delta: bicubic 0.1635, noisy 0.1667. Merge order (A then B vs B then A): max weight difference 1.5e-08.

PSNR (dB) on held-out AFHQ, 2x (256 -> 512), 180 images; noise σ = 8/255; Δ vs base in brackets.

| model | bicubic LR | noisy bicubic LR | nearest LR |
|---|---|---|---|
| base | 32.53 | 26.89 | 31.12 |
| branch: bicubic | 32.76 (+0.23) | 26.42 (-0.47) | 26.24 (-4.88) |
| branch: noisy | 32.32 (-0.20) | 30.15 (+3.26) | 30.11 (-1.01) |
| merge: base + d_bicubic + d_noisy | 32.49 (-0.03) | 29.45 (+2.56) | 27.06 (-4.06) |
| merge: base + (d_bicubic + d_noisy) / 2 | 32.64 (+0.11) | 28.78 (+1.89) | 28.08 (-3.05) |
