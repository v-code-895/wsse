# Experiment 17: rebasing the cat repo onto an upstream update

Downstream repo: `exp5_cat_subspace_out/weightrepo` (base `be2b16ff52fc4021`). Upstream: CatGAN-26 trained further from step 30000.

| upstream | relative weight change | output MAE | attribute correlation with the original (same latents) | stale sparse patches |
|---|---|---|---|---|
| +500 steps | 0.0367 | 0.0448 | fur_warmth 0.99, fur_lightness 0.98, background 0.97, eye_green 0.98, saturation 0.99 | prune2 (0.03), eye_gate (0.03) |
| +1000 steps | 0.0654 | 0.0668 | fur_warmth 0.98, fur_lightness 0.96, background 0.94, eye_green 0.95, saturation 0.99 | prune2 (0.07), eye_gate (0.07) |
| +2000 steps | 0.1114 | 0.0941 | fur_warmth 0.93, fur_lightness 0.95, background 0.89, eye_green 0.90, saturation 0.92 | prune2 (0.13), eye_gate (0.13) |
| +4000 steps | 0.1786 | 0.1268 | fur_warmth 0.89, fur_lightness 0.85, background 0.83, eye_green 0.81, saturation 0.87 | prune2 (0.20), eye_gate (0.20) |

In-context tests (`wpgit test`), value and result:

| patch | test | bound | original | replay +500 | replay +1000 | replay +2000 | replay +4000 | regenerated +4000 |
|---|---|---|---|---|---|---|---|---|
| prune2 | rest-of-image MAE | max=0.015 | 0.003 PASS | 0.006 PASS | 0.009 PASS | 0.024 FAIL | 0.025 FAIL | 0.008 PASS |
| eye_gate | eye hue (deg) | min=200 max=250 | 224.635 PASS | 226.984 PASS | 229.120 PASS | 227.802 PASS | 225.395 PASS | 25.975 FAIL |
| eye_gate | rest-of-image MAE | max=0.015 | 0.000 PASS | 0.005 PASS | 0.009 PASS | 0.024 FAIL | 0.025 FAIL | 0.004 PASS |
| lock_background | var ratio background vs base | max=0.3 | 0.114 PASS | 0.109 PASS | 0.127 PASS | 0.134 PASS | 0.294 PASS | 0.084 PASS |
| lock_fur_lightness | var ratio fur_lightness vs base | max=0.3 | 0.101 PASS | 0.104 PASS | 0.112 PASS | 0.154 PASS | 0.161 PASS | 0.127 PASS |

Regenerated on +4000 in 11 min: subspace lock ranks fur_lightness 2 (var ratio 0.17), background 4 (var ratio 0.15).
