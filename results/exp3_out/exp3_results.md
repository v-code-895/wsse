# Experiment 3: Anime DCGAN attribute weightpatches

Base weights `145ec232ed8a91e5`, fit n=1200, held-out n=400, seed 0. Shifts are alpha=2.0 latent units; effects in held-out attribute SDs.

## Linear predictability of each attribute from z (held-out R²)

| attribute | R² (raw dir) | R² (orthogonalised dir) |
|---|---|---|
| hair_warmth | 0.893 | 0.674 |
| hair_lightness | 0.895 | 0.160 |
| background | 0.851 | 0.332 |
| face_lightness | 0.659 | 0.497 |
| eye_teal | 0.808 | 0.693 |
| saturation | 0.088 | 0.023 |

## Entanglement: shift patch (row) -> attribute change (column), SD units

| patch \ measured | hair_warmth | hair_lightness | background | face_lightness | eye_teal | saturation |
|---|---|---|---|---|---|---|
| hair_warmth | **+1.50** | +0.32 | +0.23 | +0.18 | -0.32 | +1.24 |
| hair_lightness | +0.41 | **+1.50** | +1.21 | +0.39 | +0.35 | -0.20 |
| background | +0.31 | +1.28 | **+1.46** | +0.60 | +0.11 | -0.35 |
| face_lightness | +0.39 | +0.47 | +0.67 | **+1.37** | +0.03 | -0.30 |
| eye_teal | -0.47 | +0.30 | +0.09 | +0.09 | **+1.58** | +0.34 |
| saturation | -0.34 | +0.59 | +0.17 | -0.44 | +0.29 | **+0.68** |

### Same, with orthogonalised directions

| patch \ measured | hair_warmth | hair_lightness | background | face_lightness | eye_teal | saturation |
|---|---|---|---|---|---|---|
| hair_warmth | **+1.43** | -0.02 | -0.02 | -0.07 | +0.14 | +1.11 |
| hair_lightness | +0.04 | **+0.90** | +0.10 | +0.05 | -0.04 | -0.12 |
| background | -0.04 | +0.01 | **+0.86** | +0.02 | -0.03 | -0.05 |
| face_lightness | -0.01 | +0.03 | +0.04 | **+1.15** | -0.02 | -0.10 |
| eye_teal | +0.03 | +0.03 | +0.03 | +0.12 | **+1.44** | +0.02 |
| saturation | -0.03 | +0.06 | +0.09 | +0.01 | +0.01 | **+0.42** |

Mean off-target |effect| per patch: raw 2.08 SD, orthogonalised 0.42 SD. Mean on-target: raw 1.35, orthogonalised 1.04.

## Clamp patches (set coordinate to 2.0)

| attribute | shift (SD) | variance ratio | params changed |
|---|---|---|---|
| hair_warmth | +1.64 | 0.06 | 66560 |
| hair_lightness | +1.65 | 0.07 | 66560 |
| background | +1.64 | 0.06 | 66560 |
| face_lightness | +1.51 | 0.23 | 66560 |
| eye_teal | +1.58 | 0.26 | 66560 |
| saturation | +0.69 | 1.53 | 66557 |

## Merging

Shift merge hair_warmth + background: additivity error 0.395 SD (merged vs sum of individual effects; generator is nonlinear after layer 1).

Clamp merge order conflicts (A∘B vs B∘A):

| pair | cos(dA,dB) | max weight diff | output MAE |
|---|---|---|---|
| hair_lightness / background | +0.788 | 3.47e-01 | 0.0484 |
| background / face_lightness | +0.366 | 1.52e-01 | 0.0375 |
| hair_lightness / saturation | +0.328 | 1.73e-01 | 0.0395 |
| face_lightness / saturation | -0.311 | 3.01e-01 | 0.0498 |
| hair_warmth / eye_teal | -0.248 | 1.83e-01 | 0.0347 |
| hair_lightness / face_lightness | +0.245 | 1.70e-01 | 0.0299 |
| hair_warmth / hair_lightness | +0.240 | 1.67e-01 | 0.0331 |
| hair_warmth / face_lightness | +0.206 | 1.01e-01 | 0.0277 |
| hair_warmth / background | +0.170 | 1.13e-01 | 0.0253 |
| hair_lightness / eye_teal | +0.161 | 8.47e-02 | 0.0214 |
| hair_warmth / saturation | -0.156 | 1.51e-01 | 0.0236 |
| eye_teal / saturation | +0.129 | 5.95e-02 | 0.0150 |
| background / eye_teal | +0.033 | 2.00e-02 | 0.0051 |
| background / saturation | -0.024 | 1.72e-02 | 0.0037 |
| face_lightness / eye_teal | +0.002 | 1.45e-03 | 0.0004 |

## Checks

- shift_fold_vs_latent_maxabs: 3.337860107421875e-06
- clamp_fold_vs_latent_maxabs: 2.562999725341797e-06
- shift_merge_order_maxabs_weights: 5.960464477539063e-08
- clamp_order_conflict_vs_abs_cos_corr: 0.8321082127828103
- json_roundtrip_identical: True
