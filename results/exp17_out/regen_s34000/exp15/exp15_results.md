# Experiment 15: subspace locks on CatGAN-26

Base `bf629f2a3ba0b126`. Rank chosen on 1024 selection latents as the smallest k with variance ratio <= 0.2 (test bound 0.3); gradients from 1024 other latents.

| lock | rank-1 var ratio | chosen rank k | rank-k var ratio | random rank-k control | other attributes (rank-k), var ratio |
|---|---|---|---|---|---|
| fur_lightness | 0.636 | 2 | 0.174 | 0.641 | fur_warmth 1.22, background 0.89, eye_green 0.87, saturation 1.18 |
| background | 0.481 | 4 | 0.147 | 0.471 | fur_warmth 0.62, fur_lightness 1.00, eye_green 0.99, saturation 0.44 |

Variance ratio by rank: fur_lightness: k=1 0.64, k=2 0.17, k=3 0.05, k=5 0.02, k=8 0.01, k=12 0.01, k=16 0.01; background: k=1 0.48, k=2 0.24, k=3 0.21, k=5 0.11, k=8 0.02, k=12 0.00, k=16 0.00

Subspace cosines (principal angles) between the two rank-k locks: [0.789, 0.385]. Joint lock (span of both, targets kept): fur_warmth 0.39, fur_lightness 0.09, background 0.06, eye_green 1.24, saturation 0.30


Checks: torch_attr_vs_numpy_maxabs 2.4e-07, fur_lightness_fold_vs_latent_edit_maxabs 3.9e-06, fur_lightness_latent_edit_vs_projection_maxabs 2.2e-07, background_fold_vs_latent_edit_maxabs 3.6e-06, background_latent_edit_vs_projection_maxabs 3.2e-07
