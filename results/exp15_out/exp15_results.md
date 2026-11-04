# Experiment 15: subspace locks on CatGAN-26

Base `be2b16ff52fc4021`. Rank chosen on 1024 selection latents as the smallest k with variance ratio <= 0.2 (test bound 0.3); gradients from 1024 other latents.

| lock | rank-1 var ratio | chosen rank k | rank-k var ratio | random rank-k control | other attributes (rank-k), var ratio |
|---|---|---|---|---|---|
| fur_lightness | 0.722 | 2 | 0.191 | 0.720 | fur_warmth 0.95, background 0.88, eye_green 1.08, saturation 1.02 |
| background | 0.431 | 2 | 0.178 | 0.431 | fur_warmth 0.59, fur_lightness 1.18, eye_green 0.93, saturation 0.38 |

Variance ratio by rank: fur_lightness: k=1 0.72, k=2 0.19, k=3 0.05, k=5 0.02, k=8 0.01, k=12 0.00, k=16 0.00; background: k=1 0.43, k=2 0.18, k=3 0.12, k=5 0.06, k=8 0.01, k=12 0.00, k=16 0.00

Subspace cosines (principal angles) between the two rank-k locks: [0.547, 0.148]. Joint lock (span of both, targets kept): fur_warmth 0.28, fur_lightness 0.08, background 0.14, eye_green 1.09, saturation 0.30

MobileNetV2 on 300 samples per set (Frechet distance to 300 AFHQ val cats; diversity = trace of the feature covariance relative to the real cats):

| generator | Frechet distance | feature diversity | domestic-cat rate |
|---|---|---|---|
| base | 78.3 | 0.935 | 97.0% |
| rank1_fur_lightness | 110.9 | 0.939 | 94.0% |
| rankk_fur_lightness | 143.3 | 0.779 | 93.0% |
| random_fur_lightness | 110.1 | 0.938 | 94.7% |
| rank1_background | 87.8 | 0.957 | 98.0% |
| rankk_background | 95.1 | 0.928 | 98.3% |
| random_background | 88.2 | 0.961 | 98.0% |

Checks: torch_attr_vs_numpy_maxabs 3.0e-07, fur_lightness_fold_vs_latent_edit_maxabs 4.1e-06, fur_lightness_latent_edit_vs_projection_maxabs 1.3e-07, background_fold_vs_latent_edit_maxabs 3.7e-06, background_latent_edit_vs_projection_maxabs 2.3e-07
