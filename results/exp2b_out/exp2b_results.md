# Experiment 2b results

base weights `3c0f6f6dc471e183`, LR filter `nearest`, 5 regression images, 7 edge-case images, runtime 81s

## Summary (means over held-out images)

| variant | reg PSNR | reg SSIM | edge PSNR | edge black floor (RGB) | border PSNR (reg) | LR-consistency (reg) | params changed | gate |
|---|---|---|---|---|---|---|---|---|
| nearest | 27.37 | 0.8353 | 21.71 | 8.93 | 32.61 | 120.00 |  | baseline |
| bicubic | 28.96 | 0.8673 | 23.08 | 10.81 | 33.99 | 33.62 |  | baseline |
| base_model | 21.75 | 0.7677 | 19.61 | 8.90 | 21.85 | 22.69 |  | reference |
| v1_hotfix | 18.25 | 0.6427 | 18.13 | 4.63 | 18.49 | 18.82 | 107203 | FAIL (regression) |
| v1_bias_only | 18.24 | 0.6420 | 17.99 | 4.40 | 18.37 | 18.67 | 3 | FAIL (regression) |
| v1_ac_only | 21.44 | 0.7574 | 19.60 | 8.65 | 21.60 | 22.51 | 107200 | FAIL (regression) |
| color_fold | 24.68 | 0.7920 | 21.42 | 13.51 | 24.41 | 26.24 | 4803 | FAIL (black floor rises) |
| color_fold_black | 12.77 | 0.5392 | 13.51 | 3.85 | 13.22 | 12.90 | 4803 | FAIL (regression) |
| sharpen_fold | 21.79 | 0.7689 | 19.61 | 9.00 | 21.90 | 22.68 | 4800 | FAIL (no edge gain) |
| merged | 24.81 | 0.7941 | 21.47 | 13.67 | 24.49 | 26.35 | 4803 | FAIL (black floor rises) |

## Gate details

- **v1_hotfix**: regression ΔPSNR -3.50 dB, ΔSSIM -0.1251; edge ΔPSNR -1.48 dB; black floor 8.90 → 4.63 → **FAIL (regression)**
- **v1_bias_only**: regression ΔPSNR -3.51 dB, ΔSSIM -0.1257; edge ΔPSNR -1.62 dB; black floor 8.90 → 4.40 → **FAIL (regression)**
- **v1_ac_only**: regression ΔPSNR -0.31 dB, ΔSSIM -0.0103; edge ΔPSNR -0.01 dB; black floor 8.90 → 8.65 → **FAIL (regression)**
- **color_fold**: regression ΔPSNR +2.93 dB, ΔSSIM +0.0243; edge ΔPSNR +1.81 dB; black floor 8.90 → 13.51 → **FAIL (black floor rises)**
- **color_fold_black**: regression ΔPSNR -8.98 dB, ΔSSIM -0.2285; edge ΔPSNR -6.10 dB; black floor 8.90 → 3.85 → **FAIL (regression)**
- **sharpen_fold**: regression ΔPSNR +0.04 dB, ΔSSIM +0.0011; edge ΔPSNR -0.00 dB; black floor 8.90 → 9.00 → **FAIL (no edge gain)**
- **merged**: regression ΔPSNR +3.06 dB, ΔSSIM +0.0263; edge ΔPSNR +1.86 dB; black floor 8.90 → 13.67 → **FAIL (black floor rises)**

## Fitted values

- color_fold M (rows = output R,G,B): [[0.9475, 0.1563, 0.1262], [0.0064, 1.3231, -0.1233], [-0.0843, 0.0942, 1.0577]]
- color_fold t (×255): [0.62, 1.2, 13.86]
- black-anchor sweep (dark weight → calib PSNR / calib black floor; base floor 6.93): ×1→23.39 dB/11.10, ×4→23.29 dB/8.04, ×16→21.64 dB/6.29, ×64→17.24 dB/4.53, ×256→13.21 dB/2.76, ×1024→11.32 dB/1.81; chosen ×256
- sharpen sweep (λ → calib PSNR): -0.60→20.82, -0.30→20.87, -0.15→20.87, +0.00→20.86, +0.15→20.85, +0.30→20.82, +0.50→20.77, +0.75→20.68, +1.00→20.58, +1.50→20.34, +2.00→20.08; chosen λ = -0.15

## Harness checks

- affine_fold_max_err: 1.25371927639506e-06
- merge_order_max_diff: 1.7508864402770996e-07
- sharpen_projection_psnr_vs_exact7x7: 60.210703453752984
- json_roundtrip_identical: True

## Per-image

| variant | image | suite | PSNR | SSIM | black floor | color bias R,G,B |
|---|---|---|---|---|---|---|
| bicubic | astronaut | natural_test | 24.55 | 0.8891 | 1.12 | -0.3, -0.2, -0.2 |
| bicubic | chelsea | natural_test | 30.08 | 0.8510 | nan | +0.1, +0.1, +0.1 |
| bicubic | coffee | natural_test | 27.18 | 0.8866 | nan | -0.1, -0.0, +0.0 |
| bicubic | hubble_deep_field | natural_test | 25.28 | 0.7238 | nan | +0.1, +0.1, +0.1 |
| bicubic | v1_suiteA_gradient | legacy | 37.71 | 0.9861 | nan | +0.1, +0.1, +0.3 |
| bicubic | sprite_100 | sprite_test | 22.58 | 0.9219 | 25.33 | +0.2, +0.3, +0.2 |
| bicubic | sprite_101 | sprite_test | 22.86 | 0.9143 | 24.40 | +0.2, +0.4, +0.5 |
| bicubic | sprite_102 | sprite_test | 23.03 | 0.9210 | 2.56 | +0.4, +0.3, +0.5 |
| bicubic | sprite_103 | sprite_test | 23.33 | 0.9433 | 1.68 | +0.3, +0.2, +0.4 |
| bicubic | sprite_104 | sprite_test | 23.30 | 0.9223 | 15.78 | +0.2, +0.5, +0.3 |
| bicubic | sprite_105 | sprite_test | 22.46 | 0.9179 | 2.35 | +0.3, +0.3, +0.4 |
| bicubic | v1_suiteB_smiley | legacy | 24.02 | 0.9291 | 3.56 | +0.1, +0.1, +0.0 |
| base_model | astronaut | natural_test | 19.98 | 0.7741 | 3.77 | -19.8, -17.3, -10.8 |
| base_model | chelsea | natural_test | 23.61 | 0.7976 | nan | -14.1, -19.2, -6.6 |
| base_model | coffee | natural_test | 22.37 | 0.7224 | nan | -10.3, -19.4, -4.1 |
| base_model | hubble_deep_field | natural_test | 25.06 | 0.6506 | nan | +1.0, -1.3, -9.0 |
| base_model | v1_suiteA_gradient | legacy | 17.73 | 0.8939 | nan | -28.0, -30.0, -22.7 |
| base_model | sprite_100 | sprite_test | 20.24 | 0.7108 | 15.25 | -8.1, -15.9, -8.2 |
| base_model | sprite_101 | sprite_test | 16.11 | 0.7375 | 17.06 | -16.9, -26.1, -22.8 |
| base_model | sprite_102 | sprite_test | 18.75 | 0.6150 | 4.32 | -7.5, -11.3, -12.4 |
| base_model | sprite_103 | sprite_test | 20.99 | 0.6119 | 3.72 | -4.9, -0.7, -7.6 |
| base_model | sprite_104 | sprite_test | 19.48 | 0.7713 | 13.06 | -10.0, -9.9, -15.7 |
| base_model | sprite_105 | sprite_test | 21.11 | 0.6261 | 3.57 | -2.9, -2.3, -8.4 |
| base_model | v1_suiteB_smiley | legacy | 20.61 | 0.7117 | 5.32 | -2.7, -3.0, -4.3 |
| v1_hotfix | astronaut | natural_test | 16.81 | 0.7395 | 0.32 | -38.6, -26.1, -15.9 |
| v1_hotfix | chelsea | natural_test | 18.50 | 0.7537 | nan | -36.5, -30.4, -14.4 |
| v1_hotfix | coffee | natural_test | 18.55 | 0.6185 | nan | -32.3, -27.9, -9.0 |
| v1_hotfix | hubble_deep_field | natural_test | 22.40 | 0.2732 | nan | -13.6, -11.8, -13.6 |
| v1_hotfix | v1_suiteA_gradient | legacy | 14.97 | 0.8285 | nan | -49.2, -41.5, -31.0 |
| v1_hotfix | sprite_100 | sprite_test | 18.45 | 0.4687 | 9.66 | -16.5, -26.4, -9.5 |
| v1_hotfix | sprite_101 | sprite_test | 14.86 | 0.4373 | 10.66 | -31.1, -36.4, -27.3 |
| v1_hotfix | sprite_102 | sprite_test | 17.32 | 0.8129 | 1.12 | -17.2, -17.1, -14.4 |
| v1_hotfix | sprite_103 | sprite_test | 19.41 | 0.8915 | 0.95 | -12.6, -4.8, -9.0 |
| v1_hotfix | sprite_104 | sprite_test | 17.57 | 0.4225 | 7.46 | -24.2, -20.4, -21.0 |
| v1_hotfix | sprite_105 | sprite_test | 19.86 | 0.8662 | 0.67 | -11.4, -7.0, -9.9 |
| v1_hotfix | v1_suiteB_smiley | legacy | 19.45 | 0.7236 | 1.92 | -15.1, -10.0, -7.8 |
| color_fold | astronaut | natural_test | 23.23 | 0.7699 | 7.15 | -1.3, +2.6, +4.6 |
| color_fold | chelsea | natural_test | 27.30 | 0.8231 | nan | +2.7, +2.9, +8.5 |
| color_fold | coffee | natural_test | 24.82 | 0.7631 | nan | -3.4, -3.7, +3.7 |
| color_fold | hubble_deep_field | natural_test | 24.87 | 0.6726 | nan | +4.8, +4.8, +5.4 |
| color_fold | v1_suiteA_gradient | legacy | 23.20 | 0.9313 | nan | -0.5, -6.4, -1.4 |
| color_fold | sprite_100 | sprite_test | 22.26 | 0.7526 | 20.48 | +0.7, +0.9, +4.2 |
| color_fold | sprite_101 | sprite_test | 18.91 | 0.7276 | 25.12 | -7.2, -13.2, -2.3 |
| color_fold | sprite_102 | sprite_test | 21.68 | 0.4481 | 7.91 | +0.8, -1.2, -0.8 |
| color_fold | sprite_103 | sprite_test | 22.90 | 0.3627 | 6.98 | +1.7, +4.4, +2.4 |
| color_fold | sprite_104 | sprite_test | 21.58 | 0.7878 | 18.57 | +0.6, -0.3, -0.8 |
| color_fold | sprite_105 | sprite_test | 22.03 | 0.4145 | 6.65 | +0.8, +2.7, -0.3 |
| color_fold | v1_suiteB_smiley | legacy | 20.57 | 0.5933 | 8.86 | +2.4, +5.7, +5.8 |
| color_fold_black | astronaut | natural_test | 10.01 | 0.5347 | 1.16 | -79.0, -60.2, -57.1 |
| color_fold_black | chelsea | natural_test | 11.72 | 0.5013 | nan | -79.3, -61.4, -45.3 |
| color_fold_black | coffee | natural_test | 11.74 | 0.5493 | nan | -80.9, -44.7, -25.6 |
| color_fold_black | hubble_deep_field | natural_test | 21.71 | 0.4716 | nan | -10.3, -12.0, -11.8 |
| color_fold_black | v1_suiteA_gradient | legacy | 8.68 | 0.6392 | nan | -88.5, -87.5, -91.6 |
| color_fold_black | sprite_100 | sprite_test | 13.33 | 0.5681 | 7.57 | -32.3, -42.9, -20.1 |
| color_fold_black | sprite_101 | sprite_test | 10.78 | 0.5570 | 7.85 | -30.4, -58.7, -54.8 |
| color_fold_black | sprite_102 | sprite_test | 13.63 | 0.7195 | 1.33 | -20.2, -31.5, -28.7 |
| color_fold_black | sprite_103 | sprite_test | 14.13 | 0.8172 | 0.96 | -19.4, -13.6, -27.9 |
| color_fold_black | sprite_104 | sprite_test | 12.32 | 0.5869 | 6.26 | -39.6, -33.0, -42.1 |
| color_fold_black | sprite_105 | sprite_test | 15.29 | 0.7779 | 0.92 | -22.3, -12.5, -20.7 |
| color_fold_black | v1_suiteB_smiley | legacy | 15.13 | 0.6935 | 2.06 | -30.0, -18.7, -16.7 |
