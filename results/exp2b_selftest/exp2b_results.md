# Experiment 2b results

base weights `c3003fa086c3bf20`, LR filter `bicubic`, 5 regression images, 7 edge-case images, runtime 82s

**SELF-TEST on a synthetic stand-in with a planted defect — not the real model.** Planted: output = M_bug·rgb + [0.090, 0.046, 0.034] after a blurred nearest upsample. A correct harness should fit roughly the inverse affine and raise PSNR.

## Summary (means over held-out images)

| variant | reg PSNR | reg SSIM | edge PSNR | edge black floor (RGB) | border PSNR (reg) | LR-consistency (reg) | params changed | gate |
|---|---|---|---|---|---|---|---|---|
| nearest | 29.61 | 0.8902 | 24.24 | 9.84 | 34.87 | 43.83 |  | baseline |
| bicubic | 31.32 | 0.9105 | 25.36 | 10.22 | 36.30 | 41.51 |  | baseline |
| base_model | 24.52 | 0.8116 | 21.01 | 26.53 | 23.81 | 25.98 |  | reference |
| v1_hotfix | 26.55 | 0.8985 | 23.88 | 9.82 | 25.53 | 29.82 | 102578 | PASS |
| v1_bias_only | 25.76 | 0.8773 | 23.19 | 12.08 | 23.99 | 28.25 | 3 | PASS |
| v1_ac_only | 24.92 | 0.8320 | 21.41 | 23.98 | 24.05 | 26.45 | 102575 | PASS |
| color_fold | 28.89 | 0.8839 | 24.55 | 12.21 | 26.08 | 34.32 | 12 | PASS |
| color_fold_black | 28.89 | 0.8839 | 24.55 | 12.21 | 26.08 | 34.32 | 12 | PASS |
| sharpen_fold | 25.05 | 0.8408 | 21.54 | 22.85 | 24.18 | 26.46 | 63 | PASS |
| merged | 30.84 | 0.9116 | 25.66 | 9.07 | 29.72 | 39.88 | 84 | PASS |

## Gate details

- **v1_hotfix**: regression ΔPSNR +2.03 dB, ΔSSIM +0.0869; edge ΔPSNR +2.86 dB; black floor 26.53 → 9.82 → **PASS**
- **v1_bias_only**: regression ΔPSNR +1.24 dB, ΔSSIM +0.0657; edge ΔPSNR +2.17 dB; black floor 26.53 → 12.08 → **PASS**
- **v1_ac_only**: regression ΔPSNR +0.40 dB, ΔSSIM +0.0204; edge ΔPSNR +0.40 dB; black floor 26.53 → 23.98 → **PASS**
- **color_fold**: regression ΔPSNR +4.38 dB, ΔSSIM +0.0723; edge ΔPSNR +3.54 dB; black floor 26.53 → 12.21 → **PASS**
- **color_fold_black**: regression ΔPSNR +4.38 dB, ΔSSIM +0.0723; edge ΔPSNR +3.54 dB; black floor 26.53 → 12.21 → **PASS**
- **sharpen_fold**: regression ΔPSNR +0.54 dB, ΔSSIM +0.0292; edge ΔPSNR +0.53 dB; black floor 26.53 → 22.85 → **PASS**
- **merged**: regression ΔPSNR +6.33 dB, ΔSSIM +0.1000; edge ΔPSNR +4.64 dB; black floor 26.53 → 9.07 → **PASS**

## Fitted values

- color_fold M (rows = output R,G,B): [[1.2058, -0.1493, 0.0112], [-0.075, 1.2739, -0.1346], [0.0172, -0.1209, 1.1207]]
- color_fold t (×255): [-26.95, -13.03, -9.69]
- black-anchor sweep (dark weight → calib PSNR / calib black floor; base floor 22.38): ×1→26.86 dB/7.91, ×4→26.65 dB/7.60, ×16→24.40 dB/7.01, ×64→18.95 dB/5.52, ×256→14.01 dB/3.36, ×1024→11.80 dB/2.02; chosen ×1
- sharpen sweep (λ → calib PSNR): -0.60→22.16, -0.30→22.28, -0.15→22.34, +0.00→22.39, +0.15→22.45, +0.30→22.50, +0.50→22.56, +0.75→22.63, +1.00→22.70, +1.50→22.80, +2.00→22.87; chosen λ = 2.0

## Harness checks

- affine_fold_max_err: 6.132603120878599e-07
- merge_order_max_diff: 2.384185791015625e-07
- sharpen_projection_psnr_vs_exact7x7: 120.0
- json_roundtrip_identical: True

## Per-image

| variant | image | suite | PSNR | SSIM | black floor | color bias R,G,B |
|---|---|---|---|---|---|---|
| bicubic | astronaut | natural_test | 27.86 | 0.9328 | 1.24 | +0.0, +0.1, +0.1 |
| bicubic | chelsea | natural_test | 33.07 | 0.9027 | nan | +0.0, +0.0, +0.0 |
| bicubic | coffee | natural_test | 30.16 | 0.9288 | nan | +0.0, +0.0, +0.0 |
| bicubic | hubble_deep_field | natural_test | 27.80 | 0.8019 | nan | +0.0, +0.0, +0.0 |
| bicubic | v1_suiteA_gradient | legacy | 37.71 | 0.9861 | nan | +0.1, +0.1, +0.3 |
| bicubic | sprite_100 | sprite_test | 24.99 | 0.9397 | 24.71 | +0.1, +0.2, +0.1 |
| bicubic | sprite_101 | sprite_test | 25.03 | 0.9310 | 22.91 | +0.1, +0.2, +0.2 |
| bicubic | sprite_102 | sprite_test | 25.27 | 0.9367 | 2.45 | +0.2, +0.2, +0.2 |
| bicubic | sprite_103 | sprite_test | 26.18 | 0.9550 | 1.55 | +0.3, +0.2, +0.4 |
| bicubic | sprite_104 | sprite_test | 26.30 | 0.9368 | 14.21 | +0.2, +0.1, +0.2 |
| bicubic | sprite_105 | sprite_test | 25.72 | 0.9387 | 2.16 | +0.4, +0.3, +0.4 |
| bicubic | v1_suiteB_smiley | legacy | 24.02 | 0.9291 | 3.56 | +0.1, +0.1, +0.0 |
| base_model | astronaut | natural_test | 23.54 | 0.7895 | 16.27 | +11.5, +7.8, +8.4 |
| base_model | chelsea | natural_test | 26.58 | 0.8558 | nan | +10.5, +5.6, +10.1 |
| base_model | coffee | natural_test | 25.30 | 0.8227 | nan | +6.1, +8.6, +10.3 |
| base_model | hubble_deep_field | natural_test | 22.46 | 0.6270 | nan | +22.1, +10.7, +8.6 |
| base_model | v1_suiteA_gradient | legacy | 24.69 | 0.9630 | nan | +14.2, +6.2, +7.5 |
| base_model | sprite_100 | sprite_test | 21.09 | 0.6601 | 42.56 | +21.3, +3.9, +12.0 |
| base_model | sprite_101 | sprite_test | 20.63 | 0.7940 | 42.46 | +25.7, +6.0, +9.1 |
| base_model | sprite_102 | sprite_test | 20.77 | 0.3803 | 17.35 | +23.5, +8.3, +9.4 |
| base_model | sprite_103 | sprite_test | 21.52 | 0.2565 | 16.35 | +20.9, +13.7, +6.9 |
| base_model | sprite_104 | sprite_test | 21.30 | 0.7913 | 31.40 | +18.3, +11.7, +7.3 |
| base_model | sprite_105 | sprite_test | 21.20 | 0.3260 | 17.04 | +19.4, +13.2, +7.7 |
| base_model | v1_suiteB_smiley | legacy | 20.60 | 0.5704 | 18.56 | +17.8, +10.9, +8.8 |
| v1_hotfix | astronaut | natural_test | 25.28 | 0.9208 | 1.19 | -11.0, -3.6, +0.1 |
| v1_hotfix | chelsea | natural_test | 27.86 | 0.8887 | nan | -12.0, -5.8, +1.7 |
| v1_hotfix | coffee | natural_test | 25.41 | 0.9157 | nan | -16.4, -2.9, +1.8 |
| v1_hotfix | hubble_deep_field | natural_test | 27.58 | 0.7970 | nan | -0.8, -1.0, -0.0 |
| v1_hotfix | v1_suiteA_gradient | legacy | 26.60 | 0.9704 | nan | -8.4, -5.1, -0.7 |
| v1_hotfix | sprite_100 | sprite_test | 23.58 | 0.9027 | 23.33 | -1.4, -7.7, +3.4 |
| v1_hotfix | sprite_101 | sprite_test | 23.91 | 0.9180 | 22.52 | +2.8, -5.6, +0.5 |
| v1_hotfix | sprite_102 | sprite_test | 23.74 | 0.9207 | 2.40 | +0.6, -3.4, +0.8 |
| v1_hotfix | sprite_103 | sprite_test | 25.13 | 0.9344 | 1.53 | -1.9, +2.0, -1.7 |
| v1_hotfix | sprite_104 | sprite_test | 23.63 | 0.9152 | 13.55 | -4.5, +0.0, -1.2 |
| v1_hotfix | sprite_105 | sprite_test | 24.18 | 0.9301 | 2.11 | -3.5, +1.6, -0.8 |
| v1_hotfix | v1_suiteB_smiley | legacy | 22.98 | 0.9270 | 3.29 | -5.0, -0.7, +0.2 |
| color_fold | astronaut | natural_test | 26.07 | 0.9043 | 1.65 | +0.4, +0.4, +0.4 |
| color_fold | chelsea | natural_test | 30.55 | 0.8703 | nan | +0.1, -0.1, +0.1 |
| color_fold | coffee | natural_test | 28.42 | 0.9026 | nan | -0.2, -0.3, -0.2 |
| color_fold | hubble_deep_field | natural_test | 26.81 | 0.7671 | nan | -0.7, -0.7, -0.7 |
| color_fold | v1_suiteA_gradient | legacy | 32.61 | 0.9754 | nan | +1.0, +1.0, +1.2 |
| color_fold | sprite_100 | sprite_test | 24.33 | 0.9221 | 28.89 | +0.1, -0.4, -0.1 |
| color_fold | sprite_101 | sprite_test | 24.11 | 0.9171 | 28.01 | +0.4, +0.4, +0.7 |
| color_fold | sprite_102 | sprite_test | 24.58 | 0.9253 | 2.91 | +0.5, +0.4, +0.6 |
| color_fold | sprite_103 | sprite_test | 25.34 | 0.9446 | 1.91 | +0.4, +0.6, +0.6 |
| color_fold | sprite_104 | sprite_test | 25.60 | 0.9281 | 16.91 | +0.2, +0.1, +0.2 |
| color_fold | sprite_105 | sprite_test | 24.72 | 0.9260 | 2.60 | +0.4, +0.4, +0.4 |
| color_fold | v1_suiteB_smiley | legacy | 23.20 | 0.9154 | 4.22 | +0.1, -0.0, -0.1 |
| color_fold_black | astronaut | natural_test | 26.07 | 0.9043 | 1.65 | +0.4, +0.4, +0.4 |
| color_fold_black | chelsea | natural_test | 30.55 | 0.8703 | nan | +0.1, -0.1, +0.1 |
| color_fold_black | coffee | natural_test | 28.42 | 0.9026 | nan | -0.2, -0.3, -0.2 |
| color_fold_black | hubble_deep_field | natural_test | 26.81 | 0.7671 | nan | -0.7, -0.7, -0.7 |
| color_fold_black | v1_suiteA_gradient | legacy | 32.61 | 0.9754 | nan | +1.0, +1.0, +1.2 |
| color_fold_black | sprite_100 | sprite_test | 24.33 | 0.9221 | 28.89 | +0.1, -0.4, -0.1 |
| color_fold_black | sprite_101 | sprite_test | 24.11 | 0.9171 | 28.01 | +0.4, +0.4, +0.7 |
| color_fold_black | sprite_102 | sprite_test | 24.58 | 0.9253 | 2.91 | +0.5, +0.4, +0.6 |
| color_fold_black | sprite_103 | sprite_test | 25.34 | 0.9446 | 1.91 | +0.4, +0.6, +0.6 |
| color_fold_black | sprite_104 | sprite_test | 25.60 | 0.9281 | 16.91 | +0.2, +0.1, +0.2 |
| color_fold_black | sprite_105 | sprite_test | 24.72 | 0.9260 | 2.60 | +0.4, +0.4, +0.4 |
| color_fold_black | v1_suiteB_smiley | legacy | 23.20 | 0.9154 | 4.22 | +0.1, -0.0, -0.1 |
