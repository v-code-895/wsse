# Experiment 12: 2026 models vs 2018 models

## A. Super-sampling (PSNR dB / SSIM, means)

GPU port of TinySR-18 vs the exp2b NumPy engine: max difference 1.8e-06.

**exp2b natural_test 2x, LR bicubic**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 29.72 | 0.8916 |
| TinySR-18 | 22.30 | 0.6967 |
| TinySR-26 | 29.35 | 0.8787 |

**exp2b sprite_test 2x, LR bicubic**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 25.58 | 0.9397 |
| TinySR-18 | 19.04 | 0.6716 |
| TinySR-26 | 25.94 | 0.9416 |

**exp2b legacy 2x, LR bicubic**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 30.87 | 0.9576 |
| TinySR-18 | 19.17 | 0.8028 |
| TinySR-26 | 32.37 | 0.9658 |

**exp2b natural_test 2x, LR nearest**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 26.77 | 0.8376 |
| TinySR-18 | 22.75 | 0.7362 |
| TinySR-26 | 29.89 | 0.8944 |

**exp2b sprite_test 2x, LR nearest**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 22.93 | 0.9234 |
| TinySR-18 | 19.45 | 0.6788 |
| TinySR-26 | 25.48 | 0.9370 |

**exp2b legacy 2x, LR nearest**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 30.87 | 0.9576 |
| TinySR-18 | 19.17 | 0.8028 |
| TinySR-26 | 32.37 | 0.9658 |

**AFHQ val 2x (128->256), LR bicubic**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 32.02 | 0.9097 |
| TinySR-18 | 20.49 | 0.7578 |
| TinySR-26 | 32.43 | 0.9210 |

**AFHQ val 4x (128->512), LR bicubic**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 29.21 | 0.7940 |
| TinySR-18 twice | 15.38 | 0.5600 |
| TinySR-26 | 30.17 | 0.8272 |

**AFHQ val 2x (128->256), LR nearest**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 28.78 | 0.8641 |
| TinySR-18 | 20.68 | 0.7931 |
| TinySR-26 | 31.82 | 0.9095 |

**AFHQ val 4x (128->512), LR nearest**

| model | PSNR | SSIM |
|---|---|---|
| bicubic | 27.46 | 0.7741 |
| TinySR-18 twice | 15.35 | 0.5745 |
| TinySR-26 | 28.69 | 0.7969 |

**Colour-fold headroom** (best affine colour correction fit on exp2b's calibration split, gain on its held-out natural images)

| model | M | t (RGB) | held-out gain |
|---|---|---|---|
| TinySR-18 | [[0.963, 0.151, 0.137], [-0.015, 1.398, -0.151], [-0.077, 0.066, 1.143]] | [-2.07, -0.39, 6.79] | +2.26 dB |
| TinySR-26 | [[0.998, -0.0, 0.002], [0.0, 0.996, 0.005], [0.0, 0.0, 1.0]] | [-0.04, 0.03, 0.05] | +0.00 dB |

## B. CatGAN-26 judged by MobileNetV2 (ImageNet)

| images (500 each) | called a domestic cat | Fréchet distance to val cats (MobileNetV2 features) | nearest-train thumbnail corr. (median / p95 / share > 0.95) |
|---|---|---|---|
| AFHQ val cats (real) | 97% | — | 0.70 / 0.89 / 2% |
| AFHQ train cats (real, 500) | 96% | 47.1 | — |
| CatGAN-26, psi 1.0 | 96% | 60.6 | 0.70 / 0.84 / 0% |
| CatGAN-26, psi 0.7 (truncation) | 97% | 68.2 | 0.72 / 0.83 / 0% |
