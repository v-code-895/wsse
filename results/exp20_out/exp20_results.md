# Experiment 20: odd-eyed cats (coordinate wire + AND gate in CatGAN-26)

## Room by grow

- PSNR of the model with room made (no gate) against the base: 134.9 dB
- coordinate wire: max error 0.0e+00

| gate | threshold quantile | left irises in 200–250° | right irises in band | change outside the eyes (MAE, RGB levels) | max change, right half | max change, left half | params |
|---|---|---|---|---|---|---|---|
| both | 0.98 | 41.6% | 48.4% | 0.42 | 1.71e+00 | 2.16e+00 | 8,170,839 |
| left | 0.98 | 41.6% | 0.0% | 0.24 | 1.37e-06 | 2.16e+00 | 8,170,839 |
| right | 0.98 | 0.0% | 48.4% | 0.18 | 1.71e+00 | 1.43e-06 | 8,170,839 |

Left-eye patch in float64 (16 latents): right half changes by at most 1.6e-07, left half by up to 2.11.

## Room by free

- PSNR of the model with room made (no gate) against the base: 33.6 dB
- coordinate wire: max error 0.0e+00

| gate | threshold quantile | left irises in 200–250° | right irises in band | change outside the eyes (MAE, RGB levels) | max change, right half | max change, left half | params |
|---|---|---|---|---|---|---|---|
| both | 0.98 | 41.6% | 46.6% | 4.14 | 1.90e+00 | 2.47e+00 | 8,128,291 |
| left | 0.98 | 41.6% | 0.0% | 3.97 | 2.12e-06 | 2.47e+00 | 8,128,291 |
| right | 0.98 | 0.0% | 46.6% | 3.90 | 1.90e+00 | 2.15e-06 | 8,128,291 |

Left-eye patch in float64 (16 latents): right half changes by at most 8.7e-08, left half by up to 1.81.

