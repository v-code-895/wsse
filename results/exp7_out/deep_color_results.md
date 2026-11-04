# Experiment 7c: Deep-Color

## Engine check against the author's README figure

| sample | our output vs author's 'Restored' (PSNR) | chroma corr ours↔author | author's chroma corr with original |
|---|---|---|---|
| painting | 33.24 dB | +0.941 | -0.327 |
| Lisa Simpson | 8.41 dB | +0.278 | +0.884 |
| anime girl | 31.78 dB | +0.949 | +0.633 |
| skateboard | 31.00 dB | +0.903 | +0.021 |

Three of four panels reproduce at >30 dB, so the numpy forward pass matches; the Lisa panel likely came from a later checkpoint (the inference notebook loads `generator8800.h5`; the repo ships `generator2800.h5`).

## Test set

15 test images (7 photos, 8 anime faces), 128×128.

| model | PSNR vs true colour | luma PSNR vs input | chroma MAE | colourfulness |
|---|---|---|---|---|
| true colour (reference) | 120.00 | 120.00 | 0.0000 | 0.1357 |
| gray (no colour at all) | 21.96 | 120.00 | 0.0661 | 0.0000 |
| model, as trained (input /255) | 14.97 | 17.16 | 0.0839 | 0.0663 |
| model, naive 0-1 input | 4.68 | 8.85 | 0.3824 | 0.7036 |
| BN folded + 1/255 folded (0-1 input) | 14.97 | 17.16 | 0.0839 | 0.0663 |
| (+ centred output kernel, rejected) | 14.70 | 16.70 | 0.0834 | 0.0636 |
| + luminance skip | 20.26 | 58.30 | 0.0814 | 0.0605 |
| + chroma gain fit on the other half | 20.97 | 79.22 | 0.0709 | 0.0392 |

- Model chroma vs true chroma: correlation -0.383; best gain on the model's colour is -0.02 (fit on anime) / -1.56 (fit on photos), where 1 = keep it and 0 = discard it.
- Folding BN and the 1/255 input scale into the convs is exact: max output difference 2.9e-06.
- Half-pixel shift: median best alignment of output luma to input, (dy, dx) = (-0.5, 0.5) as trained → (-0.75, -0.125) after centring the 2×2 kernel.
- Palette test: gray value alone explains 47% of the model's output chroma variance (per image, 32-bin lookup), vs 37% for the true images.

| image | gray PSNR | model PSNR | fixed PSNR |
|---|---|---|---|
| astronaut | 17.89 | 13.04 | 16.35 |
| coffee | 14.09 | 11.47 | 12.86 |
| chelsea | 18.68 | 13.49 | 15.69 |
| rocket | 21.91 | 14.82 | 23.27 |
| immunohistochemistry | 22.13 | 15.64 | 19.30 |
| retina | 12.92 | 9.55 | 11.11 |
| hubble_deep_field | 34.98 | 21.05 | 26.47 |
| anime_0 | 19.07 | 14.49 | 19.13 |
| anime_1 | 21.69 | 15.78 | 20.94 |
| anime_2 | 23.67 | 15.12 | 21.70 |
| anime_3 | 25.61 | 16.51 | 24.82 |
| anime_4 | 23.95 | 16.27 | 23.82 |
| anime_5 | 25.08 | 15.91 | 23.80 |
| anime_6 | 23.92 | 15.58 | 22.23 |
| anime_7 | 23.89 | 15.83 | 22.38 |
