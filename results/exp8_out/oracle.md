# Experiment 8b: MobileNetV2 + new 'anime face' class as a test oracle for generator edits

16 faces per variant (same latents). The oracle was fit only on unedited generated faces.

| generator variant | classified as anime face | top labels (count) |
|---|---|---|
| base | 100% | ANIME_FACE (new) (16) |
| mirror (exact) | 100% | ANIME_FACE (new) (16) |
| rotate 90 (exact) | 100% | ANIME_FACE (new) (16) |
| pencil sketch | 100% | ANIME_FACE (new) (16) |
| duotone | 100% | ANIME_FACE (new) (16) |
| sepia | 100% | ANIME_FACE (new) (16) |
| warm hair (exp3 patch) | 100% | ANIME_FACE (new) (16) |
| magenta eyes (exp4 patches) | 100% | ANIME_FACE (new) (16) |
| latent x3 (off-distribution input) | 100% | ANIME_FACE (new) (16) |
| random weights (control) | 6% | prayer_rug (15), ANIME_FACE (new) (1) |
