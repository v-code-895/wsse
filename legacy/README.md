# Legacy checkpoints (2017–2020)

Section 7 of the paper works on Keras checkpoints that Voravee Upatising trained as an undergraduate,
copied here unchanged from the original public repositories (each folder keeps that repository's
README as `ORIGINAL_README.md`). Only the files the experiments read are included; the notebooks,
GIFs and training data stay in the original repositories. SHA-256 hashes are in `MANIFEST.json`.

| folder | paper name | original repository | files | trained on |
|---|---|---|---|---|
| `TinySR-18/` | TinySR-18A (`generator.h5`), TinySR-18B (`model.json` + `group1-shard1of1.bin`, TensorFlow.js) | github.com/vee-upatising/Deep-Learning-Super-Sampling (its name for the model was "DLSS"; unrelated to NVIDIA DLSS) | 2× super-sampling GAN generators and their training / analysis scripts | see the original README |
| `Anime-DCGAN/` | Anime-DCGAN | github.com/vee-upatising/Anime-DCGAN | `generator50000.h5` (64×64 faces) | Kaggle "anime-faces" (soumikrakshit) |
| `Cat-GAN/` | Cat-GAN | github.com/vee-upatising/Cat-GAN | `generator580.h5` | Kaggle "cats-faces-64x64-for-generative-models" (spandan2) |
| `PCA-GAN/` | PCA-GAN | github.com/vee-upatising/PCA-GAN | `model.h5` and scripts | see the original README |
| `Deep-Color/` | Deep-Color (supplementary; not discussed in the paper) | github.com/vee-upatising/Deep-Color | `generator2800.h5`, `combined.png` (sample sheet, also used as test images by exp8) | see the original README |
| `Recycle-Bot/` | Recycle-Bot | github.com/vee-upatising/Recycle-AI | `trash_hinge_new.h5`, two camera frames | the first author's own photos |

The ungated "v1" hotfix studied in §7.2 of the paper writes its patched copy of TinySR-18A to
`results/exp2_out/generator_patched.h5` (`src/exp2_tinysr18_patch_v1.py`), not here.
