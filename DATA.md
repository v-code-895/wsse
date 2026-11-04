# Data

No dataset is redistributed in this repository. `python scripts/download_data.py <name>` fetches
each one into `data/` in the layout the scripts expect. Check each source's licence before use.

| name | used by | source | licence (per the source) | path |
|---|---|---|---|---|
| MobileNetV2 ImageNet weights + class index | exp8, exp8b, exp12, exp13, exp15 (refactors, oracle, Fréchet distance) | Keras Applications release on storage.googleapis.com | see the Keras Applications project | `data/mobilenet_v2/` |
| AFHQ (animal faces, 512×512; 5,153 training cats) [Choi et al., StarGAN v2, CVPR 2020] | exp10–exp17 | original: github.com/clovaai/stargan-v2; downloaded via the Kaggle mirror `andrewmvd/animal-faces` | CC BY-NC 4.0 (NAVER Corporation) | `data/afhq/data/afhq/{train,val}/{cat,dog,wild}` |

The 2017–2020 generators are sampled from their latent spaces, and the CPU experiments' test
images are synthetic sprites, scikit-image's sample photos and images from the original
repositories in `legacy/`, so §7 needs no dataset beyond MobileNetV2.

**Caches written by the scripts** (not in git): `results/exp11_out/afhq_cat_train_128.npy` (AFHQ
cats at 128×128).
