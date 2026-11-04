# Models

All weights are Keras-style dicts in `.npz` files (`layer/kernel` in HWIO order, `layer/bias`), read by
the NumPy engine and by `modern_np.py`. Licence: **CC BY-NC 4.0** (the 2026 models are trained on
AFHQ, which is CC BY-NC 4.0). "hash" is `weights_hash`, the 16-character content hash that patches pin.

## In this repository

| model | file | parameters | hash | sha256 (file) |
|---|---|---|---|---|
| TinySR-26 2× | `results/exp10_out/tinysr26_x2.npz` | 276,732 | `8cb6601c1b449ffc` | `b92f340d0a0c36c3e060fb068671106dbb37e5ba366b4742197428770eec73fb` |
| TinySR-26 4× | `results/exp10_out/tinysr26_x4.npz` | 292,320 | `5f26597cb48da190` | `6d06b070092d1462b8d716d05b96013c583e64d38b63a921ea1836cc607b8052` |
| 2017–2020 Keras checkpoints | `legacy/*/` | 150K–741K | see the paper | `legacy/MANIFEST.json` |

## Release assets (too large for git)

| model | path the scripts expect | size | hash | sha256 |
|---|---|---|---|---|
| CatGAN-26 (EMA generator, BatchNorm folded) | `results/exp11_out/catgan26.npz` | 32.5 MB | `be2b16ff52fc4021` | `3aa01a82bffc6a82f3ec1f007625bb8fe67602386347124a406b878b7fa9b23a` |
| CatGAN-26 with its op list (base of the cat weight repositories) | `results/exp13_out/catgan26_with_ops.npz` | 32.5 MB | `be2b16ff52fc4021` | `abf82f3dbaffc7cbabd70440413e7ea10a3ff4adb49aa55d1eeff62b3ffb73ae` |
| CatGAN-26 training checkpoint (G, D, EMA, both Adam states; step 30,000) | `results/exp11_out/catgan26.pt` | 217 MB | — | `b1fb0182295916086e9a2a0f8572a94e4cd9dec449df8d329f79ec03b7815bc6` |

`python scripts/fetch_models.py` downloads the two weight files from the v1.0.0 release and checks
their SHA-256 (`--with-checkpoint` adds the training checkpoint, `--base-url` selects another mirror
such as the Zenodo record). `catgan26_with_ops.npz` can also be rebuilt from `catgan26.npz` by
running `src/exp13_catgan26_patches.py`. The training checkpoint is needed only to continue training
(`src/exp11_catgan26.py`, `src/exp17_rebase.py`).

## Intended use

Research on editing, testing and versioning model weights. CatGAN-26 generates 128×128 cat faces
and TinySR-26 upsamples photos 2× or 4×. Neither is meant for production use. CatGAN-26 samples are
recognisably cats (an ImageNet classifier calls 96% of them domestic cats). In MobileNetV2 feature
space their distribution is further from held-out cats than real training cats are (Fréchet
distance 60.6 vs 47.1). §4.1 of the paper reports the evaluation and a memorisation check.
