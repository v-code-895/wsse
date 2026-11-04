# Experiment 7a: blind audit (weights and config only, no forward passes)

Each bullet marked **Prediction** is tested by running the model in exp7b–d.

## Recycle-Bot (`trash_hinge_new.h5`): trash vs recycling classifier

Architecture: conv2d_1[[3, 3],valid,linear] -> batch_normalization_1 -> max_pooling2d_1 -> conv2d_2[[3, 3],valid,relu] -> batch_normalization_2 -> max_pooling2d_2 -> conv2d_3[[3, 3],valid,relu] -> batch_normalization_3 -> max_pooling2d_3 -> conv2d_4[[3, 3],valid,relu] -> batch_normalization_4 -> max_pooling2d_4 -> conv2d_5[[3, 3],valid,relu] -> batch_normalization_5 -> max_pooling2d_5 -> flatten_1 -> dense_1[256,linear] -> dropout_1 -> dense_2[1,linear]

- **741,121 parameters, 80% of them in `dense_1`**, which feeds a 6×6×64 map into 256 units.
- `dense_1` is **linear** (linear) and so is `dense_2` (linear); the Dropout between them does nothing at inference. Two stacked linear layers are one linear layer: the whole head is a single 2,304-weight vector plus a bias. Prediction: folding it removes 588,032 parameters (79% of the model) with identical outputs.
- `dense_1`'s 256 units are not even used as 256 directions: only the 1-D projection through `dense_2` matters (its singular values run 4.02 … 0.91, irrelevant to the output).
- All 320 BatchNorm scales are positive (min γ/σ = 0.834). Positive per-channel scaling commutes with ReLU-free max-pooling, and every conv uses 'valid' padding, so **every BN folds exactly** into the next conv/dense. Prediction: a BN-free model with identical outputs.
- γ ranges 0.986–1.013 and β -0.014–+0.014: the BN affine parameters barely moved from their initial (1, 0). Either training was short or the learning rate on them tiny.
- Receptive field of each of the 6×6 output cells: 94 px of the 256-px image, so the classifier sees the whole object, and `conv2d_1` is linear (no activation), so layer 1 + BN + max-pool is just a learned colour/edge pooling.
- `conv2d_1` colour: 17/64 filters respond with the same sign to R, G and B (brightness/edge detectors); 47 are colour-opponent.
- **Where it looks** (share of |weight| per cell of the 6×6 grid the head reads; rows top→bottom):

```
 3.4%  2.9%  2.7%  2.4%  2.7%  2.8%
 3.2%  2.8%  2.4%  2.2%  2.4%  2.4%
 3.1%  3.1%  2.4%  2.3%  2.2%  2.5%
 3.1%  3.4%  2.4%  2.0%  2.7%  2.9%
 3.0%  3.4%  2.7%  2.2%  2.8%  2.7%
 3.6%  3.4%  2.6%  2.9%  3.1%  3.3%
```
  Centre 4×4 holds 41% of the weight (it would hold 44% if uniform); the outer ring 59%. Left column 19%, right column 17%.
- From the training code: label +1 = folder `r` (recycling), −1 = `t` (trash), inputs /255, hinge loss, 50/50 train/test split. **Prediction:** a positive score means recyclable.
- The deployed script (`linux CNN.py`) is for a different model: it loads `trash.h5`, which expects a 2-unit softmax and un-scaled 0–255 input, and still names the classes 'Dog' and 'Cat'. This hinge checkpoint has a 1-unit output and would always land in class 0 under that script's argmax.

## Deep-Color (`generator2800.h5`): grayscale → colour

Architecture: conv2d_5[[3, 3],same,linear] -> batch_normalization_4 -> leaky_re_lu_5 -> conv2d_6[[3, 3],same,linear] -> batch_normalization_5 -> leaky_re_lu_6 -> conv2d_7[[3, 3],same,linear] -> batch_normalization_6 -> leaky_re_lu_7 -> conv2d_8[[3, 3],same,linear] -> batch_normalization_7 -> leaky_re_lu_8 -> conv2d_9[[3, 3],same,linear] -> batch_normalization_8 -> leaky_re_lu_9 -> conv2d_10[[2, 2],same,linear] -> leaky_re_lu_10

- 150,403 parameters, 5 conv+BN+LeakyReLU blocks at full 128×128 resolution, no downsampling, no skip connection from input to output.
- **Receptive field: 12 px.** Every output pixel's colour is decided by a 12×12 gray patch. Prediction: it cannot know whether a region is sky, skin or grass. It can only map local brightness/texture to colour, so it behaves like a tone-mapping palette, not a semantic colorizer.
- **Input scale, read from BatchNorm:** the first BN's running variance has median 2.4e-08 (std ≈ 1.6e-04). `conv2d_5` weights have std 0.059 over 9 taps, so the inputs it saw had magnitude ~1e-3, not ~0.5. That matches the code: skimage's `rgb2gray` already returns 0–1, and the code divides by 255 again, so **the model was trained on gray values in [0, 0.0039]**. Prediction: feeding normal 0–1 gray images breaks it; the fix is an exact fold of 1/255 into `conv2d_5`.
- BN running variances climb from 2e-08 to 8e+00 through the blocks: the network re-amplifies the tiny input signal by roughly 10^4 in amplitude.
- **Output conv is 2×2 with 'same' padding.** TensorFlow pads even kernels on the bottom/right only, so each output pixel is computed from pixels (i, i+1): the colour is shifted half a pixel up-left relative to the input.
- Output bias = [-0.006000000052154064, -0.004999999888241291, 0.003000000026077032] (RGB): where features are near zero the image defaults to that colour.
- From the training code: the generator's only loss is fooling a discriminator that sees the colour image **without** the gray input. Nothing ties the colours, or even the brightness, to the input. Prediction: output brightness drifts from the input, and a fixed colour-space fix (keep the input's luminance, take only chroma from the model) can only help.

## Cat-GAN `generator580.h5` vs PCA-GAN `model.h5`

Same architecture (dense_2[512,linear] -> reshape_1 -> conv2d_7[[5, 5],same,linear] -> leaky_re_lu_1 -> up_sampling2d_1 -> conv2d_8[[5, 5],same,linear] -> leaky_re_lu_2 -> up_sampling2d_2 -> conv2d_9[[5, 5],same,linear] -> leaky_re_lu_3 -> up_sampling2d_3 -> conv2d_10[[5, 5],same,linear] -> leaky_re_lu_4 -> conv2d_11[[5, 5],same,linear] -> leaky_re_lu_5), same 587,715 parameters. Cat-GAN's notebook saves `generator<epoch>.h5` every 10 epochs from a PCA-input GAN on the same cat dataset, so these could be two checkpoints of one run.

| tensor | correlation | relative distance |
|---|---|---|
| conv2d_10/kernel | -0.002 | 1.54 |
| conv2d_11/kernel | +0.025 | 1.42 |
| conv2d_7/kernel | -0.003 | 1.28 |
| conv2d_8/kernel | -0.001 | 1.55 |
| conv2d_9/kernel | -0.004 | 1.56 |
| dense_2/kernel | +0.004 | 1.88 |

- Same-index filters in `conv2d_8`: mean |cos| 0.019; best match anywhere: mean 0.070. Random 1,600-d vectors give |cos| ≈ 0.02.
- Effective rank of `dense_2`: Cat-GAN 266/512, PCA-GAN 217/512.
- **Reading:** kernel correlations average +0.00 and same-index filters don't match, so these are **separate training runs**, not checkpoints of one run. Interpolating between them would mix unrelated coordinate systems; merging them first needs the channels aligned (permutation matching on activations, which needs PCA inputs).
- Both take 512 PCA coefficients of the Kaggle cat-faces dataset as input. Neither repo ships the PCA basis or dataset (`pca_dataset.npy` lived on the original PC), so their outputs can't be sampled faithfully without re-downloading that dataset.

