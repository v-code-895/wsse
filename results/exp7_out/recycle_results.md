# Experiment 7b: Recycle-Bot

| image | score (positive = recycling) |
|---|---|
| 44.gif cup (trash) | -3.666 |
| 63.gif can (recycling) | +0.652 |

## Exact compression

- 741,121 → **151,809 parameters** (80% smaller), no BatchNorm, one 2,304-weight linear head.
- Max score difference on 10 inputs (2 real, 4 noise, 4 brightened): 2.9e-06
- New architecture: conv2d_1[[3, 3],valid,linear] -> max_pooling2d_1 -> conv2d_2[[3, 3],valid,relu] -> max_pooling2d_2 -> conv2d_3[[3, 3],valid,relu] -> max_pooling2d_3 -> conv2d_4[[3, 3],valid,relu] -> max_pooling2d_4 -> conv2d_5[[3, 3],valid,relu] -> max_pooling2d_5 -> flatten_1 -> head[1,linear]

## BatchNorm forensics

BN1 running variance median 0.8347 ≈ 0.99^n  ⇒  **n ≈ 18.0 update steps** read from the weights. The notebook log: 63 training images, batch 25 (3 steps/epoch), early stopping restored epoch 6 ⇒ 18 steps.

So the stored running stats are ~83% initial values: at inference the network does not normalise the way it did in training. Fix (weights only): bias-correct each running stat, true = (stored − 0.99^n·init)/(1 − 0.99^n), n = 18.

## Probes (score per image; positive = recycling)

| input | cup (base) | can (base) | cup (BN bias-corrected) | can (BN bias-corrected) |
|---|---|---|---|---|
| original | -3.666 | +0.652 | -60.613 | +51.420 |
| brightness x0.7 | +0.403 | +3.843 | -35.633 | +43.855 |
| brightness x1.3 | -6.948 | -1.019 | -66.959 | +53.398 |
| horizontal flip | -4.074 | +1.191 | -18.278 | +46.121 |
| vertical flip | -4.339 | +0.807 | -49.740 | +47.287 |
| rotate 90 | -4.281 | +2.365 | -37.635 | +52.183 |
| grayscale | -5.538 | -1.599 | -70.521 | -2.042 |
| shift right 32px | -3.122 | +2.665 | -34.463 | +70.126 |
| unscaled 0-255 input | -2657.860 | -1902.683 | -14895.375 | -1610.855 |

## Where it looks

Occlusion (32-px gray patch, 8×8 grid) vs the blind weight map: correlation -0.21 (cup), -0.21 (can). See recycle_occlusion.png.
