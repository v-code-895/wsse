# Experiment 5: git for weights (CatGAN-26)


Base weights `be2b16ff52fc4021` (results/exp13_out/catgan26_with_ops.npz). The repo holds only text: `base.json` and `patches/*.weightpatch.json`. `wpgit test` runs each patch's tests *in context*: the full build against the full build without that patch.


## 1. Base repo

```
$ git log -1 "--format=%h %s"
fdda2c6 Base: CatGAN-26 (AFHQ cats, 128x128)
```

## 2. Three feature branches

```
$ git log -1 "--format=%h %s"
4c473dc Warm fur: shift latent along orthogonalised fur_warmth direction
```

```
$ git log -1 "--format=%h %s"
f5267b8 Bright background: shift latent along orthogonalised background direction
```

```
$ git log -1 "--format=%h %s"
b306f40 Prune 2 redundant b5c2 channels (fold into out)
```

```
$ git log -1 "--format=%h %s"
a2a2198 Blue eyes: gated eye-colour circuit in the freed slots
```

## 3. Merge all three into main

Three `git merge --no-ff` calls, all clean (each branch adds its own files).

```
$ git log --graph --oneline --all
*   296eda8 Merge feature/blue-eyes
|\  
| * a2a2198 Blue eyes: gated eye-colour circuit in the freed slots
| * b306f40 Prune 2 redundant b5c2 channels (fold into out)
* |   83b624d Merge feature/bright-background
|\ \  
| * | f5267b8 Bright background: shift latent along orthogonalised background direction
| |/  
* |   9a61881 Merge feature/warm-fur
|\ \  
| |/  
|/|   
| * 4c473dc Warm fur: shift latent along orthogonalised fur_warmth direction
|/  
* fdda2c6 Base: CatGAN-26 (AFHQ cats, 128x128)
```

```
$ git diff --stat HEAD~3 HEAD
patches/10_warm_fur.weightpatch.json          |  156 ++
 patches/20_bright_background.weightpatch.json |  156 ++
 patches/30_prune2.weightpatch.json            | 2954 +++++++++++++++++++++++++
 patches/31_eye_gate.weightpatch.json          | 1227 ++++++++++
 4 files changed, 4493 insertions(+)
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| eye_gate | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | shift_fur_warmth_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_fur_warmth_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| shift_background_orthnd | shift_fur_warmth_orthnd | dense/bias | 1.2e-07 | commutes (numerically) |
0 semantic conflict(s)
```

```
$ python wpgit.py test .
model hash 553a8ea202934cbb
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.009 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 229.809 | min=200 max=250 | PASS |
| eye_gate | rest-of-image MAE | 0.004 | max=0.015 | PASS |
| shift_background_orthnd | Δbackground (SD) | 1.012 | min=0.6 | PASS |
| shift_background_orthnd | Δfur_warmth (SD) | -0.085 | min=-0.3 max=0.3 | PASS |
| shift_fur_warmth_orthnd | Δfur_warmth (SD) | 0.922 | min=1.0 | FAIL |
| shift_fur_warmth_orthnd | Δbackground (SD) | -0.546 | min=-0.3 max=0.3 | FAIL |
TESTS FAILED
```

All patches compose: NO. Merged model hash `553a8ea202934cbb`.


## 4. Revert one feature

```
$ git log -1 "--format=%h %s"
4e56cc6 Revert warm fur
```

```
$ python wpgit.py test .
model hash b563872899972665
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.005 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 196.972 | min=200 max=250 | FAIL |
| eye_gate | rest-of-image MAE | 0.001 | max=0.015 | PASS |
| shift_background_orthnd | Δbackground (SD) | 1.416 | min=0.6 | PASS |
| shift_background_orthnd | Δfur_warmth (SD) | -0.059 | min=-0.3 max=0.3 | PASS |
TESTS FAILED
```

After revert: hash `b563872899972665`. Built directly from {bright background, prune2, eye gate} without git: `b563872899972665` (match: True). Tests pass: False.


## 5. A merge git accepts but the weights do not

```
$ git log -1 "--format=%h %s"
ca2fa76 Lock fur lightness: clamp latent coordinate
```

```
$ git log -1 "--format=%h %s"
54d93fa Lock background: clamp latent coordinate
```

git merged both without complaint. The weight-level check:

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| clamp_background | clamp_fur_lightness | dense/bias,dense/kernel | 3.1e-01 | CONFLICT: order changes output, MAE 0.0519 |
| clamp_background | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | shift_background_orthnd | dense/bias | 9.9e-01 | CONFLICT: order changes output, MAE 0.1032 |
| clamp_fur_lightness | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_fur_lightness | shift_background_orthnd | dense/bias | 1.2e-07 | commutes (numerically) |
| eye_gate | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
2 semantic conflict(s)
```

cos(fur_lightness, background) = -0.154: the two clamps project out overlapping directions, so whichever is applied second partly undoes the first.


### Resolution: orthogonalise the background direction against fur lightness

New direction is background minus its fur-lightness component; target recomputed from 2.0 to 2.336 so that fur_lightness = 2.0 and background = 2.0 hold together.

```
$ git log -1 "--format=%h %s"
dd9b5b3 Resolve semantic conflict: orthogonalise background clamp
```

```
$ git diff --stat HEAD~1 HEAD
patches/50_lock_background.weightpatch.json | 260 ++++++++++++++--------------
 1 file changed, 130 insertions(+), 130 deletions(-)
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| clamp_background | clamp_fur_lightness | dense/bias,dense/kernel | 1.2e-07 | commutes (numerically) |
| clamp_background | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | shift_background_orthnd | dense/bias | 9.7e-01 | CONFLICT: order changes output, MAE 0.0829 |
| clamp_fur_lightness | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_fur_lightness | shift_background_orthnd | dense/bias | 1.2e-07 | commutes (numerically) |
| eye_gate | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
1 semantic conflict(s)
```

The clamp pair now commutes. The check still flags lock-background against bright-background: one freezes the background coordinate, the other moves it, so whichever runs last wins. That is a genuine contradiction in intent, not a numerical one, so the resolution is a decision: the lock supersedes the brighten.


```
$ git log -1 "--format=%h %s"
bd384be Resolve: lock-background supersedes bright-background
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| clamp_background | clamp_fur_lightness | dense/bias,dense/kernel | 1.2e-07 | commutes (numerically) |
| clamp_background | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_fur_lightness | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
0 semantic conflict(s)
```

```
$ python wpgit.py test .
model hash 8d5ab1c4e4f80aac
| patch | test | value | bound | result |
|---|---|---|---|---|
| clamp_background | var ratio background vs base | 0.677 | max=0.3 | FAIL |
| clamp_fur_lightness | var ratio fur_lightness vs base | 0.708 | max=0.3 | FAIL |
| prune2 | rest-of-image MAE | 0.007 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 221.909 | min=200 max=250 | PASS |
| eye_gate | rest-of-image MAE | 0.003 | max=0.015 | PASS |
TESTS FAILED
```

```
$ git log --graph --oneline
* bd384be Resolve: lock-background supersedes bright-background
* dd9b5b3 Resolve semantic conflict: orthogonalise background clamp
*   4269c01 Merge feature/lock-background
|\  
| * 54d93fa Lock background: clamp latent coordinate
* |   31fc3d6 Merge feature/lock-fur-lightness
|\ \  
| |/  
|/|   
| * ca2fa76 Lock fur lightness: clamp latent coordinate
|/  
* 4e56cc6 Revert warm fur
*   296eda8 Merge feature/blue-eyes
|\  
| * a2a2198 Blue eyes: gated eye-colour circuit in the freed slots
| * b306f40 Prune 2 redundant b5c2 channels (fold into out)
* |   83b624d Merge feature/bright-background
|\ \  
| * | f5267b8 Bright background: shift latent along orthogonalised background direction
| |/  
* |   9a61881 Merge feature/warm-fur
|\ \  
| |/  
|/|   
| * 4c473dc Warm fur: shift latent along orthogonalised fur_warmth direction
|/  
* fdda2c6 Base: CatGAN-26 (AFHQ cats, 128x128)
```
