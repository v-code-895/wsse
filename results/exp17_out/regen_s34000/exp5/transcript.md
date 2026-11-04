# Experiment 5: git for weights (CatGAN-26)


Base weights `bf629f2a3ba0b126` (exp17_out\regen_s34000\exp13\catgan26_with_ops.npz). The repo holds only text: `base.json` and `patches/*.weightpatch.json`. `wpgit test` runs each patch's tests *in context*: the full build against the full build without that patch.


## 1. Base repo

```
$ git log -1 "--format=%h %s"
e1da9fc Base: CatGAN-26 (AFHQ cats, 128x128)
```

## 2. Three feature branches

```
$ git log -1 "--format=%h %s"
8c4cc45 Warm fur: shift latent along orthogonalised fur_warmth direction
```

```
$ git log -1 "--format=%h %s"
7d36c3e Bright background: shift latent along orthogonalised background direction
```

```
$ git log -1 "--format=%h %s"
b71f5ec Prune 2 redundant b5c2 channels (fold into out)
```

```
$ git log -1 "--format=%h %s"
72a9471 Blue eyes: gated eye-colour circuit in the freed slots
```

## 3. Merge all three into main

Three `git merge --no-ff` calls, all clean (each branch adds its own files).

```
$ git log --graph --oneline --all
*   124475a Merge feature/blue-eyes
|\  
| * 72a9471 Blue eyes: gated eye-colour circuit in the freed slots
| * b71f5ec Prune 2 redundant b5c2 channels (fold into out)
* |   66fc074 Merge feature/bright-background
|\ \  
| * | 7d36c3e Bright background: shift latent along orthogonalised background direction
| |/  
* |   ffabd55 Merge feature/warm-fur
|\ \  
| |/  
|/|   
| * 8c4cc45 Warm fur: shift latent along orthogonalised fur_warmth direction
|/  
* e1da9fc Base: CatGAN-26 (AFHQ cats, 128x128)
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
model hash ad9c861bf9736e58
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.014 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 230.512 | min=200 max=250 | PASS |
| eye_gate | rest-of-image MAE | 0.009 | max=0.015 | PASS |
| shift_background_orthnd | Δbackground (SD) | 0.846 | min=0.6 | PASS |
| shift_background_orthnd | Δfur_warmth (SD) | -0.014 | min=-0.3 max=0.3 | PASS |
| shift_fur_warmth_orthnd | Δfur_warmth (SD) | 0.933 | min=1.0 | FAIL |
| shift_fur_warmth_orthnd | Δbackground (SD) | -0.554 | min=-0.3 max=0.3 | FAIL |
TESTS FAILED
```

All patches compose: NO. Merged model hash `ad9c861bf9736e58`.


## 4. Revert one feature

```
$ git log -1 "--format=%h %s"
865f036 Revert warm fur
```

```
$ python wpgit.py test .
model hash 8ab0a9a81dc67613
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.006 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 201.878 | min=200 max=250 | PASS |
| eye_gate | rest-of-image MAE | 0.001 | max=0.015 | PASS |
| shift_background_orthnd | Δbackground (SD) | 1.312 | min=0.6 | PASS |
| shift_background_orthnd | Δfur_warmth (SD) | -0.051 | min=-0.3 max=0.3 | PASS |
ALL TESTS PASS
```

After revert: hash `8ab0a9a81dc67613`. Built directly from {bright background, prune2, eye gate} without git: `8ab0a9a81dc67613` (match: True). Tests pass: True.


## 5. A merge git accepts but the weights do not

```
$ git log -1 "--format=%h %s"
ce3c03a Lock fur lightness: clamp rank-2 latent subspace
```

```
$ git log -1 "--format=%h %s"
025f4ff Lock background: clamp rank-4 latent subspace
```

git merged both without complaint. The weight-level check:

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| eye_gate | lock_background | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | lock_fur_lightness | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_background | lock_fur_lightness | dense/bias,dense/kernel | 9.0e-01 | CONFLICT: order changes output, MAE 0.1081 |
| lock_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_background | shift_background_orthnd | dense/bias | 1.1e+00 | CONFLICT: order changes output, MAE 0.0861 |
| lock_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_fur_lightness | shift_background_orthnd | dense/bias | 9.3e-02 | CONFLICT: order changes output, MAE 0.0177 |
| prune2 | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
3 semantic conflict(s)
```

Cosines of the principal angles between the fur_lightness subspace (rank 2) and the background subspace (rank 4): 0.789, 0.385. The two clamps overlap, so whichever is applied second partly undoes the first.


### Resolution: orthogonalise the background subspace against fur lightness

New subspace is the background subspace minus its fur-lightness components (Gram-Schmidt); targets recomputed from [2.0, 0.0, 0.0, 0.0] to [2.128, -0.525, 1.469, -1.424] so that both locks keep their original targets together.

```
$ git log -1 "--format=%h %s"
fc64eb5 Resolve semantic conflict: orthogonalise background subspace lock
```

```
$ git diff --stat HEAD~1 HEAD
patches/50_lock_background.weightpatch.json | 1034 +++++++++++++--------------
 1 file changed, 517 insertions(+), 517 deletions(-)
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| eye_gate | lock_background | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | lock_fur_lightness | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_background | lock_fur_lightness | dense/bias,dense/kernel | 2.4e-07 | commutes (numerically) |
| lock_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_background | shift_background_orthnd | dense/bias | 1.1e+00 | CONFLICT: order changes output, MAE 0.0741 |
| lock_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_fur_lightness | shift_background_orthnd | dense/bias | 9.3e-02 | CONFLICT: order changes output, MAE 0.0177 |
| prune2 | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
2 semantic conflict(s)
```

The clamp pair now commutes. The check still flags lock-background against bright-background: one freezes the background coordinate, the other moves it, so whichever runs last wins. That is a genuine contradiction in intent, not a numerical one, so the resolution is a decision: the lock supersedes the brighten.


```
$ git log -1 "--format=%h %s"
5254900 Resolve: lock-background supersedes bright-background
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| eye_gate | lock_background | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | lock_fur_lightness | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_background | lock_fur_lightness | dense/bias,dense/kernel | 2.4e-07 | commutes (numerically) |
| lock_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
0 semantic conflict(s)
```

```
$ python wpgit.py test .
model hash b80a1cce95f647ee
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.008 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 25.975 | min=200 max=250 | FAIL |
| eye_gate | rest-of-image MAE | 0.004 | max=0.015 | PASS |
| lock_background | var ratio background vs base | 0.084 | max=0.3 | PASS |
| lock_fur_lightness | var ratio fur_lightness vs base | 0.127 | max=0.3 | PASS |
TESTS FAILED
```

```
$ git log --graph --oneline
* 5254900 Resolve: lock-background supersedes bright-background
* fc64eb5 Resolve semantic conflict: orthogonalise background subspace lock
*   ee88804 Merge feature/lock-background
|\  
| * 025f4ff Lock background: clamp rank-4 latent subspace
* |   dd3bba0 Merge feature/lock-fur-lightness
|\ \  
| |/  
|/|   
| * ce3c03a Lock fur lightness: clamp rank-2 latent subspace
|/  
* 865f036 Revert warm fur
*   124475a Merge feature/blue-eyes
|\  
| * 72a9471 Blue eyes: gated eye-colour circuit in the freed slots
| * b71f5ec Prune 2 redundant b5c2 channels (fold into out)
* |   66fc074 Merge feature/bright-background
|\ \  
| * | 7d36c3e Bright background: shift latent along orthogonalised background direction
| |/  
* |   ffabd55 Merge feature/warm-fur
|\ \  
| |/  
|/|   
| * 8c4cc45 Warm fur: shift latent along orthogonalised fur_warmth direction
|/  
* e1da9fc Base: CatGAN-26 (AFHQ cats, 128x128)
```
