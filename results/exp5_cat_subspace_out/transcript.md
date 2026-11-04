# Experiment 5: git for weights (CatGAN-26)


Base weights `be2b16ff52fc4021` (results/exp13_out/catgan26_with_ops.npz). The repo holds only text: `base.json` and `patches/*.weightpatch.json`. `wpgit test` runs each patch's tests *in context*: the full build against the full build without that patch.


## 1. Base repo

```
$ git log -1 "--format=%h %s"
17a7b76 Base: CatGAN-26 (AFHQ cats, 128x128)
```

## 2. Three feature branches

```
$ git log -1 "--format=%h %s"
b4dc461 Warm fur: shift latent along orthogonalised fur_warmth direction
```

```
$ git log -1 "--format=%h %s"
164dbae Bright background: shift latent along orthogonalised background direction
```

```
$ git log -1 "--format=%h %s"
d978fcb Prune 2 redundant b5c2 channels (fold into out)
```

```
$ git log -1 "--format=%h %s"
0225663 Blue eyes: gated eye-colour circuit in the freed slots
```

## 3. Merge all three into main

Three `git merge --no-ff` calls, all clean (each branch adds its own files).

```
$ git log --graph --oneline --all
*   ed1d6a1 Merge feature/blue-eyes
|\  
| * 0225663 Blue eyes: gated eye-colour circuit in the freed slots
| * d978fcb Prune 2 redundant b5c2 channels (fold into out)
* |   f9d978e Merge feature/bright-background
|\ \  
| * | 164dbae Bright background: shift latent along orthogonalised background direction
| |/  
* |   3b8a0c5 Merge feature/warm-fur
|\ \  
| |/  
|/|   
| * b4dc461 Warm fur: shift latent along orthogonalised fur_warmth direction
|/  
* 17a7b76 Base: CatGAN-26 (AFHQ cats, 128x128)
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
c1a5c6d Revert warm fur
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
1db987a Lock fur lightness: clamp rank-2 latent subspace
```

```
$ git log -1 "--format=%h %s"
31c6ee9 Lock background: clamp rank-2 latent subspace
```

git merged both without complaint. The weight-level check:

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| eye_gate | lock_background | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | lock_fur_lightness | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_background | lock_fur_lightness | dense/bias,dense/kernel | 7.5e-01 | CONFLICT: order changes output, MAE 0.1286 |
| lock_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_background | shift_background_orthnd | dense/bias | 1.1e+00 | CONFLICT: order changes output, MAE 0.0826 |
| lock_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_fur_lightness | shift_background_orthnd | dense/bias | 2.6e-01 | CONFLICT: order changes output, MAE 0.0467 |
| prune2 | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
3 semantic conflict(s)
```

Cosines of the principal angles between the fur_lightness subspace (rank 2) and the background subspace (rank 2): 0.547, 0.148. The two clamps overlap, so whichever is applied second partly undoes the first.


### Resolution: orthogonalise the background subspace against fur lightness

New subspace is the background subspace minus its fur-lightness components (Gram-Schmidt); targets recomputed from [2.0, 0.0] to [2.345, 1.271] so that both locks keep their original targets together.

```
$ git log -1 "--format=%h %s"
3cde449 Resolve semantic conflict: orthogonalise background subspace lock
```

```
$ git diff --stat HEAD~1 HEAD
patches/50_lock_background.weightpatch.json | 518 ++++++++++++++--------------
 1 file changed, 259 insertions(+), 259 deletions(-)
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
| lock_background | shift_background_orthnd | dense/bias | 1.0e+00 | CONFLICT: order changes output, MAE 0.0796 |
| lock_fur_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| lock_fur_lightness | shift_background_orthnd | dense/bias | 2.6e-01 | CONFLICT: order changes output, MAE 0.0467 |
| prune2 | shift_background_orthnd | - | 0.0e+00 | commutes (disjoint tensors) |
2 semantic conflict(s)
```

The clamp pair now commutes. The check still flags lock-background against bright-background: one freezes the background coordinate, the other moves it, so whichever runs last wins. That is a genuine contradiction in intent, not a numerical one, so the resolution is a decision: the lock supersedes the brighten.


```
$ git log -1 "--format=%h %s"
886bf1e Resolve: lock-background supersedes bright-background
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
model hash 0606c0c758456bef
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.003 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 224.635 | min=200 max=250 | PASS |
| eye_gate | rest-of-image MAE | 0.000 | max=0.015 | PASS |
| lock_background | var ratio background vs base | 0.114 | max=0.3 | PASS |
| lock_fur_lightness | var ratio fur_lightness vs base | 0.101 | max=0.3 | PASS |
ALL TESTS PASS
```

```
$ git log --graph --oneline
* 886bf1e Resolve: lock-background supersedes bright-background
* 3cde449 Resolve semantic conflict: orthogonalise background subspace lock
*   d612032 Merge feature/lock-background
|\  
| * 31c6ee9 Lock background: clamp rank-2 latent subspace
* |   5fdeb04 Merge feature/lock-fur-lightness
|\ \  
| |/  
|/|   
| * 1db987a Lock fur lightness: clamp rank-2 latent subspace
|/  
* c1a5c6d Revert warm fur
*   ed1d6a1 Merge feature/blue-eyes
|\  
| * 0225663 Blue eyes: gated eye-colour circuit in the freed slots
| * d978fcb Prune 2 redundant b5c2 channels (fold into out)
* |   f9d978e Merge feature/bright-background
|\ \  
| * | 164dbae Bright background: shift latent along orthogonalised background direction
| |/  
* |   3b8a0c5 Merge feature/warm-fur
|\ \  
| |/  
|/|   
| * b4dc461 Warm fur: shift latent along orthogonalised fur_warmth direction
|/  
* 17a7b76 Base: CatGAN-26 (AFHQ cats, 128x128)
```
