# Experiment 5: git for weights (Anime DCGAN)


Base weights `145ec232ed8a91e5` (legacy/Anime-DCGAN/generator50000.h5). The repo holds only text: `base.json` and `patches/*.weightpatch.json`. `wpgit test` runs each patch's tests *in context*: the full build against the full build without that patch.


## 1. Base repo

```
$ git log -1 "--format=%h %s"
d858aeb Base: Anime DCGAN generator50000
```

## 2. Three feature branches

```
$ git log -1 "--format=%h %s"
6b6dde8 Warm hair: shift latent along orthogonalised hair_warmth direction
```

```
$ git log -1 "--format=%h %s"
997e9a3 Bright background: shift latent along orthogonalised background direction
```

```
$ git log -1 "--format=%h %s"
26b6ba5 Prune 2 redundant conv2d_9 channels (fold into conv2d_10)
```

```
$ git log -1 "--format=%h %s"
9492b6d Magenta eyes: gated eye-colour circuit in the freed slots
```

## 3. Merge all three into main

Three `git merge --no-ff` calls, all clean (each branch adds its own files).

```
$ git log --graph --oneline --all
*   f278077 Merge feature/magenta-eyes
|\  
| * 9492b6d Magenta eyes: gated eye-colour circuit in the freed slots
| * 26b6ba5 Prune 2 redundant conv2d_9 channels (fold into conv2d_10)
* |   3d3974a Merge feature/bright-background
|\ \  
| * | 997e9a3 Bright background: shift latent along orthogonalised background direction
| |/  
* |   13ee7e6 Merge feature/warm-hair
|\ \  
| |/  
|/|   
| * 6b6dde8 Warm hair: shift latent along orthogonalised hair_warmth direction
|/  
* d858aeb Base: Anime DCGAN generator50000
```

```
$ git diff --stat HEAD~3 HEAD
patches/10_warm_hair.weightpatch.json         |   92 +
 patches/20_bright_background.weightpatch.json |   92 +
 patches/30_prune2.weightpatch.json            | 5834 +++++++++++++++++++++++++
 patches/31_eye_gate.weightpatch.json          | 2379 ++++++++++
 4 files changed, 8397 insertions(+)
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| eye_gate | shift_background_orth | - | 0.0e+00 | commutes (disjoint tensors) |
| eye_gate | shift_hair_warmth_orth | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_background_orth | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_hair_warmth_orth | - | 0.0e+00 | commutes (disjoint tensors) |
| shift_background_orth | shift_hair_warmth_orth | dense_2/bias | 6.0e-08 | commutes (numerically) |
0 semantic conflict(s)
```

```
$ python wpgit.py test .
model hash 9af39ba83374f269
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.012 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 298.684 | min=270 max=330 | PASS |
| eye_gate | rest-of-image MAE | 0.000 | max=0.015 | PASS |
| shift_background_orth | Δbackground (SD) | 0.760 | min=0.6 | PASS |
| shift_background_orth | Δhair_warmth (SD) | -0.013 | min=-0.3 max=0.3 | PASS |
| shift_hair_warmth_orth | Δhair_warmth (SD) | 1.538 | min=1.0 | PASS |
| shift_hair_warmth_orth | Δbackground (SD) | -0.166 | min=-0.3 max=0.3 | PASS |
ALL TESTS PASS
```

All patches compose: yes. Merged model hash `9af39ba83374f269`.


## 4. Revert one feature

```
$ git log -1 "--format=%h %s"
cef6ce3 Revert warm hair
```

```
$ python wpgit.py test .
model hash 9e2ec73b95ef0d34
| patch | test | value | bound | result |
|---|---|---|---|---|
| prune2 | rest-of-image MAE | 0.007 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 297.380 | min=270 max=330 | PASS |
| eye_gate | rest-of-image MAE | 0.001 | max=0.015 | PASS |
| shift_background_orth | Δbackground (SD) | 0.895 | min=0.6 | PASS |
| shift_background_orth | Δhair_warmth (SD) | -0.031 | min=-0.3 max=0.3 | PASS |
ALL TESTS PASS
```

After revert: hash `9e2ec73b95ef0d34`. Built directly from {bright background, prune2, eye gate} without git: `9e2ec73b95ef0d34` (match: True). Tests pass: True.


## 5. A merge git accepts but the weights do not

```
$ git log -1 "--format=%h %s"
445c99d Lock hair lightness: clamp latent coordinate
```

```
$ git log -1 "--format=%h %s"
23caa85 Lock background: clamp latent coordinate
```

git merged both without complaint. The weight-level check:

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| clamp_background | clamp_hair_lightness | dense_2/bias,dense_2/kernel | 3.5e-01 | CONFLICT: order changes output, MAE 0.0520 |
| clamp_background | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | shift_background_orth | dense_2/bias | 2.6e-01 | CONFLICT: order changes output, MAE 0.0652 |
| clamp_hair_lightness | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_hair_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_hair_lightness | shift_background_orth | dense_2/bias | 6.0e-08 | commutes (numerically) |
| eye_gate | shift_background_orth | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_background_orth | - | 0.0e+00 | commutes (disjoint tensors) |
2 semantic conflict(s)
```

cos(hair_lightness, background) = +0.788: the two clamps project out overlapping directions, so whichever is applied second partly undoes the first.


### Resolution: orthogonalise the background direction against hair lightness

New direction is background minus its hair-lightness component; target recomputed from 2.0 to 0.689 so that hair_lightness = 2.0 and background = 2.0 hold together.

```
$ git log -1 "--format=%h %s"
c7de2f1 Resolve semantic conflict: orthogonalise background clamp
```

```
$ git diff --stat HEAD~1 HEAD
patches/50_lock_background.weightpatch.json | 132 ++++++++++++++--------------
 1 file changed, 66 insertions(+), 66 deletions(-)
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| clamp_background | clamp_hair_lightness | dense_2/bias,dense_2/kernel | 6.0e-08 | commutes (numerically) |
| clamp_background | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | shift_background_orth | dense_2/bias | 5.1e-01 | CONFLICT: order changes output, MAE 0.0931 |
| clamp_hair_lightness | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_hair_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_hair_lightness | shift_background_orth | dense_2/bias | 6.0e-08 | commutes (numerically) |
| eye_gate | shift_background_orth | - | 0.0e+00 | commutes (disjoint tensors) |
| prune2 | shift_background_orth | - | 0.0e+00 | commutes (disjoint tensors) |
1 semantic conflict(s)
```

The clamp pair now commutes. The check still flags lock-background against bright-background: one freezes the background coordinate, the other moves it, so whichever runs last wins. That is a genuine contradiction in intent, not a numerical one, so the resolution is a decision: the lock supersedes the brighten.


```
$ git log -1 "--format=%h %s"
0e60268 Resolve: lock-background supersedes bright-background
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| clamp_background | clamp_hair_lightness | dense_2/bias,dense_2/kernel | 6.0e-08 | commutes (numerically) |
| clamp_background | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_background | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_hair_lightness | eye_gate | - | 0.0e+00 | commutes (disjoint tensors) |
| clamp_hair_lightness | prune2 | - | 0.0e+00 | commutes (disjoint tensors) |
0 semantic conflict(s)
```

```
$ python wpgit.py test .
model hash dab4148291fd48c5
| patch | test | value | bound | result |
|---|---|---|---|---|
| clamp_background | var ratio background vs base | 0.055 | max=0.3 | PASS |
| clamp_hair_lightness | var ratio hair_lightness vs base | 0.067 | max=0.3 | PASS |
| prune2 | rest-of-image MAE | 0.007 | max=0.015 | PASS |
| eye_gate | eye hue (deg) | 298.064 | min=270 max=330 | PASS |
| eye_gate | rest-of-image MAE | 0.001 | max=0.015 | PASS |
ALL TESTS PASS
```

```
$ git log --graph --oneline
* 0e60268 Resolve: lock-background supersedes bright-background
* c7de2f1 Resolve semantic conflict: orthogonalise background clamp
*   4fb2870 Merge feature/lock-background
|\  
| * 23caa85 Lock background: clamp latent coordinate
* |   1841197 Merge feature/lock-hair-lightness
|\ \  
| |/  
|/|   
| * 445c99d Lock hair lightness: clamp latent coordinate
|/  
* cef6ce3 Revert warm hair
*   f278077 Merge feature/magenta-eyes
|\  
| * 9492b6d Magenta eyes: gated eye-colour circuit in the freed slots
| * 26b6ba5 Prune 2 redundant conv2d_9 channels (fold into conv2d_10)
* |   3d3974a Merge feature/bright-background
|\ \  
| * | 997e9a3 Bright background: shift latent along orthogonalised background direction
| |/  
* |   13ee7e6 Merge feature/warm-hair
|\ \  
| |/  
|/|   
| * 6b6dde8 Warm hair: shift latent along orthogonalised hair_warmth direction
|/  
* d858aeb Base: Anime DCGAN generator50000
```
