# Experiment 24: the odd-eye circuit under wpgit

```
$ git log --oneline -1
886bf1e Resolve: lock-background supersedes bright-background
```

Base be2b16ff52fc4021; patches at HEAD: eye_gate, lock_background, lock_fur_lightness, prune2.

```
$ git checkout -q -b odd-eyes

```

Patch: {"grow_ops": 11, "sparse_ops": 13, "entries": 603, "params_added": 42548, "rebuild_max_err_vs_exp20": 0.0, "exp20_left_share": 0.41605839416058393, "exp20_right_share": 0.0, "quantile": 0.98}

```
$ git log --oneline -1
0cf4794 Replace the blue-eye circuit with a grown-channel left-eye circuit
```

```
$ python wpgit.py test .
model hash a8c7f2d50ae04911
| patch | test | value | bound | result |
|---|---|---|---|---|
| left_eye_blue | left irises in 200-250 deg (share) | 0.341 | min=0.3 | PASS |
| left_eye_blue | right irises in 200-250 deg (share) | 0.000 | max=0.0 | PASS |
| left_eye_blue | max change, right half | 2.9e-06 | max=1e-05 | PASS |
| left_eye_blue | rest-of-image MAE | 1.2e-04 | max=0.015 | PASS |
| lock_background | var ratio background vs base | 0.114 | max=0.3 | PASS |
| lock_fur_lightness | var ratio fur_lightness vs base | 0.099 | max=0.3 | PASS |
ALL TESTS PASS
```

```
$ python wpgit.py check .
| patch A | patch B | shared tensors | max weight diff A∘B vs B∘A | verdict |
|---|---|---|---|---|
| left_eye_blue | lock_background | dense/bias,dense/kernel | 0.0e+00 | commutes (numerically) |
| left_eye_blue | lock_fur_lightness | dense/bias,dense/kernel | 0.0e+00 | commutes (numerically) |
| lock_background | lock_fur_lightness | dense/bias,dense/kernel | 2.4e-07 | commutes (numerically) |
0 semantic conflict(s)
```

Synthetic update, output layer only: left_eye_blue clean, lock_fur_lightness clean, lock_background clean

Synthetic update, a layer the gate reads (b3c1): left_eye_blue STALE, lock_fur_lightness clean, lock_background clean
