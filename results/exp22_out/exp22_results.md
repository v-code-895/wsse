# Experiment 22: the fine-tune merge conflict, resolved by a routed union

PSNR (dB) on 180 held-out AFHQ photos per input type; Δ vs base.

| model | bicubic LR | noisy bicubic LR | nearest LR | meets spec |
|---|---|---|---|---|
| base | 32.53 (+0.00) | 26.89 (+0.00) | 31.12 (+0.00) | False |
| branch: bicubic | 32.76 (+0.23) | 26.42 (-0.47) | 26.24 (-4.88) | False |
| branch: noisy | 32.32 (-0.20) | 30.15 (+3.26) | 30.11 (-1.01) | False |
| union (input statistics) | 32.73 (+0.21) | 29.59 (+2.69) | 30.16 (-0.96) | False |
| union (self-consistency) | 32.76 (+0.23) | 30.15 (+3.26) | 31.12 (+0.00) | True |
| union (oracle) | 32.76 (+0.23) | 30.15 (+3.26) | 31.12 (+0.00) | True |

Routers: {"input statistics": {"fit_accuracy": 0.8138888888888889, "params": 43, "features": 8}, "self-consistency": {"fit_accuracy": 1.0, "params": 33, "features": 6}}

Confusion on test (rows: input; cols: base / bicubic branch / noisy branch): {"input statistics": [[8, 166, 6], [23, 18, 139], [127, 30, 23]], "self-consistency": [[0, 180, 0], [0, 0, 180], [180, 0, 0]]}

Multiplexer vs plain selection, max abs difference: 6.0e-08

Joint retraining (exp16): {"bicubic LR": 0.02881729161620683, "noisy bicubic LR": 3.1908093491931844, "nearest LR": 0.07528812915342087}
