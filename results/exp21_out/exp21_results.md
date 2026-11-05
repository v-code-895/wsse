# Experiment 21: TinySR-18A rewired with a reference path and a min-gate

Gate of exp2b (nearest-neighbour LR): regression ΔPSNR ≥ −0.10 dB and ΔSSIM ≥ −0.005; edge PSNR gain; black floor below base.

| variant | regression PSNR | edge PSNR | edge black floor | params | gate |
|---|---|---|---|---|---|
| base_model | 21.75 | 19.61 | 8.90 | 317,059 | reference |
| color_fold | 24.68 | 21.42 | 13.51 | 317,059 | FAIL (black floor rises) |
| guard | 21.43 | 19.47 | 5.84 | 362,119 | FAIL (regression) |
| fold+guard | 25.11 | 21.63 | 7.33 | 362,119 | PASS |
| fold+guard (freed channels) | 25.20 | 21.44 | 7.11 | 331,507 | PASS |
| fold+dark guard | 25.12 | 22.01 | 7.88 | 372,403 | PASS |
| fold+dark guard (freed channels) | 25.18 | 21.80 | 7.61 | 341,341 | PASS |
| fine-tune 250 steps | 24.69 | 22.03 | 1.87 | 317,059 | FAIL (regression) |
| fine-tune 500 steps | 26.09 | 22.49 | 1.62 | 317,059 | PASS |
| fine-tune 750 steps | 25.76 | 22.51 | 1.59 | 317,059 | PASS |
| fine-tune 1000 steps | 24.81 | 22.35 | 1.44 | 317,059 | PASS |
| fine-tune 1250 steps | 24.79 | 22.35 | 1.45 | 317,059 | FAIL (regression) |
| fine-tune 1500 steps | 25.26 | 22.51 | 1.52 | 317,059 | PASS |
| fine-tune 1750 steps | 26.43 | 22.83 | 1.69 | 317,059 | PASS |
| fine-tune 2000 steps | 26.54 | 22.96 | 1.92 | 317,059 | PASS |
| fine-tune 2250 steps | 25.28 | 22.67 | 1.74 | 317,059 | PASS |
| fine-tune 2500 steps | 25.51 | 22.71 | 1.77 | 317,059 | PASS |
| fine-tune 2750 steps | 26.34 | 22.96 | 1.94 | 317,059 | PASS |
| fine-tune 3000 steps | 26.85 | 23.21 | 2.19 | 317,059 | PASS |
| fine-tune 3250 steps | 25.89 | 23.15 | 2.34 | 317,059 | PASS |
| fine-tune 3500 steps | 25.15 | 23.02 | 2.28 | 317,059 | PASS |
| fine-tune 3750 steps | 26.68 | 23.29 | 2.53 | 317,059 | PASS |
| fine-tune 4000 steps | 26.82 | 23.24 | 2.26 | 317,059 | PASS |

Chosen guard settings (calibration only): {"guard": {"k": 0.0, "delta_rgb": 1.0, "calib_psnr": 20.814626413216928, "calib_black_floor": 4.1240823917490035, "unguarded_calib_psnr": 20.864470861652833, "unguarded_calib_black_floor": 6.930146370349186}, "fold+guard": {"k": 0.0, "delta_rgb": 1.0, "calib_psnr": 23.979984791552866, "calib_black_floor": 5.266188865355625, "unguarded_calib_psnr": 23.393406745883127, "unguarded_calib_black_floor": 11.103722307084938}, "fold+dark guard": {"delta_rgb": 1.0, "t": 0.2, "K": 2, "calib_psnr": 24.028554879166254, "calib_black_floor": 5.6691496499078715}}

Exactness: {"guard": {"max_err_vs_formula": 2.4437904357910156e-06, "max_violation_of_bound": 2.980232238769531e-07}, "fold+guard": {"max_err_vs_formula": 2.649488693551305e-06, "max_violation_of_bound": 3.5762786865234375e-07}, "fold+dark guard": {"max_err_vs_formula": 3.0523098945023364e-06, "max_violation_of_bound": 7.748603820800781e-07}, "grown_network_without_guard_vs_base": 2.5033950805664062e-06}

Seconds: {"guard_fit_on_calibration": 11.723578214645386, "free_channels": 4.3831565380096436, "fine_tune_4000_steps": 529.1470251083374, "total": 585.7454535961151}
