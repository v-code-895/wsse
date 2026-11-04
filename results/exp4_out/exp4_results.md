# Experiment 4: eye-colour edit (teal -> magenta)

Base `145ec232ed8a91e5`, fit n=150, held-out n=150, seed 0. Eye pixels = teal iris mask on the base image; 'rest' = pixels more than 3 px from any eye pixel.

| edit | params changed | eye hue (base 181°, target ~300°) | eye dist to target | rest MAE | rest p99 px change |
|---|---|---|---|---|---|
| base | 0 | 183° | 0.240 | 0.0000 | 0.000 |
| latent_redness_2 | 0 (input edit) | 181° | 0.205 | 0.0993 | 0.637 |
| latent_redness_4 | 0 (input edit) | 115° | 0.250 | 0.1676 | 0.839 |
| fold_1x1_nogate | 195 | 297° | 0.012 | 0.0524 | 0.251 |
| prune2_only | 2885 | 183° | 0.243 | 0.0067 | 0.046 |
| prune2_eye_gate | 4045 | 296° | 0.032 | 0.0076 | 0.068 |

## Checks

- gate_identity_maxerr: 4.440892098500626e-16
- json_roundtrip_maxabs: 3.725290298461914e-09
- params_changed_prune2: 2885
- params_changed_gate: 1160
- params_changed_fold: 195
- pruned_slots: [15, 36]
- prune_cost_rank_of_slots: [0, 1]
- theta_quantile: 0.95
- colour_vector: [0.6612, -0.6558, -0.0099]
