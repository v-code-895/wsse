# Experiment 18: exact identities for rewriting trained networks

Each identity is compiled into a one-hidden-layer network and compared with its target on 100,000 random inputs.
`step` is tested outside its transition band [t, t + 1/k]; `gate` and `mux` within their big-M range.

| identity | slope | hidden channels | max error (float64) | max error (float32) | max |target| |
|---|---|---|---|---|---|
| carry | 0.2 | 1 | 0.0e+00 | 2.4e-07 | 5.00 |
| carry | 0.0 | 1 | 0.0e+00 | 2.4e-07 | 5.00 |
| pass | 0.2 | 2 | 1.8e-15 | 1.4e-06 | 12.97 |
| pass | 0.0 | 2 | 0.0e+00 | 4.8e-07 | 13.14 |
| relu | 0.2 | 2 | 8.9e-16 | 1.4e-06 | 13.19 |
| relu | 0.0 | 2 | 0.0e+00 | 4.8e-07 | 13.17 |
| abs | 0.2 | 2 | 1.8e-15 | 1.1e-06 | 14.04 |
| abs | 0.0 | 2 | 0.0e+00 | 4.8e-07 | 13.83 |
| min | 0.2 | 4 | 3.6e-15 | 3.2e-06 | 13.66 |
| min | 0.0 | 4 | 1.8e-15 | 1.4e-06 | 14.99 |
| max | 0.2 | 4 | 3.6e-15 | 3.3e-06 | 13.65 |
| max | 0.0 | 4 | 1.8e-15 | 1.4e-06 | 13.82 |
| clamp | 0.2 | 4 | 2.0e-15 | 1.4e-06 | 0.50 |
| clamp | 0.0 | 4 | 8.9e-16 | 4.8e-07 | 0.50 |
| step | 0.2 | 4 | 3.7e-14 | 3.1e-05 | 1.00 |
| step | 0.0 | 4 | 0.0e+00 | 0.0e+00 | 1.00 |
| and | 0.2 | 2 | 1.1e-17 | 6.0e-08 | 1.00 |
| and | 0.0 | 2 | 0.0e+00 | 0.0e+00 | 1.00 |
| or | 0.2 | 4 | 0.0e+00 | 0.0e+00 | 1.00 |
| or | 0.0 | 4 | 0.0e+00 | 0.0e+00 | 1.00 |
| gate | 0.2 | 2 | 1.3e-15 | 8.3e-07 | 4.00 |
| gate | 0.0 | 2 | 8.9e-16 | 6.0e-07 | 4.00 |
| mux | 0.2 | 6 | 2.7e-15 | 1.9e-06 | 4.00 |
| mux | 0.0 | 6 | 1.8e-15 | 1.1e-06 | 4.00 |

## Topology edits on CatGAN-26 that must change nothing (64 latents, float64)

- widen every hidden conv by 16 channels (random incoming weights, zero outgoing): 8,128,291 → 8,716,403 parameters; max output change 0.0e+00
- insert an identity conv + LeakyReLU after every hidden conv (two-channel `pass`): 12 → 22 weight layers; max output change 9.2e-07
