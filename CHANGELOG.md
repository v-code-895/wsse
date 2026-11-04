# Changelog

## v1.0.1 — 2026-11-04

- Paper: related work now covers version control for models (Git-Theta; Raffel, "Building ML
  models like open-source software"), behavioural testing (CheckList; Sculley et al.) and weight
  space as a data modality (the 2026 survey; the ICML 2026 position paper), with how WSSE differs
  from Git-Theta.
- Paper: the §7.3 table uses plain-text numbers; no blank last page; DOI added.
- Paper site on GitHub Pages: https://v-code-895.github.io/wsse/
- `paper/build_paper.py` accepts relative paths and retries a print that happens before the page renders.

Models are unchanged: the CatGAN-26 files stay attached to the v1.0.0 release, where
`scripts/fetch_models.py` downloads them.

## v1.0.0 — 2026-11-04

First public release, accompanying the paper *Weight-Space Software Engineering: Version Control and Unit-Tested Patches for Neural Network Weights*
(`paper/wsse.pdf`).

- `.weightpatch.json` format (v0) and `wpgit` with `build`, `test`, `check` and `rebase`.
- NumPy engine for Keras HDF5 / TensorFlow.js checkpoints and for exported PyTorch models.
- Experiments exp2–exp17 with their outputs (`results/`), including three weight repositories
  with full git history (`results/exp5*_out/weightrepo.bundle`).
- Two models trained for the paper: TinySR-26 (2×, 4×; in the repository) and CatGAN-26 (release
  assets; `scripts/fetch_models.py`).
- The 2017–2020 Keras checkpoints studied in §7 (`legacy/`).

Release assets: `catgan26.npz`, `catgan26_with_ops.npz`, `catgan26.pt`, `wsse.pdf`, and
`SHA256SUMS.txt` (see `MODELS.md`).
