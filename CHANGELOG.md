# Changelog

## v2.0.1 — 2026-11-05

More figures, and one spelling fix.
- A new Figure 1 shows the odd-eye patch on three cats where both gates fire fully, next to a map of
  what the left-eye patch changed: the left eye, and nothing in the right half.
- New figures for the other experiments: the identities compiled into real networks (§3.1), freeing a
  quarter of a layer with and without the fold (§3.2), the routed union on the three input types (§4.4),
  and every cat MobileNetV2 gets wrong, before and after the exception unit (§4.5). They are drawn by
  `src/make_figures_v2.py`, which re-runs each experiment without writing its result files.
- Three figures from v1.0.2 are back: the 128 → 512 pipeline, the blue-eye gate at 512 px, and the
  2018 and 2026 generators. The v2.0.0 Figure 1 is now the unselected odd-eye examples in §4.2.
- Table 7: the quick fine-tune changes 377 dog and 338 wild-animal predictions, 375 and 336 of them to
  "cat". v2.0.0 gave 375 and 336 for both. The new figure also shows that the one cat error the exception
  unit leaves is a wild cat that AFHQ files under cats.
- §2 now engages with Raffel's *Building Machine Learning Models Like Open Source Software* [25] point by
  point (cheap updates, merging, testing proposed changes, versioning), and adds Kästner's view of
  machine learning as requirements engineering [43, 44].
- The paper build caps figure height, so tall figures no longer leave half-empty pages.
- "artefact" is now "artifact", in the paper and in two code comments.
- `exp19`, `exp20`, `exp22` and `exp23` gain small hooks (`fold_and_zero`, `write=False`,
  `fit_only=True`) for the figure script; their results are unchanged.

## v2.0.0 — 2026-11-05

The paper is reframed around its new experiments, under a new title: *Beyond Gradient Descent: An LLM
Agent Rewires Trained Neural Networks with Exact, Tested Patches*. The git workflow of v1 is now one
section (§5), on keeping many patches working.

New code (`src/`):
- `circuits.py`: exact (Leaky)ReLU identities (carry, pass, relu, abs, min, max, clamp, step, and,
  or, gate, mux) and channel freeing by regression.
- `exp18_circuit_primitives.py`: exactness tests of every identity, plus exact width and depth growth
  of CatGAN-26.
- `exp19_free_capacity.py`: how many channels each CatGAN-26 layer can free, with and without the fold.
- `exp20_odd_eyes.py`: a coordinate wire and an AND gate compiled into CatGAN-26 (odd-eyed cats).
- `exp21_guarded_sr.py`: TinySR-18A rewired with a reference path and a min-gate; passes the v1
  regression gate; compared with fine-tuning.
- `exp22_union_router.py`: the v1 fine-tune merge conflict resolved by a routed union with a
  self-consistency router.
- `exp23_exception_patch.py`: a gated exception unit fixes MobileNetV2's cat errors; compared with
  head fine-tunes.

Earlier results are unchanged. The weights of the new experiments are produced by their scripts
and are not attached to the release.

## v1.0.2 — 2026-11-04

- Paper: the abstract and introduction describe how patches are written. An AI coding agent
  writes the fitting scripts for classical algebraic operators, and `wpgit` accepts or rejects
  the results.
- Paper: new §8.2 traces each proposal in the study to the check that caught it and how it was
  resolved. The old §8.2–8.5 become §8.3–8.6.
- Paper: related work covers learned editors and lifelong editing (MEND, MEMIT, GRACE,
  AlphaEdit, Text-to-LoRA) and says where WSSE differs.

Code, results and models are unchanged.

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
