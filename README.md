# Weight-Space Software Engineering (WSSE)

**Version control and unit-tested patches for neural network weights.**

Weight edits are stored as small, hash-pinned text files (`.weightpatch.json`) that carry their own
tests. `wpgit` builds a model from a base checkpoint plus patches, runs every patch's tests *in the
context of all the others*, flags pairs of patches whose order changes the model, and rebases a
patch set onto an updated checkpoint. Because everything is text, ordinary `git` does the rest:
branches, merges, reverts, history.

Paper: [`paper/wsse.pdf`](paper/wsse.pdf) (source [`paper/wsse.md`](paper/wsse.md)).

![CatGAN-26 under version control](paper/figures/fig1_cat_repository.png)

*CatGAN-26, an 8.1M-parameter cat generator trained for the paper, through successive builds of its
weight repository: base; a blue-eye circuit wired into two freed channels; a rank-2 lock on fur
lightness; HEAD, with a background lock, where every test passes. Same random latents in every row.*

## What the paper shows

- **A cat generator under version control** (§4). A 15-commit git history edits CatGAN-26 with four
  readable patches. Branches merge, `git revert` reproduces a direct build bit for bit, an in-context
  test catches two shifts that pass alone but fail together, and `wpgit check` catches two conflicts
  that git merges silently. HEAD passes every test.
- **Fix the patch, not the test** (§4.2). One-direction latent locks leave 68–71% of an attribute's
  variance; a rank-2 *subspace lock* (one extra active-subspace direction) leaves 10–11%.
- **Gated circuit wiring** (§4.2). Freeing two channels and wiring an exact ReLU gate into them turns
  irises blue with 23× less collateral change than an ungated edit (on 40% of cats per image).
- **Rebasing onto an updated model** (§5). After 4,000 more training steps, latent patches still
  pass; channel-level patches lose their guarantees, and `wpgit rebase` flags them as stale first.
- **Fine-tunes as patches** (§6). Two TinySR-26 fine-tunes merge as weight deltas in either order but
  break a regression test; no weighted merge, layer cherry-pick or repair fine-tune meets the merge's
  specification, only joint retraining does.
- **Old checkpoints as legacy code** (§7). On 2017–2020 Keras models, weights alone reveal training
  history, refactors compile exactly, and a regression gate rejects a plausible hotfix.

## Quick start

```bash
git clone https://github.com/v-code-895/wsse.git && cd wsse
pip install -r requirements.txt
python src/exp3_anime_weightpatch.py          # fit latent patches for the 2018 anime generator
python src/exp4_anime_eye_circuit.py          # prune two channels, wire the eye-colour circuit
python src/exp5_git_demo.py                   # script a 15-commit git history over the patches
python src/wpgit.py test  results/exp5_out/weightrepo
python src/wpgit.py check results/exp5_out/weightrepo
```

These run on CPU. Run every script from the repository root. The CatGAN-26 and TinySR-26
experiments need PyTorch with CUDA (`requirements-gpu.txt`), the AFHQ data
(`python scripts/download_data.py afhq`) and the CatGAN-26 weights (`python scripts/fetch_models.py`;
see [`MODELS.md`](MODELS.md)). To inspect the cat repository directly:

```bash
git clone results/exp5_cat_subspace_out/weightrepo.bundle results/exp5_cat_subspace_out/weightrepo
python src/wpgit.py test results/exp5_cat_subspace_out/weightrepo
```

`bash scripts/reproduce_cpu.sh` re-runs every CPU experiment, and `python paper/build_paper.py`
rebuilds the paper's HTML and PDF.

## `wpgit`

```
python src/wpgit.py build  <repo>            apply all patches in dependency order -> build/model.npz
python src/wpgit.py test   <repo>            each patch's tests: full build vs full build without it
python src/wpgit.py check  <repo>            does A-then-B equal B-then-A for every pair of patches?
python src/wpgit.py rebase <repo> <weights>  move the repo onto new base weights; mark stale patches
```

A weight repository is a git repository holding only text: `base.json` (checkpoint path, its
hash, optionally a test-domain module) and `patches/*.weightpatch.json`. A patch looks like this
(abridged):

```json
{
 "format": "weightpatch/v0",
 "id": "lock_fur_lightness",
 "base": "be2b16ff52fc4021",
 "description": "lock fur_lightness: clamp a rank-2 latent subspace (2018 direction + 1 active-subspace directions at the latent mean)",
 "ops": [{"type": "latent_clamp", "layer": "dense", "direction": [-0.0326139, 0.0343906, ...], "target": 2.0},
         {"type": "latent_clamp", "layer": "dense", "direction": [-0.0062195, 0.0191019, ...], "target": 0.0}],
 "tests": [{"type": "attr_var_ratio", "attr": "fur_lightness", "vs": "base", "max": 0.3}]
}
```

## Repository layout

```
paper/          the paper: wsse.md (source), wsse.pdf, wsse.html, figures, build_paper.py
src/            all code: the engine, wpgit, and one script per experiment (exp2 ... exp17)
results/        outputs of every experiment, as reported in the paper (Markdown, JSON, figures, patches)
legacy/         the 2017-2020 checkpoints studied in §7 (see legacy/README.md)
scripts/        data and model download, CPU reproduction
data/           third-party datasets (not in git; see DATA.md)
```

| paper | script(s) in `src/` | outputs in `results/` | needs |
|---|---|---|---|
| Figure 1 | `make_figure1.py` | `figure1/` | GPU, CatGAN-26 |
| §4.1 | `exp11_catgan26.py`, `exp12_modern_eval.py` | `exp11_out/`, `exp12_out/` | GPU, AFHQ |
| §4.2 | `cat_domain.py`, `exp13_catgan26_patches.py`, `exp15_subspace_locks.py` | `exp13_out/`, `exp15_out/` | GPU, CatGAN-26 |
| §4.3 | `wpgit.py`, `exp5_git_demo.py --profile cat_subspace` (and `--profile cat`) | `exp5_cat_subspace_out/`, `exp5_cat_out/` | GPU, CatGAN-26 |
| §5 | `exp17_rebase.py` | `exp17_out/` | GPU, CatGAN-26 training checkpoint |
| §6 | `exp10_tinysr26.py`, `exp14_tinysr26_branches.py`, `exp16_merge_resolution.py` | `exp10_out/`, `exp14_out/`, `exp16_out/` | GPU, AFHQ |
| §7.1 | `extract_and_audit.py`, `exp7_blind_audit.py`, `exp7b_recycle_bot.py`, `exp8_mobilenet.py` | `audit/`, `exp7_out/`, `exp8_out/` | CPU, MobileNetV2 |
| §7.2 | `exp2_tinysr18_patch_v1.py`, `exp2b_tinysr18_weightpatch.py`, `exp12_modern_eval.py A` | `exp2_out/`, `exp2b_out/`, `exp12_out/` | CPU (TinySR-26 row: GPU) |
| §7.3 | `exp7b_recycle_bot.py`, `exp8_mobilenet.py`, `exp7d_anime_styles.py` | `exp7_out/`, `exp8_out/` | CPU, MobileNetV2 |
| §7.4 | `exp6_tinysr18_pair_transfer.py` | `exp6_out/` | CPU |
| §7.5 | `exp3_anime_weightpatch.py`, `exp4_anime_eye_circuit.py`, `exp5_git_demo.py` | `exp3_out/`, `exp4_out/`, `exp5_out/` | CPU |
| §7.6 | `exp8_mobilenet.py`, `exp8b_oracle.py` | `exp8_out/` | CPU, MobileNetV2 |

Not discussed in the paper: `exp7c_deep_color.py` and the Deep-Color parts of `exp7_blind_audit.py`
and `exp7d_anime_styles.py` (a 2020 grayscale colouriser, kept as supplementary material).

The weight repositories built by `exp5_git_demo.py` are nested git repositories, so they are
shipped as git bundles with their full history: `git clone results/<exp5 dir>/weightrepo.bundle
results/<exp5 dir>/weightrepo` restores one in place (its `base.json` uses paths relative to that
location).

## Data, models and licences

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **Paper text:** CC BY 4.0.
- **Model weights and figures:** CC BY-NC 4.0. TinySR-26 and CatGAN-26 are trained on AFHQ
  (© NAVER Corporation, CC BY-NC 4.0, Choi et al., CVPR 2020), and several figures show AFHQ
  photos (resized or cropped) or images generated by models trained on it.
- **Third-party data** (AFHQ, MobileNetV2 weights) is not redistributed: see [`DATA.md`](DATA.md)
  and `scripts/download_data.py`.

## Citation and authorship

See [`CITATION.cff`](CITATION.cff). Authors: **Voravee Upatising** and **Claude** (Anthropic,
Claude Opus 5.5). The paper's contribution statement says who did what; Voravee Upatising directed
the work and is responsible for its content.
