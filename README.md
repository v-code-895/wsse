# Beyond Fixed Architectures

**An LLM agent rewires trained neural networks with exact, tested patches.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23956114.svg)](https://doi.org/10.5281/zenodo.23956114)
[![Paper](https://img.shields.io/badge/paper-read%20online-0969da)](https://v-code-895.github.io/wsse/)
[![PDF](https://img.shields.io/badge/PDF-wsse.pdf-cf222e)](paper/wsse.pdf)
[![License: MIT](https://img.shields.io/badge/code-MIT-2da44e)](LICENSE)

Gradient descent changes the values of a network's weights, inside an architecture fixed before
training. It cannot add a unit or a connection, and what it learns holds only approximately, near its
data. This repository shows a trained network being **rewired** after training instead, by an LLM
coding agent (Claude Opus 5.5, working in Claude Code and directed by a person). The agent builds new
channels, carriers, gates, guards and switches out of exact identities of the (Leaky)ReLU activations
the networks already use. Every patch is a text file that carries its own tests.

**[Read the paper](https://v-code-895.github.io/wsse/)** · [PDF](paper/wsse.pdf) ·
[source](paper/wsse.md) · [cite](#citation)

![Odd-eyed cats from a GAN never trained on any](paper/figures/fig_odd_eyes_hero.png)

*Odd-eyed cats from a GAN never trained on any. The agent compiled a coordinate wire and an AND gate
into CatGAN-26, an 8.1M-parameter cat generator. Columns: the base model; a patch that turns only the
left iris blue; the same patch with its logic inverted; both irises; and how much the left-eye patch
changed each pixel (float64). The change is the left eye; in the right half it is at most 1.6×10⁻⁷,
rounding error. Of 300 held-out cats, 92 end with both irises blue; these three were drawn at random
from the 30 of those with the least change outside the eyes. Per eye, the gates turn 42–48% of irises blue; unselected examples are further down.*

## Contents

- [The idea](#the-idea)
- [What the patches do](#what-the-patches-do)
  1. [An instruction set of exact identities](#1-an-instruction-set-of-exact-identities)
  2. [Making room: free or grow](#2-making-room-free-or-grow)
  3. [Logic in a GAN: odd-eyed cats](#3-logic-in-a-gan-odd-eyed-cats)
  4. [A repair with a bound](#4-a-repair-with-a-bound)
  5. [Grow instead of average: a merge conflict resolved](#5-grow-instead-of-average-a-merge-conflict-resolved)
  6. [An exception in a classifier](#6-an-exception-in-a-classifier)
  7. [Keeping many patches working](#7-keeping-many-patches-working)
  8. [Old checkpoints as legacy code](#8-old-checkpoints-as-legacy-code)
- [What gradient descent still does better](#what-gradient-descent-still-does-better)
- [Quick start](#quick-start) · [`wpgit`](#wpgit) · [Reproducing the paper](#reproducing-the-paper) ·
  [Licences](#data-models-and-licences) · [Citation](#citation)

## The idea

A LeakyReLU with slope $\alpha$ satisfies

```math
\mathrm{LReLU}(u) - \mathrm{LReLU}(-u) = (1+\alpha)\,u, \qquad
\mathrm{LReLU}(u) + \alpha\,\mathrm{LReLU}(-u) = (1-\alpha^2)\,\mathrm{ReLU}(u).
```

Two channels driven with opposite signs therefore carry any signal through a nonlinear layer
unchanged, or turn it into an exact ReLU. Min, max, clamps, thresholds, AND, OR and a multiplexer
follow, each a few channels wide. What remains is room, since a trained layer has no idle channels.
The agent makes room in one of two ways:

- **Freeing** channels: it regresses them onto the rest of their layer and folds the regression into
  the next layer.
- **Growing** the layer: it adds channels whose outgoing weights are zero, which changes nothing at all.

**Who writes the patches.** Every patch here was designed and compiled by an LLM coding agent, on
small models, with a person in the loop:

- **The person** states the goal and its specification, for example "one blue eye, nothing else
  changes", or "lower the black floor without losing 0.1 dB anywhere".
- **The agent** reads the architecture and the weights through scripts, chooses identities and where
  to make room, writes the code that compiles and fits the patch, runs the tests and looks at the images.
- **The tests decide.** A failing test goes back to the agent, which changes the patch, not the bound.

Around the agent we built an instruction set of identities with exactness tests
([`src/circuits.py`](src/circuits.py)); two ways to make room, with their costs measured; a patch
format in which every edit is a hash-pinned text file with its own tests; and [`wpgit`](#wpgit), which
tests each patch in the context of all the others, finds pairs whose order matters and rebases patches
onto updated models. The patches are small: most of their numbers are wiring constants the agent
writes by hand, and the rest are fitted in closed form, in seconds to minutes.

## What the patches do

| patch | model | what training the model does not give | result |
|---|---|---|---|
| [odd eyes](#3-logic-in-a-gan-odd-eyed-cats) | CatGAN-26 (8.1M) | new logic with no data for it; one half of every image exactly unchanged | 42% of left irises blue, 0% of right; other half moves ≤ 1.6×10⁻⁷ |
| [guard](#4-a-repair-with-a-bound) | TinySR-18A (2018, 0.3M) | an output bound on every input | passes the regression gate no output-layer patch passed (+3.37 dB) |
| [union](#5-grow-instead-of-average-a-merge-conflict-resolved) | TinySR-26 ×3 | routing between whole networks | +0.23 / +3.26 / +0.00 dB, against +0.03 / +3.19 / +0.08 for joint retraining |
| [exception](#6-an-exception-in-a-classifier) | MobileNetV2 (3.5M) | a fix that is exactly local where it is off | 22 of 23 cat errors fixed, 1 of 1,000 other predictions changed (a quick fine-tune: 711) |

### 1. An instruction set of exact identities

Each identity is one hidden layer: weights into it, the activation, weights out of it. The weights out
fold into whatever linear layer follows, so a circuit costs channels, not layers.

| identity | computes | channels | max error, float64 / float32 |
|---|---|---|---|
| `carry` | $u$, for $u \ge 0$ | 1 | 0 / 2.4×10⁻⁷ |
| `pass` | $u$ | 2 | 1.8×10⁻¹⁵ / 1.4×10⁻⁶ |
| `relu`, `abs` | $\max(u,0)$, $\lvert u\rvert$ | 2 | ≤ 1.8×10⁻¹⁵ / ≤ 1.4×10⁻⁶ |
| `min`, `max`, `clamp` | $x - \mathrm{relu}(x-y)$, … | 4 | ≤ 3.6×10⁻¹⁵ / 3.3×10⁻⁶ |
| `step` | 0 below $t$, 1 above $t + 1/k$ | 4 | 3.7×10⁻¹⁴ / 3.1×10⁻⁵ |
| `and`, `or` | $\mathrm{relu}(b_1 + b_2 - 1)$, … | 2, 4 | 1.1×10⁻¹⁷ / 6.0×10⁻⁸ |
| `gate` | $p - t$ where $g = 1$, 0 where $g = 0$ | 2 | 1.3×10⁻¹⁵ / 8.3×10⁻⁷ |
| `mux` | $g\ ?\ x : y$ | 6 | 2.7×10⁻¹⁵ / 1.9×10⁻⁶ |

![Eight identities compiled into real networks](paper/figures/fig_identities.png)

*Eight identities, each compiled into a one-hidden-layer LeakyReLU network (dots) over its target
(grey). Dashed lines are the two channels' terms, whose sum is the identity.*

Topology edits that change nothing: adding 16 channels with zero outgoing weights to every hidden
convolution of CatGAN-26 (8.13M → 8.72M parameters) changes its output by exactly 0, and inserting a
`pass` layer after every convolution (12 → 22 weight layers) changes it by at most 9.2×10⁻⁷.

### 2. Making room: free or grow

Trained layers have spare capacity, and the regression fold is what recovers it. Freeing 1/64 of one
CatGAN-26 layer's channels with the fold leaves the images at 37.9–56.8 dB against the unedited
generator, against 26.7–33.8 dB when the same channels are simply zeroed.

![Freeing channels with and without the fold](paper/figures/fig_room.png)

*Left: PSNR against the unedited generator after freeing a share of one layer's channels, with the
fold (blue) or by zeroing (orange), over all 11 layers. Right: a quarter of layer b1c2 (128 of 512
channels) freed. Zeroed, the cats change (14.6 dB); with the fold they stay close (29.7 dB).*

Freeing costs add up, so where exactness matters the agent grows instead. The odd-eye wire needs one
channel in each of ten layers: freed, the change outside the eyes was 3.97 RGB levels; grown, it was
0.24, for 42,548 more parameters (+0.5%).

### 3. Logic in a GAN: odd-eyed cats

Convolutions are local, so nothing in a 3×3 window says which half of the image it is in. Only the
dense layer, with a separate bias for each 4×4 cell, can hold a spatial pattern. The agent built a
coordinate wire:

1. **Room.** Grow one channel in the dense layer and in each of the nine convolutions from b1c1 to b5c1.
2. **Wire.** The dense channel's bias holds RIGHT = 1 on the right half of the 4×4 grid and 0 on the
   left. Each convolution copies it with a centre tap of 1; `carry` passes it through every LeakyReLU
   exactly. At 128 px it is exactly the right-half indicator.
3. **Gate.** The iris gate from the earlier blue-eye circuit gets $-M \cdot \mathrm{RIGHT}$ in its
   pre-activation: $\mathrm{relu}(p - t - M\cdot\mathrm{RIGHT})$ is the iris detector AND NOT RIGHT.

| patch (300 held-out cats) | left irises blue | right irises blue | change outside the eyes (RGB levels) | largest change in the other half |
|---|---|---|---|---|
| **left eye only** | **42%** | **0%** | 0.24 | **1.6×10⁻⁷** (float64) |
| right eye only | 0% | 48% | 0.18 | 1.4×10⁻⁶ (float32) |
| both eyes | 42% | 48% | 0.42 | — |
| left eye, room freed instead of grown | 42% | 0% | 3.97 | — |

The logic is exact; the detector is not. Which irises turn blue is decided by a fitted 3×3 linear
detector, which fires on 42–48% of eyes and leaks onto striped fur:

![Odd-eye patches on less selected cats](paper/figures/fig_odd_eyes.png)

*The first six held-out cats on which the left-eye patch changes the left eye at all, not selected for
success. Rows: base; left iris; right iris; both.*

The gate extends a blue-eye circuit first built for a 2018 anime generator and ported unchanged to
CatGAN-26:

![The blue-eye gate at 512 px](paper/figures/fig_cat_blue_eyes_512.png)

*The blue-eye gate on the first four held-out cats (not selected): base (top) and patched (bottom),
upscaled 4× by TinySR-26. It fires fully on the fourth, not at all on the first and third, and leaks
onto the second cat's ears.*

Why not gradient descent? Fine-tuning toward odd-eyed cats would need a loss that defines them and
examples or a critic for it, it would move every output, and nothing would keep the right half fixed.

### 4. A repair with a bound

TinySR-18A, a 2× super-sampler trained in 2018, lifts black. Its regression gate asks a patch to lose
at most 0.10 dB on five regression images, gain on seven edge cases, and lower the black floor below
8.90 RGB levels. Every output-layer patch failed it: the tone error is nonlinear. The agent kept a
12-number colour fold and gave the network a view of its own input:

1. **Reference path.** Three channels per hidden layer carry the low-resolution image through to the
   output, smoothed by a 5×5 binomial kernel: the reference $R$.
2. **Guard.** One new layer computes $\min(f, R + \delta)$ from the identities, where $f$ is the
   colour-folded output. Wherever the input is black around a pixel, $R = 0$ and the output is at most
   1/255, **on every input**, not only on test images.

| variant (TinySR-18A) | regression ΔPSNR | edge ΔPSNR | black floor | gate |
|---|---|---|---|---|
| base | — | — | 8.90 | |
| colour fold (12 numbers) | +2.93 | +1.81 | 13.51 | FAIL |
| **colour fold + guard** | **+3.37** | **+2.02** | **7.33** | **PASS** |
| colour fold + dark-only guard | +3.37 | +2.39 | 7.88 | PASS |
| fine-tune all weights, 500 steps | +4.34 | +2.88 | 1.62 | PASS |

![TinySR-18A outputs](paper/figures/fig_guarded_sr.png)

*Outputs on a smiley, a sprite, a photo and a gradient. Columns: base; the colour fold; fold + guard;
fold + dark-only guard; a 4,000-step fine-tune; ground truth.*

A fine-tune on the same nine calibration images scores better, but its verdict flips between
checkpoints and nothing bounds its output on other inputs. The two compose: a guard can sit on top of
a fine-tune.

### 5. Grow instead of average: a merge conflict resolved

Two fine-tunes of TinySR-26, one for bicubic inputs and one for noisy inputs, could not be merged:
25 weightings, cherry-picking layers and a repair fine-tune all failed the merge's specification.
Within one network of fixed size, the two specialisations compete. The agent kept all three networks
side by side and added a **self-consistency router**: each network's output is degraded again and
compared with the input, and the network that agrees with its input wins. Two `mux` identities make
the switch exact.

| model | bicubic inputs | noisy inputs | nearest inputs | meets spec |
|---|---|---|---|---|
| branch: bicubic | +0.23 | −0.47 | −4.88 | no |
| branch: noisy | −0.20 | +3.26 | −1.01 | no |
| merge: base + both deltas | −0.03 | +2.56 | −4.06 | no |
| joint retraining | +0.03 | +3.19 | +0.08 | yes |
| **union, self-consistency router** | **+0.23** | **+3.26** | **+0.00** | **yes** |

![The routed union on three kinds of input](paper/figures/fig_union.png)

*One photo, downsampled three ways and upscaled by each network. Each branch helps only on its own
input type; the router (blue frame) picks the right one, as it did for all 540 test images.*

The union keeps both branches' whole gains and beats joint retraining on bicubic inputs, at three
networks' parameters and compute.

### 6. An exception in a classifier

MobileNetV2 gets 23 of 500 AFHQ validation cats wrong. The agent added one ReLU unit beside the
classifier,

```math
\mathrm{logits}' = \mathrm{logits} + \beta\,\mathrm{relu}(w\cdot\phi + c - \theta)\,\mathbf{1}_{\text{cats}},
```

fitted in 0.17 s. Wherever it does not fire, the logits are exactly the original's.

| fix (500 validation images per class) | cat errors fixed | dog predictions changed | wild-animal predictions changed |
|---|---|---|---|
| **exception unit** (1,287 parameters), closed form | **22/23** | **0** | **1** |
| the same unit trained by Adam, preservation loss, lr 10⁻² | 22/23 | 0 | 0 |
| the same unit trained by Adam, errors only, lr 10⁻³ | 23/23 | 166 | 145 |
| head fine-tune with a preservation loss | 22/23 | 2 | 2 |
| head fine-tune on the errors only | 23/23 | 377 (375 now "cat") | 338 (336 now "cat") |

![Every cat MobileNetV2 gets wrong, before and after](paper/figures/fig_exception.png)

Gradient descent on the same one-unit structure does as well when it gets the same preservation data and
the right learning rate (lr 10⁻³ leaves it firing on 975 of the 1,000 animals). What the patch adds is the
structure, a unit that is exactly silent where it is off; once it exists, gradient descent can fit it.

*All 23 cats MobileNetV2 gets wrong, with its top-1 class before → after the patch. The one it leaves
(red) is a wild cat, apparently a clouded leopard, that AFHQ files under cats. Last (orange): the one
dog or wild-animal prediction of 1,000 that changed.*

### 7. Keeping many patches working

Models in use collect many edits from many authors. A scripted 15-commit git history of patches on
CatGAN-26 shows what single-patch tests miss, and what `wpgit` catches:

- a patch that passes alone fails once merged (warm fur: +1.25 SD alone, +0.92 SD in context, against
  a required 1.0);
- two lock branches merge without a textual conflict, and `wpgit check` reports three semantic
  conflicts; orthogonalisation resolves one (the locks then commute to 2.4×10⁻⁷), and a recorded
  decision resolves the other two;
- with the one-direction locks of 2018 the history fails at HEAD; with rank-2 subspace locks every test
  passes. The patches at HEAD are 88 KB of text on a 32.5 MB model.

![CatGAN-26 through its weight repository](paper/figures/fig1_cat_repository.png)

*CatGAN-26 through successive builds of its weight repository: base; the blue-eye circuit; a rank-2
lock on fur lightness; HEAD, where every test passes. Same latents in every row.*

![Rank-1 and rank-2 locks](paper/figures/fig_cat_subspace_locks.png)

*The same eight latents under one-direction and rank-2 locks: the rank-1 fur-lightness lock still lets
a black cat through; at rank 2 every cat is light.*

With a `grow` operator, a topology patch joins the same repository. The odd-eye patch becomes one file
of 11 grow operations and 13 sparse wirings; in a branch where it replaces the 2018 blue-eye circuit, it
passes its tests next to the rank-2 locks (34% of left irises blue, right half unchanged to 2.9×10⁻⁶)
and commutes with them exactly.

When the base model trains 4,000 more steps, `wpgit rebase` moves the repository onto each new
checkpoint. Patches written against an interface (the latent) survive; patches written against an
implementation (specific hidden channels) fail their tests from +2,000 steps, and the stale-context
check flags them at +500, before any test fails.

### 8. Old checkpoints as legacy code

The toolkit started on checkpoints trained in 2017–2020 in Keras, with no tests or documentation.
Reading their weights without running them finds a defect (TinySR-18A's positive output bias, repaired
in §4), recovers training history (Recycle-Bot trained for 18 steps; MobileNetV2's training-set mean
colour to within 1.4 levels), and supports exact refactors and stitching.

![Anime-DCGAN (2018) and CatGAN-26 (2026)](paper/figures/fig_2018_vs_2026_generators.png)

*Anime-DCGAN (64 px, 2018, top), where the latent operators, the eye circuit and the git workflow were
first built, and CatGAN-26 (128 px, 2026, bottom), 32× larger, where they ran with only two
backward-compatible additions.*

## What gradient descent still does better

- **Learning features.** With data in hand, a short fine-tune beat the TinySR-18A repair on every
  metric. Patches compose behaviour from features the network already has.
- **Compactness.** Joint retraining resolves the merge conflict at one network's size; the union needs
  three.
- **Detectors.** Every circuit contains a fitted detector, and fitted detectors are imperfect.

The two are complementary: a fine-tune can be guarded, an exception can sit on a fine-tuned head, and a
union can hold networks that were each trained well. The models here have 0.25M–8.1M parameters;
circuits in large models and transformers are untested, and the topology patches are scripts, not yet
operators of the patch format.

## Quick start

```bash
git clone https://github.com/v-code-895/wsse.git && cd wsse
pip install -r requirements.txt
python src/exp3_anime_weightpatch.py          # fit latent patches for the 2018 anime generator
python src/exp4_anime_eye_circuit.py          # prune two channels, wire the eye-colour circuit
python src/exp5_git_demo.py                   # script a 15-commit git history over the patches
python src/wpgit.py test  results/exp5_out/weightrepo
python src/wpgit.py check results/exp5_out/weightrepo
python src/exp18_circuit_primitives.py         # the identities' exactness (the width/depth part needs a GPU)
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

## Reproducing the paper

```
paper/          the paper: wsse.md (source), wsse.pdf, wsse.html, figures, build_paper.py
src/            all code: the engine, wpgit, circuits.py, and one script per experiment (exp2 ... exp23)
results/        outputs of every experiment, as reported in the paper (Markdown, JSON, figures, patches)
legacy/         the 2017-2020 checkpoints studied in §6 (see legacy/README.md)
scripts/        data and model download, CPU reproduction
data/           third-party datasets (not in git; see DATA.md)
```

| paper | script(s) in `src/` | outputs in `results/` | needs |
|---|---|---|---|
| §3.1 | `circuits.py`, `exp18_circuit_primitives.py` | `exp18_out/` | GPU for the CatGAN-26 part |
| §3.2 | `exp19_free_capacity.py` | `exp19_out/` | GPU, CatGAN-26 |
| §4.2, Figures 5–6 | `exp20_odd_eyes.py` (`exp13_catgan26_patches.py` for the blue-eye gate) | `exp20_out/`, `exp13_out/` | GPU, CatGAN-26 |
| Figures 1–3, 8, 9 | `make_figures_v2.py` (re-runs exp18–exp23 without writing their results) | `paper/figures/` | GPU, CatGAN-26, AFHQ, MobileNetV2 |
| §4.1, Figures 4, 12 | `exp11_catgan26.py`, `exp12_modern_eval.py`, `exp10_tinysr26.py` | `exp11_out/`, `exp12_out/`, `exp10_out/` | GPU, AFHQ |
| §4.3 | `exp2b_tinysr18_weightpatch.py --lr-filter nearest`, `exp21_guarded_sr.py` | `exp2b_out/`, `exp21_out/` | CPU (exp21: GPU) |
| §4.4 | `exp14_tinysr26_branches.py`, `exp16_merge_resolution.py`, `exp22_union_router.py` | `exp14_out/`, `exp16_out/`, `exp22_out/` | GPU, AFHQ |
| §4.5 | `exp23_exception_patch.py` | `exp23_out/` | CPU or GPU, MobileNetV2, AFHQ |
| §5.1 | `cat_domain.py`, `exp13_catgan26_patches.py`, `exp15_subspace_locks.py` | `exp13_out/`, `exp15_out/` | GPU, CatGAN-26 |
| §5.2, Figure 10 | `wpgit.py`, `exp5_git_demo.py --profile cat_subspace` (and `--profile cat`), `make_figure1.py` | `exp5_cat_subspace_out/`, `exp5_cat_out/`, `figure1/` | GPU, CatGAN-26 |
| §5.3 | `exp17_rebase.py` | `exp17_out/` | GPU, CatGAN-26 training checkpoint |
| §5.4 | `wpgit.py`, `exp24_odd_eyes_wpgit.py` | `exp24_out/` | GPU, CatGAN-26 |
| §6 | `extract_and_audit.py`, `exp7_blind_audit.py`, `exp7b_recycle_bot.py`, `exp8_mobilenet.py`, `exp8b_oracle.py`, `exp6_tinysr18_pair_transfer.py`, `exp3_anime_weightpatch.py`, `exp4_anime_eye_circuit.py`, `exp5_git_demo.py` | `audit/`, `exp7_out/`, `exp8_out/`, `exp6_out/`, `exp3_out/`, `exp4_out/`, `exp5_out/` | CPU, MobileNetV2 |

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

## Citation

Archived on Zenodo: [doi:10.5281/zenodo.23956114](https://doi.org/10.5281/zenodo.23956114) (resolves
to the latest version). See also [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{upatising2026beyond,
  title  = {Beyond Fixed Architectures: An {LLM} Agent Rewires Trained Neural Networks with Exact, Tested Patches},
  author = {Upatising, Voravee and {Claude (Anthropic; model Claude Opus 5.5)}},
  year   = {2026},
  doi    = {10.5281/zenodo.23956114},
  url    = {https://v-code-895.github.io/wsse/}
}
```

Authors: **Voravee Upatising** and **Claude** (Anthropic, Claude Opus 5.5). The paper's contribution
statement says who did what; Voravee Upatising directed the work and is responsible for its content.
