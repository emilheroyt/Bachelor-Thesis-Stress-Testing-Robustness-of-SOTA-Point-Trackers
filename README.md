# Stress-Testing the Robustness of State-of-the-Art Point Trackers

Bachelor thesis project (LMU Munich, Informatik) investigating how CoTracker3,
TapNext, and CoWTracker degrade under synthetic motion blur, evaluated on the
TAP-Vid-DAVIS benchmark.

## Overview

- **Experiment 1:** Degradation of all three models (CoTracker3, TapNext,
  CoWTracker) under increasing motion blur (none / medium / strong), measured
  via Average Jaccard (AJ), Occlusion Accuracy (OA), and Position Accuracy (δ).
- **Experiment 2:** Detailed comparison of CoTracker3 vs. TapNext regarding
  occlusion accuracy, outlier magnitude, and trajectory length vs. ground truth.

## Method

Motion blur is synthesized by averaging consecutive sharp frames in linear
(gamma-corrected) space, following the approach described in Chen et al.,
["Image as an IMU"](https://arxiv.org/abs/2503.17358), Eq. 2. Note: this
implementation omits the RIFE-based frame interpolation step used in the
original paper; blur is synthesized directly from real consecutive video
frames. This simplification is documented as a limitation in the thesis.

## Repository structure

- `notebooks/` — Colab notebooks for each model (setup, inference, evaluation)
- `results/` — CSV result tables and plots (per-video and aggregated metrics)

## Reproducing this work

These notebooks were developed and run on Google Colab. To reproduce:

1. Open a notebook in Colab (GPU runtime recommended: T4 is sufficient for
   CoTracker3/TapNext, CoWTracker benefits from A100 due to higher memory use)
2. Run cells top to bottom — each notebook downloads its own model checkpoint
   and the TAP-Vid-DAVIS dataset automatically
3. Known environment quirks (documented in-notebook where relevant): Git
   operations may require bypassing credential prompts in some Colab
   sessions; TensorFlow/JAX version conflicts are worked around where needed

## Models evaluated

- [CoTracker3](https://cotracker3.github.io/) (Meta AI)
- [TapNext](https://arxiv.org/abs/2504.05579) (Google DeepMind)
- [CoWTracker](https://cowtracker.github.io/) (Meta AI)

## Status

Work in progress — part of an ongoing bachelor thesis (submission: 22.09.2026).
