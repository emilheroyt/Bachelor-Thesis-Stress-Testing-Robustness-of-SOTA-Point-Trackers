# Stress-Testing the Robustness of State-of-the-Art Point Trackers

Bachelor thesis project (LMU Munich, Informatik) investigating how CoTracker3, TapNext, and CoWTracker degrade under synthetic motion blur, evaluated on the TAP-Vid-DAVIS benchmark.

## Overview

- **Experiment 1:** Degradation of all three models (CoTracker3, TapNext, CoWTracker) under increasing motion blur (none / medium / strong), measured via Average Jaccard (AJ), Occlusion Accuracy (OA), and Position Accuracy (δ_avg), with per-video statistics (95 % CI, paired Wilcoxon tests, relative degradation).
- **Experiment 2:** Detailed comparison of CoTracker3 vs. TapNext regarding occlusion accuracy, outlier magnitude, and visibility error types (false-visible vs. false-occluded), plus qualitative trajectory examples.

## Method

Motion blur is synthesized by averaging consecutive sharp frames in linear (gamma-corrected, γ = 2.2) space, following Eq. 2 of Chen et al., ["Image as an IMU"](https://arxiv.org/abs/2503.17358). Three levels are used: none (1 frame), medium (3 frames), strong (7 frames). This implementation omits the RIFE-based frame interpolation of the original paper; blur is synthesized directly from consecutive real frames (documented as a limitation).

**Evaluation.** TAP-Vid-DAVIS, "first" query mode, 30 videos, standard metrics (AJ, δ_avg, OA) computed with the official TAP-Vid code at 256×256. All blur augmentation is applied at 256×256 before model-specific preprocessing, so all trackers see identical degraded frames.

**Model-specific preprocessing.** CoTracker3 (`scaled_offline.pth`, window 60) and TapNext (`bootstapnext_ckpt.npz`) receive 256×256 input as in their respective papers. CoWTracker (`cowtracker_model.pth`) receives input upscaled to 448×448 (a multiple of its 112-px patch/stride structure); predictions are mapped back to 256×256 coordinates. Query points are read from the dense CoWTracker output by bilinear interpolation; points with query time > 0 use a forward and a time-reversed backward pass. All three baselines reproduce the published TAP-Vid-DAVIS numbers within about 2 AJ.

**Additional metrics (Experiment 2).** Outlier magnitude (maximum frame-to-frame displacement) and its ratio to the ground-truth displacement are computed only over frames at or after the query frame in which the ground-truth point is visible. False-visible and false-occluded rates quantify over-confident versus conservative visibility errors. Final result files carry the suffix `_v3` (CoWTracker) and `_v4` (CoTracker3, TapNext); earlier versions are kept for traceability.

## Repository structure

- `experiment1/` — Colab notebooks for each model (setup, inference, evaluation) and the Experiment 1 analysis notebook
- `experiment2/` — Extended CoTracker3 / TapNext notebooks and the Experiment 2 analysis notebook
- `results_experiment1and2/` — CSV result tables (per-video and aggregated metrics) and plots
- `Blur_Pipeline.ipynb` — Standalone motion blur augmentation pipeline and visual check

## Reproducing this work

These notebooks were developed and run on Google Colab. To reproduce:

1. Open a notebook in Colab (GPU runtime: T4 is sufficient for CoTracker3/TapNext, CoWTracker requires an A100 due to higher memory use)
2. Run cells top to bottom — each notebook downloads its own model checkpoint and the TAP-Vid-DAVIS dataset automatically
3. Known environment quirks (handled in-notebook): repositories are fetched as ZIP archives because `git clone` may fail in some Colab sessions; a TensorFlow import conflict in the TAP-Vid evaluation utilities is bypassed; a FlashAttention compatibility patch is applied to CoWTracker for recent `timm` versions

## Models evaluated

- [CoTracker3](https://cotracker3.github.io/) (Meta AI)
- [TapNext](https://arxiv.org/abs/2504.05579) (Google DeepMind)
- [CoWTracker](https://cowtracker.github.io/) (Meta AI)

## Status

Experiments complete (05.09.2026); thesis text in progress. Submission: 22.09.2026.
