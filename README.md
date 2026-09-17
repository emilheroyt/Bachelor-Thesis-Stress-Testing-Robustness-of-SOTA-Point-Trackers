# Stress-Testing the Robustness of State-of-the-Art Point Trackers

Bachelor thesis project (LMU Munich, Informatik). It measures how three state-of-the-art
point trackers — **CoTracker3**, **TAPNext** and **CoWTracker** — degrade under synthetic
motion blur on the TAP-Vid-DAVIS benchmark, and characterises how CoTracker3 and TAPNext fail.

## Overview

- **Experiment 1 — degradation.** All three models at three blur levels (none / medium / strong),
  measured with the official TAP-Vid metrics Average Jaccard (AJ), Occlusion Accuracy (OA) and
  position accuracy (δ_avg), with per-video statistics (mean, std, 95 % CI), paired Wilcoxon
  signed-rank tests and relative degradation.
- **Experiment 2 — failure modes of CoTracker3 vs. TAPNext.** Occlusion accuracy, outlier
  magnitude and its ratio to the ground-truth motion, false-visible / false-occluded rates,
  and a qualitative trajectory example (DAVIS sequence *bmx-trees*).

## Method

**Motion blur.** A blurred frame is the average of `w` consecutive frames centred on it, computed
in linear radiometric space (γ = 2.2), following Eq. 2 of Chen & Clark,
["Image as an IMU"](https://arxiv.org/abs/2503.17358):

    B_i = ( mean_{j in W(i)} I_j^γ )^(1/γ),    W(i) = [i - floor(w/2), i + floor(w/2)]

Levels: `none` (w = 1), `medium` (w = 3), `strong` (w = 7). Two deliberate deviations from the
reference method: the window is **centred** on the annotated frame (TAP-Vid ground truth is
defined per frame), and **no frame interpolation** (RIFE) is applied before averaging, so fast
motion produces ghosting. With DAVIS at 24 fps the windows correspond to effective exposures of
about 125 ms and 292 ms — an extreme stress test, not a calibrated camera model.

**Evaluation.** TAP-Vid-DAVIS, 30 videos, 650 trajectories, query mode `first`, 256×256, official
`tapnet` metric code. Blur is applied at 256×256 *before* any model-specific pre-processing, so
all trackers see identical degraded frames.

**Models.**

| Model | Checkpoint | Notes |
|---|---|---|
| CoTracker3 | `scaled_offline.pth` | offline, window 60, 5×5 support grid |
| TAPNext | `bootstapnext_ckpt.npz` (BootsTAPNext-B) | causal per-frame decoding |
| CoWTracker | `cowtracker_model.pth` | dense; input up-sampled 256→448, queries read by bilinear interpolation, forward + time-reversed pass, fp16 |

## Results

Mean over 30 videos (full statistics in `updated_results_experiment1and2/`):

| Model | Blur | AJ | OA | δ_avg |
|---|---|---|---|---|
| CoTracker3 | none | 62.6 | 88.1 | 77.2 |
| CoTracker3 | medium | 46.9 | 80.4 | 63.6 |
| CoTracker3 | strong | 26.0 | 66.9 | 41.6 |
| TAPNext | none | **66.7** | 91.3 | **79.6** |
| TAPNext | medium | 42.3 | 77.3 | 57.3 |
| TAPNext | strong | 19.2 | 62.1 | 30.8 |
| CoWTracker | none | 65.3 | **92.1** | 78.8 |
| CoWTracker | medium | **52.1** | **89.9** | **65.8** |
| CoWTracker | strong | **35.3** | **85.6** | **48.2** |

Clean results reproduce the published numbers within 2 AJ (CoTracker3 64.4, BootsTAPNext-B 65.2,
CoWTracker 65.5). The ranking inverts under blur: TAPNext is first on clean video and last under
strong blur (better than CoWTracker in 1 of 30 videos, Wilcoxon p < 1e-4); relative AJ loss at
strong blur is 72 % (TAPNext), 60 % (CoTracker3) and 46 % (CoWTracker).

Experiment 2: CoTracker3's largest frame-to-frame jump stays at the level of the true motion
(ratio ≈ 1.0 at every level) while its false-occluded rate rises from 0.15 to 0.39; TAPNext's
outlier ratio grows from 1.2 to 2.8 and its false-visible rate from 0.16 to 0.35.

## Repository structure

    Blur_Pipeline.ipynb                  blur function + visual check
    experiment1/
      Cotracker3Isolation.ipynb          CoTracker3: setup, inference, TAP-Vid metrics
      tapnext_demoIsolation.ipynb        TAPNext: setup, inference, TAP-Vid metrics
      CoWTrackerIsolation.ipynb          CoWTracker: setup, inference, TAP-Vid metrics
      All3CSVsUpdated.ipynb              Experiment 1 statistics and figures (final)
      AlleDreiCSVs.ipynb                 superseded first aggregation, kept for traceability
    experiment2/
      Cotracker3_Experiment2.ipynb       CoTracker3 + failure-mode metrics, incl. verification run
      TapNext_Experiment2.ipynb          TAPNext + failure-mode metrics
      Experiment2_Plot.ipynb             Experiment 2 statistics, figures, video-22 plot, blur example
    updated_results_experiment1and2/     FINAL results used in the thesis
      cowtracker_all_videos_v3.csv         per-video metrics (90 rows = 30 videos × 3 levels)
      cotracker3_all_videos_v4.csv
      tapnext_all_videos_v4.csv
      experiment1_stats_mean_std_ci.csv    mean, std, 95 % CI
      experiment1_relative_degradation.csv
      experiment2_combined_v4.csv, experiment2_summary_table_v4.csv
      *_traj_video22_strong.npy            trajectories for the qualitative figure
      *.png / *.pdf                        figures (English labels)
    results_experiment1and2/             ARCHIVE of earlier runs (v1–v4 intermediates, superseded plots)

Note on the archive: v1 used integer-rounded CoWTracker queries (clean AJ 56.9) and an outlier
metric that included occluded frames; both were replaced (v3/v4). Files in the archive are not
used in the thesis.

## Reproducing this work

The notebooks were developed and run on Google Colab.

1. Open a notebook in Colab with a GPU runtime (T4 is sufficient for CoTracker3 and TAPNext;
   CoWTracker was run on an A100).
2. Run all cells. Each model notebook downloads its checkpoint and TAP-Vid-DAVIS automatically
   and writes a per-video CSV; the analysis notebooks read those CSVs and produce tables and figures.
3. Inference is deterministic. A verification run of the CoTracker3 Experiment 2 notebook on
   17 Sep 2026 reproduced the committed result file exactly (max. absolute difference 0.0 over
   90 rows × 18 columns, and 0.0 px for the saved trajectories).

Environment work-arounds handled in the notebooks: repositories are fetched as ZIP archives,
a TensorFlow import in the TAP-Vid utilities is stubbed out, and a FlashAttention signature
patch is applied to CoWTracker for recent `timm` versions.

## Models

- [CoTracker3](https://cotracker3.github.io/) — Karaev et al., 2024
- [TAPNext](https://arxiv.org/abs/2504.05579) — Zholus et al., 2025
- [CoWTracker](https://cowtracker.github.io/) — Lai et al., 2026

## Status

Experiments complete; thesis submission 22 September 2026.
