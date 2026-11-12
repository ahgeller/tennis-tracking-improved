"""Where the model assets live, relative to the repository root.

The tracker's defaults, the evaluation scripts and the fine-tuning workspace all point at
the same files; this is the one place their paths are spelled. Everything is absolute, so a
script started from another working directory still finds them.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
# The ball detectors: the stream_s4 record of run s4_fg1 (epoch 46) with the peak visibility head of
# run s4_fg1_peak on top, which the tracker runs by default since 2026-11-10 (the same detector with
# the selector's visibility gate); that record without the head, the default from 2026-11-05; the
# stream_s4 record of epoch 60, the default before it (2026-10-30 to 2026-11-05); and the heatmap_s4
# record, the default before all and the training workspace's model. Their thresholds, the gate and
# where the files come from are in ball_weights.py.
STREAM_PEAK_WEIGHTS = MODELS / "candidates" / "stream_s4_fg1_e46_peakhead.pt"
STREAM_WEIGHTS = MODELS / "candidates" / "stream_s4_fg1_e46.pt"
STREAM_E60_WEIGHTS = MODELS / "candidates" / "stream_s4_e60.pt"
HEATMAP_WEIGHTS = MODELS / "ball_detector.pt"
BALL_WEIGHTS = STREAM_PEAK_WEIGHTS                                 # the ball detector the tracker runs by default
PLAYER_ENGINE = MODELS / "player.engine"                           # TensorRT player detector
COURT_ENGINE = MODELS / "courtdetection.engine"                    # TensorRT court keypoint detector
BOUNCE_MODEL = MODELS / "ctb_regr_bounce.cbm"                      # CatBoost bounce model (--court)
