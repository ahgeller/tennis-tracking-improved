"""Where the model assets live, relative to the repository root.

The tracker's defaults, the evaluation scripts and the fine-tuning workspace all point at
the same files; this is the one place their paths are spelled. Everything is absolute, so a
script started from another working directory still finds them.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
# The ball detectors: the stream_s4 record (epoch 60), which the tracker runs by default, and the
# heatmap_s4 record, the default before it and the training workspace's model. Their thresholds and
# where the files come from are in ball_weights.py.
STREAM_WEIGHTS = MODELS / "candidates" / "stream_s4_e60.pt"
HEATMAP_WEIGHTS = MODELS / "ball_detector.pt"
BALL_WEIGHTS = STREAM_WEIGHTS                                      # the ball detector the tracker runs by default
PLAYER_ENGINE = MODELS / "player.engine"                           # TensorRT player detector
COURT_ENGINE = MODELS / "courtdetection.engine"                    # TensorRT court keypoint detector
BOUNCE_MODEL = MODELS / "ctb_regr_bounce.cbm"                      # CatBoost bounce model (--court)
