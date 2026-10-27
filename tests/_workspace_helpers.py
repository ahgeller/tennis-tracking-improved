"""Synthetic fixtures the workspace suites share: a headless label tool, a weights template and a
probe of the trainer's metric. No match media or labels. The tracker's own helpers (``Capture``,
``write_clip``, ``cpu_detector`` and the rest) are ``system/tests/_helpers.py``, imported as
``tests._helpers``; these build on them."""
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import torch

from finetune.data.detector_metrics import DetectorMetrics
from finetune.data.workspace import VideoMeta
from finetune.training import metrics
from tennis_tracker.engines import ball_detectors
from tests._helpers import Capture, cpu_detector


# --------------------------------------------------------------------------- the label tool

def fake_source(clip: str, count: int = 6, width: int = 640, height: int = 360, origin=(100.0, 80.0),
                step=(5.0, 2.0), conf: float = 0.0, **extra) -> SimpleNamespace:
    """A ``label_source.Source`` stand-in with no video: `count` rows of `clip` along a line from
    `origin` in steps of `step`, every row visible at `conf`, and the fields the label tool reads.
    `extra` adds or overrides fields (label_cli reads video, flagged and zoom)."""
    from finetune.labeling.label_store import Row

    rows = [Row(i, origin[0] + i * step[0], origin[1] + i * step[1], True, "csv", conf) for i in range(count)]
    for row in rows:
        row.clip = clip
    capture = SimpleNamespace(release=lambda: None, isOpened=lambda: False)
    fields = dict(clip=clip, rows=rows, width=width, height=height, frame_cache={}, dirty=False, cursor=0, start=0,
                  missing=set(), saved_cursor=0, capture=capture, position=0, release=lambda: None, landings={})
    return SimpleNamespace(**{**fields, **extra})


def fake_tool(*sources: SimpleNamespace, follow: bool = True):
    """A headless ``label_tool.Tool`` over `sources` (one fake_source by default) whose frames are
    blank arrays, one per (clip, frame) as the real cache hands back, centred on its first ball
    (`follow`); no window is opened."""
    from finetune.labeling.label_tool import Tool

    tool = Tool(list(sources or [fake_source("a")]))
    frames: Dict[Tuple[str, int], np.ndarray] = {}
    for source in tool.session.sources:
        source.frame = lambda index, source=source: frames.setdefault(
            (source.clip, index), np.zeros((source.height, source.width, 3), np.uint8))
    if follow:
        tool.follow_ball()
    return tool


# --------------------------------------------------------------------------- the trainer

def blank_template(model, path: Path) -> Path:
    """An .npz with the layout ``save_weights`` expects (the BatchNorm statistics), made from the
    model's own layers; the convolutions are written by ``save_weights`` itself."""
    arrays = {}
    for index, layer in enumerate(model.batch_norms):
        name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
        for key, value in (("gamma", layer.gamma), ("beta", layer.beta), ("moving_mean", layer.mean),
                           ("moving_variance", layer.variance)):
            arrays[f"{name}/{name}/{key}:0"] = value.detach().numpy()
    np.savez(path, **arrays)
    return path


# A detector output the metric probe returns for every five-frame unit.
_PROBE_POINT = (100, 100)
_PROBE_CONFIDENCE = 0.9
_PROBE_FPS = 30
_PROBE_SIZE = (1920, 1080)


def metric_probe(count: int, labels: Dict[int, Optional[List[float]]]) -> DetectorMetrics:
    """``score_weights`` on a ``count``-frame synthetic clip whose detector always fires at (100, 100),
    run through the real prepass (``GridTrackNetBallDetector.detect_clip``) on the CPU."""
    width, height = _PROBE_SIZE
    detector = cpu_detector(lambda output, w, h, threshold: [(_PROBE_POINT, _PROBE_CONFIDENCE)] * (5 * output.reshape(-1, 5).shape[0]))
    with patch.object(metrics, "ball_detector", return_value=detector), \
         patch.object(metrics, "video_meta", return_value=VideoMeta(_PROBE_FPS, width, height, count)), \
         patch.object(metrics, "training_targets", return_value=labels), \
         patch.object(ball_detectors.cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
         patch.object(torch.cuda, "synchronize"):
        return metrics.score_weights(
            Path("unused"), [("probe", Path("unused"), Path("unused"), "custom")], torch.device("cpu")
        )
