"""Running the tracker from another program with nothing drawn: one ``Config`` for every caller
that scores or drafts from the pipeline's answer (on ``train``: ``finetune.scoring.archive``, ``finetune.training.pretrack``),
and the shape the tracking JSON gives that answer, built from the ``RunResult`` instead of
re-read from disk.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from .config import Config
from .paths import BOUNCE_MODEL, COURT_ENGINE, PLAYER_ENGINE
from .export import RunResult
from .tracking_schema import TrackingFrames


def headless_config(video: Path, *, model_path: Path, conf: float, device: str, tracking_json: Path,
                    output_video: Optional[Path] = None, court_events: bool = False,
                    window_step: Optional[int] = None, prepass_mode: Optional[str] = None) -> Config:
    """The tracker's configuration for a run that writes the tracking JSON and no video: the
    bundled player and court engines, the selector's table silenced. ``output_video`` only names
    the video the run would have written (its debug outputs are derived from it); ``court_events``
    also computes and exports the court events, with the bundled bounce model; ``window_step`` is
    ``Config.gridtracknet_window_step``, the prepass's frames between unit starts, and ``prepass_mode``
    ``Config.ball_prepass_mode``, where the prepass runs (None, their defaults)."""
    config = Config(
        input_video=str(video),
        output_video=str(output_video or tracking_json.with_suffix(".mp4")),
        tracking_json=str(tracking_json),
        model_path=str(model_path),
        player_model_path=str(PLAYER_ENGINE),
        court_model_path=str(COURT_ENGINE),
        bounce_model_path=str(BOUNCE_MODEL),
        conf=conf,
        device=device,
        save_tracking_video=False,
        court_events=court_events,
        print_selector_tracks=False,
    )
    if window_step is not None:
        config = replace(config, gridtracknet_window_step=window_step)
    return config if prepass_mode is None else replace(config, ball_prepass_mode=prepass_mode)


def run_headless(video: Path, *, model_path: Path, conf: float, device: str, tracking_json: Path,
                 output_video: Optional[Path] = None, court_events: bool = False,
                 window_step: Optional[int] = None, prepass_mode: Optional[str] = None) -> RunResult:
    """Track ``video`` with ``headless_config``'s settings and return what was found."""
    from .pipeline import run     # the pipeline imports CUDA, TensorRT and the renderers: only when a run is wanted

    return run(headless_config(video, model_path=model_path, conf=conf, device=device, tracking_json=tracking_json,
                               output_video=output_video, court_events=court_events, window_step=window_step,
                               prepass_mode=prepass_mode))


def tracking_payload(result: RunResult) -> TrackingFrames:
    """The part of the tracking JSON a scorer reads (``frames`` and the court as last seen), from
    the run's result, so a caller need not read the file the run wrote."""
    return {"frames": result.frame_rows(), "last_valid_court_keypoints": result.last_valid_court_keypoints}
