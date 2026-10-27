"""The workspace scorers: ``archive`` scores the detector or the whole pipeline on the workspace
clips (``evaluate_archive.py`` at the repository root), with ``pipeline_score`` scoring one
pipeline run's clip; ``court_events_score`` scores a ``--court`` run's landings against the
landing labels (``score_court_events.py``); ``track_breaks`` buckets a pipeline run's misses by
cause (``track_breaks.py``); ``replay_selector`` reruns the selector on the CPU over a run's
recorded detections; ``compare_eval_logs`` sets two evaluator logs side by side.

They read the workspace and its labels, so they live here and not in the tracker's
``evaluation`` package (``system/evaluation``: the tracking JSON scorer and the parity gate).
"""
