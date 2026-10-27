"""Scoring the tracker and the detector against hand labels.

Two scorers live here on purpose, because they answer different questions and one of them is
frozen:

- ``evaluation.validate`` scores an exported tracking JSON against an annotation JSON in native
  pixels (presence recall, 5/10/20 px localisation buckets, absence false positives, jumps).
  The parity gate freezes its numbers, so they are not to change.
- ``evaluation.parity`` is that gate: it runs the tracker on the bundled Pomona benchmark and
  compares the scores with the frozen baseline.
- ``evaluation.archive`` scores the raw detector or the whole pipeline on the fine-tuning
  workspace's labelled clips with ``finetune.data.detector_metrics`` (1080p-normalised hit / wrong-object /
  false-alarm), the same metric training uses to pick a winner; ``evaluation.pipeline_score``
  scores one clip of a pipeline run for it and says where each miss landed.

``validate`` and ``parity`` run as modules from the repository root (``python -m evaluation.validate``,
``python -m evaluation.parity``); the root keeps ``evaluate_archive.py``, ``score_court_events.py`` and
``track_breaks.py`` as thin entry points to the workspace scorers.
"""
