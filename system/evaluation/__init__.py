"""Scoring the tracker against hand labels.

- ``evaluation.validate`` scores an exported tracking JSON against an annotation JSON in native
  pixels (presence recall, 5/10/20 px localisation buckets, absence false positives, jumps).
  The parity gate freezes its numbers, so they are not to change.
- ``evaluation.parity`` is that gate: it runs the tracker on the bundled Pomona benchmark and
  compares the scores with the frozen baseline.

Both run as modules from the repository root: ``python -m evaluation.validate`` and
``python -m evaluation.parity``. The detector and pipeline scorers over the labelling workspace's
clips (``evaluate_archive.py``, ``score_court_events.py``, ``track_breaks.py``) live on the
``train`` branch with the workspace they read.
"""
