"""Scoring the tracker against hand labels.

- ``evaluation.validate`` scores an exported tracking JSON against an annotation JSON in native
  pixels (presence recall, 5/10/20 px localisation buckets, absence false positives, jumps).
  ``check_parity`` gates the bundled Pomona benchmark on its numbers, so they are not to change.
- ``evaluation.parity`` is that gate: it runs the tracker on the sample and compares the scores
  with the frozen baseline.

The repository root keeps ``validate_tracking.py`` and ``check_parity.py`` as thin entry points to
these modules, so the documented commands keep working. The detector and pipeline scorers over the
labelling workspace's clips (``evaluate_archive.py``, ``score_court_events.py``, ``track_breaks.py``)
live on the ``train`` branch with the workspace they read.
"""
