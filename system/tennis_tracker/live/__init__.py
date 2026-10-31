"""The live tracker: frames in one at a time, each frame's ball out a fixed delay later.

``pipeline.run_live`` is the entry point (``clean_tracker.py --live``); ``source`` delivers the
frames and holds the drop policy, ``fixed_lag`` commits the selector's output on a fixed lag and
``evidence`` records the selector's input so another lag can be replayed on the CPU, and
``selector_process`` runs the selector in a child process. Only ``pipeline`` runs anything on the
GPU; ``source`` (page-locked buffers) and ``evidence`` (through ``pass1``) import torch, and
``fixed_lag`` imports neither torch nor the engines.
"""
