"""The live tracker: frames in one at a time, each frame's ball out a fixed delay later.

``pipeline.run_live`` is the entry point (``clean_tracker.py --live``); ``source`` delivers the
frames and holds the drop policy, ``fixed_lag`` commits the selector's output on a fixed lag and
``evidence`` records the selector's input so another lag can be replayed on the CPU. The
modules import CUDA only through ``pipeline``.
"""
