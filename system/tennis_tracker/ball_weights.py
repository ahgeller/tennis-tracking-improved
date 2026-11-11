"""The ball-detector weights this tree knows by file: each one's operating threshold, its visibility gate
and where it comes from.

A detector's confidence threshold belongs to its weights, not to the tracker: each file here was
passed by the held-out pipeline gate (``evaluate_archive.py --mode pipeline --holdout mixed_v1``)
at its own threshold, so a run that names the weights and no threshold gets that one
(``operating_conf``). Weights this table does not know take ``UNKNOWN_CONF``, the heat-map
network's. The selector's visibility gate (``Config.ball_visibility_gate``) belongs to the weights
the same way: only a file carrying the visibility head can answer it, and the gate it was passed
at is its own (``operating_visibility_gate``; 0, off, for every other file). No file is read here: the table is keyed by path, so ``Config`` can take its defaults
from it without torch.

Keyed by path, the table cannot tell what is in a file. Weights at another path (a retrained
stream_s4 in finetune/models/) are unknown and run at ``UNKNOWN_CONF``, which the callers say
(``unknown_weights_note``); weights copied over a known path (``ft.py promote`` over
``models/ball_detector.pt``) inherit that entry's threshold and origin, which ``ft.py promote``
warns about. A new file gated at another threshold needs its own entry here, or ``--conf``.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Union

from .paths import BALL_WEIGHTS, HEATMAP_WEIGHTS, ROOT, STREAM_E60_WEIGHTS, STREAM_PEAK_WEIGHTS, STREAM_WEIGHTS

PathLike = Union[str, Path]


class KnownWeights(NamedTuple):
    """One weights file: its path, the threshold it was gated at, where to get it and the
    visibility gate it was gated with (0: none, the file has no visibility head or runs without)."""
    path: Path
    conf: float
    origin: str
    visibility_gate: float = 0.0


# On mixed_v1 (2026-11-10, integration tree): gate off 0.957 recall / 94 false alarms of 518, 0.30
# 0.956 / 70, 0.40 0.955 / 67, 0.50 0.954 / 63; live mode 0.956 / 94 off and 0.954 / 67 at 0.40.
# Clip-paired at 0.40: 27 false alarms removed and none added, 17 hits lost and 1 gained of 8,515.
STREAM_S4_PEAK_WEIGHTS = KnownWeights(
    STREAM_PEAK_WEIGHTS, 0.38,
    "the stream_s4 ball detector of run s4_fg1 (epoch 46) with the peak visibility head of run s4_fg1_peak "
    "(the head trained alone on the frozen detector, branch phase-vis-only: finetune/train_gridtracknet.py "
    "--arch stream_s4 --vis-only); like every weights file it is not in the repository",
    visibility_gate=0.40)
STREAM_S4_WEIGHTS = KnownWeights(
    STREAM_WEIGHTS, 0.38,
    "the stream_s4 ball detector (run s4_fg1, epoch 46: trained from scratch on both label sets with 30% "
    "pseudo-labelled clips, distilled from ball_detector.pt, its visibility head stripped), trained on the "
    "phase-colab-fg branch (finetune/train_gridtracknet.py --arch stream_s4); like every weights file it "
    "is not in the repository")
STREAM_S4_E60_WEIGHTS = KnownWeights(
    STREAM_E60_WEIGHTS, 0.42,
    "the stream_s4 ball detector of epoch 60, the default before 2026-11-05, trained on the train branch "
    "(finetune/train_gridtracknet.py --arch stream_s4); not in the repository")
HEATMAP_S4_WEIGHTS = KnownWeights(
    HEATMAP_WEIGHTS, 0.40,
    "the heatmap_s4 ball detector, trained on the train branch (models/README.md); not in the repository")
KNOWN = (STREAM_S4_PEAK_WEIGHTS, STREAM_S4_WEIGHTS, STREAM_S4_E60_WEIGHTS, HEATMAP_S4_WEIGHTS)
UNKNOWN_CONF = HEATMAP_S4_WEIGHTS.conf


def known_weights(weights: PathLike) -> Optional[KnownWeights]:
    """The table's entry for `weights` (compared as resolved paths), or None."""
    resolved = Path(weights).expanduser().resolve()
    return next((entry for entry in KNOWN if entry.path.resolve() == resolved), None)


def operating_conf(weights: PathLike) -> float:
    """The confidence threshold `weights` run at when none is given: the gated one, else ``UNKNOWN_CONF``."""
    entry = known_weights(weights)
    return entry.conf if entry is not None else UNKNOWN_CONF


def operating_visibility_gate(weights: PathLike) -> float:
    """The visibility gate `weights` run with when none is given: the one the file was gated with,
    0 (off) for a file gated without one or one this table does not know."""
    entry = known_weights(weights)
    return entry.visibility_gate if entry is not None else 0.0


def unknown_weights_note(weights: PathLike) -> Optional[str]:
    """For weights the table does not know, the line saying which threshold they run at and why
    (to print where no threshold was given); None for known weights."""
    if known_weights(weights) is not None:
        return None
    return (f"{Path(weights).name} is not a gated file this tree knows (ball_weights.py), so it runs at "
            f"{UNKNOWN_CONF:.2f}; pass --conf with the threshold it was gated at")


def missing_weights(weights: PathLike) -> FileNotFoundError:
    """The error for weights that are not on disk; for a known file it says what the file is, where
    it goes and how to run the other known detector instead."""
    path = Path(weights).expanduser().resolve()
    entry = known_weights(path)
    message = f"Ball-detector weights not found: {path}"
    if entry is not None:
        others = " or ".join(f"--weights {other.path.relative_to(ROOT).as_posix()}"
                             for other in KNOWN if other is not entry)
        message += (f"\n  This is {entry.origin}. Copy the file to {path} from a checkout or machine that has "
                    f"it (models/README.md lists the files), or run another detector with {others}.")
    return FileNotFoundError(message)


DEFAULT_CONF = operating_conf(BALL_WEIGHTS)        # the default detector's threshold, Config.conf's default
