"""What the scorers of the new label types share: the command line, the clips that have labels of a
type, the metric lines and comparing two logs of them.

The hit, rally, player and court scorers (``hits_score``, ``rallies_score``, ``players_score``,
``court_score``) follow the landing scorer's conventions (``court_events_score``): a pipeline run's
folder of tracking JSONs (``--out-dir``, written by ``evaluate_archive.py --mode pipeline --court``),
the archive's scorable clips narrowed by ``--clips`` and ``--holdout``, one metric line per clip and a
pooled ``ALL`` line, and ``--compare before.log`` for the deltas against an earlier log.

Their archive defaults to the label home's (``workspace.label_home``): the main checkout's labels,
which is where the label tool writes, even when a scorer runs from a worktree.

A metric line is the clip's name and then ``key=value`` pairs, some values with a unit
(``0.913``, ``2.0fr``, ``0.41s``, ``3.2px``, ``0.08m``) or ``n/a``; ``read_metric_lines`` reads any of
them back, so one ``compare`` serves every scorer.
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from finetune.data import workspace as ws
from finetune.scoring.archive import archive_folders, list_clips
from finetune.scoring.tracker_events import RUN_MODEL

ALL = "ALL"
NAME_WIDTH = 14
METRIC = re.compile(r"(?P<key>[a-z_@.0-9]+)=\s*(?P<value>-?[\d.]+|n/a)(?P<unit>fr|ms|px|m|s)?")

Metrics = Dict[str, Optional[float]]
LabelledClip = Tuple[str, Path, Path]      # (clip, ball label csv, the type's label file)


def label_archive() -> Path:
    """The archive the new scorers read by default: the folder holding the label home."""
    return ws.label_home().parent


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments every scorer of a pipeline run takes."""
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="the pipeline run's folder of <clip>_<model>.json files (evaluate_archive.py --out-dir)")
    parser.add_argument("--models", nargs="+", default=[RUN_MODEL])
    parser.add_argument("--clips", nargs="*", default=None, help="Clip stems or globs, e.g. video6 'grid_match*'")
    parser.add_argument("--holdout", default=None, help="Only the clips of this held-out set (holdouts.json), e.g. mixed_v1")
    parser.add_argument("--archive", type=Path, default=None,
                        help="Label set: a finetune-style folder with videos/ and labels/ (default: the label home's, "
                             "the main checkout's finetune/)")
    parser.add_argument("--include-excluded", action="store_true",
                        help="Score every labelled clip, including those the review policy holds back")
    parser.add_argument("--compare", type=Path, default=None, help="an earlier log of this script to print deltas against")


def labelled_clips(names: Optional[List[str]], archive: Path, include_excluded: bool, holdout: Optional[str],
                   label_path: Callable[[str, Path], Path]) -> List[LabelledClip]:
    """(clip, ball labels, the type's labels) for every scorable clip that has the type's label file."""
    label_dir, _ = archive_folders(archive)
    found = []
    for video, csv_path in list_clips(names, archive, include_excluded, holdout):
        path = label_path(video.stem, label_dir)
        if path.is_file():
            found.append((video.stem, csv_path, path))
    return found


# --------------------------------------------------------------------------- numbers

def median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)] if ordered else None


def share(part: float, whole: float) -> Optional[float]:
    return part / whole if whole else None


# --------------------------------------------------------------------------- metric lines

def metric(key: str, value: Optional[float], spec: str = ".3f", unit: str = "") -> str:
    """``key=value``, ``n/a`` for a value there is nothing to measure with."""
    return f"{key}={'  n/a' if value is None else format(value, spec) + unit}"


def metric_line(name: str, groups: Iterable[Iterable[str]]) -> str:
    """A clip's name and its metrics, groups split by ``|``."""
    return f"{name:{NAME_WIDTH}s} " + " | ".join(" ".join(group) for group in groups)


def read_metric_lines(text: str) -> Dict[str, Metrics]:
    """{clip or ALL: {key: value}} for every metric line in a log (lines with a ``key=`` pair after
    the name; indented detail lines are skipped)."""
    out: Dict[str, Metrics] = {}
    for line in text.splitlines():
        if not line or line[0].isspace() or line.startswith(("=", "[")):
            continue
        name, _, rest = line.partition(" ")
        values = {match["key"]: None if match["value"] == "n/a" else float(match["value"]) for match in METRIC.finditer(rest)}
        if values:
            out[name] = values
    return out


def compare(before: Dict[str, Metrics], after: Dict[str, Metrics], keys: Sequence[str]) -> str:
    """One row per clip in both logs, the pooled line last: each of `keys` as it is now, with its
    change since the first log in brackets."""
    shared = [name for name in before if name in after]
    names = [name for name in shared if name != ALL] + ([ALL] if ALL in shared else [])
    width = max([len("clip")] + [min(30, len(name)) for name in names])
    header = f"{'clip':{width}s} " + " ".join(f"{key:>18s}" for key in keys)
    rows = [header, "-" * len(header)]
    for name in names:
        if name == ALL:
            rows.append("")
        cells = []
        for key in keys:
            new, old = after[name].get(key), before[name].get(key)
            if new is None:
                cells.append(f"{'n/a':>18s}")
            elif old is None:
                cells.append(f"{new:>9.3f} {'(new)':>8s}")
            else:
                cells.append(f"{new:>9.3f} ({new - old:+.3f})")
        rows.append(f"{name[:30]:{width}s} " + " ".join(cells))
    only_before = [name for name in before if name not in after]
    only_after = [name for name in after if name not in before]
    if only_before:
        rows.append(f"\nonly in the first log: {', '.join(only_before)}")
    if only_after:
        rows.append(f"only in the second log: {', '.join(only_after)}")
    return "\n".join(rows)


def finish(lines: List[str], compare_path: Optional[Path], keys: Sequence[str]) -> int:
    """Print the report and, with --compare, the deltas against the earlier log."""
    print("\n".join(lines))
    if compare_path is None:
        return 0
    before = read_metric_lines(compare_path.read_text(encoding="utf-8"))
    if not before:
        print(f"no metric lines found in {compare_path}", file=sys.stderr)
        return 1
    print(f"\n=== against {compare_path} ===")
    print(compare(before, read_metric_lines("\n".join(lines)), keys))
    return 0


def no_labels(kind: str, archive: Path, pattern: str) -> int:
    """The note for a label type nobody has labelled yet (`kind` as in "hit labels"): not an error,
    there is nothing to score."""
    print(f"[none] no scorable clip in {archive} has {kind} labels ({pattern}) yet: nothing to score")
    return 0
