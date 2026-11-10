"""What the label scorers share: the command line (``run_scorer``), the clips that have labels of a
type, the metric lines and comparing two logs of them.

The landing, hit, rally, player and court scorers (``court_events_score``, ``hits_score``,
``rallies_score``, ``players_score``, ``court_score``) each score a pipeline run's folder of
tracking JSONs (``--out-dir``, written by ``evaluate_archive.py --mode pipeline --court``), one
metric line per clip and a pooled ``ALL`` line, with ``--compare before.log`` for the deltas
against an earlier log.

Which labels they score is one rule for all five (``labelled_clips``):

- the archive is ``--archive``, else the label home's (``workspace.label_home``): the main
  checkout's labels, where the label tool writes, even when a scorer runs from a worktree;
- a clip is scored for a type when it has that type's label file, narrowed by ``--clips`` and
  ``--holdout``, unless it is listed in exclude.txt (``--include-excluded`` scores it anyway);
- whether its ball labels have been reviewed (``verified_clips`` in the review policy) is about the
  ball labels alone. It does not hold the clip's other labels back, and a clip needs no ball labels
  for them to count. It only decides whether a landing or a hit without a point of its own may
  borrow one from the ball labels (``LabelledClip``'s ball path is None when it may not): such a
  label is then still scored for its timing, and left out of the pixel error.

A metric line is the clip's name and then ``key=value`` pairs, some values with a unit
(``0.913``, ``2.0fr``, ``0.41s``, ``3.2px``, ``0.08m``) or ``n/a``; ``read_metric_lines`` reads any of
them back, so one ``compare`` serves every scorer.
"""
from __future__ import annotations

import argparse
import fnmatch
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from finetune.data import workspace as ws
from finetune.data.data_policy import is_verified, read_review_status
from finetune.data.holdouts import holdout_clips
from finetune.data.labels import Labels, read_labels
from finetune.scoring.archive import archive_folders
from finetune.scoring.tracker_events import RUN_MODEL

ALL = "ALL"
NAME_WIDTH = 14
METRIC = re.compile(r"(?P<key>[a-z_@.0-9]+)=\s*(?P<value>-?[\d.]+|n/a)(?P<unit>fr|ms|px|m|s)?")

Metrics = Dict[str, Optional[float]]
# (clip, its ball labels when points may be borrowed from them (None: unreviewed, or none), the type's label file)
LabelledClip = Tuple[str, Optional[Path], Path]
LabelPath = Callable[[str, Path], Path]    # clip, labels folder -> the type's label file (workspace.hits_csv)
ScoreRun = Callable[[Sequence[LabelledClip], Path, str, argparse.Namespace], List[str]]


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
                   label_path: LabelPath) -> List[LabelledClip]:
    """Every clip with the type's label file that may be scored, by the rule of the module
    docstring, in natural order; the clips left out, and those whose ball labels lend no point,
    said on a ``[skip]`` or ``[note]`` line."""
    label_dir, _ = archive_folders(archive)
    suffix = label_path("", label_dir).name
    clips = sorted((path.name[:-len(suffix)] for path in label_dir.glob(f"*{suffix}")), key=ws.natural_key)
    if names:
        clips = [clip for clip in clips if any(fnmatch.fnmatchcase(clip, pattern) for pattern in names)]
    if holdout:
        held_out = set(holdout_clips(holdout, clips, ws.read_manifest(archive / ws.MANIFEST_NAME), archive / ws.HOLDOUTS_NAME))
        clips = [clip for clip in clips if clip in held_out]
    excluded = set(ws.read_exclusions(archive))
    status = read_review_status(archive)
    found: List[LabelledClip] = []
    for clip in clips:
        if clip in excluded and not include_excluded:
            print(f"[skip] {clip}: listed in exclude.txt")
            continue
        ball = ws.label_csv(clip, label_dir)
        lends = ball.is_file() and (include_excluded or is_verified(clip, status))
        if ball.is_file() and not lends:
            print(f"[note] {clip}: its ball labels are still under review, so no point is borrowed from them")
        found.append((clip, ball if lends else None, label_path(clip, label_dir)))
    return found


def ball_labels(path: Optional[Path], width: int, height: int) -> Labels:
    """A clip's ball labels for the points its landings or hits borrow; none when it may lend none."""
    return read_labels(path, width, height) if path is not None else {}


def run_scorer(argv: Optional[Sequence[str]], description: str, kind: str, label_path: LabelPath, score_run: ScoreRun,
               keys: Sequence[str] = (), add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None,
               report: Optional[Callable[[List[str], argparse.Namespace], int]] = None) -> int:
    """A scorer's command: its arguments (``add_run_arguments`` and `add_arguments`), the clips with
    `kind` labels (``labelled_clips``), `score_run` per model, then `report` (by default the lines
    printed and ``--compare`` read back with ``read_metric_lines`` for `keys`). A missing or
    unreadable run is 1, with the error; no clip labelled is a note, and 0."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)
    archive = args.archive or label_archive()
    clips = labelled_clips(args.clips, archive, args.include_excluded, args.holdout, label_path)
    if not clips:
        return no_labels(kind, archive, label_path("<clip>", Path("labels")).as_posix())
    print(f"[archive] {archive}  ({len(clips)} clip{'' if len(clips) == 1 else 's'} with {kind} labels: "
          f"{', '.join(clip for clip, _, _ in clips)})")
    try:
        lines = [line for model in args.models for line in score_run(clips, args.out_dir, model, args)]
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return report(lines, args) if report is not None else finish(lines, args.compare, keys)


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
