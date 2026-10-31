"""Every label type's scorecard for one pipeline run: landings, hits, rallies, player identity and
the court, each from its own scorer, on the same clips.

    python score_labels.py --out-dir output/label_candidates [--holdout mixed_v1] [--only hits rallies]

The run folder is ``evaluate_archive.py --mode pipeline --court --out-dir <folder>``'s. The labels
are read from the label home (``workspace.label_home``) unless ``--archive`` says otherwise, the
landing scorer's included, so every section reads the labels the label tool writes. A type nobody
has labelled yet says so and the rest carry on (the landing scorer counts that as a failure of
its own: it is listed at the end, and nothing else is affected). For ``--compare``, run a type's
scorer alone (``python -m finetune.scoring.<type>_score``) and give it an earlier log of that
scorer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from finetune.scoring import court_events_score, court_score, hits_score, players_score, rallies_score
from finetune.scoring.label_scoring import label_archive

SCORERS: Dict[str, Callable[[Optional[Sequence[str]]], int]] = {
    "landings": court_events_score.main, "hits": hits_score.main, "rallies": rallies_score.main,
    "players": players_score.main, "court": court_score.main}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True, help="the pipeline run's folder of tracking JSONs")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--clips", nargs="*", default=None, help="Clip stems or globs")
    parser.add_argument("--holdout", default=None, help="Only the clips of this held-out set, e.g. mixed_v1")
    parser.add_argument("--archive", type=Path, default=None, help="Label set (default: the label home's)")
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--only", nargs="+", choices=list(SCORERS), default=list(SCORERS))
    args = parser.parse_args(argv)
    archive = args.archive or label_archive()
    shared: List[str] = ["--out-dir", str(args.out_dir), "--archive", str(archive)]
    if args.models:
        shared += ["--models", *args.models]
    if args.clips:
        shared += ["--clips", *args.clips]
    if args.include_excluded:
        shared.append("--include-excluded")
    if args.holdout:
        shared += ["--holdout", args.holdout]
    failed = []
    for name in args.only:
        print(f"\n######## {name} ########")
        if SCORERS[name](shared) != 0:
            failed.append(name)
    if failed:
        print(f"\n[scorecard] failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
