"""Tier 0 test B: the e60_0.42 gate run re-scored under stricter gap-filling rules (CPU only).

    python experiments/tier0/B_fill_rules/run.py [--run DIR]

The recorded run's `frames` table tags every output row with its `source`: `det` (a selected
track's detection), `interp` (a straight/parabolic fill between detections, or a closed single
hole) or `motion` (a fill placed on a motion blob, or a tail extended on motion). Switching a
fill rule off is therefore the same as dropping the rows it produced, up to the refit (which
weights interp at 0.15 and motion at 0.30 of a detection when it smooths neighbours):

    baseline    the run as written (must reproduce recall 0.953, FA 95 = 79 bridged / 16 false)
    no_interp   (a) rows with source == interp removed
    no_motion   (b) rows with source == motion removed
    det_only    (c) both removed: only detector observations the chooser kept
    inferred    (d) fills kept but tagged: the per-source contribution table (hits, wrong, FA
                bridged / false, unlabelled) so recall and FA can be read with and without them

The refit effect is measured separately: the selector is replayed over the recorded detections
(finetune.scoring.replay_selector, `chain`) once as it is and once with `_fill_internal_gaps`
and `_close_single_frame_holes` patched to no-ops (the replay has no motion masks, so its
motion rows are absent in both), and the patched replay is compared with the post-filtered one.
No selector or replay code is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import finetune  # noqa: F401
from ball_in_play_selector import core
from finetune.data.detector_metrics import HIT_PX, WRONG_PX, DetectorMetrics, bridged_alarms, scale_for_width
from finetune.data.labels import read_labels
from finetune.data.workspace import label_csv, read_manifest
from finetune.data.data_policy import clip_source
from finetune.scoring.pipeline_score import score_pipeline_clip
from finetune.scoring.replay_selector import replay_variant

RUN = Path(r"C:\Users\Andrew\AppData\Local\Temp\claude\C--Users-Andrew-Desktop-clean-version"
           r"\a6549017-5a6a-4af2-bd14-c1b09a105d2c\scratchpad\opt1_gate\e60_0.42")
RULES = {"baseline": (), "no_interp": ("interp",), "no_motion": ("motion",), "det_only": ("interp", "motion")}
GROUP = {"custom": "own", "grid": "grid", "prof": "prof"}


def without(payload, sources):
    """The payload with every output row of these sources turned into an absent frame."""
    out = dict(payload)
    out["frames"] = [dict(row, present=False) if row.get("present") and row.get("source") in sources else row
                     for row in payload["frames"]]
    return out


def load(run_dir, manifest):
    clips = []
    for jp in sorted(run_dir.glob("*_gridtracknet.json")):
        clip = jp.name[:-len("_gridtracknet.json")]
        payload = json.loads(jp.read_text(encoding="utf-8"))
        W, H = int(payload["video"]["width"]), int(payload["video"]["height"])
        clips.append((clip, GROUP.get(clip_source(clip, manifest), "other"), payload, read_labels(label_csv(clip), W, H)))
    return clips


def metrics(clips, transform):
    by_group = {}
    pairs_all = []
    for clip, group, payload, labels in clips:
        scored = score_pipeline_clip(labels, transform(payload), int(payload["video"]["width"]))
        by_group.setdefault(group, []).append(scored.pairs)
        pairs_all.append(scored.pairs)
    rows = {g: DetectorMetrics.of_clips(p, 1.0) for g, p in sorted(by_group.items())}
    rows["all"] = DetectorMetrics.of_clips(pairs_all, 1.0)
    return rows


def row_dict(rule, group, m):
    return dict(rule=rule, group=group, vis=m.visible, hits=round(m.recall * m.visible), recall=round(m.recall, 4),
                wrong=round(m.wrong * m.visible), invis=m.invisible, fa=m.false_alarm_count,
                bridged=m.bridged_count, false=m.false_alarm_count - m.bridged_count)


def contribution(clips):
    """What every output source contributes: hits, loc, wrong on visible labels; bridged / false
    on invisible labels; rows on unlabelled frames."""
    tally = {}
    for clip, group, payload, labels in clips:
        W = int(payload["video"]["width"])
        scale = scale_for_width(W)
        frames = {r["frame"]: r for r in payload["frames"]}
        ordered = sorted(labels.items())
        pairs = [(lab, (frames[f]["x"], frames[f]["y"]) if f in frames and frames[f].get("present") else None)
                 for f, lab in ordered]
        flags = bridged_alarms(pairs, scale)
        labelled = set(labels)
        for (f, lab), (_, pred), br in zip(ordered, pairs, flags):
            if pred is None:
                continue
            src = frames[f].get("source")
            c = tally.setdefault(src, Counter())
            if lab is None:
                c["bridged" if br else "false"] += 1
            else:
                e = math.hypot(pred[0] - lab[0], pred[1] - lab[1]) * scale
                c["hit" if e <= HIT_PX else ("loc" if e <= WRONG_PX else "wrong")] += 1
        for f, r in frames.items():
            if r.get("present") and f not in labelled:
                tally.setdefault(r.get("source"), Counter())["unlabelled"] += 1
    return tally


def fmt(rows):
    head = f"{'rule':12s} {'grp':5s} {'vis':>5s} {'hits':>5s} {'recall':>7s} {'wrong':>5s} {'invis':>5s} {'FA':>4s} {'brdg':>4s} {'false':>5s}"
    lines = [head]
    for r in rows:
        lines.append(f"{r['rule']:12s} {r['group']:5s} {r['vis']:5d} {r['hits']:5d} {r['recall']:7.3f} {r['wrong']:5d} "
                     f"{r['invis']:5d} {r['fa']:4d} {r['bridged']:4d} {r['false']:5d}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--no-replay", action="store_true", help="skip the refit cross-check (the slow part)")
    args = ap.parse_args(argv)
    clips = load(args.run, read_manifest())

    rows = []
    for rule, sources in RULES.items():
        for group, m in metrics(clips, lambda p, s=sources: without(p, s)).items():
            rows.append(row_dict(rule, group, m))
    print("== recorded run, fill rows removed by source ==")
    print(fmt(rows))
    base = next(r for r in rows if r["rule"] == "baseline" and r["group"] == "all")
    assert (base["hits"], base["fa"], base["bridged"], base["false"]) == (8111, 95, 79, 16), base
    print("\nrule          hits lost  FA removed (bridged/false)  FA removed per hit lost")
    for rule in ("no_interp", "no_motion", "det_only"):
        r = next(x for x in rows if x["rule"] == rule and x["group"] == "all")
        lost, gone = base["hits"] - r["hits"], base["fa"] - r["fa"]
        print(f"{rule:12s} {lost:9d}  {gone:3d} ({base['bridged'] - r['bridged']}/{base['false'] - r['false']})"
              f"{'':16s} {gone / lost if lost else float('inf'):.2f}")

    print("\n== (d) inferred rows tagged: what each output source contributes ==")
    tally = contribution(clips)
    keys = ("hit", "loc", "wrong", "bridged", "false", "unlabelled")
    print(f"{'source':8s} " + " ".join(f"{k:>10s}" for k in keys))
    for src in ("det", "interp", "motion"):
        c = tally.get(src, Counter())
        print(f"{src:8s} " + " ".join(f"{c[k]:10d}" for k in keys))
    with open(args.out / "fill_rules.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({k: dict(v) for k, v in tally.items()}, open(args.out / "source_contribution.json", "w"), indent=1)

    if args.no_replay:
        return 0
    print("\n== refit cross-check: selector replayed over the recorded detections (no motion masks) ==")
    chain = {clip: replay_variant("chain", payload) for clip, _, payload, _ in clips}
    noop = lambda *a, **k: None
    with patch.object(core, "_fill_internal_gaps", noop), patch.object(core, "_close_single_frame_holes", noop):
        chain_nofill = {clip: replay_variant("chain", payload) for clip, _, payload, _ in clips}
    rep = []
    for name, table, sources in (("replay_chain", chain, ()), ("replay_chain-interp", chain, ("interp",)),
                                 ("replay_nofill", chain_nofill, ())):
        m = metrics(clips, lambda p, t=table, s=sources: without(t[_clip_of(p)], s))["all"]
        rep.append(row_dict(name, "all", m))
    print(fmt(rep))
    with open(args.out / "replay_crosscheck.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rep[0]))
        w.writeheader()
        w.writerows(rep)
    return 0


def _clip_of(payload):
    return Path(payload["video"]["input"]).stem


if __name__ == "__main__":
    raise SystemExit(main())
