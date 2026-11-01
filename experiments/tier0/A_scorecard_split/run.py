"""Tier 0 test A: the gate's scorecard split into in-flight balls, held/static balls and speed
bands, per source group, for the recorded gate runs (CPU only, reads the tracking JSONs).

    python experiments/tier0/A_scorecard_split/run.py [--gate DIR] [--runs e60_0.42,opt1_0.38,...]

Definitions (all distances in 1080p px, frame = source frame):
- A visible label's speed is the label's displacement per source frame from the neighbouring
  labels (both sides when present, one side otherwise; `None` when it has no labelled neighbour
  within one label stride). This is finetune.scoring's miss_overlap logic.
- "held/static": speed < 6 px/frame AND the label is inside a player box of the run's own JSON
  (`frames[f].player_boxes`, no margin). Everything else visible is "in-flight".
- An invisible label has no position, so a false alarm is classed by the run's own output: its
  speed from the neighbouring output rows (frame +- stride) and whether the output point sits
  inside a player box. A quiet invisible frame (no output) has no speed and is counted only in
  the bucket's `invis` denominator for the `all` row.
- Speed bands: 0-6, 6-30, 30-45, 45-70, 70+ px/frame. 30 FPS clips move about twice as far per
  frame as 60 FPS clips; the band is per source frame, as the selector sees it.
- recall = hits (<= 10 px) / visible labels in the bucket; wrong = outputs > 30 px off / visible;
  FA = outputs on invisible labels, split bridged (within 10 px of the line between the flanking
  visible labels, run of <= 3) / false, with finetune.data.detector_metrics.bridged_alarms.
The `all` row of each run must reproduce the gate log (e60_0.42: recall 0.953, FA 95 = 79/16).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import finetune  # noqa: F401  (puts system/ on the path)
from finetune.data.detector_metrics import HIT_PX, WRONG_PX, bridged_alarms, scale_for_width
from finetune.data.labels import read_labels
from finetune.data.workspace import label_csv, read_manifest
from finetune.data.data_policy import clip_source
from finetune.scoring.pipeline_score import _inside_player

GATE = Path(r"C:\Users\Andrew\AppData\Local\Temp\claude\C--Users-Andrew-Desktop-clean-version"
            r"\a6549017-5a6a-4af2-bd14-c1b09a105d2c\scratchpad\opt1_gate")
RESULTS_CSV = Path(r"C:\Users\Andrew\Desktop\label_backups\models\stream_s4_new\results.csv")
STATIC_SPEED = 6.0
BANDS = [(0, 6), (6, 30), (30, 45), (45, 70), (70, math.inf)]
GROUP = {"custom": "own", "grid": "grid", "prof": "prof"}


def band_of(speed):
    if speed is None:
        return "n/a"
    for lo, hi in BANDS:
        if lo <= speed < hi:
            return f"{lo}-{hi}" if hi != math.inf else f"{lo}+"
    return "n/a"


def label_speed(labels, f, stride, scale):
    lab = labels[f]
    prev, nxt = labels.get(f - stride), labels.get(f + stride)
    if prev and nxt:
        return math.hypot(nxt[0] - prev[0], nxt[1] - prev[1]) * scale / (2 * stride)
    if prev:
        return math.hypot(lab[0] - prev[0], lab[1] - prev[1]) * scale / stride
    if nxt:
        return math.hypot(nxt[0] - lab[0], nxt[1] - lab[1]) * scale / stride
    return None


def output_speed(frames, f, stride, scale):
    row = frames.get(f)
    prev, nxt = frames.get(f - stride), frames.get(f + stride)
    pts = [(r["x"], r["y"]) for r in (prev, nxt) if r and r.get("present")]
    if len(pts) == 2:
        return math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) * scale / (2 * stride)
    if pts and row and row.get("present"):
        return math.hypot(pts[0][0] - row["x"], pts[0][1] - row["y"]) * scale / stride
    return None


def score_run(run_dir, manifest):
    """One record per labelled frame of the run: outcome, bucket keys, source group."""
    records = []
    for jp in sorted(run_dir.glob("*_gridtracknet.json")):
        clip = jp.name[:-len("_gridtracknet.json")]
        payload = json.loads(jp.read_text(encoding="utf-8"))
        W, H = int(payload["video"]["width"]), int(payload["video"]["height"])
        scale = scale_for_width(W)
        labels = read_labels(label_csv(clip), W, H)
        stride = int(manifest[clip]["stride"] or 1) if clip in manifest else 1
        frames = {r["frame"]: r for r in payload["frames"]}
        group = GROUP.get(clip_source(clip, manifest), "other")
        ordered = sorted(labels.items())
        pairs = [(lab, ((frames.get(f) or {}).get("x"), (frames.get(f) or {}).get("y"))
                  if (frames.get(f) or {}).get("present") else None) for f, lab in ordered]
        flags = bridged_alarms(pairs, scale)
        for (f, lab), (_, pred), bridged in zip(ordered, pairs, flags):
            row = frames.get(f) or {}
            boxes = row.get("player_boxes")
            if lab is not None:
                speed = label_speed(labels, f, stride, scale)
                in_player = _inside_player(lab[0], lab[1], boxes)
                if pred is None:
                    outcome = "miss"
                else:
                    e = math.hypot(pred[0] - lab[0], pred[1] - lab[1]) * scale
                    outcome = "hit" if e <= HIT_PX else ("loc" if e <= WRONG_PX else "wrong")
            else:
                if pred is None:
                    outcome, speed, in_player = "quiet", None, False
                else:
                    outcome = "bridged" if bridged else "false"
                    speed = output_speed(frames, f, stride, scale)
                    in_player = _inside_player(pred[0], pred[1], boxes)
            held = speed is not None and speed < STATIC_SPEED and in_player
            records.append(dict(clip=clip, group=group, frame=f, visible=lab is not None, outcome=outcome,
                                speed=speed, in_player=in_player, held=held, band=band_of(speed),
                                src=row.get("source")))
    return records


def tally(recs):
    vis = [r for r in recs if r["visible"]]
    inv = [r for r in recs if not r["visible"]]
    n = len(vis)
    hits = sum(r["outcome"] == "hit" for r in vis)
    wrong = sum(r["outcome"] == "wrong" for r in vis)
    miss = sum(r["outcome"] == "miss" for r in vis)
    br = sum(r["outcome"] == "bridged" for r in inv)
    fa = sum(r["outcome"] == "false" for r in inv)
    return dict(vis=n, hits=hits, recall=hits / n if n else float("nan"), wrong=wrong, miss=miss,
                invis=len(inv), fa=br + fa, bridged=br, false=fa)


def rows_for(run, recs):
    groups = ("all", "grid", "own", "prof")
    out = []

    def add(bucket, sub):
        for g in groups:
            part = sub if g == "all" else [r for r in sub if r["group"] == g]
            out.append(dict(run=run, bucket=bucket, group=g, **tally(part)))

    add("all", recs)
    add("in_flight", [r for r in recs if not r["held"] and (r["visible"] or r["outcome"] != "quiet")])
    add("held_static", [r for r in recs if r["held"]])
    for lo, hi in BANDS:
        name = f"speed {lo}-{hi}" if hi != math.inf else f"speed {lo}+"
        add(name, [r for r in recs if r["band"] == band_of(lo)])
    add("speed n/a", [r for r in recs if r["band"] == "n/a" and (r["visible"] or r["outcome"] != "quiet")])
    return out


def fmt_rows(rows):
    lines = [f"{'run':10s} {'bucket':12s} {'grp':5s} {'vis':>5s} {'recall':>7s} {'wrong':>5s} {'miss':>5s} "
             f"{'invis':>5s} {'FA':>4s} {'brdg':>4s} {'false':>5s}"]
    for r in rows:
        lines.append(f"{r['run']:10s} {r['bucket']:12s} {r['group']:5s} {r['vis']:5d} {r['recall']:7.3f} "
                     f"{r['wrong']:5d} {r['miss']:5d} {r['invis']:5d} {r['fa']:4d} {r['bridged']:4d} {r['false']:5d}")
    return "\n".join(lines)


def noise_band(path, first_epoch=70):
    """sd of the raw held-out recall over the scored epochs from `first_epoch` (the training
    run's own epoch-to-epoch wobble), and the binomial sd of the pipeline recall."""
    recalls = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["recall"] and int(row["epoch"]) >= first_epoch:
                recalls.append(float(row["recall"]))
    sd = statistics.stdev(recalls)
    return dict(epochs=len(recalls), first_epoch=first_epoch, mean=statistics.fmean(recalls), sd=sd,
                diff_sd=sd * math.sqrt(2), meaningful=1.96 * sd * math.sqrt(2),
                binomial_sd=math.sqrt(0.953 * 0.047 / 8515))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", type=Path, default=GATE)
    ap.add_argument("--runs", default="e60_0.42,e60_0.34,opt1_0.38,opt1_0.30,heatmap_0.40")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args(argv)
    manifest = read_manifest()
    all_rows, frames_out = [], {}
    for run in args.runs.split(","):
        run_dir = args.gate / run
        if not run_dir.is_dir():
            print(f"[skip] {run}: not in {args.gate}")
            continue
        recs = score_run(run_dir, manifest)
        frames_out[run] = recs
        all_rows += rows_for(run, recs)
    print(fmt_rows(all_rows))
    with open(args.out / "scorecard.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    # Which visible frames does opt1_0.38 hit that e60_0.42 misses, and the reverse, in flight?
    if {"e60_0.42", "opt1_0.38"} <= set(frames_out):
        a = {(r["clip"], r["frame"]): r for r in frames_out["e60_0.42"] if r["visible"]}
        b = {(r["clip"], r["frame"]): r for r in frames_out["opt1_0.38"] if r["visible"]}
        gained = [k for k in a if a[k]["outcome"] != "hit" and b[k]["outcome"] == "hit"]
        lost = [k for k in a if a[k]["outcome"] == "hit" and b[k]["outcome"] != "hit"]
        print(f"\nopt1_0.38 vs e60_0.42 on visible labels: gained {len(gained)} lost {len(lost)} "
              f"(in-flight gained {sum(not a[k]['held'] for k in gained)} lost {sum(not a[k]['held'] for k in lost)})")
    if RESULTS_CSV.is_file():
        nb = noise_band(RESULTS_CSV)
        print(f"\nnoise band from {RESULTS_CSV}: epochs>={nb['first_epoch']} n={nb['epochs']} raw recall mean="
              f"{nb['mean']:.4f} sd={nb['sd']:.4f}; sd of a difference of two runs={nb['diff_sd']:.4f}; "
              f"a difference is meaningful (95%) above {nb['meaningful']:.3f}; "
              f"binomial sd of pipeline recall at n=8515: {nb['binomial_sd']:.4f}")
        json.dump(nb, open(args.out / "noise_band.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
