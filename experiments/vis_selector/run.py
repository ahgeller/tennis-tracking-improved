"""The visibility head as a fill gate, measured offline on the CPU (no GPU, no tracker run).

    python experiments/vis_selector/run.py [--weights system/models/candidates/s4_fg1_e10_vis.pt]
                                           [--run DIR] [--recompute] [--threads 12]

Step 1 (`vis_logits.csv`, cached): the headed stream_s4 record is run over every frame of the 24
held-out clips of mixed_v1 exactly as the prepass schedules it (``frame_rate.phase_stride``: a 60
FPS clip is two 30 FPS phases; a unit at every frame of a phase, ``Config.gridtracknet_window_step``
1), on the CPU in float32 from the deployed (reparameterized) network. Each unit answers one
visibility logit per frame of the unit; a frame's score is the mean of the logits of the up-to-five
units holding it, the same averaging the heat ring applies to the grids. The CSV has one row per
decoded frame: the mean logit, its sigmoid, the peak confidence of the frame's mean heat grid (what
this record's detector would say), the label (visible / absent / unlabelled) and the e60 0.42
tracking JSON's row for the frame (present, source, and the scorer's verdict: hit, loc, wrong,
bridged, false).

Step 2 (`RESULTS.md`, `gate_sweep.csv`): (a) the AUC of the logit against visible / absent on the
held-out labels; (b) per probability threshold, how many of the run's fill false alarms (interp,
motion) and detector false alarms sit on frames below it, how many fill hits would be lost if fills
below it were dropped, and the recall / FA the recorded run would reach with those fills turned into
absent frames (scored with the real pipeline scorer, like Tier 0 B); (c) the same for a softer rule
that keeps a fill below the threshold when it closes a single-frame hole ("discount").

The run is e60 (no head) and the logits are the fine-tuned record's, so every post-filtered
number is an estimate of what the gate does inside the pipeline on a headed model; the AUC and
the counts of rows above / below a threshold are measured.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import finetune  # noqa: F401,E402
from finetune.data.data_policy import clip_source  # noqa: E402
from finetune.data.detector_metrics import HIT_PX, WRONG_PX, DetectorMetrics, bridged_alarms, scale_for_width  # noqa: E402
from finetune.data.labels import read_labels  # noqa: E402
from finetune.data.workspace import label_csv, read_manifest  # noqa: E402
from finetune.scoring.pipeline_score import score_pipeline_clip  # noqa: E402
from tennis_tracker import streamnet  # noqa: E402
from tennis_tracker.ball_models import deployed  # noqa: E402
from tennis_tracker.frame_rate import require_phase_stride  # noqa: E402
from tennis_tracker.gridtracknet import resize_to_model  # noqa: E402

RUN = Path(r"C:\Users\Andrew\AppData\Local\Temp\claude\C--Users-Andrew-Desktop-clean-version"
           r"\a6549017-5a6a-4af2-bd14-c1b09a105d2c\scratchpad\opt1_gate\e60_0.42")
WEIGHTS = ROOT / "system" / "models" / "candidates" / "s4_fg1_e10_vis.pt"
VIDEOS = ROOT / "finetune" / "videos"
GROUP = {"custom": "own", "grid": "grid", "prof": "prof"}
FILLS = ("interp", "motion")
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
FIELDS = ("clip", "group", "frame", "fps", "label", "label_x", "label_y", "vis_logit", "vis_prob", "e10_peak_conf",
          "present", "source", "verdict", "run_len")


def frame_input(frame_bgr: np.ndarray) -> torch.Tensor:
    """One BGR frame as the model's (1, 3, 432, 768) RGB float32 input, through the tracker's resize."""
    rgb = cv2.cvtColor(resize_to_model(frame_bgr), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def clip_scores(model: streamnet.StreamNet, video: Path, fps: float) -> Tuple[Dict[int, float], Dict[int, float]]:
    """frame -> mean visibility logit and frame -> peak confidence of the mean heat grid, over
    every frame of every phase, as the window-step-1 prepass would average them."""
    stride = require_phase_stride(fps)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    pairs = streamnet.PAIRS_PER_UNIT
    previous: List[Optional[torch.Tensor]] = [None] * stride
    cached: List[List[streamnet.PairFeatures]] = [[] for _ in range(stride)]
    logit_sum: Dict[int, float] = defaultdict(float)
    logit_count: Dict[int, int] = defaultdict(int)
    heat_sum: Dict[int, torch.Tensor] = {}
    heat_count: Dict[int, int] = defaultdict(int)
    peak: Dict[int, float] = {}
    index = 0
    with torch.inference_mode():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            phase = index % stride
            current = frame_input(frame)
            if previous[phase] is not None:
                cached[phase].append(model.encode_frame(current, previous[phase]))
                if len(cached[phase]) > pairs:
                    del cached[phase][0]
                if len(cached[phase]) == pairs:
                    grids, logits = model.decode(cached[phase], return_vis=True)
                    unit = [index - stride * (4 - k) for k in range(5)]
                    for k, f in enumerate(unit):
                        logit_sum[f] += float(logits[0, k])
                        logit_count[f] += 1
                        heat = grids[0, 3 * k]
                        heat_sum[f] = heat.clone() if f not in heat_sum else heat_sum[f] + heat
                        heat_count[f] += 1
                    done = unit[0]                     # the oldest frame of the unit is final now
                    peak[done] = float((heat_sum.pop(done) / heat_count.pop(done)).max())
            previous[phase] = current
            index += 1
    capture.release()
    for f in list(heat_sum):
        peak[f] = float((heat_sum.pop(f) / heat_count.pop(f)).max())
    return {f: logit_sum[f] / logit_count[f] for f in logit_sum}, peak


def verdicts(payload: dict, labels: Dict[int, Optional[Tuple[float, float]]]) -> Dict[int, str]:
    """The scorer's verdict for every labelled frame the run output: hit, loc, wrong (visible
    labels) or bridged, false (absent labels)."""
    scale = scale_for_width(int(payload["video"]["width"]))
    frames = {r["frame"]: r for r in payload["frames"]}
    ordered = sorted(labels.items())
    pairs = [(lab, (frames[f]["x"], frames[f]["y"]) if f in frames and frames[f].get("present") else None)
             for f, lab in ordered]
    out: Dict[int, str] = {}
    for (f, lab), (_, pred), bridged in zip(ordered, pairs, bridged_alarms(pairs, scale)):
        if pred is None:
            continue
        if lab is None:
            out[f] = "bridged" if bridged else "false"
        else:
            error = math.hypot(pred[0] - lab[0], pred[1] - lab[1]) * scale
            out[f] = "hit" if error <= HIT_PX else ("loc" if error <= WRONG_PX else "wrong")
    return out


def run_lengths(payload: dict) -> Dict[int, int]:
    """frame -> the length of the run of consecutive non-detector output rows the frame sits in
    (0 for a detector row or an absent frame)."""
    rows = sorted(payload["frames"], key=lambda r: r["frame"])
    out: Dict[int, int] = {}
    run: List[int] = []
    for row in rows + [{"frame": -1, "present": False}]:
        if row.get("present") and row.get("source") in FILLS:
            run.append(row["frame"])
            continue
        for f in run:
            out[f] = len(run)
        run = []
    return out


def load_clips(run_dir: Path):
    manifest = read_manifest()
    clips = []
    for jp in sorted(run_dir.glob("*_gridtracknet.json")):
        clip = jp.name[:-len("_gridtracknet.json")]
        payload = json.loads(jp.read_text(encoding="utf-8"))
        W, H = int(payload["video"]["width"]), int(payload["video"]["height"])
        clips.append((clip, GROUP.get(clip_source(clip, manifest), "other"), payload, read_labels(label_csv(clip), W, H)))
    return clips


def compute(weights: Path, clips, out_csv: Path, threads: int) -> None:
    torch.set_num_threads(threads)
    model = deployed(streamnet.load_model(weights, torch.device("cpu"))).float().eval()
    assert model.has_vis_head, f"{weights} has no visibility head"
    started = time.perf_counter()
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for clip, group, payload, labels in clips:
            fps = float(payload["video"]["fps"])
            t0 = time.perf_counter()
            logits, peaks = clip_scores(model, VIDEOS / f"{clip}.mp4", fps)
            verdict = verdicts(payload, labels)
            runs = run_lengths(payload)
            frames = {r["frame"]: r for r in payload["frames"]}
            for f in sorted(logits):
                row = frames.get(f, {})
                lab = labels.get(f, "none")
                writer.writerow(dict(
                    clip=clip, group=group, frame=f, fps=round(fps, 3),
                    label="none" if lab == "none" else ("absent" if lab is None else "visible"),
                    label_x="" if not isinstance(lab, tuple) else round(lab[0], 2),
                    label_y="" if not isinstance(lab, tuple) else round(lab[1], 2),
                    vis_logit=round(logits[f], 4), vis_prob=round(1.0 / (1.0 + math.exp(-logits[f])), 5),
                    e10_peak_conf=round(peaks.get(f, float("nan")), 4),
                    present=int(bool(row.get("present"))), source=row.get("source", "") if row.get("present") else "",
                    verdict=verdict.get(f, ""), run_len=runs.get(f, 0)))
            fh.flush()
            print(f"[vis] {clip}: {len(logits)} frames in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[vis] logits for {len(clips)} clips in {time.perf_counter() - started:.0f}s -> {out_csv}")


def read_rows(csv_path: Path) -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["frame"] = int(r["frame"])
        r["vis_logit"] = float(r["vis_logit"])
        r["vis_prob"] = float(r["vis_prob"])
        r["present"] = int(r["present"])
        r["run_len"] = int(r["run_len"])
    return rows


def auc(scores_pos: List[float], scores_neg: List[float]) -> float:
    """The Mann-Whitney AUC: the share of (visible, absent) pairs ranked the right way."""
    pos = np.asarray(scores_pos)
    neg = np.asarray(scores_neg)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    values = np.concatenate([pos, neg])
    for v in np.unique(values[np.r_[True, np.diff(np.sort(values)) == 0]]):
        tie = values == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gated(payload: dict, prob: Dict[int, float], threshold: float, keep_single: bool, sources=FILLS) -> dict:
    """The payload with every fill row of `sources` on a frame whose visibility probability is
    below `threshold` turned into an absent frame; with `keep_single` a fill that is alone in its
    run (a closed single hole) is kept whatever the probability."""
    runs = run_lengths(payload)
    out = dict(payload)
    rows = []
    for row in payload["frames"]:
        f = row["frame"]
        if (row.get("present") and row.get("source") in sources and prob.get(f, 1.0) < threshold
                and not (keep_single and runs.get(f, 0) == 1)):
            row = dict(row, present=False)
        rows.append(row)
    out["frames"] = rows
    return out


def metrics(clips, transform) -> Dict[str, DetectorMetrics]:
    by_group: Dict[str, list] = {}
    pairs_all = []
    for clip, group, payload, labels in clips:
        scored = score_pipeline_clip(labels, transform(clip, payload), int(payload["video"]["width"]))
        by_group.setdefault(group, []).append(scored.pairs)
        pairs_all.append(scored.pairs)
    rows = {g: DetectorMetrics.of_clips(p, 1.0) for g, p in sorted(by_group.items())}
    rows["all"] = DetectorMetrics.of_clips(pairs_all, 1.0)
    return rows


def analyse(rows: List[dict], clips, out_dir: Path, weights: Path) -> None:
    labelled = [r for r in rows if r["label"] != "none"]
    vis = [r["vis_logit"] for r in labelled if r["label"] == "visible"]
    absent = [r["vis_logit"] for r in labelled if r["label"] == "absent"]
    overall_auc = auc(vis, absent)
    lines = [f"# Visibility head as a fill gate: offline evidence (CPU, {weights.name})", "",
             "Numbers from the per-frame CSV `vis_logits.csv` (measured) and the e60 0.42 run re-scored with fills removed "
             "(an estimate of the gate inside the pipeline: the run's detections are e60's, the logits are this record's).", "",
             "## (a) AUC of the visibility logit against the held-out labels", "",
             "| group | visible | absent | AUC |", "|---|---|---|---|"]
    for group in sorted({r["group"] for r in labelled}):
        g = [r for r in labelled if r["group"] == group]
        gv = [r["vis_logit"] for r in g if r["label"] == "visible"]
        ga = [r["vis_logit"] for r in g if r["label"] == "absent"]
        lines.append(f"| {group} | {len(gv)} | {len(ga)} | {auc(gv, ga):.3f} |" if gv and ga
                     else f"| {group} | {len(gv)} | {len(ga)} | n/a |")
    lines.append(f"| all | {len(vis)} | {len(absent)} | {overall_auc:.3f} |")
    # AUC restricted to frames the run output as a fill (the frames the gate actually judges)
    fill_rows = [r for r in labelled if r["present"] and r["source"] in FILLS]
    fv = [r["vis_logit"] for r in fill_rows if r["label"] == "visible"]
    fa = [r["vis_logit"] for r in fill_rows if r["label"] == "absent"]
    lines += ["", f"On the frames the run filled (the ones a gate judges): {len(fv)} visible, {len(fa)} absent, "
                  f"AUC {auc(fv, fa):.3f}." if fv and fa else ""]
    det_rows = [r for r in labelled if r["present"] and r["source"] == "det"]
    dv = [r["vis_logit"] for r in det_rows if r["label"] == "visible"]
    da = [r["vis_logit"] for r in det_rows if r["label"] == "absent"]
    if dv and da:
        lines.append(f"On the frames the run output a detection: {len(dv)} visible, {len(da)} absent, AUC {auc(dv, da):.3f}.")
    # The absent labels by the length of the run of absent labels they sit in: a short run between
    # visible labels is the occlusion a bridged fill crosses; a long one is the ball out of frame.
    by_clip: Dict[str, List[dict]] = defaultdict(list)
    for r in labelled:
        by_clip[r["clip"]].append(r)
    short_absent, long_absent = [], []
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda r: r["frame"])
        i = 0
        while i < len(clip_rows):
            if clip_rows[i]["label"] != "absent":
                i += 1
                continue
            j = i
            while j < len(clip_rows) and clip_rows[j]["label"] == "absent":
                j += 1
            flanked = 0 < i and j < len(clip_rows)
            (short_absent if flanked and j - i <= 3 else long_absent).extend(r["vis_logit"] for r in clip_rows[i:j])
            i = j
    lines += ["", f"Absent labels in a run of at most 3 flanked by visible labels (what a bridged fill crosses): "
                  f"{len(short_absent)}, AUC against the visible labels {auc(vis, short_absent):.3f}; "
                  f"in longer runs or at a clip's edge: {len(long_absent)}, AUC {auc(vis, long_absent):.3f}."]

    # Probability quantiles for reading the sweep
    def q(values, p):
        return float(np.quantile(values, p)) if values else float("nan")
    pv = [r["vis_prob"] for r in labelled if r["label"] == "visible"]
    pa = [r["vis_prob"] for r in labelled if r["label"] == "absent"]
    lines += ["", f"Probability (sigmoid of the logit) quantiles, visible labels: p5 {q(pv, .05):.3f}, p25 {q(pv, .25):.3f}, "
                  f"median {q(pv, .5):.3f}; absent labels: p50 {q(pa, .5):.3f}, p75 {q(pa, .75):.3f}, p95 {q(pa, .95):.3f}."]

    # Baseline counts of the recorded run (must match Tier 0 B)
    prob = {(r["clip"], r["frame"]): r["vis_prob"] for r in rows}
    base = metrics(clips, lambda clip, p: p)["all"]
    tally = Counter((r["source"], r["verdict"]) for r in labelled if r["present"])
    lines += ["", "## Baseline (the recorded e60 0.42 run, as Tier 0 B read it)", "",
              f"recall {base.recall:.4f} ({round(base.recall * base.visible)} hits of {base.visible}), "
              f"FA {base.false_alarm_count} of {base.invisible} ({base.bridged_count} bridged / "
              f"{base.false_alarm_count - base.bridged_count} false); "
              f"fill rows: interp {tally[('interp', 'hit')]} hits / {tally[('interp', 'bridged')] + tally[('interp', 'false')]} FA, "
              f"motion {tally[('motion', 'hit')]} hits / {tally[('motion', 'bridged')] + tally[('motion', 'false')]} FA, "
              f"det {tally[('det', 'hit')]} hits / {tally[('det', 'bridged')] + tally[('det', 'false')]} FA."]

    # (b) and (c): the sweep
    lines += ["", "## (b) Drop every fill below the probability threshold; (c) keep a fill that closes a single-frame hole", "",
              "Columns: rows of the run below the threshold (fill FA = interp + motion false alarms, det FA = detector false "
              "alarms the gate cannot touch, fill hits = visible labels a fill got right, lost if dropped), then the re-scored run.", "",
              "| threshold | fill FA below (interp/motion) | det FA below | fill hits below (interp/motion) | drop: recall / FA (bridged/false) | "
              "drop: grid / own / prof | keep-single: recall / FA | keep-single hits lost |",
              "|---|---|---|---|---|---|---|---|"]
    sweep = []
    for t in THRESHOLDS:
        below = [r for r in labelled if r["present"] and r["vis_prob"] < t]
        fill_fa = Counter(r["source"] for r in below if r["source"] in FILLS and r["verdict"] in ("bridged", "false"))
        det_fa = sum(1 for r in below if r["source"] == "det" and r["verdict"] in ("bridged", "false"))
        fill_hit = Counter(r["source"] for r in below if r["source"] in FILLS and r["verdict"] == "hit")
        drop = metrics(clips, lambda clip, p, t=t: gated(p, {f: v for (c, f), v in prob.items() if c == clip}, t, False))
        soft = metrics(clips, lambda clip, p, t=t: gated(p, {f: v for (c, f), v in prob.items() if c == clip}, t, True))
        d, s = drop["all"], soft["all"]
        groups = " / ".join(f"{drop[g].recall:.3f}" if g in drop else "-" for g in ("grid", "own", "prof"))
        lines.append(f"| {t:.2f} | {fill_fa['interp'] + fill_fa['motion']} ({fill_fa['interp']}/{fill_fa['motion']}) | {det_fa} | "
                     f"{fill_hit['interp'] + fill_hit['motion']} ({fill_hit['interp']}/{fill_hit['motion']}) | "
                     f"{d.recall:.4f} / {d.false_alarm_count} ({d.bridged_count}/{d.false_alarm_count - d.bridged_count}) | {groups} | "
                     f"{s.recall:.4f} / {s.false_alarm_count} ({s.bridged_count}/{s.false_alarm_count - s.bridged_count}) | "
                     f"{round(base.recall * base.visible) - round(s.recall * s.visible)} |")
        sweep.append(dict(threshold=t, fill_fa_below=fill_fa['interp'] + fill_fa['motion'], interp_fa_below=fill_fa['interp'],
                          motion_fa_below=fill_fa['motion'], det_fa_below=det_fa,
                          fill_hits_below=fill_hit['interp'] + fill_hit['motion'], interp_hits_below=fill_hit['interp'],
                          motion_hits_below=fill_hit['motion'],
                          drop_recall=round(d.recall, 4), drop_hits=round(d.recall * d.visible), drop_fa=d.false_alarm_count,
                          drop_bridged=d.bridged_count, keep_single_recall=round(s.recall, 4),
                          keep_single_hits=round(s.recall * s.visible), keep_single_fa=s.false_alarm_count,
                          **{f"drop_recall_{g}": round(drop[g].recall, 4) for g in drop if g != "all"},
                          **{f"drop_fa_{g}": drop[g].false_alarm_count for g in drop if g != "all"}))
    with open(out_dir / "gate_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)

    # Per-source gating: interp only, motion only
    lines += ["", "## Interp-only and motion-only gates (drop)", "", "| threshold | interp only: recall / FA | motion only: recall / FA |",
              "|---|---|---|"]
    for t in (0.10, 0.20, 0.30, 0.40, 0.50):
        parts = []
        for src in FILLS:
            m = metrics(clips, lambda clip, p, t=t, src=src: gated(p, {f: v for (c, f), v in prob.items() if c == clip}, t, False, (src,)))["all"]
            parts.append(f"{m.recall:.4f} / {m.false_alarm_count}")
        lines.append(f"| {t:.2f} | {parts[0]} | {parts[1]} |")

    # Where the fill FAs sit vs the fill hits: the distributions
    lines += ["", "## Where the run's fill rows sit on the probability axis", "",
              "| bucket | fill FA | fill hits | det FA | det hits |", "|---|---|---|---|---|"]
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [r for r in labelled if r["present"] and lo <= r["vis_prob"] < hi]
        lines.append(f"| [{lo:.2f}, {min(hi, 1.0):.2f}) | "
                     f"{sum(1 for r in b if r['source'] in FILLS and r['verdict'] in ('bridged', 'false'))} | "
                     f"{sum(1 for r in b if r['source'] in FILLS and r['verdict'] == 'hit')} | "
                     f"{sum(1 for r in b if r['source'] == 'det' and r['verdict'] in ('bridged', 'false'))} | "
                     f"{sum(1 for r in b if r['source'] == 'det' and r['verdict'] == 'hit')} |")
    # Detector misses the head calls visible (the recall the fills bring that the gate must keep)
    missed = [r for r in labelled if r["label"] == "visible" and not r["present"]]
    lines += ["", f"Visible labels the run missed outright: {len(missed)}; the head calls {sum(1 for r in missed if r['vis_prob'] >= 0.5)} "
                  f"of them visible at 0.5 (these are the frames a fill may still earn)."]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--recompute", action="store_true", help="recompute the logits even when the CSV exists")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--compute-only", action="store_true")
    ap.add_argument("--clips", nargs="*", default=None, help="clip stems to restrict to (a smoke test)")
    args = ap.parse_args(argv)
    clips = [c for c in load_clips(args.run) if not args.clips or c[0] in args.clips]
    out_csv = args.out / "vis_logits.csv"
    if args.recompute or not out_csv.is_file():
        compute(args.weights, clips, out_csv, args.threads)
    if args.compute_only:
        return 0
    analyse(read_rows(out_csv), clips, args.out, args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
