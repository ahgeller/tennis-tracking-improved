"""Test C, steps 2 to 4: score the top-K peak dump (dump_peaks.py) against the visible labels.

Step 2  the oracle ceiling (any of the top K peaks at or above the floor within HIT_PX of the
        label), the rank-1 threshold sweep (must reproduce raw e60: 0.952 / 55 FA at 0.40 and
        0.930 / 27 FA at 0.50, audit/records/opt_stream_prepass.md), and where the true ball sits
        (rank, confidence) on the frames the pipeline (e60 at 0.42) and the raw rank-1 rule missed,
        by source and by label speed.
Step 3  junk: peaks per no-ball frame above 0.05 and 0.20, and how many sit inside the player
        boxes the e60_0.42 pipeline run recorded.
Step 4  (only if the ceiling is at or above 0.97) a causal sliding dynamic programme over the
        last W frames: constant velocity plus gravity, summed confidence minus residual costs,
        scored on the raw metric against rank-1 thresholding at matched false-alarm counts.

Distances are 1080p pixels (detector_metrics.scale_for_width). Writes results.json and
per_frame.csv beside the dump and prints the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
import finetune  # noqa: E402,F401

import numpy as np  # noqa: E402

from finetune.data.data_policy import clip_source  # noqa: E402
from finetune.data.detector_metrics import HIT_PX, WRONG_PX, DetectorMetrics, scale_for_width  # noqa: E402
from finetune.data.labels import read_labels  # noqa: E402
from finetune.data.workspace import MANIFEST_NAME, read_manifest  # noqa: E402
from finetune.scoring.archive import list_clips  # noqa: E402

GATE_DIR = Path(r"C:\Users\Andrew\AppData\Local\Temp\claude\C--Users-Andrew-Desktop-clean-version"
                r"\a6549017-5a6a-4af2-bd14-c1b09a105d2c\scratchpad\opt1_gate\e60_0.42")
SPEED_BINS = [(0, 6), (6, 30), (30, 70), (70, float("inf"))]
CONF_BINS = [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.42), (0.42, 1.01)]
CEILING_FLOOR = 0.05
OPERATING = 0.42
# Sanity: raw rank-1 numbers recorded for stream_s4 e60 on mixed_v1 (opt_stream_prepass.md, section 2).
KNOWN_RAW = {0.40: (0.952, 55), 0.50: (0.930, 27)}

# Step 4: the path scorer's constants (confidence units; distances 1080p px).
DP_WINDOW = 8
DP_SIGMA_PX = 10.0        # residual of the constant-velocity-plus-gravity prediction that costs DP_RESIDUAL_W
DP_RESIDUAL_W = 0.30
DP_RESIDUAL_CAP = 3.0     # residuals beyond cap * sigma cost the same (an outlier, not infinitely bad)
DP_MISS = 0.35            # choosing no candidate on a frame
DP_REACQUIRE = 0.50       # a candidate right after a frame with none
DP_FIRST_LINK = 0.30      # a candidate after one candidate with no velocity yet, plus a speed term
DP_SPEED_PX = 80.0


class Frame:
    __slots__ = ("clip", "frame", "label", "conf", "x", "y", "speed", "pipeline")

    def __init__(self, clip, frame, label, conf, x, y, speed, pipeline):
        self.clip, self.frame, self.label = clip, frame, label
        self.conf, self.x, self.y = conf, x, y          # (K,) 1080p, conf 0 and nan where no peak
        self.speed = speed                              # 1080p px per source frame, or None
        self.pipeline = pipeline                        # the e60_0.42 pipeline's point (1080p) or None


def label_speeds(labels: Dict[int, Optional[Tuple[float, float]]], scale: float) -> Dict[int, Optional[float]]:
    """Per visible label, the distance per source frame to its nearest visible labelled neighbour
    within 3 frames (1080p px); None when there is none."""
    frames = sorted(labels)
    speeds: Dict[int, Optional[float]] = {}
    for index, frame in enumerate(frames):
        point = labels[frame]
        if point is None:
            continue
        best = None
        for other in (frames[index - 1] if index > 0 else None, frames[index + 1] if index + 1 < len(frames) else None):
            if other is None or labels[other] is None or abs(other - frame) > 3:
                continue
            value = math.hypot(point[0] - labels[other][0], point[1] - labels[other][1]) * scale / abs(other - frame)
            best = value if best is None else min(best, value)
        speeds[frame] = best
    return speeds


def speed_bin(speed: Optional[float]) -> str:
    if speed is None:
        return "unknown"
    for low, high in SPEED_BINS:
        if low <= speed < high:
            return f"{low}-{high if high != float('inf') else '+'}".replace("-+", "+")
    return "unknown"


def group_of(source: str) -> str:
    return {"custom": "own", "grid": "grid", "prof": "prof"}.get(source, source)


def load_clip(npz_path: Path, csv_path: Path, source: str) -> Tuple[List[Frame], dict]:
    data = np.load(npz_path)
    width, height, fps = int(data["width"]), int(data["height"]), float(data["fps"])
    scale = scale_for_width(width)
    labels = read_labels(csv_path, width, height)
    speeds = label_speeds(labels, scale)
    answered = {int(frame): index for index, frame in enumerate(data["frame"])}
    conf, xs, ys = data["conf"], data["x"] * scale, data["y"] * scale
    # Identity check: rank 1 of the dump is the tracker's own answer on every frame whose argmax
    # cleared the dump's floor (below it the dump holds nothing and the tracker's point is junk).
    tracker_conf = data["tracker_conf"]
    rank1 = np.zeros_like(tracker_conf)
    rank1[data["frame"]] = conf[:, 0]
    above = tracker_conf >= CEILING_FLOOR
    if not np.array_equal(rank1[above], tracker_conf[above]) or (rank1[~above] != 0).any():
        raise RuntimeError(f"{npz_path.stem}: rank-1 of the dump differs from the tracker's answer")
    json_path = GATE_DIR / f"{npz_path.stem}_gridtracknet.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = {row["frame"]: row for row in payload["frames"]}
    frames = []
    for frame in sorted(labels):
        label = labels[frame]
        index = answered.get(frame)
        if index is None:
            c, x, y = np.zeros(conf.shape[1], np.float32), np.full(conf.shape[1], np.nan), np.full(conf.shape[1], np.nan)
        else:
            c, x, y = conf[index], xs[index], ys[index]
        row = rows.get(frame)
        pipeline = (row["x"] * scale, row["y"] * scale) if row is not None and row.get("present") else None
        scaled = None if label is None else (label[0] * scale, label[1] * scale)
        frames.append(Frame(npz_path.stem, frame, scaled, c, x, y, speeds.get(frame), pipeline))
    clip_info = {"clip": npz_path.stem, "source": source, "group": group_of(source), "width": width, "height": height,
                 "fps": fps, "scale": scale, "rows": rows, "total": int(data["total"]),
                 "all_conf": conf, "all_x": xs, "all_y": ys, "all_frames": data["frame"]}
    return frames, clip_info


def true_rank(frame: Frame, floor: float = CEILING_FLOOR) -> Tuple[int, float]:
    """(rank 1..K of the first peak at or above `floor` within HIT_PX of the label, 0 if none;
    that peak's confidence)."""
    lx, ly = frame.label
    for rank in range(len(frame.conf)):
        if frame.conf[rank] < floor or frame.conf[rank] <= 0:
            continue
        if math.hypot(frame.x[rank] - lx, frame.y[rank] - ly) <= HIT_PX:
            return rank + 1, float(frame.conf[rank])
    return 0, 0.0


def rank1_pairs(frames: List[Frame], threshold: float):
    pairs = []
    for frame in frames:
        pred = None
        if frame.conf[0] >= threshold and frame.conf[0] > 0:
            pred = (float(frame.x[0]), float(frame.y[0]))
        pairs.append((frame.label, pred))
    return pairs


def metrics_line(name: str, stats: DetectorMetrics) -> str:
    return (f"{name:28s} recall={stats.recall:.3f} wrong={stats.wrong:.3f} FA={stats.false_alarm_count:3d}/{stats.invisible} "
            f"(bridged={stats.bridged_count} false={stats.false_alarm_count - stats.bridged_count})")


def conf_bin(conf: float) -> str:
    for low, high in CONF_BINS:
        if low <= conf < high:
            return f"{low:.2f}-{min(high, 1.0):.2f}"
    return "below 0.05"


# ---------------------------------------------------------------- step 4: the path scorer

def transition_costs(conf: np.ndarray, x: np.ndarray, y: np.ndarray, gravity: float) -> np.ndarray:
    """``(T, S, S, S)`` cost of choosing candidate c at frame i+1 after (a at i-1, b at i); the
    last candidate index S-1 is "none". `conf`, `x`, `y` are ``(T, K)`` for every source frame."""
    T, K = conf.shape
    S = K + 1
    real = conf > 0                                                   # (T, K)
    px = np.where(real, x, 0.0)
    py = np.where(real, y, 0.0)
    costs = np.zeros((T, S, S, S), np.float32)
    for i in range(0, T - 1):
        ra = real[i - 1] if i > 0 else np.zeros(K, bool)
        rb, rc = real[i], real[i + 1]
        # a none, b real, c real: a first link, speed only.
        speed = np.hypot(px[i + 1][None, :] - px[i][:, None], py[i + 1][None, :] - py[i][:, None])   # (b, c)
        first = DP_FIRST_LINK + 0.3 * np.minimum(speed / DP_SPEED_PX, 1.0)
        block = costs[i]
        if i > 0:
            # a real, b real, c real: constant velocity plus gravity.
            pred_x = 2 * px[i][None, :] - px[i - 1][:, None]                       # (a, b)
            pred_y = 2 * py[i][None, :] - py[i - 1][:, None] + gravity
            res = np.hypot(px[i + 1][None, None, :] - pred_x[:, :, None], py[i + 1][None, None, :] - pred_y[:, :, None])
            block[:K, :K, :K] = DP_RESIDUAL_W * np.minimum(res / DP_SIGMA_PX, DP_RESIDUAL_CAP)
        block[K, :K, :K] = first
        block[:, K, :K] = DP_REACQUIRE                                          # b none -> c real
        block[:, :, K] = DP_MISS                                                # c none
        # Non-existent candidates (conf 0 rows) are unreachable: infinite cost in and out.
        block[~np.append(ra, True), :, :] = np.inf
        block[:, ~np.append(rb, True), :] = np.inf
        block[:, :, ~np.append(rc, True)] = np.inf
    return costs


def path_scores(conf: np.ndarray, x: np.ndarray, y: np.ndarray, gravity: float, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per source frame t, the best causal path over frames t-window+1..t: (the candidate index it
    ends on, K for none; its summed score). A Viterbi over pair states (a at i-1, b at i)."""
    T, K = conf.shape
    S = K + 1
    gain = np.concatenate([np.where(conf > 0, conf, -np.inf), np.zeros((T, 1), np.float32)], axis=1)   # (T, S)
    costs = transition_costs(conf, x, y, gravity)
    choice = np.full(T, K, np.int64)
    score = np.full(T, -np.inf, np.float32)
    for t in range(T):
        start = max(0, t - window + 1)
        V = np.full((S, S), -np.inf, np.float32)      # V[a, b]: a at i-1 (none at the start), b at i
        V[K, :] = gain[start]
        for i in range(start + 1, t + 1):
            # V_new[b, c] = max_a V[a, b] + gain[i, c] - costs[i-1][a, b, c]   (costs indexed by the middle frame)
            step = V[:, :, None] - costs[i - 1]                    # (a, b, c)
            V = step.max(axis=0) + gain[i][None, :]
        best = np.unravel_index(np.argmax(V), V.shape)
        choice[t] = best[1]
        score[t] = V[best]
    return choice, score


def estimate_gravity(clips: List[dict], frames_by_clip: Dict[str, List[Frame]]) -> Dict[float, float]:
    """Median second difference of the labelled y per fps (1080p px per source frame squared),
    over triples of consecutive labels with equal spacing."""
    samples: Dict[float, List[float]] = defaultdict(list)
    for info in clips:
        frames = [frame for frame in frames_by_clip[info["clip"]] if frame.label is not None]
        for a, b, c in zip(frames, frames[1:], frames[2:]):
            gap = b.frame - a.frame
            if gap != c.frame - b.frame or gap > 2:
                continue
            samples[round(info["fps"])].append((c.label[1] - 2 * b.label[1] + a.label[1]) / gap ** 2)
    return {fps: float(np.median(values)) for fps, values in samples.items()}


def run_dp(clips: List[dict], frames_by_clip: Dict[str, List[Frame]], window: int, out_rows: List[dict]) -> dict:
    gravity_by_fps = estimate_gravity(clips, frames_by_clip)
    print(f"\n=== step 4: path scorer over the last {window} frames ===")
    print(f"gravity (median labelled d2y, 1080p px/frame^2) by fps: {gravity_by_fps}")
    # Two decision statistics on the best path ending at frame t: the summed path score itself, and
    # the current candidate's confidence lifted by the mean gain of the rest of the path
    # (conf_t + lam * (S - conf_t) / (window - 1)), lam in LAMBDAS.
    LAMBDAS = (0.25, 0.5, 1.0)
    statistics = ["sum"] + [f"lift{lam}" for lam in LAMBDAS]
    decided: Dict[str, List[Tuple[str, Optional[tuple], Optional[tuple], float]]] = {name: [] for name in statistics}
    for info in clips:
        conf, xs, ys = info["all_conf"], info["all_x"], info["all_y"]
        full_conf = np.zeros((info["total"], conf.shape[1]), np.float32)
        full_x = np.full_like(full_conf, np.nan)
        full_y = np.full_like(full_conf, np.nan)
        full_conf[info["all_frames"]] = conf
        full_x[info["all_frames"]] = xs
        full_y[info["all_frames"]] = ys
        gravity = gravity_by_fps.get(round(info["fps"]), 0.0)
        choice, score = path_scores(full_conf, full_x, full_y, gravity, window)
        for frame in frames_by_clip[info["clip"]]:
            c = choice[frame.frame]
            real = c < conf.shape[1]
            pred = (float(full_x[frame.frame, c]), float(full_y[frame.frame, c])) if real else None
            frame_score = float(score[frame.frame]) if real else -np.inf
            conf_t = float(full_conf[frame.frame, c]) if real else 0.0
            out_rows.append({"clip": info["clip"], "frame": frame.frame, "dp_choice": int(c) + 1 if real else 0,
                             "dp_score": round(frame_score, 4) if real else ""})
            decided["sum"].append((info["clip"], frame.label, pred, frame_score))
            for lam in LAMBDAS:
                lifted = conf_t + lam * (frame_score - conf_t) / (window - 1) if real else -np.inf
                decided[f"lift{lam}"].append((info["clip"], frame.label, pred, lifted))
    curves = {}
    for name in statistics:
        values = sorted({round(v, 2) for _, _, _, v in decided[name] if np.isfinite(v)})
        thresholds = values[:: max(1, len(values) // 150)]
        curve = []
        for threshold in thresholds:
            by_clip = defaultdict(list)
            for clip, label, pred, value in decided[name]:
                by_clip[clip].append((label, pred if value >= threshold else None))
            stats = DetectorMetrics.of_clips(list(by_clip.values()), 1.0)
            curve.append({"threshold": threshold, "recall": stats.recall, "wrong": stats.wrong,
                          "fa": stats.false_alarm_count, "bridged": stats.bridged_count})
        curves[name] = curve
        print(f"  statistic {name}:")
        for row in curve[:: max(1, len(curve) // 12)]:
            print(f"    >= {row['threshold']:.2f}: recall={row['recall']:.3f} wrong={row['wrong']:.3f} "
                  f"FA={row['fa']} (bridged {row['bridged']})")
    return {"gravity_by_fps": gravity_by_fps, "curves": curves}


# ---------------------------------------------------------------- the report

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks", default=str(HERE.parent / "peaks"))
    parser.add_argument("--holdout", default="mixed_v1")
    parser.add_argument("--window", type=int, default=DP_WINDOW)
    parser.add_argument("--force-dp", action="store_true", help="Run step 4 whatever the ceiling")
    args = parser.parse_args(argv)

    archive = REPO / "finetune"
    manifest = read_manifest(archive / MANIFEST_NAME)
    clips_info: List[dict] = []
    frames_by_clip: Dict[str, List[Frame]] = {}
    for video, csv_path in list_clips(None, archive, False, args.holdout):
        npz = Path(args.peaks) / f"{video.stem}.npz"
        if not npz.is_file():
            raise SystemExit(f"missing {npz}; run dump_peaks.py first")
        frames, info = load_clip(npz, csv_path, clip_source(video.stem, manifest))
        clips_info.append(info)
        frames_by_clip[info["clip"]] = frames
    every = [frame for info in clips_info for frame in frames_by_clip[info["clip"]]]
    visible = [frame for frame in every if frame.label is not None]
    invisible = [frame for frame in every if frame.label is None]
    group_of_clip = {info["clip"]: info["group"] for info in clips_info}
    K = len(visible[0].conf)
    results: dict = {"clips": len(clips_info), "visible": len(visible), "invisible": len(invisible), "K": K}
    print(f"[score] {len(clips_info)} clips, {len(visible)} visible labels, {len(invisible)} absent-ball labels, K={K}")
    print("[score] identity check passed: rank 1 of the dump equals the tracker's answer on every frame of every clip")

    # ---- sanity: the rank-1 threshold sweep must reproduce the recorded raw numbers
    print("\n=== rank-1 thresholding (the raw metric, no selector) ===")
    sweep = {}
    for threshold in (0.30, 0.34, 0.38, 0.40, OPERATING, 0.46, 0.50):
        stats = DetectorMetrics.of_clips([rank1_pairs(frames_by_clip[info["clip"]], threshold) for info in clips_info], 1.0)
        sweep[threshold] = stats
        print(metrics_line(f"rank1 conf>={threshold:.2f}", stats))
    for threshold, (recall, fa) in KNOWN_RAW.items():
        got = sweep[threshold]
        ok = abs(got.recall - recall) < 0.0015 and got.false_alarm_count == fa
        print(f"  sanity at {threshold:.2f}: recorded recall {recall:.3f} FA {fa}; got {got.recall:.3f} FA {got.false_alarm_count}"
              f" -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("the rank-1 sweep does not reproduce the recorded raw e60 numbers")
    results["rank1_sweep"] = {str(t): {"recall": s.recall, "wrong": s.wrong, "fa": s.false_alarm_count, "bridged": s.bridged_count}
                              for t, s in sweep.items()}

    # ---- the oracle ceiling
    print("\n=== oracle ceiling: label within 10 px of any of the top-K peaks ===")
    ranks = {frame: true_rank(frame) for frame in visible}
    ceilings = {}
    for k in range(1, K + 1):
        for floor in (0.05, 0.10, 0.20, OPERATING):
            hits = sum(1 for frame in visible if 0 < true_rank(frame, floor)[0] <= k)
            ceilings[f"top{k}_floor{floor:.2f}"] = hits / len(visible)
    for k in range(1, K + 1):
        print(f"  top-{k}: " + "  ".join(f"floor {floor:.2f}: {ceilings[f'top{k}_floor{floor:.2f}']:.4f}"
                                        for floor in (0.05, 0.10, 0.20, OPERATING)))
    ceiling = ceilings[f"top{K}_floor{CEILING_FLOOR:.2f}"]
    results["ceilings"] = ceilings
    results["ceiling"] = ceiling
    print(f"  ceiling (top-{K}, floor {CEILING_FLOOR}): {ceiling:.4f} = {round(ceiling * len(visible))}/{len(visible)}")

    # ---- where the true ball sits on the missed frames
    def miss_table(name: str, missed: List[Frame]) -> dict:
        print(f"\n=== {name}: {len(missed)} frames ===")
        rank_counts = Counter(ranks[frame][0] for frame in missed)
        conf_counts = Counter(conf_bin(ranks[frame][1]) for frame in missed if ranks[frame][0])
        print("  rank of the true ball: " + ", ".join(f"rank {r}: {rank_counts.get(r, 0)}" for r in range(1, K + 1))
              + f", not in top-{K} above 0.05: {rank_counts.get(0, 0)}")
        print("  its confidence when present: " + ", ".join(f"{b}: {conf_counts.get(b, 0)}"
                                                           for b in [f"{lo:.2f}-{min(hi, 1.0):.2f}" for lo, hi in CONF_BINS]))
        table = {}
        for key_name, key in (("group", lambda f: group_of_clip[f.clip]), ("speed", lambda f: speed_bin(f.speed))):
            print(f"  by {key_name}:")
            for value in sorted({key(f) for f in visible}, key=str):
                total = sum(1 for f in visible if key(f) == value)
                sub = [f for f in missed if key(f) == value]
                in_topk = sum(1 for f in sub if ranks[f][0])
                r1 = sum(1 for f in sub if ranks[f][0] == 1)
                print(f"    {value:10s} labels={total:5d} missed={len(sub):4d} ({len(sub) / max(total, 1):.3f})  "
                      f"recoverable in top-{K}: {in_topk} (rank 1 below threshold: {r1}, rank 2+: {in_topk - r1})  "
                      f"not present: {len(sub) - in_topk}")
                table[f"{key_name}:{value}"] = {"labels": total, "missed": len(sub), "in_topk": in_topk, "rank1": r1}
        return {"count": len(missed), "rank": {str(k): v for k, v in rank_counts.items()},
                "conf": dict(conf_counts), "table": table}

    pipeline_missed = [frame for frame in visible
                       if frame.pipeline is None or math.hypot(frame.pipeline[0] - frame.label[0], frame.pipeline[1] - frame.label[1]) > HIT_PX]
    raw_missed = [frame for frame in visible
                  if not (frame.conf[0] >= OPERATING and math.hypot(frame.x[0] - frame.label[0], frame.y[0] - frame.label[1]) <= HIT_PX)]
    results["pipeline_missed"] = miss_table("frames the e60_0.42 pipeline missed (no output or beyond 10 px)", pipeline_missed)
    results["raw_missed"] = miss_table(f"frames rank-1 at {OPERATING} missed (raw metric)", raw_missed)

    # ---- recall by group and speed: rank-1 at 0.42 vs the ceiling
    print("\n=== recall by group and speed: rank-1 at 0.42 vs the top-5 ceiling ===")
    breakdown = {}
    for key_name, key in (("group", lambda f: group_of_clip[f.clip]), ("speed", lambda f: speed_bin(f.speed))):
        for value in sorted({key(f) for f in visible}, key=str):
            sub = [f for f in visible if key(f) == value]
            r1 = sum(1 for f in sub if f.conf[0] >= OPERATING
                     and math.hypot(f.x[0] - f.label[0], f.y[0] - f.label[1]) <= HIT_PX)
            top = sum(1 for f in sub if ranks[f][0])
            print(f"  {key_name}={value:10s} n={len(sub):5d}  rank1@0.42={r1 / len(sub):.3f}  ceiling={top / len(sub):.3f}")
            breakdown[f"{key_name}:{value}"] = {"n": len(sub), "rank1": r1 / len(sub), "ceiling": top / len(sub)}
    results["breakdown"] = breakdown

    # ---- step 3: junk on the no-ball frames
    print("\n=== step 3: peaks on the 518 no-ball frames ===")
    junk = {}
    for floor in (0.05, 0.20, OPERATING):
        counts, in_player, with_boxes = [], 0, 0
        for frame in invisible:
            info = next(i for i in clips_info if i["clip"] == frame.clip)
            row = info["rows"].get(frame.frame)
            boxes = (row or {}).get("player_boxes") or {}
            boxes = [b for b in (boxes.values() if isinstance(boxes, dict) else boxes) if b and len(b) >= 4]
            scale = info["scale"]
            n = 0
            for rank in range(K):
                if frame.conf[rank] >= floor and frame.conf[rank] > 0:
                    n += 1
                    x, y = frame.x[rank] / scale, frame.y[rank] / scale      # native px, as the boxes
                    if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes):
                        in_player += 1
            counts.append(n)
            with_boxes += bool(boxes)
        total = sum(counts)
        junk[f"{floor:.2f}"] = {"mean_per_frame": total / len(invisible), "total": total, "in_player": in_player,
                                "frames_with_any": sum(1 for c in counts if c), "frames_with_boxes": with_boxes}
        print(f"  conf>={floor:.2f}: {total / len(invisible):.2f} peaks/frame ({total} in all; {sum(1 for c in counts if c)} "
              f"frames with at least one; capped at K={K}), {in_player} inside a player box "
              f"({in_player / max(total, 1):.1%}); {with_boxes} of {len(invisible)} frames have player boxes")
    # The same on the visible frames, the true peak left out, for contrast.
    for floor in (0.05, 0.20):
        n = sum(1 for frame in visible for rank in range(K)
                if frame.conf[rank] >= floor and frame.conf[rank] > 0 and rank + 1 != ranks[frame][0])
        print(f"  (visible frames, the true ball's peak excluded: {n / len(visible):.2f} other peaks/frame above {floor:.2f})")
        junk[f"visible_other_{floor:.2f}"] = n / len(visible)
    results["junk"] = junk

    # ---- per-frame CSV
    out_rows = []
    for frame in every:
        rank, conf = (ranks[frame] if frame.label is not None else (None, None))
        out_rows.append({"clip": frame.clip, "frame": frame.frame, "group": group_of_clip[frame.clip],
                         "visible": int(frame.label is not None),
                         "speed": "" if frame.speed is None else round(frame.speed, 2),
                         "rank1_conf": round(float(frame.conf[0]), 4),
                         "true_rank": "" if rank is None else rank, "true_conf": "" if conf is None else round(conf, 4),
                         "pipeline_hit": "" if frame.label is None else int(frame not in pipeline_missed),
                         "n_peaks_0.05": int((frame.conf >= 0.05).sum()), "n_peaks_0.20": int((frame.conf >= 0.20).sum())})

    # ---- step 4
    dp_rows: List[dict] = []
    if ceiling >= 0.97 or args.force_dp:
        results["dp"] = run_dp(clips_info, frames_by_clip, args.window, dp_rows)
        print("\n=== step 4: matched comparison (raw metric) ===")
        matched = []
        for name, curve in results["dp"]["curves"].items():
            print(f"  statistic {name}:")
            for threshold in (0.34, OPERATING, 0.50):
                base = sweep[threshold]
                fa = base.false_alarm_count
                candidates = [row for row in curve if row["fa"] <= fa]
                best = max(candidates, key=lambda row: row["recall"]) if candidates else None
                same_recall = [row for row in curve if row["recall"] >= base.recall]
                cheapest = min(same_recall, key=lambda row: row["fa"]) if same_recall else None
                print(f"    rank-1 at {threshold:.2f}: recall {base.recall:.3f} FA {fa}  |  path scorer at FA<={fa}: "
                      + (f"recall {best['recall']:.3f} FA {best['fa']} (>={best['threshold']:.2f})" if best else "none")
                      + "  |  at recall>=rank-1: " + (f"FA {cheapest['fa']} (>={cheapest['threshold']:.2f})" if cheapest else "none"))
                matched.append({"statistic": name, "threshold": threshold, "rank1": {"recall": base.recall, "fa": fa},
                                "dp_at_fa": best, "dp_at_recall": cheapest})
        results["dp"]["matched"] = matched
        by_key = {(row["clip"], row["frame"]): row for row in dp_rows}
        for row in out_rows:
            extra = by_key.get((row["clip"], row["frame"]), {})
            row["dp_choice"] = extra.get("dp_choice", "")
            row["dp_score"] = extra.get("dp_score", "")
    else:
        print(f"\n=== step 4 skipped: the ceiling {ceiling:.4f} is under 0.97 ===")
        results["dp"] = None

    out_dir = HERE.parent
    with open(out_dir / "per_frame.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'per_frame.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
