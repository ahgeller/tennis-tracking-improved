"""Tier 0 test E: per-session BatchNorm re-estimation for stream_s4 e60 (Schneider et al. 2020,
"Improving robustness against common corruptions by covariate shift adaptation", arXiv 2006.16971).

Question: does re-estimating the BatchNorm running statistics on a clip's own first seconds, with no
training, raise the raw detector's recall on that clip (grid / prof in particular) at equal false alarms?

How it runs (``--score``, GPU):
  * the PyTorch trainable form of ``system/models/candidates/stream_s4_e60.pt`` (``streamnet.load_model``,
    BatchNorm intact) through the tracker's own prepass, ``GridTrackNetBallDetector`` with
    ``Config.ball_prepass_runtime="torch"`` and ``ball_prepass_streaming=False`` (the streaming path would
    fold BatchNorm away through ``load_deployed``), at threshold 0 so thresholds sweep offline;
  * for each variant and each held-out clip (mixed_v1, 24 clips) separately: a float32 copy of the record,
    every BatchNorm's running statistics reset with ``momentum=None`` (cumulative average), train mode under
    ``no_grad``, the clip's first N frames fed as the same five-frame units the prepass cuts (a unit every
    frame of each phase, window step 1), then eval, FP16, and that clip is scored with it;
  * variants: ``baseline`` (the record as it is), ``bn60`` (N=60 source frames), ``bn300`` (N=300),
    ``bnall`` (the whole clip; an upper bound, not usable live), and optionally ``bnN_pK``: the adapted
    statistics blended with the record's as a prior of weight K (the paper's N / (N + n) blend).
  * every labelled frame's (label, prediction, confidence) row lands in ``rows_<variant>.csv``.

How it is analysed (``--analyze``, CPU, from the CSVs): the pooled threshold sweep 0.30 to 0.50, per
source group and per clip at 0.42, and the equal-false-alarm comparison: for each variant the threshold
on a 0.01 grid whose pooled false-alarm count does not exceed the baseline's at 0.42, and the recall there,
overall and per group. Writes ``summary.md`` and ``summary.json``. The verdict is written by hand in
RESULTS.md from these.

Usage, from the repository root (the tests_AE worktree), with the tennis-opt interpreter:
  python experiments/tier0/E_session_batchnorm/run.py --score --analyze
  python experiments/tier0/E_session_batchnorm/run.py --analyze            # CSVs already there
  python experiments/tier0/E_session_batchnorm/run.py --score --variants baseline bn60 --clips grid_match8
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import finetune  # noqa: E402,F401  (puts system/ on sys.path)
from finetune.data import workspace as ws  # noqa: E402
from finetune.data.data_policy import clip_source  # noqa: E402
from finetune.data.detect import detect_points  # noqa: E402
from finetune.data.detector_metrics import DetectorMetrics, scale_for_width  # noqa: E402
from finetune.data.labels import read_labels  # noqa: E402
from finetune.data.workspace import MANIFEST_NAME, read_manifest, video_meta  # noqa: E402
from finetune.scoring.archive import list_clips  # noqa: E402

WEIGHTS = REPOSITORY / "system" / "models" / "candidates" / "stream_s4_e60.pt"
HOLDOUT = "mixed_v1"
DEFAULT_VARIANTS = ["baseline", "bn60", "bn300", "bnall", "bn300_p1"]
OPERATING = 0.42
SWEEP = [round(0.30 + 0.02 * i, 2) for i in range(11)]            # 0.30 to 0.50
FINE = [round(0.10 + 0.01 * i, 2) for i in range(71)]             # 0.10 to 0.80, the equal-false-alarm grid
ADAPT_BATCH = 8                                                   # units per forward pass while adapting
GROUPS = ("grid", "custom", "prof")

VARIANT_RE = re.compile(r"^bn(\d+|all)(?:_p(\d+(?:\.\d+)?))?$")


def parse_variant(name: str) -> Tuple[Optional[int], float]:
    """(frames to adapt on, None for the whole clip; prior weight on the record's statistics, 0 for none)."""
    if name == "baseline":
        raise ValueError("baseline has no adaptation")
    match = VARIANT_RE.match(name)
    if not match:
        raise ValueError(f"variant {name!r}: expected baseline, bn<N>, bnall or bn<N>_p<K>")
    frames = None if match.group(1) == "all" else int(match.group(1))
    return frames, float(match.group(2) or 0.0)


# ------------------------------------------------------------------------------------- scoring (GPU)

def build_detector(device: str):
    import torch
    from tennis_tracker.config import FOREGROUND, Config
    from tennis_tracker.engines import ball_detectors as backends
    torch.backends.cudnn.benchmark = True
    cfg = Config(conf=0.0, device=device, ball_prepass_mode=FOREGROUND, ball_prepass_runtime="torch",
                 ball_prepass_streaming=False)
    detector = backends.GridTrackNetBallDetector(str(WEIGHTS), cfg)
    assert detector.engine is None and detector.stream_halves is None, "the PyTorch model must run, not TensorRT"
    return detector


def source_model(device):
    """The record's trainable network in float32 on `device`, eval mode: the master every variant copies."""
    import torch
    from tennis_tracker import streamnet
    from tennis_tracker.heatmapnet import read_record
    model = streamnet.StreamNet()
    model.load_state_dict(read_record(WEIGHTS, streamnet.ARCH)["state_dict"])
    return model.eval().to(device)


def clip_units(video: Path, meta, frames: Optional[int], device):
    """Generator of float32 ``(ADAPT_BATCH, 15, H, W)`` unit batches cut from the clip's first `frames`
    frames (all of them for None) exactly as the prepass cuts them: one stream of consecutive frames per
    phase (stride 2 at 60 FPS, 1 at 30), a unit at every frame from the fifth of its phase on."""
    import cv2
    import torch
    from tennis_tracker.engines.ball_gpu import ModelResize
    from tennis_tracker.frame_rate import require_phase_stride
    from tennis_tracker.gridtracknet import HEIGHT, WIDTH
    stride = require_phase_stride(meta.fps)
    resize = ModelResize(meta.width, meta.height, WIDTH, HEIGHT, device, dtype=torch.float32)
    capture = cv2.VideoCapture(str(video))
    phases: List[List[torch.Tensor]] = [[] for _ in range(stride)]
    pending: List[torch.Tensor] = []
    index = 0
    seen = 0
    try:
        while frames is None or index < frames:
            ok, frame = capture.read()
            if not ok:
                break
            tensor = resize(torch.from_numpy(frame).to(device))
            phase = phases[index % stride]
            phase.append(tensor)
            if len(phase) > 5:
                del phase[0]
            if len(phase) == 5:
                pending.append(torch.cat(phase, dim=0))
                if len(pending) == ADAPT_BATCH:
                    yield torch.stack(pending)
                    pending = []
            index += 1
            seen += 1
    finally:
        capture.release()
    if pending:
        yield torch.stack(pending)


def adapt(model, batches, prior: float) -> Tuple[int, int]:
    """Re-estimate every BatchNorm's running statistics of `model` (float32, in place) on `batches`
    (train mode, no gradient, momentum None: the cumulative average over the batches). With `prior` > 0
    the result is the blend ``(prior * record + 1 * adapted) / (prior + 1)`` per statistic. Returns
    (units seen, batches seen)."""
    import torch
    from torch import nn
    norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    saved = [(norm.running_mean.clone(), norm.running_var.clone()) for norm in norms]
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None
    model.train()
    units = count = 0
    with torch.no_grad():
        for batch in batches:
            model(batch)
            units += batch.shape[0]
            count += 1
    model.eval()
    if units == 0:
        raise RuntimeError("no units to adapt on")
    if prior > 0.0:
        with torch.no_grad():
            for norm, (mean, var) in zip(norms, saved):
                norm.running_mean.copy_((prior * mean + norm.running_mean) / (prior + 1.0))
                norm.running_var.copy_((prior * var + norm.running_var) / (prior + 1.0))
    for norm in norms:
        norm.momentum = 0.1
    return units, count


def score(variants: Sequence[str], clip_filter: Optional[List[str]], device: str) -> None:
    import torch
    archive = ws.current().root
    clips = list_clips(clip_filter, archive, holdout=HOLDOUT)
    manifest = read_manifest(archive / MANIFEST_NAME)
    print(f"[E] {len(clips)} clips of {HOLDOUT}; weights {WEIGHTS.name}; variants {list(variants)}")
    detector = build_detector(device)
    torch_device = detector.device
    master = source_model(torch_device)
    original_half = detector.model
    for variant in variants:
        out_path = HERE / f"rows_{variant}.csv"
        write_header = not out_path.exists() or not clip_filter
        mode = "a" if (clip_filter and out_path.exists()) else "w"
        with out_path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["clip", "source", "frame", "label_x", "label_y", "pred_x", "pred_y", "conf"])
            for video, csv_path in clips:
                clip = video.stem
                meta = video_meta(video)
                started = time.perf_counter()
                note = "record statistics"
                if variant == "baseline":
                    detector.model = original_half
                else:
                    frames, prior = parse_variant(variant)
                    model = copy.deepcopy(master)
                    units, batches = adapt(model, clip_units(video, meta, frames, torch_device), prior)
                    detector.model = model.half().eval()
                    note = f"adapted on {units} units ({batches} batches, first {frames or 'all'} frames, prior {prior:g})"
                points = detect_points(detector, video, meta)
                labels = read_labels(csv_path, meta.width, meta.height)
                source = clip_source(clip, manifest)
                for frame, label in sorted(labels.items()):
                    point = points.get(frame)
                    writer.writerow([clip, source, frame,
                                     "" if label is None else f"{label[0]:.3f}", "" if label is None else f"{label[1]:.3f}",
                                     "" if point is None else f"{point[0]:.3f}", "" if point is None else f"{point[1]:.3f}",
                                     "" if point is None else f"{point[2]:.5f}"])
                handle.flush()
                if variant != "baseline":
                    detector.model = original_half
                    del model
                torch.cuda.empty_cache()
                print(f"[E] {variant:10s} {clip:14s} {meta.total} frames, {note}, {time.perf_counter() - started:.1f}s",
                      flush=True)
    detector.model = original_half


# ------------------------------------------------------------------------------------ analysis (CPU)

Row = Tuple[str, str, int, Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]


def read_rows(variant: str) -> Dict[str, List[Row]]:
    per_clip: Dict[str, List[Row]] = {}
    with (HERE / f"rows_{variant}.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = (float(row["label_x"]), float(row["label_y"])) if row["label_x"] else None
            pred = (float(row["pred_x"]), float(row["pred_y"])) if row["pred_x"] else None
            conf = float(row["conf"]) if row["conf"] else 0.0
            per_clip.setdefault(row["clip"], []).append((row["clip"], row["source"], int(row["frame"]), label, pred, conf))
    return per_clip


def widths() -> Dict[str, int]:
    archive = ws.current().root
    return {video.stem: video_meta(video).width for video, _ in list_clips(None, archive, holdout=HOLDOUT)}


def pairs_at(rows: List[Row], threshold: float, scale: float) -> list:
    return [(None if label is None else (label[0] * scale, label[1] * scale),
             (pred[0] * scale, pred[1] * scale) if pred is not None and conf >= threshold else None)
            for _, _, _, label, pred, conf in rows]


def pooled(per_clip: Dict[str, List[Row]], width_of: Dict[str, int], threshold: float,
           clips: Optional[Sequence[str]] = None) -> DetectorMetrics:
    chosen = [clip for clip in per_clip if clips is None or clip in clips]
    return DetectorMetrics.of_clips([pairs_at(per_clip[clip], threshold, scale_for_width(width_of[clip])) for clip in chosen], 1.0)


def metric_dict(stats: DetectorMetrics) -> dict:
    return {"visible": stats.visible, "recall": round(stats.recall, 4), "wrong": round(stats.wrong, 4),
            "invisible": stats.invisible, "false_alarm": round(stats.false_alarm, 4),
            "fa_count": stats.false_alarm_count, "bridged": stats.bridged_count,
            "false": stats.false_alarm_count - stats.bridged_count,
            "median_px": None if math.isnan(stats.median_px) else round(stats.median_px, 2)}


def equal_fa_threshold(per_clip, width_of, budget: int, clips: Optional[Sequence[str]] = None) -> float:
    """The threshold on the fine grid with the highest recall whose false-alarm count is at most `budget`
    (the lowest threshold that fits, as recall falls with the threshold)."""
    fitting = [t for t in FINE if pooled(per_clip, width_of, t, clips).false_alarm_count <= budget]
    return min(fitting) if fitting else 1.0


def analyze(variants: Sequence[str]) -> None:
    width_of = widths()
    data = {variant: read_rows(variant) for variant in variants if (HERE / f"rows_{variant}.csv").exists()}
    variants = [variant for variant in variants if variant in data]
    if "baseline" not in data:
        raise SystemExit("rows_baseline.csv is missing: run --score first")
    sources = {clip: rows[0][1] for clip, rows in data["baseline"].items()}
    by_group = {group: [clip for clip, source in sources.items() if source == group] for group in GROUPS}
    out: dict = {"weights": WEIGHTS.name, "holdout": HOLDOUT, "variants": {}, "equal_false_alarm": {}}
    lines: List[str] = []
    lines.append(f"# Test E summary: {WEIGHTS.name} on {HOLDOUT}, PyTorch FP16 trainable form\n")
    lines.append("Pooled sweep (24 clips, raw detector; FA = false alarms of the invisible labels, bridged / false).\n")
    header = "| thr | " + " | ".join(f"{v} recall / FA" for v in variants) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(variants) + 1))
    for threshold in SWEEP:
        cells = []
        for variant in variants:
            stats = pooled(data[variant], width_of, threshold)
            cells.append(f"{stats.recall:.4f} / {stats.false_alarm_count} ({stats.bridged_count}+{stats.false_alarm_count - stats.bridged_count})")
            out["variants"].setdefault(variant, {}).setdefault("pooled", {})[str(threshold)] = metric_dict(stats)
        lines.append(f"| {threshold:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Per source group at {OPERATING:.2f} (recall / FA count / wrong).\n")
    lines.append("| group | clips | vis | " + " | ".join(variants) + " |")
    lines.append("|" + "---|" * (len(variants) + 3))
    for group in GROUPS:
        clips = by_group[group]
        cells = []
        for variant in variants:
            stats = pooled(data[variant], width_of, OPERATING, clips)
            cells.append(f"{stats.recall:.4f} / {stats.false_alarm_count} / {stats.wrong:.4f}")
            out["variants"][variant].setdefault("groups", {})[group] = metric_dict(stats)
        vis = pooled(data["baseline"], width_of, OPERATING, clips).visible
        lines.append(f"| {group} | {len(clips)} | {vis} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Per clip at {OPERATING:.2f} (recall / FA count), baseline then each variant's recall delta.\n")
    lines.append("| clip | source | vis | invis | baseline | " + " | ".join(v for v in variants if v != "baseline") + " |")
    lines.append("|" + "---|" * (len(variants) + 4))
    for clip in sorted(sources, key=lambda c: (sources[c], c)):
        base = pooled(data["baseline"], width_of, OPERATING, [clip])
        cells = []
        for variant in variants:
            stats = pooled(data[variant], width_of, OPERATING, [clip])
            out["variants"][variant].setdefault("clips", {})[clip] = metric_dict(stats)
            if variant != "baseline":
                cells.append(f"{stats.recall - base.recall:+.4f} / {stats.false_alarm_count - base.false_alarm_count:+d}")
        lines.append(f"| {clip} | {sources[clip]} | {base.visible} | {base.invisible} | {base.recall:.4f} / {base.false_alarm_count} | "
                     + " | ".join(cells) + " |")
    lines.append("")
    budget = pooled(data["baseline"], width_of, OPERATING).false_alarm_count
    lines.append(f"Equal false alarms: the baseline fires {budget} times on the {pooled(data['baseline'], width_of, OPERATING).invisible} "
                 f"invisible labels at {OPERATING:.2f}; each variant at the lowest 0.01-grid threshold whose pooled FA count "
                 f"is at most {budget}.\n")
    lines.append("| variant | thr | recall | FA | grid | custom | prof |")
    lines.append("|---|---|---|---|---|---|---|")
    base_groups = {group: pooled(data["baseline"], width_of, OPERATING, by_group[group]).recall for group in GROUPS}
    for variant in variants:
        threshold = OPERATING if variant == "baseline" else equal_fa_threshold(data[variant], width_of, budget)
        stats = pooled(data[variant], width_of, threshold)
        groups = {group: pooled(data[variant], width_of, threshold, by_group[group]) for group in GROUPS}
        out["equal_false_alarm"][variant] = {"threshold": threshold, "pooled": metric_dict(stats),
                                             "groups": {g: metric_dict(s) for g, s in groups.items()},
                                             "group_recall_delta": {g: round(groups[g].recall - base_groups[g], 4) for g in GROUPS}}
        lines.append(f"| {variant} | {threshold:.2f} | {stats.recall:.4f} | {stats.false_alarm_count} | "
                     + " | ".join(f"{groups[g].recall:.4f} ({groups[g].recall - base_groups[g]:+.4f})" for g in GROUPS) + " |")
    lines.append("")
    lines.append("Equal false alarms per group: each group's own threshold matched to the baseline's FA count in that group at 0.42.\n")
    lines.append("| variant | " + " | ".join(f"{g} thr / recall (delta) / FA" for g in GROUPS) + " |")
    lines.append("|" + "---|" * (len(GROUPS) + 1))
    for variant in variants:
        cells = []
        per_group = {}
        for group in GROUPS:
            clips = by_group[group]
            group_budget = pooled(data["baseline"], width_of, OPERATING, clips).false_alarm_count
            threshold = OPERATING if variant == "baseline" else equal_fa_threshold(data[variant], width_of, group_budget, clips)
            stats = pooled(data[variant], width_of, threshold, clips)
            per_group[group] = {"threshold": threshold, **metric_dict(stats), "recall_delta": round(stats.recall - base_groups[group], 4)}
            cells.append(f"{threshold:.2f} / {stats.recall:.4f} ({stats.recall - base_groups[group]:+.4f}) / {stats.false_alarm_count}")
        out["equal_false_alarm"][variant]["per_group_matched"] = per_group
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (HERE / "summary.md").write_text(text, encoding="utf-8")
    (HERE / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true", help="run the detector (GPU) and write rows_<variant>.csv")
    parser.add_argument("--analyze", action="store_true", help="score the CSVs and write summary.md / summary.json")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--clips", nargs="*", default=None, help="score only these clips (appends to the CSVs)")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    if args.score:
        score(args.variants, args.clips, args.device)
    if args.analyze:
        analyze(args.variants)
    if not (args.score or args.analyze):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
