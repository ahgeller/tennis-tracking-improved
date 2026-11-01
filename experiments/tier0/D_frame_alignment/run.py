"""Test D, step 4: stream_s4 e60 in raw mode on the mixed_v1 holdout (24 clips), the transition as
trained (baseline) and with the previous frame moved by the estimated camera translation before
the difference (``Config.ball_transition_align_px``, ``ball_gpu.estimate_shift`` and
``aligned_previous``). Same detector, same clips, same scorer as ``evaluate_archive.py --mode raw``
(``finetune.scoring.archive``); the option is set on the detector's Config after it is built, the
way a flag on the evaluator would set it, so the evaluator itself is untouched.

Prints the raw report of both runs, then recall / wrong / false alarm per source and for the three
grid clips at 0.42 (the gate's threshold) and 0.40 (the raw table's), a paired hit count, and the
shift estimate's CPU cost. Writes per-clip detection CSVs under ``out/<variant>/`` and
``summary.txt``. Waits for the GPU to have under 6000 MiB in use before each run.

    python experiments/tier0/D_frame_alignment/run.py [--align-px 3] [--skip-baseline]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(REPOSITORY))
import finetune  # noqa: E402,F401
from finetune.data.data_policy import clip_source  # noqa: E402
from finetune.data.detect import ball_detector  # noqa: E402
from finetune.data.detector_metrics import DetectorMetrics  # noqa: E402
from finetune.data.workspace import MANIFEST_NAME, clip_of, read_manifest  # noqa: E402
from finetune.scoring import archive  # noqa: E402

WEIGHTS = REPOSITORY / "system" / "models" / "candidates" / "stream_s4_e60.pt"
GRID = ("grid_match8", "grid_match16", "grid_match22", "grid_match85", "grid_match89")
THRESHOLDS = [0.3, 0.4, 0.42, 0.5]


def wait_for_gpu(limit_mib: int = 6000) -> None:
    while True:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.split()[0]
        if int(used) < limit_mib:
            return
        print(f"[gpu] {used} MiB in use, waiting", flush=True)
        time.sleep(30)


def save_rows(folder: Path, rows) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for clip in rows:
        with open(folder / f"{clip.clip}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label_x", "label_y", "pred_x", "pred_y", "conf"])
            for label, pred, conf in clip.rows:
                cells = [f"{v:.2f}" for v in label] if label else ["", ""]
                cells += [f"{v:.2f}" for v in pred] if pred else ["", ""]
                writer.writerow(cells + [f"{conf:.4f}"])


def group_lines(rows, sources, threshold: float) -> str:
    out = []
    for source in sorted(set(sources.values())):
        of_source = [clip for clip in rows if sources.get(clip.clip) == source]
        out.append(archive.fmt(f"{source}@{threshold:.2f}", DetectorMetrics.of_clips(archive._pooled_pairs(of_source, threshold), 1.0)))
    for clip in rows:
        if clip.clip in GRID:
            out.append(archive.fmt(f"{clip.clip}@{threshold:.2f}", archive._clip_metrics(clip, threshold)))
    out.append(archive.fmt(f"ALL@{threshold:.2f}", DetectorMetrics.of_clips(archive._pooled_pairs(rows, threshold), 1.0)))
    return "\n".join(out)


def paired(base, variant, threshold: float) -> str:
    gained = lost = changed_conf = 0
    for b, v in zip(base, variant):
        for (label, bp, bc), (_, vp, vc) in zip(b.rows, v.rows):
            if label is None:
                continue
            scale = archive.scale_for_width(b.width)
            hit_b = bp is not None and bc >= threshold and archive.math.hypot(bp[0] - label[0], bp[1] - label[1]) * scale <= archive.HIT_PX
            hit_v = vp is not None and vc >= threshold and archive.math.hypot(vp[0] - label[0], vp[1] - label[1]) * scale <= archive.HIT_PX
            gained += hit_v and not hit_b
            lost += hit_b and not hit_v
            changed_conf += abs(bc - vc) > 1e-3
    return f"paired at {threshold:.2f}: aligned gained {gained} hits, lost {lost}; labelled frames whose confidence moved: {changed_conf}"


def cpu_cost(radius: int) -> str:
    import cv2
    import numpy as np
    import torch
    from tennis_tracker.engines.ball_gpu import estimate_shift
    frame, previous = torch.rand(3, 432, 768), torch.rand(3, 432, 768)
    estimate_shift(frame, previous, radius)
    started = time.perf_counter()
    for _ in range(10):
        estimate_shift(frame, previous, radius)
    search = (time.perf_counter() - started) / 10 * 1000
    small = (np.random.rand(54, 96).astype(np.float32), np.random.rand(54, 96).astype(np.float32))
    window = cv2.createHanningWindow((96, 54), cv2.CV_32F)
    started = time.perf_counter()
    for _ in range(200):
        cv2.phaseCorrelate(small[0], small[1], window)
    phase = (time.perf_counter() - started) / 200 * 1000
    return (f"CPU cost per transition: exhaustive integer search radius {radius} at 768x432 (torch, {torch.get_num_threads()} threads) "
            f"{search:.1f} ms; OpenCV phaseCorrelate at 96x54 {phase:.3f} ms (plus the downscale)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--align-px", type=int, default=3)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--clips", nargs="*", default=None,
                        help="clip stems instead of the mixed_v1 holdout (supporting runs on other clips)")
    args = parser.parse_args()
    out = Path(args.out)
    archive_root = REPOSITORY / "finetune"
    clips = archive.list_clips(args.clips, archive_root, holdout=None if args.clips else "mixed_v1")
    manifest = read_manifest(archive_root / MANIFEST_NAME)
    sources = {clip_of(csv): clip_source(clip_of(csv), manifest) for _, csv in clips}
    print(f"[archive] {archive_root} ({len(clips)} clips, mixed_v1)")
    results = {}
    for variant in ([] if args.skip_baseline else ["baseline"]) + [f"aligned{args.align_px}"]:
        wait_for_gpu()
        detector = ball_detector(WEIGHTS, device="0")
        if variant != "baseline":
            detector.cfg = dataclasses.replace(detector.cfg, ball_transition_align_px=args.align_px)
        print(f"\n##### {variant}: ball_transition_align_px={detector.cfg.ball_transition_align_px}", flush=True)
        rows = archive.raw_rows(detector, clips, variant)
        del detector
        save_rows(out / variant, rows)
        archive.report_raw(variant, rows, THRESHOLDS, sources)
        results[variant] = rows
    summary = io.StringIO()
    with redirect_stdout(summary):
        for variant, rows in results.items():
            print(f"\n=== {variant}: per source and the grid clips ===")
            for threshold in (0.42, 0.40):
                print(group_lines(rows, sources, threshold))
        if "baseline" in results and len(results) == 2:
            base, variant = results["baseline"], results[f"aligned{args.align_px}"]
            print()
            print(paired(base, variant, 0.42))
            print(paired(base, variant, 0.40))
        print()
        print(cpu_cost(args.align_px))
    print(summary.getvalue())
    (out / "summary.txt").write_text(summary.getvalue(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
