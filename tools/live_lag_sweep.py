"""Score the live tracker's fixed-lag selector at several commit delays, on recorded evidence.

    python tools/live_lag_sweep.py --evidence output/live_gate --delays 0.1 0.2 0.33 0.5 1 2 3 clip
    python tools/live_lag_sweep.py --evidence output/live_gate --delays 0.5 --history 2 4 8

A live run (``clean_tracker.py --live``, or ``evaluate_archive.py --mode pipeline --live``) records
each clip's per-frame evidence, the input the selector reads, beside its tracking JSON
(``<json stem>.evidence.pkl``, ``tennis_tracker.live.pipeline.save_evidence``). Everything upstream
of the selector is causal and does not depend on the delay, so replaying
``tennis_tracker.live.fixed_lag.FixedLagSelector`` over that evidence on the CPU gives exactly the
rows a live run at another delay would commit. ``clip`` is the whole clip as the delay: the offline
selector's answer.

For every setting it prints the held-out gate's pooled line (the archive scorer's metric), recall
per source, and how many frames differ from the whole-clip answer (present or not, or more than
0.5 px apart). ``--write-json DIR`` also writes each replay's tracking frames for the other scorers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import finetune  # noqa: E402,F401  (puts system/ on the path)
from finetune.data.data_policy import clip_source  # noqa: E402
from finetune.data.detector_metrics import DetectorMetrics  # noqa: E402
from finetune.data.labels import read_labels  # noqa: E402
from finetune.data.workspace import MANIFEST_NAME, read_manifest, video_meta  # noqa: E402
from finetune.scoring.archive import fmt, list_clips  # noqa: E402
from finetune.scoring.pipeline_score import ClipScore, score_pipeline_clip  # noqa: E402
from tennis_tracker.court.court_geometry import last_valid_court_keypoints  # noqa: E402
from tennis_tracker.export import frame_result_to_json, json_safe  # noqa: E402
from tennis_tracker.live.evidence import EVIDENCE_SUFFIX, load_evidence, replay  # noqa: E402
from tennis_tracker.live.fixed_lag import LagSettings  # noqa: E402
from tennis_tracker.tracking_schema import TrackingFrames  # noqa: E402

MOVED_PX = 0.5      # a frame whose output moved more than this from the whole-clip answer differs


class Setting(NamedTuple):
    delay_s: Optional[float]     # None: the whole clip
    history_s: float
    every_s: float

    @property
    def label(self) -> str:
        delay = "clip" if self.delay_s is None else f"{self.delay_s:g}s"
        return f"L={delay} H={self.history_s:g}s K={self.every_s:g}s"


class Replay(NamedTuple):
    clip: str
    setting: Setting
    score: ClipScore
    rows: TrackingFrames
    runs: int
    seconds: float          # the replay's selector time (one process, so only roughly a live run's)


def _delay(value: str) -> Optional[float]:
    return None if value == "clip" else float(value)


def replay_rows(evidence_path: Path, setting: Setting) -> Tuple[TrackingFrames, int, float]:
    """The tracking frames the fixed-lag selector commits over one recorded clip, its run count and
    the seconds the replay took."""
    recorded = load_evidence(evidence_path)
    settings = LagSettings.at_fps(recorded.fps, setting.delay_s, setting.history_s, setting.every_s)
    started = time.perf_counter()
    selector = replay(recorded, settings)
    seconds = time.perf_counter() - started
    frames = []
    for index, result in enumerate(selector.committed):
        row = frame_result_to_json(index, result)
        row["court_keypoints"] = json_safe(recorded.court_keypoints[index])
        row["player_boxes"] = json_safe(recorded.player_boxes[index])
        frames.append(row)
    payload: TrackingFrames = {"frames": frames,
                               "last_valid_court_keypoints": last_valid_court_keypoints(recorded.court_keypoints)}
    return payload, selector.runs, seconds


def _job(args: Tuple[Path, Path, Path, Setting]) -> Replay:
    evidence, video, labels_csv, setting = args
    payload, runs, seconds = replay_rows(evidence, setting)
    meta = video_meta(video)
    labels = read_labels(labels_csv, meta.width, meta.height)
    return Replay(video.stem, setting, score_pipeline_clip(labels, payload, meta.width), payload, runs,
                  seconds)


def differing_frames(rows: TrackingFrames, reference: TrackingFrames) -> int:
    """Frames whose output is present in one and not the other, or more than ``MOVED_PX`` apart."""
    count = 0
    for row, ref in zip(rows["frames"], reference["frames"]):
        if bool(row.get("present")) != bool(ref.get("present")):
            count += 1
        elif row.get("present") and math.hypot(row["x"] - ref["x"], row["y"] - ref["y"]) > MOVED_PX:
            count += 1
    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, required=True, help="Folder of <clip>*.evidence.pkl files")
    parser.add_argument("--delays", nargs="+", type=_delay, default=[0.1, 0.2, 0.33, 0.5, 1.0, 2.0, 3.0, None],
                        help="Commit delays in seconds; 'clip' is the whole clip")
    parser.add_argument("--history", nargs="+", type=float, default=[4.0], help="Seconds of committed context")
    parser.add_argument("--every", nargs="+", type=float, default=[0.1], help="Seconds between selector runs")
    parser.add_argument("--archive", type=Path, default=ROOT / "finetune")
    parser.add_argument("--holdout", default="mixed_v1")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--write-json", type=Path, help="Write each replay's tracking frames here")
    args = parser.parse_args(argv)

    manifest = read_manifest(args.archive / MANIFEST_NAME)
    clips = []
    for video, labels_csv in list_clips(None, args.archive, False, args.holdout):
        found = sorted(args.evidence.glob(f"{video.stem}_*{EVIDENCE_SUFFIX}")) or sorted(
            args.evidence.glob(f"{video.stem}{EVIDENCE_SUFFIX}"))
        if not found:
            print(f"[sweep] no evidence for {video.stem}; skipped")
            continue
        clips.append((found[0], video, labels_csv))
    # The whole clip is one run whatever the history and the cadence: once.
    settings = [Setting(delay, history, every) for delay in args.delays if delay is not None
                for history in args.history for every in args.every]
    settings += [Setting(None, args.history[0], args.every[0])]
    jobs = [(evidence, video, labels_csv, setting) for setting in settings for evidence, video, labels_csv in clips]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        replays = list(pool.map(_job, jobs))

    whole = {item.clip: item.rows for item in replays if item.setting.delay_s is None}
    sources = sorted({clip_source(item.clip, manifest) for item in replays})
    print(f"\n=== fixed-lag selector over {len(clips)} clips ({args.holdout}) ===")
    for setting in settings:
        mine = [item for item in replays if item.setting == setting]
        pooled = DetectorMetrics.of_clips([item.score.pairs for item in mine], 1.0)
        per_source = []
        for source in sources:
            of_source = [item.score.pairs for item in mine if clip_source(item.clip, manifest) == source]
            stats = DetectorMetrics.of_clips(of_source, 1.0)
            per_source.append(f"{source}={stats.recall:.3f} FA {stats.false_alarm_count}/{stats.invisible}")
        differ = sum(differing_frames(item.rows, whole[item.clip]) for item in mine)
        frames = sum(len(item.rows["frames"]) for item in mine)
        runs = sum(item.runs for item in mine)
        per_run_ms = 1000 * sum(item.seconds for item in mine) / max(runs, 1)
        print(fmt(setting.label, pooled))
        print(f"    {', '.join(per_source)} | differs from the whole clip on {differ}/{frames} frames | "
              f"{runs} selector runs, {per_run_ms:.0f} ms a run")
        if args.write_json is not None:
            folder = args.write_json / setting.label.replace(" ", "_").replace("=", "")
            folder.mkdir(parents=True, exist_ok=True)
            for item in mine:
                (folder / f"{item.clip}.json").write_text(json.dumps(item.rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
