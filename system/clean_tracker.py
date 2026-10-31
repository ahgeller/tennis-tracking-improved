"""Run the isolated clean tennis tracker."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from tennis_tracker.ball_weights import KNOWN, UNKNOWN_CONF, operating_conf, unknown_weights_note
from tennis_tracker.config import Config
from tennis_tracker.paths import BOUNCE_MODEL, COURT_ENGINE, BALL_WEIGHTS, PLAYER_ENGINE, ROOT
from tennis_tracker.live.fixed_lag import WHOLE_CLIP, parse_delay
from tennis_tracker.live.pipeline import LiveOptions, run_live
from tennis_tracker.live.source import PACINGS, DropPolicy
from tennis_tracker.pipeline import run
from evaluation.validate import main as validate_tracking


OUTPUT = ROOT / "output"
# What a run in an output folder is named after its input: ``<stem>_tracked.mp4``
# and ``<stem>_tracking.json``.
TRACKED_VIDEO_SUFFIX = "_tracked.mp4"
TRACKING_JSON_SUFFIX = "_tracking.json"
_SEPARATORS = tuple(sep for sep in (os.sep, os.altsep) if sep)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated clean tennis tracker")
    parser.add_argument("-i", "--input", default=str(ROOT / "sample" / "pomona.mp4"))
    parser.add_argument("-o", "--output", default=str(OUTPUT / "tracking.mp4"),
                        help="The tracking video, or a folder (existing, or ending with a separator) "
                             f"for <input stem>{TRACKED_VIDEO_SUFFIX}")
    parser.add_argument("--tracking-json",
                        help=f"Default: <input stem>{TRACKING_JSON_SUFFIX} beside a folder's video, "
                             f"else {OUTPUT / 'tracking.json'}")
    parser.add_argument("--weights", default=str(BALL_WEIGHTS),
                        help="Ball-detector weights, a .pt record (default: the stream_s4 detector, "
                             f"{BALL_WEIGHTS.relative_to(ROOT).as_posix()}) or a GridTrackNet .npz")
    parser.add_argument("--conf", type=float, default=None,
                        help="Ball-detector confidence threshold (default: the one the weights were gated at: "
                             + ", ".join(f"{known.conf:.2f} for {known.path.relative_to(ROOT).as_posix()}"
                                         for known in KNOWN)
                             + f", {UNKNOWN_CONF:.2f} for other weights)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--court", action="store_true",
                        help="Compute the court events (landings, contacts, rally legs), export them "
                             "in the tracking JSON and draw the top-right 2D court minimap")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--showtracks", action="store_true",
                        help="Draw every selector track on the tracking video, chosen or not, each in "
                             "its own colour under the chosen track's trail and marker")
    parser.add_argument("--motion-diagnostics", action="store_true",
                        help="Add per-frame motion diagnostics to the tracking JSON "
                             "(costs ~1.7 ms/frame)")
    parser.add_argument("--annotations", help="Validate against this annotation JSON")
    parser.add_argument("--report-json", default=str(OUTPUT / "validation.json"))
    parser.add_argument("--self-test", action="store_true")
    live = parser.add_argument_group("live", "Track frame by frame as they arrive (tennis_tracker/live); "
                                             "writes the tracking JSON and <json stem>.live.json, no video")
    live.add_argument("--live", action="store_true", help="Run the live tracker on --input or --camera")
    live.add_argument("--camera", type=int, help="Capture from this camera index (implies --live; the JSON "
                                                 f"defaults to {OUTPUT / 'camera<N>_live_tracking.json'})")
    live.add_argument("--camera-fps", type=float, metavar="FPS",
                      help="The camera's frame rate (default: measured over its first frames)")
    live.add_argument("--pace", choices=PACINGS, default="realtime",
                      help="realtime: a file's frames arrive at its frame rate; fast: as fast as the tracker takes "
                           "them (files only: a camera is realtime)")
    live.add_argument("--live-delay", type=parse_delay, default=LiveOptions.delay_s, metavar="SECONDS",
                      help="Commit each frame's output this long after the prepass has it (beyond the ball "
                           f"model's own lookahead, 4 phase frames); '{WHOLE_CLIP}': only at the end, the offline answer")
    live.add_argument("--live-history", type=float, default=LiveOptions.history_s, metavar="SECONDS",
                      help="Committed context each selector run keeps")
    live.add_argument("--live-every", type=float, default=LiveOptions.every_s, metavar="SECONDS",
                      help="Time between selector runs")
    live.add_argument("--live-drop", choices=[policy.value for policy in DropPolicy], default=LiveOptions.drop.value,
                      help="backlog: drop the oldest frames once more than --live-backlog wait; never: report the lag")
    live.add_argument("--live-backlog", type=float, default=LiveOptions.max_backlog_s, metavar="SECONDS",
                      help="With --live-drop backlog: how many seconds of frames may wait before the oldest are "
                           "dropped (a phase group at a time)")
    live.add_argument("--record-evidence", action="store_true",
                      help="Also write <json stem>.evidence.pkl, the selector's input, for tools/live_lag_sweep.py")
    live.add_argument("--live-selector-inline", action="store_true",
                      help="Run the selector on the tracking thread instead of its own process")
    return parser


def live_options(args: argparse.Namespace) -> LiveOptions:
    """The ``LiveOptions`` the command line asks for."""
    return LiveOptions(delay_s=args.live_delay, history_s=args.live_history, every_s=args.live_every,
                       pacing=args.pace, drop=DropPolicy(args.live_drop), max_backlog_s=args.live_backlog,
                       camera=args.camera, camera_fps=args.camera_fps, record_evidence=bool(args.record_evidence),
                       selector_process=not args.live_selector_inline)


# What ``--self-test`` runs: the selector, court-event, ROI-tracker and
# GridTrackNet suites. ``python -m pytest -q`` from the root runs every test.
SELF_TEST_MODULES = (
    "tests/test_selector_scenarios.py",
    "tests/test_court_scenarios.py",
    "tests/test_roi_tracker.py",
    "tests/test_gridtracknet.py",
)


def _self_test() -> int:
    """Run the tracker's synthetic regression tests under pytest; returns pytest's exit code."""
    try:
        import pytest  # optional: only the self-test needs it
    except ImportError:
        print("--self-test runs the tests under pytest, which is not installed "
              "(pip install pytest)", file=sys.stderr)
        return 2
    return int(pytest.main(["-q", "-p", "no:cacheprovider", "--rootdir", str(ROOT),
                            *(str(ROOT / module) for module in SELF_TEST_MODULES)]))


def resolve_output_paths(input_path: str, output: str, tracking_json: Optional[str]) -> Tuple[Path, Path]:
    """Where the tracking video and the tracking JSON go, as absolute paths.

    ``output`` names the video file, unless it is an existing directory or ends
    with a path separator: then it is a folder and the video is
    ``<input stem>_tracked.mp4`` in it.  ``tracking_json`` is used as given;
    when it is None the JSON goes beside a folder's video as
    ``<input stem>_tracking.json``, otherwise to ``OUTPUT / "tracking.json"``.
    """
    stem = Path(input_path).stem
    if output.endswith(_SEPARATORS) or Path(output).is_dir():
        folder = Path(output).resolve()
        video = folder / f"{stem}{TRACKED_VIDEO_SUFFIX}"
        default_json = folder / f"{stem}{TRACKING_JSON_SUFFIX}"
    else:
        video = Path(output).resolve()
        default_json = OUTPUT / "tracking.json"
    json_path = Path(tracking_json).resolve() if tracking_json else default_json
    return video, json_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.camera is not None and args.pace == "fast":
        parser.error("--pace fast is for files: a camera is realtime")
    if args.self_test:
        status = _self_test()
        if status == 0:
            print("clean_version self-test passed")
        return status

    OUTPUT.mkdir(parents=True, exist_ok=True)
    live = bool(args.live or args.camera is not None)
    tracking_json_arg = args.tracking_json
    if args.camera is not None and not tracking_json_arg:    # not over a file run's default JSON
        tracking_json_arg = str(OUTPUT / f"camera{args.camera}_live{TRACKING_JSON_SUFFIX}")
    output_video, tracking_json = resolve_output_paths(args.input, args.output, tracking_json_arg)
    options = live_options(args) if live else None
    if not args.no_video and not live:
        print(f"[init] Tracking video: {output_video}")
    print(f"[init] Tracking JSON: {tracking_json}")
    note = unknown_weights_note(args.weights) if args.conf is None else None
    if note:
        print(f"[init] {note}")
    cfg = Config(
        input_video=str(Path(args.input).resolve()),
        output_video=str(output_video),
        tracking_json=str(tracking_json),
        model_path=args.weights,
        player_model_path=str(PLAYER_ENGINE),
        court_model_path=str(COURT_ENGINE),
        conf=float(operating_conf(args.weights) if args.conf is None else args.conf),
        device=str(args.device),
        save_tracking_video=not (args.no_video or live),
        court_events=bool(args.court),
        draw_court_minimap=bool(args.court),
        bounce_model_path=str(BOUNCE_MODEL),
        print_selector_tracks=False,
        info_timing=bool(args.info),
        motion_diagnostics=bool(args.motion_diagnostics),
        show_all_tracks=bool(args.showtracks),
    )
    if options is not None:
        run_live(cfg, options)
    else:
        run(cfg)
    if not args.annotations:
        return 0
    return validate_tracking([
        "--predictions", cfg.tracking_json,
        "--annotations", str(Path(args.annotations).resolve()),
        "--report-json", str(Path(args.report_json).resolve()),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
