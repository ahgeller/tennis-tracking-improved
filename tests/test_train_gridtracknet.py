"""finetune/training/train_gridtracknet.py and the modules under it: the script's own self-test, and the data
path - which clips train, the per-clip frame cache, and the flip augmentation's label transform - on
synthetic clips under tmp_path. No GPU, no real label or video."""
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from finetune.training import train_gridtracknet as training
from finetune.data.detector_metrics import DetectorMetrics
from finetune.data.labels import write_labels
from finetune.training.clips import ClipSelection, find_clips
from finetune.training.dataset import (TRANSLATE_PX, Augmenter, UnitDataset, compress, make_target, reverse, translate,
                                       translate_limits)
from finetune.training.frame_cache import DATA_VERSION, TrainingUnit, prepare_clip, training_targets
from finetune.training.loop import evaluate_loss, fresh_model, save_weights, train_epoch
from finetune.training.metrics import promotion_improves
from tennis_tracker.gridtracknet import GRID_COLS, GRID_ROWS, HEIGHT, WIDTH, load_model
from tests._helpers import blank_template, metric_probe, write_clip

SIZE = (64, 36)


def test_training_self_test_passes():
    training.self_test()


# --------------------------------------------------------------------------- the architecture switch

def test_a_fine_tune_starts_from_its_own_architectures_weights_or_is_refused_in_one_line(tmp_path, capsys):
    """Without --from-scratch, --weights is the starting model: a heatmap_s4 fine-tune from the
    GridTrackNet .npz (the default) is refused before any cache or run folder is written, as is a
    gridtracknet fine-tune from a .pt; a .pt starts a heatmap_s4 fine-tune."""
    videos, labels = workspace(tmp_path, ["a", "b"])
    argv = ["--videos", str(videos), "--labels", str(labels), "--data-dir", str(tmp_path / "cache"),
            "--run-dir", str(tmp_path / "run"), "--arch", "heatmap_s4"]
    assert training.main(argv) == 1
    error = capsys.readouterr().err
    assert "[error] --weights" in error and "is a gridtracknet file, but the starting model of a heatmap_s4 fine-tune " \
                                           "must be a heatmap_s4 one (.pt); --from-scratch trains a new network instead" in error
    assert not (tmp_path / "cache").exists() and not (tmp_path / "run").exists()
    parser = training.build_parser()
    args = parser.parse_args(["--arch", "gridtracknet", "--weights", str(tmp_path / "net.pt")])
    with pytest.raises(ValueError, match="is a heatmap_s4 file, but the starting model of a gridtracknet fine-tune"):
        training.resolve_arch(args, None)
    args = parser.parse_args(["--arch", "heatmap_s4", "--weights", str(tmp_path / "best.pt"), "--save", str(tmp_path / "w.pt")])
    training.resolve_arch(args, None)
    assert args.arch == "heatmap_s4" and args.weights == tmp_path / "best.pt"
    args = parser.parse_args(["--weights", str(tmp_path / "ball.engine")])
    with pytest.raises(ValueError, match="end in .npz"):
        training.resolve_arch(args, None)
    with pytest.raises(SystemExit):
        parser.parse_args(["--arch", "yolo"])


def test_a_scratch_run_reads_either_weights_for_the_prepass_but_gridtracknet_needs_its_template(tmp_path):
    parser = training.build_parser()
    args = parser.parse_args(["--arch", "heatmap_s4", "--from-scratch", "--weights", str(tmp_path / "grid.npz"),
                              "--save", str(tmp_path / "w.pt")])
    training.resolve_arch(args, None)                                       # the .npz feeds the prepass alone
    args = parser.parse_args(["--arch", "heatmap_s4", "--from-scratch", "--weights", str(tmp_path / "old.pt"),
                              "--save", str(tmp_path / "w.pt")])
    training.resolve_arch(args, None)
    args = parser.parse_args(["--from-scratch", "--weights", str(tmp_path / "old.pt")])
    with pytest.raises(ValueError, match="the template the trained weights are written into must be a gridtracknet one"):
        training.resolve_arch(args, None)


def test_train_fine_tunes_a_heatmap_winner_from_its_pt(tmp_path, capsys):
    """The deployment path: a heatmap_s4 fine-tune loads the .pt as its starting model (the
    prepass and the fit are stood in for; the loaded network is what fit receives)."""
    from tennis_tracker.heatmapnet import HeatmapNet, save_model
    videos, labels = workspace(tmp_path, ["a", "b"])
    start = tmp_path / "best.pt"
    save_model(HeatmapNet(), start)
    received = {}

    def fit(args, model, *rest):
        received["model"] = model
        return training.Selection(args.run_dir / "best.pt", 1, training.metrics_of(0.9, 0.0), 1, 0.1)
    manifest = {"train": [{"clip": "a"}], "val": [{"clip": "b"}], "validation_clips": ["b"]}
    with patch.object(training, "prepare_data", return_value=manifest), patch.object(training, "DataLoader"), \
            patch.object(training, "UnitDataset"), patch.object(training, "make_sampler", return_value=None), \
            patch.object(training, "fit", side_effect=fit), patch.object(training, "pick_winner"):
        assert training.main(["--videos", str(videos), "--labels", str(labels), "--data-dir", str(tmp_path / "cache"),
                              "--run-dir", str(tmp_path / "run"), "--arch", "heatmap_s4", "--weights", str(start),
                              "--save", str(tmp_path / "winner.npz"), "--device", "cpu", "--epochs", "1"]) == 0
    assert isinstance(received["model"], HeatmapNet) and received["model"].training
    out = capsys.readouterr().out
    assert f"[train] fine-tuning heatmap_s4 from {start}" in out and "winner.pt" in out
    assert json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))["arch"] == "heatmap_s4"


def test_resolve_arch_defaults_inherits_and_corrects_the_save_suffix(tmp_path, capsys):
    """No --arch: GridTrackNet, or the selection run's architecture; one that contradicts the
    selection run is refused; --save takes the architecture's suffix, with a line saying so."""
    parser = training.build_parser()
    args = parser.parse_args(["--save", str(tmp_path / "best.npz")])
    training.resolve_arch(args, None)
    assert args.arch == "gridtracknet" and args.save == tmp_path / "best.npz" and capsys.readouterr().out == ""
    args = parser.parse_args(["--arch", "heatmap_s4", "--from-scratch", "--save", str(tmp_path / "gridtracknet_best.npz")])
    training.resolve_arch(args, None)
    assert args.arch == "heatmap_s4" and args.save == tmp_path / "gridtracknet_best.pt"
    assert "is given .pt, the heatmap_s4 weights suffix" in capsys.readouterr().out
    heat_recipe = {"recipe": {"arch": "heatmap_s4"}}
    args = parser.parse_args(["--from-scratch", "--all-clips", "--save", str(tmp_path / "w.npz")])
    training.resolve_arch(args, heat_recipe)
    assert args.arch == "heatmap_s4" and args.save == tmp_path / "w.pt"
    args = parser.parse_args(["--arch", "gridtracknet", "--from-scratch", "--all-clips"])
    with pytest.raises(ValueError, match="does not follow the selection run, which trained heatmap_s4"):
        training.resolve_arch(args, heat_recipe)
    args = parser.parse_args(["--all-clips", "--weights", str(tmp_path / "h1" / "best.pt"), "--save", str(tmp_path / "w.pt")])
    training.resolve_arch(args, heat_recipe)                                 # the winner fine-tuned on every clip
    assert args.arch == "heatmap_s4"


def test_a_resume_without_arch_continues_the_runs_architecture(tmp_path, capsys):
    """--resume reads the architecture off the run's config.json as it reads --epochs: a heatmap_s4
    run resumed from its best.pt with no --arch goes on as heatmap_s4 (not refused as a gridtracknet
    fine-tune from a .pt), and --arch gridtracknet on it is refused in a line naming the run."""
    videos, labels = workspace(tmp_path, ["a", "b"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.pt").write_bytes(b"read after the data, not here")
    argv = ["--videos", str(videos), "--labels", str(labels), "--data-dir", str(tmp_path / "cache"),
            "--run-dir", str(run_dir), "--weights", str(tmp_path / "best.pt"), "--save", str(tmp_path / "w.pt")]
    parser = training.build_parser()
    config = training.run_config(parser.parse_args([*argv, "--arch", "heatmap_s4"]), ["a", "b"], ["b"], {})
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    args = parser.parse_args([*argv, "--resume"])
    training.resolve_arch(args, None, training.read_run_config(run_dir))
    assert args.arch == "heatmap_s4"
    args = parser.parse_args([*argv, "--resume", "--arch", "gridtracknet"])
    with pytest.raises(ValueError, match="--arch gridtracknet does not follow the run in .*, which trains heatmap_s4"):
        training.resolve_arch(args, None, config)
    with patch.object(training, "prepare_data", side_effect=ValueError("the data step")) as data:
        assert training.main([*argv, "--resume"]) == 1
    assert data.called and capsys.readouterr().err.strip() == "[error] the data step"      # the options all passed


def test_config_json_carries_the_arch_and_same_run_refuses_a_change():
    parser = training.build_parser()
    grid = training.run_config(parser.parse_args([]), ["a"], ["a"], {})
    heat = training.run_config(parser.parse_args(["--arch", "heatmap_s4", "--from-scratch"]), ["a"], ["a"], {})
    assert grid["arch"] == "gridtracknet" and heat["arch"] == "heatmap_s4"
    training.same_run(grid, grid)
    with pytest.raises(ValueError, match="config.json has arch='gridtracknet', this run arch='heatmap_s4'"):
        training.same_run(grid, {**heat, "from_scratch": False})
    before_the_field = {key: value for key, value in grid.items() if key != "arch"}
    training.same_run(before_the_field, grid)                               # an older run trained GridTrackNet
    assert training.arch_of_config(before_the_field) == "gridtracknet"


def test_a_heatmap_run_writes_every_file_with_the_pt_suffix(tmp_path, capsys):
    """fit on a HeatmapNet: last/best/best_loss and epochs/epoch_NNN are .pt records the tracker's
    loader reads, and no .npz appears."""
    model = fresh_model(torch.device("cpu"), "heatmap_s4")
    args = training.build_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--arch", "heatmap_s4", "--from-scratch",
                                               "--epochs", "2", "--weights", str(tmp_path / "unused.npz")])
    args.run_dir.mkdir()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with patch.object(training, "train_epoch", side_effect=[0.5, 0.4]), patch.object(training, "evaluate_loss", side_effect=[0.3, 0.2]):
        selection = training.fit(args, model, None, object(), optimizer, scaler, torch.device("cpu"),
                                 lambda weights: training.metrics_of(0.8 if weights.suffix == ".pt" else -1.0, 0.0))
    assert selection.best_path == args.run_dir / "best.pt" and selection.best_metrics.recall == 0.8
    assert sorted(path.name for path in args.run_dir.iterdir() if path.suffix in (".pt", ".npz")) == \
        ["best.pt", "best_loss.pt", "checkpoint.pt", "last.pt"]
    assert sorted(path.name for path in (args.run_dir / training.EPOCHS_DIR).iterdir()) == ["epoch_001.pt", "epoch_002.pt"]
    assert training.epoch_weights(args.run_dir, 3, ".pt") == args.run_dir / "epochs" / "epoch_003.pt"
    from tennis_tracker.ball_models import load_ball_model
    assert isinstance(load_ball_model(args.run_dir / "best.pt", torch.device("cpu")), type(model))
    args.save = tmp_path / "save.pt"
    training.pick_winner(args, [], ["video10"], selection, torch.device("cpu"))
    assert "[eval] best.pt (epoch 1) on ['video10']" in capsys.readouterr().out and args.save.is_file()


def test_main_reports_a_run_that_cannot_start_as_status_1(tmp_path, capsys):
    """No clips, or an option the clips cannot satisfy: one line on stderr and exit status 1, the
    same as the other scripts' setup errors, before any cache or run folder is written."""
    videos, labels = workspace(tmp_path, ["a", "b"])
    folders = ["--data-dir", str(tmp_path / "cache"), "--run-dir", str(tmp_path / "run")]
    assert training.main(["--videos", str(videos), "--labels", str(tmp_path / "empty"), *folders]) == 1
    assert "[error] No *_ball.csv files" in capsys.readouterr().err
    assert training.main(["--videos", str(videos), "--labels", str(labels), "--cameras", "fixed", *folders]) == 1
    assert "[error] --cameras needs camera tags" in capsys.readouterr().err
    assert not (tmp_path / "cache").exists() and not (tmp_path / "run").exists()


def test_holdout_and_val_clips_are_alternatives_and_the_config_records_the_choice(capsys):
    parser = training.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--holdout", "mixed_v1", "--val-clips", "video10"])
    assert "not allowed with argument" in capsys.readouterr().err
    args = parser.parse_args(["--holdout", "mixed_v1", "--epochs", "3"])
    config = training.run_config(args, ["video1", "video13"], ["video1"], {"custom": 2.0})
    assert config["holdout"] == "mixed_v1" and config["val_clips"] is None and config["validation_clips"] == ["video1"]
    assert config["clips"] == ["video1", "video13"] and config["epochs"] == 3 and config["oversample"] == {"custom": 2.0}
    args = parser.parse_args(["--val-clips", "video10", "grid_match*"])
    config = training.run_config(args, ["video10"], ["video10"], {})
    assert config["holdout"] is None and config["val_clips"] == ["video10", "grid_match*"]


# --------------------------------------------------------------------------- which clips train

def workspace(root: Path, clips, manifest=None, exclude=(), review_status=None) -> tuple:
    """labels/ and videos/ under `root` with a label file and an (empty) video per clip, plus the
    three records when given. Returns (videos, labels)."""
    (root / "labels").mkdir(exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)
    for clip in clips:
        write_labels(root / "labels" / f"{clip}_ball.csv", {0: (1.0, 2.0)}, *SIZE)
        (root / "videos" / f"{clip}.mp4").write_bytes(b"")
    if manifest is not None:
        lines = ["clip,source,camera"] + [f"{clip},{source},{camera}" for clip, source, camera in manifest]
        (root / "clips.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if exclude:
        (root / "exclude.txt").write_text("\n".join(exclude) + "\n", encoding="utf-8")
    if review_status is not None:
        (root / "review_status.json").write_text(json.dumps(review_status), encoding="utf-8")
    return root / "videos", root / "labels"


def names(clips) -> list:
    return [clip for clip, _, _, _ in clips]


def test_find_clips_applies_exclusions_review_policy_sources_and_cameras(tmp_path, capsys):
    videos, labels = workspace(
        tmp_path, ["video1", "video13", "video14", "grid_match4", "tnv2_a"],
        manifest=[("video1", "custom", "fixed"), ("video13", "prof", "moving"), ("video14", "custom-uncorrected", ""),
                  ("grid_match4", "grid", "fixed")],
        exclude=["grid_match4  # audit"], review_status={"verified_video_min": 13, "verified_clips": []})
    assert names(find_clips(videos, labels)) == ["video13"]        # video1 unverified, video14 untrusted, grid excluded, tnv2 unknown
    assert [line for line in capsys.readouterr().out.splitlines() if line.startswith("[skip]")] == [
        "[skip] grid_match4: listed in exclude.txt (--include-excluded keeps it)",
        "[skip] tnv2_a: labels are still under review (--include-excluded keeps it)",     # unreviewed before unlisted
        "[skip] video1: labels are still under review (--include-excluded keeps it)",
        "[skip] video14: source 'custom-uncorrected' is machine labels nobody checked (--include-excluded keeps it)",
    ]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, sources=["prof"]))) == ["video13"]
    assert "[skip]" not in capsys.readouterr().out                 # a run's own choices are not announced
    everything = ["grid_match4", "tnv2_a", "video1", "video13", "video14"]      # natural clip order
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True))) == everything
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, sources=["custom", "grid"]))) == ["grid_match4", "video1"]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, cameras=["fixed"]))) == ["grid_match4", "video1"]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, cameras=["untagged"]))) == ["tnv2_a", "video14"]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, exclude_clips=["video1*", "tnv2_a"]))) == ["grid_match4"]
    for clip, video, csv_path, source in find_clips(videos, labels, ClipSelection(include_excluded=True)):
        assert video == videos / f"{clip}.mp4" and csv_path == labels / f"{clip}_ball.csv"
    assert [s for _, _, _, s in find_clips(videos, labels, ClipSelection(include_excluded=True))] == [
        "grid", "unknown", "custom", "prof", "custom-uncorrected"]


def test_find_clips_without_records_trusts_every_clip_as_custom(tmp_path):
    videos, labels = workspace(tmp_path, ["a", "b"])
    assert [(clip, source) for clip, _, _, source in find_clips(videos, labels)] == [("a", "custom"), ("b", "custom")]
    with pytest.raises(ValueError, match="--cameras needs camera tags"):
        find_clips(videos, labels, ClipSelection(cameras=["fixed"]))
    (videos / "b.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="No video for b_ball.csv"):
        find_clips(videos, labels)
    with pytest.raises(FileNotFoundError, match="No .*_ball.csv files"):
        find_clips(videos, tmp_path / "empty")


# --------------------------------------------------------------------------- the frame cache

def test_training_targets_clamp_onto_the_image(tmp_path):
    csv_path = tmp_path / "c_ball.csv"
    csv_path.write_text("frame,ball_x,ball_y\nframe_000,-3,2000\nframe_001,10.5,20.5\nframe_002,63,0\n", encoding="utf-8")
    assert training_targets(csv_path, *SIZE) == {0: [0.0, 35.0], 1: [10.5, 20.5], 2: None}


def test_prepare_clip_caches_the_needed_frames_and_describes_the_units(tmp_path):
    video = write_clip(tmp_path / "c.avi", 14, SIZE, 30.0)
    csv_path = tmp_path / "c_ball.csv"
    write_labels(csv_path, {f: (10.0 + f, 20.0) if f != 3 else None for f in range(12)}, *SIZE)
    data = tmp_path / "data"
    record = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert record["version"] == DATA_VERSION and record["clip"] == "c" and record["source"] == "custom"
    assert (record["stride"], record["size"]) == (1, list(SIZE))
    units = record["samples"]
    assert [unit["frames"] for unit in units] == [[f"frames/c/{i:06d}.png" for i in range(start, start + 5)]
                                                  for start in (0, 2, 4, 6)]
    assert units[0]["points"] == [[10.0, 20.0], [11.0, 20.0], [12.0, 20.0], None, [14.0, 20.0]]
    assert units[0] == TrainingUnit(clip="c", source="custom", indices=[0, 1, 2, 3, 4], frames=units[0]["frames"],
                                    points=units[0]["points"], size=list(SIZE))
    assert record["visible"] == sum(point is not None for unit in units for point in unit["points"])
    cached = sorted(p.name for p in (data / "frames" / "c").iterdir())
    assert cached == [f"{i:06d}.png" for i in range(11)]            # the last unit is 6..10; frame 11 is in none
    assert record["frames"] == list(range(11))
    image = cv2.imread(str(data / "frames" / "c" / "000007.png"))
    assert image.shape == (training.HEIGHT, training.WIDTH, 3) and int(image.mean()) == 7      # lossless
    assert json.loads((data / "clips" / "c.json").read_text(encoding="utf-8")) == record

    # a second call is served from the cache: the frames are not re-extracted ...
    stamp = (data / "frames" / "c" / "000000.png").stat().st_mtime_ns
    again = prepare_clip(("c", video, csv_path, "elsewhere"), data, rebuild=False)
    assert [unit["frames"] for unit in again["samples"]] == [unit["frames"] for unit in units]
    assert again["source"] == "elsewhere" and {unit["source"] for unit in again["samples"]} == {"elsewhere"}   # the source follows the manifest
    assert (data / "frames" / "c" / "000000.png").stat().st_mtime_ns == stamp
    # ... and a label edit only changes the units: the signature is the video's, the frames stay
    write_labels(csv_path, {f: (10.0 + f, 20.0) for f in range(7)}, *SIZE)
    os.utime(csv_path, ns=(stamp + 10 ** 9, stamp + 10 ** 9))
    edited = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert len(edited["samples"]) == 2 and edited["signature"] == record["signature"]
    assert (data / "frames" / "c" / "000000.png").stat().st_mtime_ns == stamp
    assert sorted(p.name for p in (data / "frames" / "c").iterdir()) == cached
    # --rebuild-data extracts everything the units need afresh and drops the rest
    rebuilt = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=True)
    assert rebuilt["frames"] == list(range(7)) and (data / "frames" / "c" / "000000.png").stat().st_mtime_ns > stamp
    assert sorted(p.name for p in (data / "frames" / "c").iterdir()) == [f"{i:06d}.png" for i in range(7)]


def test_prepare_data_splits_the_units_and_needs_training_units_only(tmp_path, capsys):
    """The trainer decides whether an empty validation split is an error (--all-clips holds
    nothing out); the cache only insists on something to train on."""
    from finetune.training.frame_cache import prepare_data
    clips = []
    for name in ("a", "b"):
        video = write_clip(tmp_path / f"{name}.avi", 8, SIZE, 30.0)
        csv_path = tmp_path / f"{name}_ball.csv"
        write_labels(csv_path, {f: (1.0, 1.0) for f in range(7)}, *SIZE)
        clips.append((name, video, csv_path, "custom"))
    manifest = prepare_data(clips, tmp_path / "data", ["b"], rebuild=False, workers=1)
    assert [unit["clip"] for unit in manifest["train"]] == ["a", "a"] and [unit["clip"] for unit in manifest["val"]] == ["b", "b"]
    assert manifest["validation_clips"] == ["b"] and "2 clips ready" in capsys.readouterr().out
    everything = prepare_data(clips, tmp_path / "data", [], rebuild=False, workers=1)
    assert len(everything["train"]) == 4 and everything["val"] == []
    with pytest.raises(ValueError, match="training split"):
        prepare_data(clips, tmp_path / "data", ["a", "b"], rebuild=False, workers=1)


def test_prepare_clip_refuses_a_video_that_ends_before_its_labels(tmp_path):
    video = write_clip(tmp_path / "short.avi", 4, SIZE, 30.0)
    csv_path = tmp_path / "short_ball.csv"
    write_labels(csv_path, {f: (1.0, 1.0) for f in range(8)}, *SIZE)
    with pytest.raises(ValueError, match="ended before frames"):
        prepare_clip(("short", video, csv_path, "custom"), tmp_path / "data", rebuild=True)


# --------------------------------------------------------------------------- augmentation

def unit_on_disk(data: Path, stripe=(0, 10)) -> TrainingUnit:
    """One unit whose five cached frames carry a red stripe over the cache columns `stripe`
    (the left edge by default)."""
    (data / "frames" / "u").mkdir(parents=True, exist_ok=True)
    for index in range(5):
        image = np.zeros((training.HEIGHT, training.WIDTH, 3), np.uint8)
        image[:, stripe[0]:stripe[1]] = (0, 0, 255)               # red (BGR) stripe
        cv2.imwrite(str(data / "frames" / "u" / f"{index:06d}.png"), image)
    return TrainingUnit(clip="u", source="custom", indices=list(range(5)), frames=[f"frames/u/{i:06d}.png" for i in range(5)],
                        points=[[100.0, 200.0], None, [1919.0, 0.0], [0.0, 1079.0], [500.5, 500.5]], size=[1920, 1080])


def forced(dataset: UnitDataset, flip: float) -> None:
    """Make the next draws answer `flip` to the coin tosses (0.0: flip and jitter; 1.0: neither)."""
    dataset.rng = Augmenter(0)
    dataset.rng.draws = random.Random(0)
    dataset.rng.draws.random = lambda: flip


def scripted(dataset: UnitDataset, tosses, integers=()) -> None:
    """Make the coin tosses answer `tosses` in turn and the integer draws `integers`."""
    dataset.rng = Augmenter(0)
    dataset.rng.draws = random.Random(0)
    remaining = iter(integers)
    dataset.rng.draws.random = iter(tosses).__next__
    dataset.rng.draws.randint = lambda low, high: next(remaining)


def stripe_centre(images: torch.Tensor) -> float:
    """Where the red stripe's mass sits, in the unit's 1920-wide frame coordinates."""
    red = images[0].numpy().sum(axis=0)
    columns = np.arange(training.WIDTH) + 0.5
    return float((red * columns).sum() / red.sum()) * 1920 / training.WIDTH


def decoded_x(target: torch.Tensor, frame: int = 0) -> float:
    row, col = np.argwhere(target[frame * 3].numpy() == 1)[0]
    return (col + float(target[frame * 3 + 1, row, col])) * 1920 / training.GRID_COLS


def test_flip_mirrors_the_frames_and_the_labels_about_the_image_width(tmp_path):
    unit = unit_on_disk(tmp_path)
    plain = UnitDataset(tmp_path, [unit], augment=False)
    images, target = plain[0]
    assert images.shape == (15, training.HEIGHT, training.WIDTH) and images.dtype == torch.float32
    assert float(images[0, 0, 0]) == 1.0 and float(images[0, 0, -1]) == 0.0           # red channel first (RGB); lossless
    assert torch.equal(target, make_target(unit["points"], 1920, 1080))

    flipped = UnitDataset(tmp_path, [unit], augment=True)
    forced(flipped, 0.0)
    images, target = flipped[0]
    assert float(images[0, 0, -1]) > 0.5 and float(images[0, 0, 0]) < 0.5           # the stripe is on the right now
    mirrored = [None if p is None else [1920 - p[0], p[1]] for p in unit["points"]]
    assert torch.equal(target, make_target(mirrored, 1920, 1080))
    assert mirrored[2] == [1.0, 0.0] and mirrored[3] == [1920.0, 1079.0]   # the left edge of pixel 0 <-> the right edge of pixel 1919

    unflipped = UnitDataset(tmp_path, [unit], augment=True)
    forced(unflipped, 1.0)
    images, target = unflipped[0]
    assert float(images[0, 0, 0]) > 0.9 and torch.equal(target, make_target(unit["points"], 1920, 1080))


def test_flipped_label_lands_on_the_flipped_pixels(tmp_path):
    # The stripe covers pixels 0-9 of the 768-wide cache (0-24 at the unit's 1920), so its
    # centre is 12.5 in frame coordinates (pixel k's centre is k + 0.5). After the flip the
    # pixels sit at 758-767 and the label must follow them, to 1907.5: mirroring pixel
    # indices (1920 - 1 - 12.5) would put it a pixel to the left.
    unit = unit_on_disk(tmp_path)
    unit["points"] = [[12.5, 200.0], None, None, None, None]
    dataset = UnitDataset(tmp_path, [unit], augment=True)
    forced(dataset, 0.0)
    images, target = dataset[0]
    red = images[0].numpy().sum(axis=0)                                            # red mass per cached column
    columns = np.arange(training.WIDTH) + 0.5
    stripe_centre = float((red * columns).sum() / red.sum()) * 1920 / training.WIDTH
    row, col = np.argwhere(target[0].numpy() == 1)[0]
    decoded_x = (col + float(target[1, row, col])) * 1920 / training.GRID_COLS
    assert abs(stripe_centre - 1907.5) < 0.05
    assert decoded_x == pytest.approx(1907.5) and decoded_x == pytest.approx(stripe_centre, abs=0.05)


def test_translate_moves_the_stripe_and_the_label_by_the_same_shift(tmp_path):
    # A stripe on cache pixels 40-49 has its centre at 45 cache pixels, 112.5 in the unit's
    # 1920-wide frame coordinates; a shift of 16 cache pixels is 40 native ones, so stripe and
    # label land at 152.5. (The stripe sits away from the edge because the border is replicated,
    # which would stretch an edge stripe over the gap the shift opens.)
    unit = unit_on_disk(tmp_path, stripe=(40, 50))
    unit["points"] = [[112.5, 200.0], None, None, None, None]
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"translate"})
    scripted(dataset, tosses=[1.0, 0.0], integers=[16, 0])               # no flip; translate, dx=16, dy=0
    images, target = dataset[0]
    assert abs(stripe_centre(images) - 152.5) < 0.05
    assert decoded_x(target) == pytest.approx(152.5) and decoded_x(target) == pytest.approx(stripe_centre(images), abs=0.05)
    for frame in range(1, 5):                                             # every frame shifted alike
        assert torch.equal(images[frame * 3:frame * 3 + 3], images[:3])
    assert float(target[3:].sum()) == 0.0                                 # no ball stays no ball
    edge = unit_on_disk(tmp_path, stripe=(0, 10))
    edge["points"] = [[12.5, 200.0], None, None, None, None]
    dataset = UnitDataset(tmp_path, [edge], augment=True, augmentations={"translate"})
    scripted(dataset, tosses=[1.0, 0.0], integers=[16, 0])
    images, target = dataset[0]
    assert float(images[0, :, :26].mean()) > 0.99 and float(images[0, :, 26:].mean()) < 0.01     # replicated: red fills the gap
    assert decoded_x(target) == pytest.approx(52.5)


def test_translate_is_held_so_that_no_label_leaves_the_image(tmp_path):
    unit = unit_on_disk(tmp_path)                                         # labels at x=0 and x=1919, y=0 and y=1079
    assert translate_limits(unit["points"], 1920, 1080, training.WIDTH, training.HEIGHT) == (0, 0, 0, 0)
    assert translate_limits([[12.5, 200.0], None], 1920, 1080, training.WIDTH, training.HEIGHT) == (-5, 20, -20, 20)
    assert translate_limits([[1910.0, 1070.0]], 1920, 1080, training.WIDTH, training.HEIGHT) == (-20, 3, -20, 3)
    assert translate_limits([None] * 5, 1920, 1080, training.WIDTH, training.HEIGHT) == (-TRANSLATE_PX, TRANSLATE_PX, -TRANSLATE_PX, TRANSLATE_PX)
    images = np.zeros((15, training.HEIGHT, training.WIDTH), np.float32)
    images[:, :, :10] = 1.0
    shifted, points = translate(images, [[1919.0, 1079.0], None, [0.0, 0.0]], 0, 0, 1920, 1080)
    assert np.array_equal(shifted, images) and points == [[1919.0, 1079.0], None, [0.0, 0.0]]
    shifted, points = translate(images, [[100.0, 100.0]], -4, 8, 1920, 1080)
    assert points == [[90.0, 120.0]] and shifted.shape == images.shape
    assert float(shifted[0, 0, :6].sum()) == 6.0 and float(shifted[0, 0, 6]) == 0.0      # the stripe moved 4 left
    assert float(shifted[0, 7].sum()) == 6.0 == float(shifted[0, 8].sum())                # rows moved 8 down, the top replicated
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"translate"})
    scripted(dataset, tosses=[1.0, 0.0], integers=[0, 0])                 # the clamped draw: randint(0, 0)
    images, target = dataset[0]
    assert torch.equal(target, make_target(unit["points"], 1920, 1080))


def test_reverse_reverses_the_frames_and_the_targets(tmp_path):
    unit = unit_on_disk(tmp_path)
    plain, forward = UnitDataset(tmp_path, [unit], augment=False)[0]
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"reverse"})
    scripted(dataset, tosses=[1.0, 0.0])                                  # no flip; reverse
    images, target = dataset[0]
    assert torch.equal(target, make_target(unit["points"][::-1], 1920, 1080))
    assert torch.equal(images.reshape(5, 3, training.HEIGHT, training.WIDTH).flip(0).reshape(15, training.HEIGHT, training.WIDTH), plain)
    frames = np.arange(15, dtype=np.float32).reshape(15, 1, 1) * np.ones((15, 2, 2), np.float32)
    reversed_frames, points = reverse(frames, [[1.0, 1.0], None, [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    assert reversed_frames[:, 0, 0].tolist() == [12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]
    assert points == [[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], None, [1.0, 1.0]]


def test_compress_keeps_the_shape_and_the_labels_and_changes_only_the_pixels(tmp_path):
    unit = unit_on_disk(tmp_path)
    plain, forward = UnitDataset(tmp_path, [unit], augment=False)[0]
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"compress"})
    scripted(dataset, tosses=[1.0, 0.0, 1.0], integers=[35])             # no flip; compress at quality 35; no blur
    images, target = dataset[0]
    assert images.shape == plain.shape and images.dtype == torch.float32 and torch.equal(target, forward)
    assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0
    assert not torch.equal(images, plain) and float((images - plain).abs().mean()) < 0.05    # artefacts, not another picture
    assert float(images[0, 100, :5].mean()) > 0.8 and float(images[0, 100, 100:].mean()) < 0.1
    rough = compress(plain.numpy(), 35)
    assert np.array_equal(rough, images.numpy())                          # the dataset is the function, no more
    assert float(np.abs(compress(plain.numpy(), 95) - plain.numpy()).mean()) < float(np.abs(rough - plain.numpy()).mean())


def test_new_augmentations_draw_after_the_old_ones_so_old_runs_reproduce(tmp_path):
    """A run with --augment basic makes exactly the draws it always made: the newer augmentations
    draw nothing unless named, so the seed still reproduces an old run."""
    unit = unit_on_disk(tmp_path)
    before, after = UnitDataset(tmp_path, [unit], augment=True), UnitDataset(tmp_path, [unit], augment=True, augmentations={"basic"})
    before.rng, after.rng = Augmenter(5), Augmenter(5)
    for _ in range(3):
        images, target = before[0]
        again, target_again = after[0]
        assert torch.equal(images, again) and torch.equal(target, target_again)
    assert before.rng.draws.random() == after.rng.draws.random()         # the streams are in step
    everything = UnitDataset(tmp_path, [unit], augment=True, augmentations=set(training.AUGMENTATIONS))
    everything.rng = Augmenter(5)
    images, target = everything[0]
    assert images.shape == (15, training.HEIGHT, training.WIDTH) and target.shape == (15, training.GRID_ROWS, training.GRID_COLS)
    with pytest.raises(ValueError, match="unknown augmentation"):
        UnitDataset(tmp_path, [unit], augment=True, augmentations={"basic", "rotate"})
    assert UnitDataset(tmp_path, [unit], augment=False, augmentations={"translate"}).augmentations == set()


# --------------------------------------------------------------------------- the metric and the winner

def scripted_run(tmp_path, argv, train_losses, val_losses, scored, scoring=True, validate=True, resume=False):
    """``fit`` on a fresh CPU model with the epoch losses scripted and the held-out metric answered
    from `scored` (per scored call: a recall, wrong and false alarm 0, or a (recall, wrong,
    false_alarm) tuple); returns (args, selection, the metric calls).
    `scoring` False runs without scoring and `validate` False without a validation loader. `resume`
    continues the run dir's checkpoint instead of starting over: `train_losses` is indexed by
    epoch, so it stays the whole list, while `val_losses` and `scored` are the epochs left."""
    model = fresh_model(torch.device("cpu"))
    with torch.no_grad():
        model.convs[0].bias.zero_()
    template = blank_template(model, tmp_path / "template.npz")
    args = training.build_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--weights", str(template),
                                               *(["--resume"] if resume else []), *argv])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = training.START
    if resume:
        progress = training.read_checkpoint(args.run_dir / training.CHECKPOINT, model, optimizer, scaler)
    else:
        args.run_dir.mkdir()
    calls = []

    def score(weights):
        calls.append(weights)
        answer = scored[len(calls) - 1]
        return training.metrics_of(*(answer if isinstance(answer, tuple) else (answer, 0.0)))

    def train_epoch(model, *_):
        with torch.no_grad():
            model.convs[0].bias += 1.0          # "training": the first bias counts the epochs, so each saved file is its epoch
            for parameter in model.parameters():    # and the optimizer gets a step to remember
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        return train_losses[int(model.convs[0].bias[0]) - 1]
    with patch.object(training, "train_epoch", side_effect=train_epoch), \
            patch.object(training, "evaluate_loss", side_effect=val_losses):
        selection = training.fit(args, model, None, object() if validate else None, optimizer, scaler, torch.device("cpu"),
                                 score if scoring else None, progress)
    return args, selection, calls


def results_of(run_dir):
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tiny_loader(count: int = 2, batch: int = 1):
    """A loader of ``count`` random model-sized units (the width is fixed by the normalisation,
    the grid by the loss) with a one-cell target each, on the CPU."""
    generator = torch.Generator().manual_seed(7)
    images = torch.rand((count * batch, 15, HEIGHT, WIDTH), generator=generator)
    target = torch.zeros((count * batch, 15, GRID_ROWS, GRID_COLS))
    target[:, 0, 1, 2] = 1.0
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, target), batch_size=batch)


def test_a_scratch_training_step_moves_the_running_statistics_and_evaluate_loss_leaves_them():
    """From scratch the BatchNorm layers start at Keras's defaults; one training step normalises
    with the batch and moves every layer's running mean, and the validation pass in eval mode
    neither moves them nor leaves the model in eval mode."""
    device = torch.device("cpu")
    model = fresh_model(device)
    assert model.training
    for layer in model.batch_norms:
        assert float(layer.mean.abs().sum()) == 0 and torch.equal(layer.variance, torch.ones_like(layer.variance))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_epoch(model, _tiny_loader(), optimizer, scaler, device, 1, 1, max_steps=1)
    for layer in model.batch_norms:
        assert float(layer.mean.abs().sum()) > 0
        assert not torch.equal(layer.variance, torch.ones_like(layer.variance))
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    loss = evaluate_loss(model, _tiny_loader(count=1), device)
    assert np.isfinite(loss) and model.training
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[name]), name
    model.eval()
    evaluate_loss(model, _tiny_loader(count=1), device)
    assert not model.training


def test_save_weights_and_load_model_round_trip_every_batchnorm_array(tmp_path):
    """gamma and beta (parameters now) and the running mean and variance (buffers) all come back
    from the .npz exactly, and the loaded model runs in eval mode."""
    device = torch.device("cpu")
    model = fresh_model(device)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in model.batch_norms:
            for tensor in (layer.gamma, layer.beta, layer.mean, layer.variance):
                tensor.copy_(torch.rand(tensor.shape, generator=generator) + 0.25)
    template = blank_template(model, tmp_path / "template.npz")
    save_weights(model, template, tmp_path / "saved.npz")
    loaded = load_model(tmp_path / "saved.npz", device)
    assert not loaded.training
    for saved, back in zip(model.batch_norms, loaded.batch_norms):
        assert isinstance(back.gamma, torch.nn.Parameter) and isinstance(back.beta, torch.nn.Parameter)
        for name in ("gamma", "beta", "mean", "variance"):
            np.testing.assert_array_equal(getattr(back, name).detach().float().numpy(),
                                          getattr(saved, name).detach().half().float().numpy())


def test_best_npz_is_the_epoch_the_detector_metric_prefers_not_the_lowest_loss(tmp_path):
    """Three epochs: the validation loss keeps falling, the held-out recall peaks at epoch 2 and
    epoch 3 only ties it. best.npz is epoch 2 (a tie keeps the earlier epoch), best_loss.npz is
    epoch 3, and results.csv carries both the losses and the metric of every scored epoch."""
    args, selection, calls = scripted_run(tmp_path, ["--epochs", "3"], [0.6, 0.4, 0.3], [0.5, 0.3, 0.2], [0.80, 0.90, 0.90])
    assert calls == [args.run_dir / "last.npz"] * 3
    assert (selection.best_epoch, selection.best_loss_epoch) == (2, 3) and selection.best_loss == 0.2
    assert selection.best_metrics.recall == 0.90 and selection.best_path == args.run_dir / "best.npz"
    rows = results_of(args.run_dir)
    assert [row["epoch"] for row in rows] == ["1", "2", "3"] and list(rows[0]) == list(training.RESULTS_COLUMNS)
    assert [float(row["val_loss"]) for row in rows] == [0.5, 0.3, 0.2]
    assert [float(row["recall"]) for row in rows] == [0.80, 0.90, 0.90] and rows[1]["visible"] == "100"
    assert all(row["score_seconds"] != "" and row["seconds"] != "" for row in rows)
    # the files are the epochs they claim (the scripted epoch counts in the first bias)
    assert epoch_of(args.run_dir / "best.npz") == 2 and epoch_of(args.run_dir / "best_loss.npz") == 3
    assert epoch_of(args.run_dir / "last.npz") == 3


def epoch_of(weights: Path) -> int:
    return int(load_model(weights, torch.device("cpu")).convs[0].bias[0])


def test_score_every_leaves_unscored_epochs_blank_and_always_scores_the_last(tmp_path):
    args, selection, calls = scripted_run(tmp_path, ["--epochs", "3", "--score-every", "2"], [0.6, 0.4, 0.3],
                                          [0.5, 0.3, 0.2], [0.95, 0.90])
    assert len(calls) == 2 and selection.best_epoch == 2                 # epochs 2 and 3 scored; 2 preferred
    assert epoch_of(args.run_dir / "best.npz") == 2 and epoch_of(args.run_dir / "last.npz") == 3
    rows = results_of(args.run_dir)
    assert rows[0]["recall"] == "" and rows[0]["score_seconds"] == "" and rows[0]["val_loss"] != ""
    assert rows[1]["recall"] == "0.950000" and rows[2]["recall"] == "0.900000"
    assert training.main(["--score-every", "0", "--self-test"]) == 0        # the self-test needs no run
    assert training.main(["--score-every", "0"]) == 1
    assert training.main(["--epochs", "0"]) == 1                            # no epoch: no best.npz to score or save


def test_resume_continues_an_interrupted_run_to_the_uninterrupted_result(tmp_path):
    """Four epochs in one go against two, an interruption (a results row the checkpoint never saw)
    and --resume for the other two: the same best epochs, the same results.csv (one row per
    epoch, the duplicated row written once) and the same files, with checkpoint.pt at epoch 4."""
    losses, val_losses, scored = [0.6, 0.4, 0.3, 0.35], [0.5, 0.3, 0.2, 0.25], [0.80, 0.90, 0.90, 0.85]
    (tmp_path / "whole").mkdir()
    whole_args, whole, _ = scripted_run(tmp_path / "whole", ["--epochs", "4"], losses, val_losses, scored)
    (tmp_path / "parts").mkdir()
    args, first, _ = scripted_run(tmp_path / "parts", ["--epochs", "2"], losses, val_losses[:2], scored[:2])
    assert first.best_epoch == 2 and first.best_loss_epoch == 2
    checkpoint = torch.load(args.run_dir / "checkpoint.pt", weights_only=True)
    assert checkpoint["epoch"] == 2 and checkpoint["best_epoch"] == 2 and checkpoint["best_metrics"]["recall"] == 0.90
    assert checkpoint["best_loss_epoch"] == 2 and checkpoint["best_loss"] == 0.3
    assert checkpoint["optimizer"]["state"] and "momentum_buffer" in checkpoint["optimizer"]["state"][0]
    assert float(checkpoint["model"]["convs.0.bias"][0]) == 2.0 and checkpoint["scaler"] == {}
    with (args.run_dir / "results.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(training.results_row(3, 0.1, 0.1, 1.0, None, None))
        # ^ the interruption fell between epoch 3's row and its checkpoint
    args, resumed, calls = scripted_run(tmp_path / "parts", ["--epochs", "4"], losses, val_losses[2:], scored[2:], resume=True)
    assert args.resume and calls == [args.run_dir / "last.npz"] * 2
    assert resumed._replace(best_path=whole.best_path) == whole
    assert (resumed.best_epoch, resumed.best_loss_epoch, resumed.best_loss) == (2, 3, 0.2)
    rows, whole_rows = results_of(args.run_dir), results_of(whole_args.run_dir)
    assert [row["epoch"] for row in rows] == ["1", "2", "3", "4"]
    timings = ("seconds", "score_seconds")
    assert [{k: v for k, v in row.items() if k not in timings} for row in rows] == \
        [{k: v for k, v in row.items() if k not in timings} for row in whole_rows]
    for name in ("best.npz", "best_loss.npz", "last.npz"):
        assert epoch_of(args.run_dir / name) == epoch_of(whole_args.run_dir / name)
    assert epoch_of(args.run_dir / "best.npz") == 2 and epoch_of(args.run_dir / "last.npz") == 4
    assert torch.load(args.run_dir / "checkpoint.pt", weights_only=True)["epoch"] == 4
    assert not (args.run_dir / "checkpoint.pt.tmp").exists()


def test_resume_at_the_last_epoch_runs_nothing_and_a_lower_epoch_count_is_refused(tmp_path):
    (tmp_path / "run").mkdir()
    scripted_run(tmp_path / "run", ["--epochs", "2"], [0.6, 0.4], [0.5, 0.3], [0.80, 0.90])
    args, selection, calls = scripted_run(tmp_path / "run", ["--epochs", "2"], [0.6, 0.4], [], [], resume=True)
    assert calls == [] and (selection.best_epoch, selection.best_loss_epoch, selection.best_loss) == (2, 2, 0.3)
    assert selection.best_metrics.recall == 0.90 and selection.best_path == args.run_dir / "best.npz"
    assert [row["epoch"] for row in results_of(args.run_dir)] == ["1", "2"] and epoch_of(args.run_dir / "last.npz") == 2
    with pytest.raises(ValueError, match="below the checkpoint's epoch 2"):
        scripted_run(tmp_path / "run", ["--epochs", "1"], [0.6, 0.4], [], [], resume=True)


NOTHING = (0.0, 0.0, 0.0)                 # a scored epoch that detected nothing: recall, wrong, false alarm
EARLY_SCRATCH = (0.050, 0.276, 0.236)     # the Colab scratch run's epoch 4: finds a little, costs more
R0_EPOCH_1, R0_EPOCH_9 = (0.938, 0.011, 0.305), (0.936, 0.007, 0.230)


def test_a_zero_detection_first_epoch_is_overtaken_by_a_later_epoch_that_finds_more_than_it_costs(tmp_path):
    """The scratch run's failure: the first scored epoch detected nothing and, as the incumbent,
    promotion_improves let nothing beat it. By epoch_score it is 0, an epoch costing more than it
    finds ranks below it, and the first epoch worth something takes best.npz."""
    args, selection, _ = scripted_run(tmp_path, ["--epochs", "3"], [0.6, 0.4, 0.3], [0.5, 0.3, 0.2],
                                      [NOTHING, EARLY_SCRATCH, (0.5, 0.05, 0.1)])
    assert not promotion_improves(training.metrics_of(0.5, 0.05, 0.1), training.metrics_of(*NOTHING))
    assert selection.best_epoch == 3 and selection.best_metrics == training.metrics_of(0.5, 0.05, 0.1)
    assert epoch_of(args.run_dir / "best.npz") == 3
    assert [row["recall"] for row in results_of(args.run_dir)] == ["0.000000", "0.050000", "0.500000"]


def test_a_little_less_recall_with_far_fewer_false_alarms_wins_and_a_tie_keeps_the_earlier_epoch(tmp_path):
    """R0's question: epoch 1 had the most recall, epoch 9 far fewer false alarms. The scalar rule
    takes the false-alarm drop, and the repeat of that result at epoch 3 does not move best.npz."""
    args, selection, _ = scripted_run(tmp_path, ["--epochs", "3"], [0.6, 0.4, 0.3], [0.5, 0.3, 0.2],
                                      [R0_EPOCH_1, R0_EPOCH_9, R0_EPOCH_9])
    assert selection.best_epoch == 2 and epoch_of(args.run_dir / "best.npz") == 2
    assert selection.best_metrics == training.metrics_of(*R0_EPOCH_9)


def test_every_scored_epochs_weights_are_kept_unless_no_keep_epochs(tmp_path):
    """epochs/epoch_NNN.npz for the scored epochs only (2 and 3 of a --score-every 2 run of three),
    each the file its name says; --no-keep-epochs writes none."""
    (tmp_path / "kept").mkdir(), (tmp_path / "dropped").mkdir()
    args, _, _ = scripted_run(tmp_path / "kept",["--epochs", "3", "--score-every", "2"], [0.6, 0.4, 0.3],
                              [0.5, 0.3, 0.2], [0.9, 0.8])
    kept = args.run_dir / training.EPOCHS_DIR
    assert sorted(path.name for path in kept.iterdir()) == ["epoch_002.npz", "epoch_003.npz"]
    assert epoch_of(kept / "epoch_002.npz") == 2 and epoch_of(kept / "epoch_003.npz") == 3
    assert training.epoch_weights(args.run_dir, 12) == kept / "epoch_012.npz"
    args, _, _ = scripted_run(tmp_path / "dropped", ["--epochs", "3", "--no-keep-epochs"], [0.6, 0.4, 0.3],
                              [0.5, 0.3, 0.2], [0.9, 0.8, 0.7])
    assert not args.keep_epochs and not (args.run_dir / training.EPOCHS_DIR).exists()
    assert epoch_of(args.run_dir / "best.npz") == 1


def test_resume_derives_the_best_epoch_from_results_csv_not_from_the_checkpoint(tmp_path, capsys):
    """A checkpoint whose stored best is another rule's choice (here epoch 1 by hand, over a
    results.csv that ranks epoch 2 first) is corrected on resume: the best epoch comes from the
    rows, best.npz is made the kept epoch_002.npz, and the run carries on from there."""
    (tmp_path / "run").mkdir()
    args, first, _ = scripted_run(tmp_path / "run", ["--epochs", "3"], [0.6, 0.4, 0.3, 0.2], [0.5, 0.3, 0.2],
                                  [0.8, 0.9, 0.7])
    assert first.best_epoch == 2
    checkpoint = args.run_dir / training.CHECKPOINT
    state = torch.load(checkpoint, weights_only=True)
    state["best_epoch"], state["best_metrics"] = 1, training.metrics_of(0.8, 0.0)._asdict()
    torch.save(state, checkpoint)
    shutil.copy(args.run_dir / "last.npz", args.run_dir / "best.npz")              # the other rule's best.npz, epoch 3
    assert training.best_scored_epoch(args.run_dir / "results.csv") == (2, training.metrics_of(0.9, 0.0))
    capsys.readouterr()
    args, resumed, calls = scripted_run(tmp_path / "run", ["--epochs", "4"], [0.6, 0.4, 0.3, 0.2], [0.1], [0.85],
                                        resume=True)
    assert calls == [args.run_dir / "last.npz"] and resumed.best_epoch == 2
    assert resumed.best_metrics == training.metrics_of(0.9, 0.0) and epoch_of(args.run_dir / "best.npz") == 2
    assert "[train] best epoch so far by results.csv: 2 (the checkpoint said 1); best.npz is now that epoch" in capsys.readouterr().out
    assert torch.load(checkpoint, weights_only=True)["best_epoch"] == 2
    assert training.best_scored_epoch(tmp_path / "none.csv") == (0, None)


def test_resume_of_a_run_anchored_on_a_zero_epoch_without_kept_weights_takes_the_first_real_epoch(tmp_path, capsys):
    """The run on Colab: checkpoint from the incumbent rule, best epoch 2 with zero metrics, no
    epochs/ folder. Resumed, the zero epoch stays best (results.csv says so, and there is no kept
    file to copy) until a later epoch scores more than it costs, which then takes best.npz."""
    (tmp_path / "run").mkdir()
    args, first, _ = scripted_run(tmp_path / "run", ["--epochs", "2", "--score-every", "2"], [0.6, 0.4, 0.3, 0.35, 0.3, 0.25],
                                  [0.5, 0.3], [NOTHING])
    assert first.best_epoch == 2 and epoch_of(args.run_dir / "best.npz") == 2
    shutil.rmtree(args.run_dir / training.EPOCHS_DIR)
    args, resumed, _ = scripted_run(tmp_path / "run", ["--epochs", "4", "--score-every", "2"],
                                    [0.6, 0.4, 0.3, 0.35, 0.3, 0.25], [0.2, 0.25], [EARLY_SCRATCH], resume=True)
    assert resumed.best_epoch == 2 and epoch_of(args.run_dir / "best.npz") == 2        # epoch 4 costs more than it finds
    assert "best epoch so far by results.csv" not in capsys.readouterr().out
    args, resumed, _ = scripted_run(tmp_path / "run", ["--epochs", "6", "--score-every", "2"],
                                    [0.6, 0.4, 0.3, 0.35, 0.3, 0.25], [0.2, 0.2], [(0.5, 0.05, 0.1)], resume=True)
    assert resumed.best_epoch == 6 and epoch_of(args.run_dir / "best.npz") == 6
    assert [row["recall"] for row in results_of(args.run_dir)] == ["", "0.000000", "", "0.050000", "", "0.500000"]


def test_pick_winner_records_a_scratch_run_whose_every_epoch_detected_nothing(tmp_path):
    """Rates of zero and NaN pixel distances (no detection, no distance): winner.json is written
    with null for the distances, the best epoch is the winner and nothing crashes."""
    args, selection, _ = scripted_run(tmp_path, ["--epochs", "1", "--from-scratch", "--save", str(tmp_path / "save.npz")],
                                      [0.6], [0.5], [NOTHING])
    nothing = DetectorMetrics(100, 0.0, 0.0, 1.0, 100, 0.0, math.nan, math.nan)
    training.pick_winner(args, [], ["video10"], selection._replace(best_metrics=nothing), torch.device("cpu"))
    winner = json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))
    assert Path(winner["winner"]) == args.run_dir / "best.npz" and winner["best_epoch"] == 1
    assert winner["metrics"]["recall"] == 0.0 and winner["metrics"]["median_px"] is None
    assert epoch_of(tmp_path / "save.npz") == 1


def test_keep_results_through_drops_later_rows_and_fragments(tmp_path):
    """A results.csv as an interruption can leave it: whole rows past the checkpoint's epoch, and
    a last row cut short (fewer columns, or an epoch number missing its last digit, which must not
    pass as an earlier epoch's row) all go; a missing file becomes the header alone."""
    results = tmp_path / "results.csv"
    rows = [training.results_row(epoch, 0.5, 0.25, 90.0, None, None) for epoch in range(1, 11)]
    with results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(training.RESULTS_COLUMNS)
        writer.writerows(rows[:9])
        handle.write("1")                                    # epoch 10's row, cut after its first character
    training.keep_results_through(results, 9)
    assert [row["epoch"] for row in results_of(tmp_path)] == [str(epoch) for epoch in range(1, 10)]
    with results.open("a", encoding="utf-8", newline="") as handle:
        handle.write(",".join(rows[9][:4]) + "\n")           # epoch 10's row, cut mid-way
    training.keep_results_through(results, 7)
    assert [row["epoch"] for row in results_of(tmp_path)] == [str(epoch) for epoch in range(1, 8)]
    results.unlink()
    training.keep_results_through(results, 0)
    assert results.read_text(encoding="utf-8").splitlines() == [",".join(training.RESULTS_COLUMNS)]


def test_a_checkpoint_from_elsewhere_is_refused_in_one_line(tmp_path):
    """A readable torch file without the trainer's fields (a bare state dict, say) is the ValueError
    --resume reports, not a KeyError traceback."""
    model = fresh_model(torch.device("cpu"))
    torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
    with pytest.raises(ValueError, match="not a checkpoint this trainer wrote"):
        training.read_checkpoint(tmp_path / "checkpoint.pt", model, None, None)


def test_resume_needs_a_checkpoint_and_the_run_it_belongs_to(tmp_path, capsys):
    """--resume on a folder without checkpoint.pt, or with one from a run of other options, is a
    one-line error naming what is missing or which config.json field differs, before any cache
    or training; the run's own outcomes (epochs, the hard-negative count) may differ."""
    videos, labels = workspace(tmp_path, ["a", "b"])
    run_dir = tmp_path / "run"
    argv = ["--videos", str(videos), "--labels", str(labels), "--data-dir", str(tmp_path / "cache"), "--run-dir", str(run_dir)]
    assert training.main([*argv, "--resume"]) == 1
    assert "[error] --resume needs a checkpoint.pt in" in capsys.readouterr().err
    run_dir.mkdir()
    (run_dir / "checkpoint.pt").write_bytes(b"not a checkpoint")
    assert training.main([*argv, "--resume"]) == 1
    assert "readable config.json" in capsys.readouterr().err
    config = training.run_config(training.build_parser().parse_args(argv), ["a", "b"], ["b"], {})
    config["epochs"], config["lr"] = 3, 0.3
    del config["lr_schedule"]                      # a run recorded before the schedule existed: its lr is the rate
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert training.main([*argv, "--resume"]) == 1
    assert "[error] --resume would continue a different run: config.json has lr=0.3, this run lr=1.0" in capsys.readouterr().err
    assert not (tmp_path / "cache").exists()
    training.same_run(config, {**config, "epochs": 5, "false_alarm_units": 12})
    with pytest.raises(ValueError, match="validation_clips"):
        training.same_run(config, {**config, "validation_clips": ["a"]})
    with pytest.raises(ValueError, match="readable checkpoint.pt"):
        training.read_checkpoint(run_dir / "checkpoint.pt", fresh_model(torch.device("cpu")), None, None)


def test_resume_through_main_keeps_the_run_and_raises_its_epochs(tmp_path, capsys):
    """The whole trainer on two tiny clips, CPU, one step per epoch, the CUDA-only scoring stubbed:
    one epoch, then --resume with --epochs 2 continues the same run (results.csv rows 1 and 2,
    config.json's epochs raised, the winner picked), and a third run without --resume starts over
    after saying what it dropped."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "labels").mkdir()
    for clip in ("a", "b"):
        write_clip(tmp_path / "videos" / f"{clip}.avi", 8, SIZE, 30.0)
        write_labels(tmp_path / "labels" / f"{clip}_ball.csv", {f: (1.0 + f, 1.0) for f in range(7)}, *SIZE)
    run_dir = tmp_path / "run"
    argv = ["--videos", str(tmp_path / "videos"), "--labels", str(tmp_path / "labels"), "--data-dir", str(tmp_path / "cache"),
            "--run-dir", str(run_dir), "--save", str(tmp_path / "save.npz"), "--device", "cpu", "--workers", "0",
            "--from-scratch", "--max-train-steps", "1", "--max-val-steps", "1", "--prep-workers", "1"]
    recalls = iter([0.5, 0.6, 0.7])
    with patch.object(training, "score_weights", side_effect=lambda *_: training.metrics_of(next(recalls), 0.0)):
        assert training.main([*argv, "--epochs", "1"]) == 0
        assert [row["epoch"] for row in results_of(run_dir)] == ["1"] and "[train] epoch 1:" in capsys.readouterr().out
        assert training.main([*argv, "--resume", "--epochs", "2"]) == 0
        out = capsys.readouterr().out
        assert f"[train] resuming {run_dir} after epoch 1 of 2" in out and "[train] epoch 2:" in out and "[train] epoch 1:" not in out
        assert [row["epoch"] for row in results_of(run_dir)] == ["1", "2"]
        assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["epochs"] == 2
        winner = json.loads((run_dir / "winner.json").read_text(encoding="utf-8"))
        assert winner["best_epoch"] == 2 and winner["metrics"]["recall"] == 0.6 and (tmp_path / "save.npz").exists()
        assert training.main([*argv, "--resume"]) == 0                 # --epochs defaults to the run's: nothing left to train
        out = capsys.readouterr().out
        assert "after epoch 2 of 2: the epochs are done" in out and "[train] epoch" not in out
        assert training.main([*argv, "--epochs", "1"]) == 0
        out = capsys.readouterr().out
        assert "[train] checkpoint.pt found in" in out and "--resume would have continued it" in out
        assert [row["epoch"] for row in results_of(run_dir)] == ["1"]
        assert torch.load(run_dir / "checkpoint.pt", weights_only=True)["epoch"] == 1


def test_lower_learning_rate_changes_only_the_rate_and_keeps_adadeltas_accumulators(capsys):
    """One Adadelta step leaves a squared-gradient and an update accumulator per parameter; the
    drop sets every param group's lr, leaves that state untouched, prints its one line and
    appends the entry to the schedule it was given."""
    model = fresh_model(torch.device("cpu"))
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-7)
    for parameter in model.parameters():
        parameter.grad = torch.rand_like(parameter)
    optimizer.step()
    before = {id(p): {k: v.clone() for k, v in state.items()} for p, state in optimizer.state.items()}
    assert before and all("square_avg" in state and "acc_delta" in state for state in before.values())
    schedule = training.lower_learning_rate(optimizer, [{"epoch": 1, "lr": 1.0}], 0.1, 41)
    assert schedule == [{"epoch": 1, "lr": 1.0}, {"epoch": 41, "lr": 0.1}]
    assert [group["lr"] for group in optimizer.param_groups] == [0.1]
    assert {id(p) for p in optimizer.state} == set(before)
    for parameter, state in optimizer.state.items():
        assert all(torch.equal(value, before[id(parameter)][key]) for key, value in state.items())
    assert capsys.readouterr().out == "[train] learning rate 1 -> 0.1 from epoch 41\n"
    assert training.lr_schedule_of({"lr": 0.3}) == [{"epoch": 1, "lr": 0.3}]          # a config.json from before the schedule
    assert training.lr_schedule_of({"lr": 0.1, "lr_schedule": schedule}) == schedule


def test_same_run_reads_the_learning_rate_through_the_schedule():
    """The rate in force is the schedule's last entry: a resume at it passes with or without a
    drop pending elsewhere, a different rate is refused unless --lr-drop lowers it, and the flag
    refuses a raise and the same rate (nothing to drop)."""
    config = training.run_config(training.build_parser().parse_args(["--lr", "0.1"]), ["a", "b"], ["b"], {},
                                 lr_schedule=[{"epoch": 1, "lr": 1.0}, {"epoch": 41, "lr": 0.1}])
    assert config["lr"] == 0.1 and config["lr_schedule"][-1] == {"epoch": 41, "lr": 0.1}
    training.same_run(config, {**config, "epochs": 60})
    with pytest.raises(ValueError, match=r"config.json has lr=0.1, this run lr=1.0 \(--lr-drop lowers it on purpose\)"):
        training.same_run(config, {**config, "lr": 1.0})
    training.same_run(config, {**config, "lr": 0.01}, lr_drop=True)
    with pytest.raises(ValueError, match="--lr-drop lowers the learning rate: config.json has lr=0.1, this run lr=0.5"):
        training.same_run(config, {**config, "lr": 0.5}, lr_drop=True)
    with pytest.raises(ValueError, match="this run lr=0.1, the same, so there is nothing to drop"):
        training.same_run(config, config, lr_drop=True)
    with pytest.raises(ValueError, match="validation_clips"):                     # the other fields still have to agree
        training.same_run(config, {**config, "lr": 0.01, "validation_clips": ["a"]}, lr_drop=True)
    older = {**config, "lr": 0.3}
    del older["lr_schedule"]                                                      # written before the schedule existed
    training.same_run(older, {**config, "lr": 0.3})
    training.same_run(older, {**config, "lr": 0.1}, lr_drop=True)
    assert training.run_config(training.build_parser().parse_args([]), [], [], {})["lr_schedule"] == [{"epoch": 1, "lr": 1.0}]


def test_lr_drop_through_main_records_the_schedule_and_the_lowered_rate_resumes_plainly(tmp_path, capsys):
    """The whole trainer on two tiny clips: one epoch at lr 1.0, then --resume --lr-drop --lr 0.1
    carries on at 0.1 (the checkpoint's param group says so, Adadelta's accumulators are kept) and
    config.json records both rates; a resume at 0.1 without the flag is accepted, a second drop
    appends a third entry, and a raise, the same rate with the flag, the original rate without it
    and the flag without --resume are each refused in one line. An all-clips run following the
    recipe sees the schedule and says it is not replayed."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "labels").mkdir()
    for clip in ("a", "b"):
        write_clip(tmp_path / "videos" / f"{clip}.avi", 8, SIZE, 30.0)
        write_labels(tmp_path / "labels" / f"{clip}_ball.csv", {f: (1.0 + f, 1.0) for f in range(7)}, *SIZE)
    run_dir = tmp_path / "run"
    argv = ["--videos", str(tmp_path / "videos"), "--labels", str(tmp_path / "labels"), "--data-dir", str(tmp_path / "cache"),
            "--run-dir", str(run_dir), "--save", str(tmp_path / "save.npz"), "--device", "cpu", "--workers", "0",
            "--from-scratch", "--max-train-steps", "1", "--max-val-steps", "1", "--prep-workers", "1"]
    schedule_of = lambda: json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["lr_schedule"]
    with patch.object(training, "score_weights", return_value=training.metrics_of(0.5, 0.0)):
        assert training.main([*argv, "--epochs", "1"]) == 0
        assert schedule_of() == [{"epoch": 1, "lr": 1.0}]
        state = torch.load(run_dir / "checkpoint.pt", weights_only=True)["optimizer"]
        assert state["param_groups"][0]["lr"] == 1.0 and "square_avg" in state["state"][0]
        capsys.readouterr()
        assert training.main([*argv, "--resume", "--epochs", "2", "--lr", "0.1", "--lr-drop"]) == 0
        out = capsys.readouterr().out
        assert f"[train] resuming {run_dir} after epoch 1 of 2\n[train] learning rate 1 -> 0.1 from epoch 2\n" in out
        assert "[train] epoch 2:" in out
        assert schedule_of() == [{"epoch": 1, "lr": 1.0}, {"epoch": 2, "lr": 0.1}]
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert config["lr"] == 0.1 and config["epochs"] == 2
        state = torch.load(run_dir / "checkpoint.pt", weights_only=True)["optimizer"]
        assert state["param_groups"][0]["lr"] == 0.1 and "square_avg" in state["state"][0] and "acc_delta" in state["state"][0]
        assert training.main([*argv, "--resume", "--epochs", "3", "--lr", "0.1"]) == 0       # the lowered rate needs no flag
        assert "learning rate" not in capsys.readouterr().out
        assert schedule_of() == [{"epoch": 1, "lr": 1.0}, {"epoch": 2, "lr": 0.1}]
        assert [row["epoch"] for row in results_of(run_dir)] == ["1", "2", "3"]
        assert training.main([*argv, "--resume", "--epochs", "4", "--lr", "0.01", "--lr-drop"]) == 0
        assert "[train] learning rate 0.1 -> 0.01 from epoch 4" in capsys.readouterr().out
        assert schedule_of() == [{"epoch": 1, "lr": 1.0}, {"epoch": 2, "lr": 0.1}, {"epoch": 4, "lr": 0.01}]
        for flags, message in ((["--lr", "0.5", "--lr-drop"], "--lr-drop lowers the learning rate: config.json has lr=0.01, "
                                                               "this run lr=0.5; it is not raised on a resume"),
                               (["--lr", "0.01", "--lr-drop"], "--lr-drop lowers the learning rate: config.json has lr=0.01, "
                                                                "this run lr=0.01, the same, so there is nothing to drop"),
                               (["--lr", "1.0"], "--resume would continue a different run: config.json has lr=0.01, "
                                                 "this run lr=1.0 (--lr-drop lowers it on purpose)")):
            assert training.main([*argv, "--resume", "--epochs", "5", *flags]) == 1
            assert f"[error] {message}" in capsys.readouterr().err
        assert schedule_of()[-1] == {"epoch": 4, "lr": 0.01} and torch.load(run_dir / "checkpoint.pt", weights_only=True)["epoch"] == 4
    assert training.main([*argv, "--epochs", "5", "--lr", "0.1", "--lr-drop"]) == 1
    assert "[error] --lr-drop goes with --resume" in capsys.readouterr().err
    recipe = training.read_selection_run(run_dir)["recipe"]
    assert recipe["lr"] == 0.01 and recipe["lr_schedule"] == schedule_of()
    all_clips = ["--videos", str(tmp_path / "videos"), "--labels", str(tmp_path / "labels"), "--data-dir", str(tmp_path / "cache"),
                 "--run-dir", str(tmp_path / "all"), "--save", str(tmp_path / "all.npz"), "--device", "cpu", "--workers", "0",
                 "--from-scratch", "--max-train-steps", "1", "--prep-workers", "1", "--all-clips", "--selection-run", str(run_dir),
                 "--epochs", "1"]
    assert training.main(all_clips) == 0
    out = capsys.readouterr().out
    assert ("[train] the selection run lowered its learning rate (1 from epoch 1, 0.1 from epoch 2, 0.01 from epoch 4); "
            "this run trains at --lr 1 throughout, the schedule is not replayed") in out
    winner = json.loads((tmp_path / "all" / "winner.json").read_text(encoding="utf-8"))
    assert winner["selection_run"]["recipe"]["lr_schedule"] == schedule_of()


def test_pick_winner_reuses_the_best_epochs_metrics_and_records_how_it_chose(tmp_path, capsys):
    args, selection, _ = scripted_run(tmp_path, ["--epochs", "2", "--save", str(tmp_path / "save.npz")],
                                      [0.6, 0.4], [0.5, 0.3], [0.90, 0.85])
    clips = [("video1", Path("v1"), Path("c1"), "custom"), ("video10", Path("v10"), Path("c10"), "custom")]
    with patch.object(training, "score_weights", return_value=training.metrics_of(0.80, 0.0)) as baseline:
        training.pick_winner(args, clips, ["video10"], selection, torch.device("cpu"))
    assert baseline.call_args.args[:2] == (args.weights, clips[1:])      # the starting weights, on the held-out clips only
    winner = json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))
    assert Path(winner["winner"]) == args.run_dir / "best.npz" and winner["selected_by"] == "detector_metric"
    assert winner["best_epoch"] == 1 and winner["best_loss_epoch"] == 2 and winner["validation_clips"] == ["video10"]
    assert winner["metrics"]["recall"] == 0.90 and winner["starting_weights"]["metrics"]["recall"] == 0.80
    assert epoch_of(tmp_path / "save.npz") == 1
    assert "promote it with" in capsys.readouterr().out

    with patch.object(training, "score_weights", return_value=training.metrics_of(0.95, 0.0)):
        training.pick_winner(args, clips, ["video10"], selection, torch.device("cpu"))
    winner = json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))
    assert Path(winner["winner"]) == args.weights and winner["metrics"]["recall"] == 0.95
    assert (tmp_path / "save.npz").read_bytes() == args.weights.read_bytes()
    assert "stayed ahead" in capsys.readouterr().out
    args.from_scratch = True
    with patch.object(training, "score_weights") as never:
        training.pick_winner(args, clips, ["video10"], selection, torch.device("cpu"))
    assert not never.called and "starting_weights" not in json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))


def test_all_clips_runs_every_epoch_unscored_and_records_the_recipe(tmp_path, capsys):
    """Nothing held out: no validation loss, no scoring, last.npz is the result, and winner.json
    says so (selected_by recipe) with the selection run's details copied in; --epochs defaults
    to that run's best epoch."""
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    (selection_dir / "winner.json").write_text(json.dumps({"winner": "best.npz", "selected_by": "detector_metric", "best_epoch": 2,
                                                           "metrics": {"recall": 0.9}, "starting_weights": {"metrics": {"recall": 0.8}}}))
    (selection_dir / "config.json").write_text(json.dumps({"augment": ["basic", "translate"], "oversample": {}, "oversample_absent": 3.0,
                                                           "oversample_false_alarms": 1.0, "epoch_units": None, "batch_size": 2,
                                                           "lr": 1.0, "seed": 7, "holdout": "mixed_v1", "validation_clips": ["video1"]}))
    selection_run = training.read_selection_run(selection_dir)
    assert selection_run["best_epoch"] == 2 and selection_run["metrics"] == {"recall": 0.9}
    assert selection_run["recipe"]["augment"] == ["basic", "translate"] and selection_run["recipe"]["holdout"] == "mixed_v1"
    assert selection_run["recipe"]["lr_schedule"] == [{"epoch": 1, "lr": 1.0}]       # a run from before the schedule: its lr
    with pytest.raises(ValueError, match="no readable winner.json"):
        training.read_selection_run(tmp_path / "nowhere")

    args, selection, calls = scripted_run(tmp_path, ["--epochs", "2", "--all-clips", "--save", str(tmp_path / "save.npz")],
                                          [0.6, 0.4], [], [], scoring=False, validate=False)
    assert calls == [] and selection.best_metrics is None and selection.best_epoch == 2
    assert selection.best_path == args.run_dir / "last.npz" and not (args.run_dir / "best.npz").exists()
    assert not (args.run_dir / "best_loss.npz").exists() and epoch_of(args.run_dir / "last.npz") == 2
    rows = results_of(args.run_dir)
    assert [row["val_loss"] for row in rows] == ["", ""] and [row["recall"] for row in rows] == ["", ""]
    assert "val=" not in capsys.readouterr().out
    training.record_recipe_result(args, selection, selection_run)
    winner = json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))
    assert winner["selected_by"] == "recipe" and winner["epochs"] == 2 and winner["metrics"] is None
    assert Path(winner["winner"]) == args.run_dir / "last.npz" and winner["selection_run"] == selection_run
    assert epoch_of(tmp_path / "save.npz") == 2 and "nothing was held out" in capsys.readouterr().out

    parser = training.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--all-clips", "--holdout", "mixed_v1"])
    assert parser.parse_args([]).epochs is None and training.DEFAULT_EPOCHS == 10
    args = parser.parse_args(["--all-clips", "--selection-run", str(selection_dir)])
    assert training.run_config(args, [], [], {})["all_clips"] and training.run_config(args, [], [], {})["selection_run"] == str(selection_dir)
    assert training.main(["--selection-run", str(selection_dir)]) == 1
    assert "goes with --all-clips" in capsys.readouterr().err


def test_the_seed_is_recorded_and_defaults_to_the_fixed_one():
    parser = training.build_parser()
    assert parser.parse_args([]).seed == training.SEED == 7
    args = parser.parse_args(["--seed", "11", "--score-every", "2"])
    config = training.run_config(args, [], [], {})
    assert config["seed"] == 11 and config["score_every"] == 2


def test_augment_is_a_set_of_names_and_the_hard_negative_weights_are_recorded(capsys):
    parser = training.build_parser()
    assert parser.parse_args([]).augment == ["basic"]
    args = parser.parse_args(["--augment", "translate", "basic", "reverse", "--oversample-absent", "3",
                              "--oversample-false-alarms", "2.5"])
    config = training.run_config(args, [], [], {}, flagged=42)
    assert config["augment"] == ["basic", "reverse", "translate"] and config["oversample_absent"] == 3.0
    assert config["oversample_false_alarms"] == 2.5 and config["false_alarm_units"] == 42
    assert training.run_config(parser.parse_args([]), [], [], {})["false_alarm_units"] == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--augment", "rotate"])
    assert "invalid choice" in capsys.readouterr().err


def test_metric_counts_localization_errors():
    stats = metric_probe(5, {i: [100, 100] if i == 0 else [120, 100] for i in range(5)})
    assert stats.visible == 5
    assert stats.recall == .2


def test_metric_infers_tail():
    stats = metric_probe(7, {i: [100, 100] for i in range(7)})
    assert stats.visible == 7
    assert stats.recall == 1.


def test_false_alarms_block_promotion():
    assert not promotion_improves(training.metrics_of(.91, 0., false_alarm=1.), training.metrics_of(.90, 0., false_alarm=0.))
