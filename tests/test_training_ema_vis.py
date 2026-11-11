"""The trainer's --ema and --vis-head plumbing (finetune/training/train_gridtracknet.py): the averaged
weights' files, columns and best epoch through fit, the checkpoint round trip on resume, the older
results.csv layout kept on resume, and the recipe fields. Scripted epochs on a tiny CPU network."""
import csv
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from finetune.training import train_gridtracknet as training
from finetune.training.loop import EMA, Validation, fresh_model
from tennis_tracker.gridtracknet import load_model
from _workspace_helpers import blank_template


def ema_run(tmp_path: Path, argv, scored_live, scored_ema, resume=False, vis=False):
    """``fit`` with --ema on a fresh CPU GridTrackNet: every "epoch" adds one to the first bias
    (so a saved file is its epoch), the EMA is updated as the real loop would, the held-out metric
    answers `scored_live` for the live weights and `scored_ema` for the averaged ones (per scored
    call, a recall), and the validation pass answers a loss and, with `vis`, head metrics."""
    model = fresh_model(torch.device("cpu"))
    with torch.no_grad():
        model.convs[0].bias.zero_()
    template = blank_template(model, tmp_path / "template.npz")
    args = training.build_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--weights", str(template), "--ema",
                                               *(["--resume"] if resume else []), *argv])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = EMA(model, args.ema_decay)
    progress = training.START
    if resume:
        progress = training.read_checkpoint(args.run_dir / training.CHECKPOINT, model, optimizer, scaler, ema=ema)
    else:
        args.run_dir.mkdir()
    calls = {"live": [], "ema": []}

    def score(weights):
        kind = "ema" if weights.stem.endswith(training.EMA_SUFFIX) else "live"
        calls[kind].append(weights)
        answers = scored_ema if kind == "ema" else scored_live
        return training.metrics_of(answers[len(calls[kind]) - 1], 0.0)

    def train_epoch(model, loader, optimizer, scaler, device, epoch, epochs, max_steps, distiller=None, ema=None,
                    vis_weight=0.0, vis_gap_weight=1.0, vis_only=False):
        with torch.no_grad():
            model.convs[0].bias += 1.0
        if ema is not None:
            ema.update(model)
        return 0.5

    def evaluate(model, loader, device, max_steps=None, visibility=False):
        return Validation(0.3, 0.8, 0.7) if visibility else Validation(0.3)
    with patch.object(training, "train_epoch", side_effect=train_epoch), patch.object(training, "evaluate", side_effect=evaluate):
        selection = training.fit(args, model, None, object(), optimizer, scaler, torch.device("cpu"), score, progress, None, ema)
    return args, selection, calls, ema


def rows_of(run_dir: Path):
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bias_of(weights: Path) -> float:
    return float(load_model(weights, torch.device("cpu")).convs[0].bias.detach()[0])


def test_an_ema_run_scores_both_networks_and_keeps_best_ema_by_its_own_ranking(tmp_path):
    """Three scored epochs: the live recall peaks at 2, the averaged at 3. best.npz is epoch 2 and
    stays the winner's path; best_ema.npz is epoch 3; last_ema.npz is written every epoch; the
    ema_* columns carry the averaged metrics and validation on the scored rows; the checkpoint
    carries the average and its best epoch."""
    args, selection, calls, ema = ema_run(tmp_path, ["--epochs", "3"], [0.80, 0.90, 0.85], [0.70, 0.88, 0.92])
    assert calls["live"] == [args.run_dir / "last.npz"] * 3 and calls["ema"] == [args.run_dir / "last_ema.npz"] * 3
    assert (selection.best_epoch, selection.best_ema_epoch) == (2, 3)
    assert selection.best_path == args.run_dir / "best.npz" and selection.best_ema_metrics.recall == 0.92
    assert ema.updates == 3
    assert (args.run_dir / "last_ema.npz").is_file() and (args.run_dir / "best_ema.npz").is_file()
    assert bias_of(args.run_dir / "best.npz") == 2.0 and bias_of(args.run_dir / "last.npz") == 3.0
    # the averaged bias is 0.001 of the way toward each epoch's live bias: never a whole epoch
    assert 0.0 < bias_of(args.run_dir / "best_ema.npz") < 0.01 and bias_of(args.run_dir / "best_ema.npz") == bias_of(args.run_dir / "last_ema.npz")
    kept = sorted(path.name for path in (args.run_dir / training.EPOCHS_DIR).iterdir())
    assert kept == ["epoch_001.npz", "epoch_001_ema.npz", "epoch_002.npz", "epoch_002_ema.npz", "epoch_003.npz", "epoch_003_ema.npz"]
    rows = rows_of(args.run_dir)
    assert list(rows[0]) == list(training.RESULTS_COLUMNS)
    assert [row["ema_recall"] for row in rows] == ["0.700000", "0.880000", "0.920000"]
    assert [row["recall"] for row in rows] == ["0.800000", "0.900000", "0.850000"]
    assert rows[0]["ema_val_loss"] == "0.30000000" and rows[0]["ema_visible"] == "100" and rows[0]["vis_auc"] == ""
    checkpoint = torch.load(args.run_dir / training.CHECKPOINT, weights_only=True)
    assert checkpoint["best_ema_epoch"] == 3 and checkpoint["best_ema_metrics"]["recall"] == 0.92
    assert checkpoint["ema"]["updates"] == 3 and float(checkpoint["ema"]["model"]["convs.0.bias"][0]) == pytest.approx(bias_of(args.run_dir / "last_ema.npz"), abs=1e-6)


def test_an_ema_run_resumes_with_the_average_and_its_best_epoch(tmp_path):
    """Two epochs, an interruption, two more on resume: the averaged weights continue from the
    checkpoint (not from a fresh copy), the best EMA epoch is re-derived from results.csv, and the
    outcome equals the uninterrupted four-epoch run."""
    live, averaged = [0.80, 0.90, 0.85, 0.86], [0.70, 0.88, 0.92, 0.91]
    (tmp_path / "whole").mkdir()
    whole_args, whole, _, whole_ema = ema_run(tmp_path / "whole", ["--epochs", "4"], live, averaged)
    (tmp_path / "parts").mkdir()
    args, first, _, _ = ema_run(tmp_path / "parts", ["--epochs", "2"], live[:2], averaged[:2])
    assert (first.best_epoch, first.best_ema_epoch) == (2, 2)
    args, resumed, calls, ema = ema_run(tmp_path / "parts", ["--epochs", "4"], live[2:], averaged[2:], resume=True)
    assert ema.updates == 4 and len(calls["ema"]) == 2
    assert resumed._replace(best_path=whole.best_path) == whole and (resumed.best_epoch, resumed.best_ema_epoch) == (2, 3)
    assert bias_of(args.run_dir / "last_ema.npz") == pytest.approx(bias_of(whole_args.run_dir / "last_ema.npz"), abs=1e-7)
    assert bias_of(args.run_dir / "best_ema.npz") == pytest.approx(bias_of(whole_args.run_dir / "epochs" / "epoch_003_ema.npz"))
    assert [row["ema_recall"] for row in rows_of(args.run_dir)] == [row["ema_recall"] for row in rows_of(whole_args.run_dir)]


def test_a_checkpoint_without_the_average_cannot_continue_an_ema_run(tmp_path):
    model = fresh_model(torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    training.write_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, scaler, training.Checkpoint(1, 1, None, 1, 0.5))
    plain = training.read_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, scaler)
    assert (plain.best_ema_epoch, plain.best_ema_metrics) == (0, None)          # an older checkpoint reads with the defaults
    with pytest.raises(ValueError, match="without --ema"):
        training.read_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, scaler, ema=EMA(model))


def test_the_visibility_metrics_reach_results_csv_and_the_epoch_line(tmp_path, capsys):
    args, selection, _, _ = ema_run(tmp_path, ["--epochs", "1", "--vis-head"], [0.8], [0.7], vis=True)
    row = rows_of(args.run_dir)[0]
    assert (row["vis_auc"], row["vis_bacc"], row["ema_vis_auc"], row["ema_vis_bacc"]) == ("0.800000", "0.700000", "0.800000", "0.700000")
    out = capsys.readouterr().out
    assert "visAUC=0.800" in out and "EMA recall=0.700" in out and "best EMA epoch 1" in out


def test_an_older_results_layout_is_kept_and_padded_on_resume(tmp_path):
    """A run begun before the EMA and visibility columns existed has shorter rows; a resume keeps
    them (padded, under this layout's header) rather than dropping every epoch it scored, and
    best_scored_epoch still reads them."""
    results = tmp_path / "results.csv"
    legacy = training.results_row(1, 0.5, 0.25, 90.0, 10.0, training.metrics_of(0.9, 0.01))[:len(training.LEGACY_RESULTS_COLUMNS)]
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(training.LEGACY_RESULTS_COLUMNS)
        writer.writerow(legacy)
        writer.writerow(["2", "0.4"])                                            # a fragment
    assert training.whole_row(legacy) == legacy + [""] * (len(training.RESULTS_COLUMNS) - len(legacy))
    assert training.whole_row(["2", "0.4"]) is None
    training.keep_results_through(results, 2)
    rows = rows_of(tmp_path)
    assert len(rows) == 1 and list(rows[0]) == list(training.RESULTS_COLUMNS) and rows[0]["recall"] == "0.900000"
    assert rows[0]["ema_recall"] == "" and training.best_scored_epoch(results) == (1, training.metrics_of(0.9, 0.01))
    assert training.best_scored_epoch(results, training.EMA_PREFIX) == (0, None)


def test_config_json_records_the_average_and_the_head_and_same_run_refuses_switching_them():
    off = training.build_parser().parse_args(["--arch", "stream_s4", "--from-scratch"])
    on = training.build_parser().parse_args(["--arch", "stream_s4", "--from-scratch", "--ema", "--ema-decay", "0.99",
                                             "--vis-head", "--vis-weight", "0.2"])
    plain, flagged = training.run_config(off, [], [], {}), training.run_config(on, [], [], {})
    assert plain["ema"] is None and plain["vis_head"] is None
    assert flagged["ema"] == {"decay": 0.99} and flagged["vis_head"] == {"weight": 0.2, "pool": "global", "gap_weight": 1.0}
    assert {"ema", "vis_head"} <= set(training.RECIPE_FIELDS)
    training.same_run(flagged, flagged)
    older = {key: value for key, value in plain.items() if key not in ("ema", "vis_head")}      # recorded before the fields
    training.same_run(older, plain)
    with pytest.raises(ValueError, match="config.json has ema=None"):
        training.same_run(plain, flagged)
    with pytest.raises(ValueError, match="vis_head="):
        training.same_run(flagged, {**flagged, "vis_head": None})
    assert on.ema_decay == 0.99 and off.ema_decay == training.EMA_DECAY and off.vis_weight == training.VIS_LOSS_WEIGHT
