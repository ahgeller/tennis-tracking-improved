"""The trainer's side of the visibility head's ``peak`` variant and the gap-frame target: the
frame cache marks the absent labels inside a short absence a rally crosses (``gap``), the dataset
hands them to the loss as the frame code ``FRAME_GAP`` in ``valid``, ``--vis-gap-weight`` weights
exactly those frames in the visibility loss and nothing in the heat loss, ``evaluate`` reports
their AUC, ``--vis-pool`` picks the head and is recorded, ``same_run`` refuses switching either,
and the Colab notebook wires both like VIS_HEAD / VIS_WEIGHT."""
import ast
import csv
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from finetune.colab_zip import OPTIMIZATION_NOTEBOOK
from finetune.data.labels import write_labels
from finetune.training import frame_cache
from finetune.training import train_gridtracknet as training
from finetune.training.dataset import FRAME_GAP, FRAME_LABEL, FRAME_NONE, UnitDataset, make_target, frame_codes
from finetune.training.frame_cache import gap_frames, prepare_clip
from finetune.training.loop import (EMA, Validation, evaluate, frame_weights, fresh_model, heatmap_loss, train_epoch,
                                    visibility_loss, visible_frames, with_vis_head)
from tennis_tracker import streamnet
from tests._helpers import write_clip

SIZE = (64, 36)
CPU = torch.device("cpu")


# --------------------------------------------------------------------------- the gap frames

def test_gap_frames_are_short_absent_runs_flanked_by_visible_labels():
    """A run of at most three absent labels with a visible label on both sides, at the label
    cadence (`stride`); a longer run, a run at the clip's edge, a run against a missing label and
    a run against an ignored (pseudo-label) frame are not gaps."""
    ball = [1.0, 1.0]
    labels = {0: ball, 1: None, 2: ball,                       # run of 1
              3: None, 4: None, 5: None, 6: ball,              # run of 3
              7: None, 8: None, 9: None, 10: None, 11: ball,   # run of 4: long
              13: None, 14: ball,                              # against a missing label (12)
              15: None, 16: ball, 17: None}                    # ignored 15; 17 at the edge
    assert gap_frames(labels, {15}, 1) == {1, 3, 4, 5}
    assert gap_frames({}, set(), 1) == set()
    at_stride = {index * 2: labels[index] for index in labels}
    assert gap_frames(at_stride, {30}, 2) == {2, 6, 8, 10}
    assert gap_frames(labels, {15}, 1, max_run=1) == {1}
    assert frame_cache.GAP_RUN == 3


def test_prepare_clip_marks_the_gap_frames_in_the_units_that_hold_one(tmp_path):
    """Units carry ``gap`` (one flag per frame) only when one of their frames is a gap frame, as
    ``valid`` is carried only by pseudo-labelled units; the record is rewritten with the flags
    and the frames on disk are untouched."""
    video = write_clip(tmp_path / "c.avi", 16, SIZE, 30.0)
    csv_path = tmp_path / "c_ball.csv"
    points = {frame: (1.0, 1.0) for frame in range(16)}
    points[4] = None                                  # a one-frame occlusion
    points[12] = points[13] = points[14] = points[15] = None     # a long absence to the edge
    write_labels(csv_path, points, *SIZE)
    record = prepare_clip(("c", video, csv_path, "custom"), tmp_path / "data", rebuild=False)
    units = record["samples"]
    with_gap = [unit for unit in units if "gap" in unit]
    assert with_gap and all(4 in unit["indices"] for unit in with_gap)
    assert all(unit["gap"] == [index == 4 for index in unit["indices"]] for unit in with_gap)
    assert all("gap" not in unit for unit in units if 4 not in unit["indices"])
    assert all(unit["points"][unit["indices"].index(4)] is None for unit in with_gap)


def test_the_dataset_hands_the_frame_codes_to_the_loss_and_the_codes_mask_as_before():
    """``frame_codes``: 0 no label, 1 a label, 2 a gap label. ``frame_weights`` (every heat loss)
    reads them as the 0/1 mask they always were, so a gap frame weighs 1 in the heat loss; the
    reversal and the dataset's example carry them like ``valid``."""
    assert (FRAME_NONE, FRAME_LABEL, FRAME_GAP) == (0.0, 1.0, 2.0)
    codes = frame_codes([True, True, True, True, True], [False, False, True, False, False])
    assert codes.tolist() == [1.0, 1.0, 2.0, 1.0, 1.0]
    assert frame_codes([True, False, True, True, True], None).tolist() == [1.0, 0.0, 1.0, 1.0, 1.0]
    assert frame_codes([True, False, True, True, True], [False, True, False, False, False]).tolist() == [1.0, 0.0, 1.0, 1.0, 1.0]
    weight = frame_weights(codes.unsqueeze(0), 1, codes)
    assert weight.flatten().tolist() == [1.0] * 5
    assert frame_weights(frame_codes([True, False, True, True, True], None).unsqueeze(0), 1, codes).flatten().tolist() == [1, 0, 1, 1, 1]
    target = make_target([[10.0, 8.0], None, [5.0, 3.0], None, None], 160, 96, streamnet.ARCH)
    prediction = torch.rand(1, 15, *target.shape[1:])
    plain = heatmap_loss(prediction, target.unsqueeze(0), torch.ones(1, 5))
    assert torch.allclose(heatmap_loss(prediction, target.unsqueeze(0), codes.unsqueeze(0)), plain)   # a gap frame: a label
    unit = {"clip": "c", "source": "custom", "indices": [0, 1, 2, 3, 4], "frames": [], "points": [[1.0, 1.0], None] + [[1.0, 1.0]] * 3,
            "size": [64, 36], "gap": [False, True, False, False, False]}
    dataset = UnitDataset(Path("unused"), [unit], augment=False, arch=streamnet.ARCH)
    assert dataset.with_valid
    with patch.object(UnitDataset, "read_frames", return_value=np.zeros((15, 432, 768), np.float32)):
        images, target, valid = dataset[0]
    assert valid.tolist() == [1.0, 2.0, 1.0, 1.0, 1.0]
    plain_unit = {key: value for key, value in unit.items() if key != "gap"}
    assert not UnitDataset(Path("unused"), [plain_unit], augment=False, arch=streamnet.ARCH).with_valid


def test_the_visibility_loss_weights_the_gap_frames_alone_and_evaluate_reports_their_auc():
    """With ``gap_weight`` W the gap frames' BCE terms weigh W and every other labelled frame 1
    (an unlabelled frame 0); W = 1 is the loss as it was. ``visible_frames`` still answers the
    labels and a 0/1 weight. ``evaluate`` adds the AUC of the visible labels against the gap
    frames alone (None without a gap frame among those seen)."""
    target = torch.stack([make_target([[10.0, 8.0], None, [5.0, 3.0], None, None], 160, 96, streamnet.ARCH)[:, :16, :24]])
    codes = torch.tensor([[1.0, 2.0, 1.0, 2.0, 0.0]])
    labels, weight = visible_frames(target, codes)
    assert labels.tolist() == [[1.0, 0.0, 1.0, 0.0, 0.0]] and weight.tolist() == [[1.0, 1.0, 1.0, 1.0, 0.0]]
    logits = torch.tensor([[2.0, 1.0, -1.0, 0.5, 3.0]])
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")[0]
    plain = float(visibility_loss(logits, target, codes))
    assert plain == pytest.approx(float(entropy[:4].mean()))
    assert float(visibility_loss(logits, target, codes, gap_weight=1.0)) == plain
    weighted = float(visibility_loss(logits, target, codes, gap_weight=4.0))
    expected = (entropy[0] + 4 * entropy[1] + entropy[2] + 4 * entropy[3]) / (1 + 4 + 1 + 4)
    assert weighted == pytest.approx(float(expected)) and weighted != plain
    assert float(visibility_loss(logits, target, torch.ones(1, 5), gap_weight=4.0)) == pytest.approx(float(entropy.mean()))
    with pytest.raises(ValueError, match="vis-gap-weight"):
        visibility_loss(logits, target, codes, gap_weight=0.0)
    # evaluate: the gap AUC
    headed = fresh_model(CPU, streamnet.ARCH, vis_head=True, vis_pool="peak")
    images = torch.rand(2, 15, 64, 96)
    targets = torch.cat([target, target])
    plain_loader = DataLoader(TensorDataset(images, targets, torch.ones(2, 5)), batch_size=2)
    validation = evaluate(headed, plain_loader, CPU, visibility=True)
    assert isinstance(validation, Validation) and validation.vis_auc is not None and validation.vis_gap_auc is None
    gap_loader = DataLoader(TensorDataset(images, targets, torch.cat([codes, codes])), batch_size=2)
    with_gaps = evaluate(headed, gap_loader, CPU, visibility=True)
    assert with_gaps.vis_gap_auc is not None and 0.0 <= with_gaps.vis_gap_auc <= 1.0
    assert evaluate(headed, gap_loader, CPU) == Validation(with_gaps.loss)
    assert Validation(0.3, 0.8, 0.7).vis_gap_auc is None                      # the older three-field value still reads


def test_fresh_model_builds_the_asked_pool_and_an_epoch_trains_the_peak_head_with_the_gap_weight():
    peak = fresh_model(CPU, streamnet.ARCH, vis_head=True, vis_pool="peak")
    assert peak.vis_pool == "peak" and fresh_model(CPU, streamnet.ARCH, vis_head=True).vis_pool == "global"
    with pytest.raises(ValueError, match="vis_pool"):
        fresh_model(CPU, streamnet.ARCH, vis_head=True, vis_pool="mean")
    images = torch.rand(2, 15, 64, 96)
    target = torch.stack([make_target([[10.0, 8.0], None, [5.0, 3.0], None, None], 160, 96, streamnet.ARCH)[:, :16, :24],
                          make_target([None] * 5, 160, 96, streamnet.ARCH)[:, :16, :24]])
    codes = torch.tensor([[1.0, 2.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    loader = DataLoader(TensorDataset(images, target, codes), batch_size=2)
    before = {name: tensor.detach().clone() for name, tensor in peak.vis.named_parameters()}
    optimizer = torch.optim.SGD(peak.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = EMA(peak)
    loss = train_epoch(peak, loader, optimizer, scaler, CPU, 1, 1, None, ema=ema, vis_weight=0.1, vis_gap_weight=4.0)
    assert np.isfinite(loss) and ema.updates == 1
    assert any(not torch.equal(tensor, before[name]) for name, tensor in peak.vis.named_parameters())
    # with_vis_head: the asked pool replaces another record's head, and says so
    loaded = streamnet.StreamNet(vis_head=True)
    with patch("builtins.print") as printed:
        assert with_vis_head(loaded, "peak") is loaded and loaded.vis_pool == "peak"
    assert any("peak" in str(call) for call in printed.call_args_list)
    assert with_vis_head(loaded, "peak").vis is loaded.vis                     # the same pool: kept


# --------------------------------------------------------------------------- the trainer's record

def test_config_json_records_the_pool_and_the_gap_weight_and_same_run_refuses_switching_them():
    parser = training.build_parser()
    default = parser.parse_args(["--arch", "stream_s4", "--from-scratch", "--vis-head"])
    peak = parser.parse_args(["--arch", "stream_s4", "--from-scratch", "--vis-head", "--vis-pool", "peak",
                              "--vis-gap-weight", "4"])
    assert default.vis_pool == "global" and default.vis_gap_weight == 1.0 and peak.vis_pool == "peak" and peak.vis_gap_weight == 4.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--vis-head", "--vis-pool", "mean"])
    plain, flagged = training.run_config(default, [], [], {}), training.run_config(peak, [], [], {})
    assert plain["vis_head"] == {"weight": 0.1, "pool": "global", "gap_weight": 1.0}
    assert flagged["vis_head"] == {"weight": 0.1, "pool": "peak", "gap_weight": 4.0}
    training.same_run(flagged, flagged)
    older = {**plain, "vis_head": {"weight": 0.1}}                   # recorded by the trainer before the fields existed
    training.same_run(older, plain)
    with pytest.raises(ValueError, match="vis_head="):
        training.same_run(plain, flagged)
    with pytest.raises(ValueError, match="vis_head="):
        training.same_run(flagged, {**flagged, "vis_head": {"weight": 0.1, "pool": "peak", "gap_weight": 1.0}})
    with pytest.raises(ValueError, match="vis_head="):
        training.same_run(older, flagged)
    assert training.vis_head_of_config(older) == plain["vis_head"] and training.vis_head_of_config({}) is None
    assert training.main(["--arch", "stream_s4", "--from-scratch", "--vis-head", "--vis-gap-weight", "0"]) == 1


def test_the_gap_auc_reaches_results_csv_and_the_epoch_line(tmp_path, capsys):
    """Through ``fit`` with scripted epochs (the pattern of test_training_ema_vis): the validation
    pass's gap AUC lands in ``vis_gap_auc`` (and ``ema_vis_gap_auc``) and on the epoch line, and
    ``train_epoch`` is handed the run's gap weight."""
    from _workspace_helpers import blank_template

    model = fresh_model(CPU)
    template = blank_template(model, tmp_path / "template.npz")
    args = training.build_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--weights", str(template), "--ema",
                                               "--epochs", "1", "--vis-head", "--vis-pool", "peak", "--vis-gap-weight", "4"])
    args.run_dir.mkdir()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = EMA(model, args.ema_decay)
    seen = {}

    def train_epoch(model, loader, optimizer, scaler, device, epoch, epochs, max_steps, distiller=None, ema=None,
                    vis_weight=0.0, vis_gap_weight=1.0, vis_only=False):
        seen["vis"] = (vis_weight, vis_gap_weight)
        return 0.5

    def evaluate(model, loader, device, max_steps=None, visibility=False):
        return Validation(0.3, 0.8, 0.7, 0.55) if visibility else Validation(0.3)
    with patch.object(training, "train_epoch", side_effect=train_epoch), patch.object(training, "evaluate", side_effect=evaluate):
        training.fit(args, model, None, object(), optimizer, scaler, CPU, lambda weights: training.metrics_of(0.8, 0.0),
                     training.START, None, ema)
    assert seen["vis"] == (0.1, 4.0)
    with (args.run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[0]
    assert (row["vis_auc"], row["vis_gap_auc"], row["ema_vis_gap_auc"]) == ("0.800000", "0.550000", "0.550000")
    out = capsys.readouterr().out
    assert "visAUC=0.800" in out and "gapAUC=0.550" in out


def test_the_results_layout_gains_the_gap_columns_at_the_end_and_pads_older_rows():
    assert training.RESULTS_COLUMNS[-2:] == ("vis_gap_auc", "ema_vis_gap_auc")
    legacy = training.results_row(1, 0.5, 0.25, 90.0, 10.0, training.metrics_of(0.9, 0.01))[:len(training.LEGACY_RESULTS_COLUMNS)]
    assert training.whole_row(legacy) == legacy + [""] * (len(training.RESULTS_COLUMNS) - len(legacy))
    before_gap = training.results_row(1, 0.5, 0.25, 90.0, 10.0, training.metrics_of(0.9, 0.01))[:-2]
    assert training.whole_row(before_gap) == before_gap + ["", ""]
    row = training.results_row(1, 0.5, 0.25, 90.0, 10.0, None, Validation(0.3, 0.8, 0.7, 0.55), Validation(0.3, 0.8, 0.7, 0.6))
    assert dict(zip(training.RESULTS_COLUMNS, row))["vis_gap_auc"] == "0.550000"
    assert dict(zip(training.RESULTS_COLUMNS, row))["ema_vis_gap_auc"] == "0.600000"
    assert dict(zip(training.RESULTS_COLUMNS, training.results_row(1, 0.5, 0.25, 90.0, 10.0, None)))["vis_gap_auc"] == ""


# --------------------------------------------------------------------------- the notebook

def code_cells() -> list:
    import json
    notebook = json.loads(OPTIMIZATION_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_the_notebook_settings_wire_vis_pool_and_vis_gap_weight_like_vis_head():
    """The settings cell has the two parameters beside VIS_HEAD / VIS_WEIGHT, passes them only
    with VIS_HEAD, and the command parses with the trainer's parser; the quickstart has a row."""
    settings = next(source for source in code_cells() if source.startswith("#@title Training settings\n"))
    ast.parse(settings)
    assert 'VIS_POOL = "global"' in settings and "#@param" in settings.split('VIS_POOL = "global"', 1)[1].split("\n", 1)[0]
    options = ast.literal_eval(settings.split('VIS_POOL = "global"', 1)[1].split("#@param", 1)[1].split("\n", 1)[0].strip())
    assert tuple(options) == streamnet.VIS_POOLS
    assert "VIS_GAP_WEIGHT = 1.0" in settings
    assert "'--vis-pool', VIS_POOL" in settings and "'--vis-gap-weight', str(VIS_GAP_WEIGHT)" in settings
    # the flags follow VIS_HEAD: the two appear inside its branch
    branch = settings.split("if VIS_HEAD:", 1)[1].split("if TRAIN_DATA", 1)[0]
    assert "--vis-pool" in branch and "--vis-gap-weight" in branch
    quickstart = (OPTIMIZATION_NOTEBOOK.parent / "COLAB_QUICKSTART.md").read_text(encoding="utf-8")
    assert "`VIS_POOL`" in quickstart and "`VIS_GAP_WEIGHT`" in quickstart and "vis_gap_auc" in quickstart
