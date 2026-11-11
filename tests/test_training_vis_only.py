"""``--vis-only``: train the visibility head alone on top of a finished detector. Every other
parameter and every BatchNorm running statistic stays exactly as loaded, so the 15-channel output
the tracker reads is the starting record's to the bit while the head learns; the flag needs
``--vis-head`` and a fine-tune (not ``--from-scratch``), refuses ``--ema`` and ``--distill``
(nothing else trains), and is recorded in ``config.json`` so a resume cannot switch it."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from finetune.training import train_gridtracknet as training
from finetune.training.dataset import make_target
from finetune.training.loop import build_optimizer, freeze_all_but_vis_head, train_epoch, with_vis_head
from tennis_tracker import streamnet

CPU = torch.device("cpu")


def _loader():
    images = torch.rand(2, 15, 64, 96)
    target = torch.stack([make_target([[10.0, 8.0], None, [5.0, 3.0], None, None], 160, 96, streamnet.ARCH)[:, :16, :24],
                          make_target([None] * 5, 160, 96, streamnet.ARCH)[:, :16, :24]])
    codes = torch.tensor([[1.0, 2.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    return DataLoader(TensorDataset(images, target, codes), batch_size=2)


def test_vis_only_trains_the_head_and_leaves_every_other_weight_and_running_statistic_as_loaded():
    model = with_vis_head(streamnet.StreamNet(), "peak")
    trainable = freeze_all_but_vis_head(model)
    assert trainable and all(name.startswith("vis.") for name, _ in trainable)
    assert all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("vis."))
    frozen = {name: t.detach().clone() for name, t in [*model.named_parameters(), *model.named_buffers()]
              if not name.startswith("vis.")}
    head = {name: t.detach().clone() for name, t in model.vis.named_parameters()}
    optimizer = build_optimizer(model, 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_epoch(model, _loader(), optimizer, scaler, CPU, 1, 1, None, vis_weight=1.0, vis_gap_weight=4.0,
                       vis_only=True)
    assert np.isfinite(loss)
    after = dict([*model.named_parameters(), *model.named_buffers()])
    changed = [name for name, t in frozen.items() if not torch.equal(after[name], t)]
    assert changed == []                                   # BatchNorm statistics included: eval mode outside the head
    assert any(not torch.equal(t, head[name]) for name, t in model.vis.named_parameters())
    assert model.vis.training and not model.head.training  # the epoch leaves the detector in eval mode


def test_freeze_needs_a_head():
    with pytest.raises(ValueError, match="visibility head"):
        freeze_all_but_vis_head(streamnet.StreamNet())


def test_the_flag_is_recorded_and_refused_without_its_partners():
    parser = training.build_parser()
    base = ["--arch", "stream_s4", "--weights", "w.pt", "--vis-head", "--vis-pool", "peak"]
    args = parser.parse_args([*base, "--vis-only"])
    assert args.vis_only and not parser.parse_args(base).vis_only
    config = training.run_config(args, [], [], {})
    assert config["vis_only"] is True and training.run_config(parser.parse_args(base), [], [], {})["vis_only"] is False
    with pytest.raises(ValueError, match="vis_only="):
        training.same_run(config, {**config, "vis_only": False})
    training.same_run({k: v for k, v in config.items() if k != "vis_only"}, {**config, "vis_only": False})  # older runs
    assert training.main(["--arch", "stream_s4", "--weights", "w.pt", "--vis-only"]) == 1             # no --vis-head
    assert training.main([*base, "--vis-only", "--from-scratch"]) == 1
    assert training.main([*base, "--vis-only", "--ema"]) == 1
    assert training.main([*base, "--vis-only", "--distill", "--teacher", "t.pt"]) == 1


def test_the_notebook_has_a_vis_only_switch_inside_the_vis_head_branch_and_the_quickstart_a_row():
    """VIS_ONLY sits beside the head's settings, passes --vis-only only with VIS_HEAD, refuses the
    combinations the trainer refuses before the run starts, and the quickstart documents it."""
    import ast
    import json
    from finetune.colab_zip import OPTIMIZATION_NOTEBOOK
    cells = ["".join(c["source"]) for c in json.loads(OPTIMIZATION_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    settings = next(source for source in cells if source.startswith("#@title Training settings\n"))
    ast.parse(settings)
    assert "VIS_ONLY = False" in settings and "#@param" in settings.split("VIS_ONLY = False", 1)[1].split("\n", 1)[0]
    branch = settings.split("if VIS_HEAD:", 1)[1].split("if TRAIN_DATA", 1)[0]
    assert "'--vis-only'" in branch
    for refused in ("MODE != 'finetune'", "DISTILL", "EMA"):
        assert refused in branch.split("'--vis-only'", 1)[0]
    assert "if VIS_ONLY and not VIS_HEAD" in settings
    quickstart = (OPTIMIZATION_NOTEBOOK.parent / "COLAB_QUICKSTART.md").read_text(encoding="utf-8")
    assert "`VIS_ONLY`" in quickstart and "--vis-only" in quickstart
