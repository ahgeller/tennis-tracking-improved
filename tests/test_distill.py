"""finetune/training/distill.py: the distillation losses, the teacher online and cached, MGD's
feature taps, and the cached maps under the augmentations, on a random HeatmapNet teacher."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from finetune.training import distill
from finetune.training.dataset import UnitDataset, flipped, make_target
from finetune.training.distill import (DistillConfig, Distiller, TeacherMaps, build_teacher_maps, feature_layers,
                                       flip_maps, heat_distillation, offset_distillation, paste_into_maps, reverse_maps,
                                       shift_maps)
from finetune.training.frame_cache import TrainingUnit
from finetune.training.loop import train_epoch
from tennis_tracker.gridtracknet import HEIGHT, WIDTH
from tennis_tracker.heatmapnet import ARCH, GRID_COLS, GRID_ROWS, HeatmapNet, save_model

CPU = torch.device("cpu")


@pytest.fixture
def teacher(tmp_path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "teacher.pt"
    save_model(HeatmapNet(), path)
    return path


def cached_clip(data: Path, count: int = 7) -> list:
    """`count` cached frames of clip c (a bright square moving right) and its units at stride 1."""
    (data / "frames" / "c").mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = np.full((HEIGHT, WIDTH, 3), 60, np.uint8)
        image[200:204, 100 + 6 * index:104 + 6 * index] = 255
        cv2.imwrite(str(data / "frames" / "c" / f"{index:06d}.png"), image)
    return [TrainingUnit(clip="c", source="custom", indices=list(range(start, start + 5)),
                         frames=[f"frames/c/{index:06d}.png" for index in range(start, start + 5)],
                         points=[[(102 + 6 * index) * 2.5, 202 * 2.5] for index in range(start, start + 5)],
                         size=[1920, 1080]) for start in range(count - 4)]


# --------------------------------------------------------------------------- the losses

def test_heat_distillation_is_zero_at_the_teacher_and_grows_away_from_it():
    teacher = torch.rand(2, 5, 4, 6)
    for kind in distill.DISTILL_LOSSES:
        assert float(heat_distillation(teacher.clone(), teacher, kind)) == pytest.approx(0.0, abs=1e-4)
        assert float(heat_distillation((teacher + 0.3).clamp(0, 1), teacher, kind)) > 0.01
    student = torch.full((1, 5, 4, 6), 0.2, requires_grad=True)
    loss = heat_distillation(student, torch.zeros(1, 5, 4, 6), "mse")
    assert float(loss.detach()) == pytest.approx(24 * 0.04)                   # summed over a frame's cells, averaged over frames
    loss.backward()
    assert student.grad is not None and float(student.grad.min()) > 0
    with pytest.raises(ValueError, match="unknown distillation loss"):
        heat_distillation(student, student, "kl")


def test_offset_distillation_counts_only_where_the_teacher_sees_a_ball():
    student, teacher = torch.zeros(1, 5, 2, 3, 3), torch.ones(1, 5, 2, 3, 3)
    heat = torch.zeros(1, 5, 3, 3)
    assert float(offset_distillation(student, teacher, heat)) == 0.0
    heat[0, 0, 1, 1] = 1.0
    assert float(offset_distillation(student, teacher, heat)) == pytest.approx(2.0)     # both offsets, weight one


# --------------------------------------------------------------------------- the distiller

def test_the_online_distiller_adds_a_finite_loss_and_keeps_the_teacher_out_of_its_state(teacher):
    student = HeatmapNet()
    distiller = Distiller(DistillConfig(str(teacher)), student, CPU)
    assert list(distiller.state_dict()) == [] and list(distiller.parameters()) == []
    images = torch.rand(1, 15, HEIGHT, WIDTH)
    prediction = student(images)
    loss, parts = distiller.loss(prediction, images, None)
    assert set(parts) == {"kd_heat", "kd_off"} and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in student.parameters() if parameter.requires_grad)
    assert all(parameter.grad is None for parameter in distiller.teacher[0].parameters())
    self_taught = Distiller(DistillConfig(str(teacher), loss="mse"), distiller.teacher[0], CPU)
    with torch.no_grad():
        answer = distiller.teacher[0](images)
    assert float(self_taught.loss(answer, images, None)[1]["kd_heat"]) == pytest.approx(0.0, abs=1e-6)


def test_mgd_taps_the_decoder_features_and_trains_its_generators(teacher):
    student = HeatmapNet()
    assert feature_layers(student) == {8: ("up1", 192), 4: ("up2", 96)}
    distiller = Distiller(DistillConfig(str(teacher), mgd_weight=2e-5), student, CPU)
    assert sorted(distiller.generators) == ["4", "8"] and any(key.startswith("generators.") for key in distiller.state_dict())
    images = torch.rand(1, 15, HEIGHT, WIDTH)
    loss, parts = distiller.loss(student(images), images, None)
    assert parts["mgd"] > 0 and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in distiller.parameters())
    with pytest.raises(ValueError, match="needs --teacher-maps online"):
        Distiller(DistillConfig(str(teacher), maps="cache", mgd_weight=1e-5), student, CPU)
    with pytest.raises(ValueError, match="--teacher-maps"):
        Distiller(DistillConfig(str(teacher), maps="disk"), student, CPU)


def test_train_epoch_adds_the_distillation_terms_and_reports_them(teacher, capsys):
    student = HeatmapNet()
    distiller = Distiller(DistillConfig(str(teacher)), student, CPU)
    target = make_target([[960.0, 540.0]] * 5, 1920, 1080, ARCH)
    batch = (torch.rand(1, 15, HEIGHT, WIDTH), target.unsqueeze(0))
    optimizer = torch.optim.Adadelta(student.parameters(), lr=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_epoch(student, [batch], optimizer, scaler, CPU, 1, 1, None, distiller)
    assert np.isfinite(loss) and "kd_heat=" in capsys.readouterr().out


# --------------------------------------------------------------------------- the cached maps

def test_cached_maps_are_the_dense_average_of_the_teachers_windows(teacher, tmp_path):
    units = cached_clip(tmp_path)
    maps_dir = tmp_path / "maps"
    report = build_teacher_maps(teacher, tmp_path, units, CPU, maps_dir)
    assert report["frames"] == 7 and report["built"] == 1
    model = distill.load_ball_model(teacher, CPU).float().eval()
    reader = TeacherMaps(maps_dir)
    with torch.no_grad():
        windows = [model(UnitDataset(tmp_path, [unit], augment=False, arch=ARCH)[0][0].unsqueeze(0))[0] for unit in units]
    first_frame = windows[0][:3]                                  # frame 0 is covered by the first window only
    assert np.allclose(reader.unit(units[0])[:3], first_frame.numpy(), atol=2e-3)
    middle = (windows[0][6:9] + windows[1][3:6] + windows[2][0:3]) / 3      # frame 2: three windows
    assert np.allclose(reader.unit(units[0])[6:9], middle.numpy(), atol=2e-3)
    assert reader.unit(units[1]).shape == (15, GRID_ROWS, GRID_COLS)
    assert build_teacher_maps(teacher, tmp_path, units, CPU, maps_dir)["built"] == 0      # up to date: kept


def test_the_maps_follow_the_flip_the_shift_and_the_reversal_as_the_target_does():
    # Off the cell edges: a ball exactly on one (offset 0) mirrors to offset 1 of the cell before.
    points = [[401.3, 207.9], None, [1001.3, 700.0], [77.7, 999.1], [1500.2, 33.3]]
    target = make_target(points, 1920, 1080, ARCH).numpy()
    mirrored = make_target(flipped(points, 1920), 1920, 1080, ARCH).numpy()
    flipped_maps = flip_maps(target)
    assert np.allclose(flipped_maps[0::3], mirrored[0::3], atol=1e-6)
    peaks = mirrored[0::3] == 1.0
    assert np.allclose(flipped_maps[1::3][peaks], mirrored[1::3][peaks], atol=1e-5)
    assert np.allclose(flipped_maps[2::3][peaks], mirrored[2::3][peaks], atol=1e-5)
    cell = 1920 / GRID_COLS
    moved = make_target([None if p is None else [p[0] + 3 * cell, p[1] - 2 * cell] for p in points], 1920, 1080, ARCH)
    assert np.allclose(shift_maps(target, 3, -2)[0::3], moved.numpy()[0::3], atol=1e-6)
    assert np.array_equal(reverse_maps(reverse_maps(target)), target)
    assert np.array_equal(reverse_maps(target)[0:3], target[12:15])
    pasted = make_target([[960.0, 540.0]] * 5, 1920, 1080, ARCH).numpy()
    laid = paste_into_maps(np.zeros_like(target), pasted)
    assert np.array_equal(laid, pasted)


def test_the_dataset_carries_the_teacher_maps_and_holds_shifts_to_whole_cells(teacher, tmp_path):
    units = cached_clip(tmp_path)
    maps_dir = tmp_path / "maps"
    build_teacher_maps(teacher, tmp_path, units, CPU, maps_dir)
    dataset = UnitDataset(tmp_path, units, augment=True, augmentations={"basic", "translate", "reverse"}, arch=ARCH,
                          teacher_maps=maps_dir)
    assert dataset.cell_px == WIDTH // GRID_COLS
    images, target, valid, maps = dataset[0]
    assert images.shape == (15, HEIGHT, WIDTH) and maps.shape == target.shape and valid.tolist() == [1.0] * 5
    for _ in range(30):
        drawn = dataset.draw(units[0])
        if drawn.shift is not None:
            assert drawn.shift[0] % dataset.cell_px == 0 and drawn.shift[1] % dataset.cell_px == 0
    plain = UnitDataset(tmp_path, units, augment=False, arch=ARCH, teacher_maps=maps_dir)
    assert np.allclose(plain[1][3].numpy(), TeacherMaps(maps_dir).unit(units[1]))
