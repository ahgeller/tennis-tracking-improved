"""Check the exported Core ML models on a Mac against the PyTorch outputs saved beside them.

Run from this folder (``bash validate.sh`` sets up the environment and runs it):

    python3 validate_coreml.py                       # every package, compute units ALL and CPU_AND_NE
    python3 validate_coreml.py --compute-units CPU_ONLY

``manifest.json`` (written by ``tools/export_coreml.py``) lists the packages, the sample inputs
(``samples/unit*/``: five consecutive same-phase frames of ``system/sample/pomona.mp4`` at 768x432,
RGB in [0, 1], float16; ``samples/players/``: each unit's first frame as the player detector takes
it) and the PyTorch FP32 reference outputs for them. For every package this loads it with
coremltools, predicts on each sample and prints, per kind of case (``KINDS``):

- grids (a unit's 15-channel output): the largest heat-map difference, the largest offset
  difference (over all frames, and over the frames whose PyTorch peak reaches the weights'
  operating threshold, the confident ones), and in how many of the unit's frames the heat map's
  argmax lands on the same cell as PyTorch's (exactly, and within one cell), over all frames and
  over the confident ones;
- stream_chain: the phone's path, the Core ML encoder on the unit's four transitions and its
  outputs fed straight to the Core ML decoder, scored as grids;
- features (the stream_s4 encoder): per level the largest difference, also relative to the
  reference's largest magnitude;
- boxes (the player detector): both raw heads decoded as the tracker does (score threshold, then
  NMS), whether each frame finds as many players, and over the matched boxes the lowest IoU and
  the largest corner and score differences.

Exit status 1 when any group fails: a heat map differs by more than ``--heat-tolerance``, a
confident frame's argmax lands elsewhere or its offsets differ by more than
``--offset-tolerance``, an encoder level's largest relative difference exceeds the feature
tolerance, or a frame finds a different number of players or a player box overlaps the
reference's by less than the box IoU tolerance (``Tolerances``).

Only numpy and coremltools are needed (``requirements-mac.txt``); predictions need macOS 14 or
later (the packages target iOS 17). This file is also imported on the PC, where the same functions
score the PyTorch graphs; coremltools is imported only to predict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
FRAMES_PER_UNIT = 5
GRID_PLANES = 3                     # per frame: heat, x offset, y offset
HEAT_CHANNELS = tuple(range(0, FRAMES_PER_UNIT * GRID_PLANES, GRID_PLANES))
OFFSET_CHANNELS = tuple(c for c in range(FRAMES_PER_UNIT * GRID_PLANES) if c not in HEAT_CHANNELS)
DEFAULT_COMPUTE_UNITS = ("ALL", "CPU_AND_NE")


@dataclass(frozen=True)
class Tolerances:
    """What passes. heat: the largest heat-map difference; offset: the largest offset difference
    on a confident frame (the PC's FP16 graphs reach about 0.026); feature: an encoder level's
    largest difference over the reference's largest magnitude; box_iou: the lowest IoU a player
    box may have with the reference's."""
    heat: float = 0.05
    offset: float = 0.1
    feature: float = 0.05
    box_iou: float = 0.9


@dataclass
class GridScore:
    """How units' grids compare with the reference's: the largest heat difference, and per frame
    the largest offset difference, whether the heat argmax agrees (exact cell, within one cell)
    and whether the reference's peak reaches the threshold."""
    max_heat_diff: float
    offset_diffs: List[float]
    exact: List[bool]
    near: List[bool]
    confident: List[bool]

    @property
    def max_offset_diff(self) -> float:
        return max(self.offset_diffs, default=0.0)

    @property
    def confident_max_offset_diff(self) -> float:
        return max((d for d, c in zip(self.offset_diffs, self.confident) if c), default=0.0)

    def summary(self) -> Dict[str, Any]:
        confident = [i for i, flag in enumerate(self.confident) if flag]
        return {
            "max_heat_diff": round(self.max_heat_diff, 5),
            "max_offset_diff": round(self.max_offset_diff, 5),
            "confident_max_offset_diff": round(self.confident_max_offset_diff, 5),
            "argmax_exact": f"{sum(self.exact)}/{len(self.exact)}",
            "argmax_within_1": f"{sum(self.near)}/{len(self.near)}",
            "confident_argmax_exact": f"{sum(self.exact[i] for i in confident)}/{len(confident)}",
        }


def _peak(plane: np.ndarray) -> Tuple[int, int]:
    row, col = np.unravel_index(int(np.argmax(plane)), plane.shape)
    return int(row), int(col)


def score_grids(output: np.ndarray, reference: np.ndarray, threshold: float) -> GridScore:
    """`output` against `reference`, both a unit's ``(1, 15, rows, cols)`` grids (per frame heat,
    x offset, y offset)."""
    out = np.asarray(output, dtype=np.float32).reshape(-1, *np.shape(output)[-3:])[0]
    ref = np.asarray(reference, dtype=np.float32).reshape(-1, *np.shape(reference)[-3:])[0]
    if out.shape != ref.shape or out.shape[0] != FRAMES_PER_UNIT * GRID_PLANES:
        raise ValueError(f"grids of shape {out.shape} and {ref.shape}: expected two (15, rows, cols)")
    diff = np.abs(out - ref)
    offsets, exact, near, confident = [], [], [], []
    for channel in HEAT_CHANNELS:
        (r0, c0), (r1, c1) = _peak(out[channel]), _peak(ref[channel])
        offsets.append(float(diff[channel + 1:channel + GRID_PLANES].max()))
        exact.append((r0, c0) == (r1, c1))
        near.append(abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1)
        confident.append(bool(ref[channel].max() >= threshold))
    return GridScore(float(diff[list(HEAT_CHANNELS)].max()), offsets, exact, near, confident)


def merge_grid_scores(scores: Sequence[GridScore]) -> GridScore:
    """The scores of several units as one: the largest heat difference, every frame's values."""
    return GridScore(max(s.max_heat_diff for s in scores), [d for s in scores for d in s.offset_diffs],
                     [f for s in scores for f in s.exact], [f for s in scores for f in s.near],
                     [f for s in scores for f in s.confident])


def score_features(output: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """The largest absolute difference of encoder features, and that over the reference's largest
    magnitude."""
    out = np.asarray(output, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if out.shape != ref.shape:
        raise ValueError(f"features of shape {out.shape} and {ref.shape}")
    largest = float(np.abs(out - ref).max())
    scale = float(np.abs(ref).max()) or 1.0
    return {"max_diff": round(largest, 5), "max_diff_relative": round(largest / scale, 6)}


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices greedy NMS keeps, best score first; the arithmetic of the tracker's
    ``engine_boxes.nms_xyxy``."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        best, rest = int(order[0]), order[1:]
        keep.append(best)
        inter = (np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
                 * np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])))
        order = rest[inter / (areas[best] + areas[rest] - inter + 1e-6) <= float(iou)]
    return np.asarray(keep, dtype=np.int64)


def decode_head(head: np.ndarray, threshold: float, iou: float) -> np.ndarray:
    """A raw detection head ``(1, 4 + classes, anchors)`` as the tracker decodes it
    (``engine_boxes._raw_head_rows``): anchors whose best class scores at least `threshold`, as
    ``[x1, y1, x2, y2, score]`` float32 rows in input pixels, after greedy NMS at `iou`, best first."""
    values = np.asarray(head).reshape(np.shape(head)[-2:])
    scores = values[4:].astype(np.float32).max(axis=0)
    keep = np.flatnonzero(scores >= float(threshold))
    cx, cy, w, h = values[:4, keep].astype(np.float32)
    boxes = np.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], axis=1)
    kept = nms(boxes, scores[keep], iou)
    return np.column_stack([boxes[kept], scores[keep][kept]]).astype(np.float32).reshape(-1, 5)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    width = max(0.0, float(min(a[2], b[2]) - max(a[0], b[0])))
    height = max(0.0, float(min(a[3], b[3]) - max(a[1], b[1])))
    inter = width * height
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class BoxScore:
    """How a detector's decoded boxes compare with the reference's, over one or more frames: per
    frame whether the counts agree, and over the pairs matched by IoU the lowest IoU, the largest
    corner difference (input pixels) and the largest score difference."""
    same_count: List[bool]
    boxes: int
    reference_boxes: int
    min_iou: float
    max_corner_diff: float
    max_score_diff: float

    def summary(self) -> Dict[str, Any]:
        return {"frames_same_count": f"{sum(self.same_count)}/{len(self.same_count)}",
                "boxes": f"{self.boxes} vs {self.reference_boxes}", "min_iou": round(self.min_iou, 4),
                "max_corner_diff_px": round(self.max_corner_diff, 3), "max_score_diff": round(self.max_score_diff, 4)}


def score_boxes(output: np.ndarray, reference: np.ndarray, threshold: float, iou: float) -> BoxScore:
    """`output`'s players against `reference`'s, both raw heads decoded by ``decode_head``; each
    reference box is paired with the unpaired output box it overlaps most."""
    ours, theirs = decode_head(output, threshold, iou), decode_head(reference, threshold, iou)
    used: set = set()
    ious, corners, scores = [], [], []
    for box in theirs:
        candidates = [(box_iou(box, other), j) for j, other in enumerate(ours) if j not in used]
        if not candidates:
            ious.append(0.0)
            continue
        best_iou, j = max(candidates)
        used.add(j)
        ious.append(best_iou)
        corners.append(float(np.abs(box[:4] - ours[j][:4]).max()))
        scores.append(float(abs(box[4] - ours[j][4])))
    return BoxScore([len(ours) == len(theirs)], len(ours), len(theirs), min(ious, default=1.0),
                    max(corners, default=0.0), max(scores, default=0.0))


def merge_box_scores(scores: Sequence[BoxScore]) -> BoxScore:
    return BoxScore([f for s in scores for f in s.same_count], sum(s.boxes for s in scores),
                    sum(s.reference_boxes for s in scores), min(s.min_iou for s in scores),
                    max(s.max_corner_diff for s in scores), max(s.max_score_diff for s in scores))


def load_input(spec: Mapping[str, Any], root: Path) -> np.ndarray:
    """One model input from the manifest: ``{"npy": path}`` or ``{"npy": path, "channels": [a, b]}``
    (a channel slice of a unit's frames, an encoder's frame), as contiguous float16."""
    array = np.load(root / spec["npy"])
    if "channels" in spec:
        start, stop = spec["channels"]
        array = array[:, start:stop]
    return np.ascontiguousarray(array, dtype=np.float16)


class CoreMLRunner:
    """The manifest's packages loaded with coremltools at one compute-unit setting (macOS only)."""

    def __init__(self, manifest: Mapping[str, Any], root: Path, compute_units: str):
        import coremltools as ct            # macOS: predictions need the Core ML framework
        self.ct = ct
        self.root = root
        self.units = getattr(ct.ComputeUnit, compute_units)
        self.models: Dict[str, Any] = {}
        self.load_seconds: Dict[str, float] = {}
        self.manifest = manifest

    def model(self, name: str) -> Any:
        if name not in self.models:
            started = time.perf_counter()
            path = self.root / self.manifest["packages"][name]["file"]
            self.models[name] = self.ct.models.MLModel(str(path), compute_units=self.units)
            self.load_seconds[name] = time.perf_counter() - started
        return self.models[name]

    def predict(self, name: str, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self.model(name).predict(dict(inputs)).items()}


def _predict_case(runner: Any, case: Mapping[str, Any], root: Path) -> Dict[str, np.ndarray]:
    return runner.predict(case["package"], {name: load_input(spec, root) for name, spec in case["inputs"].items()})


def run_grids(runner: Any, case: Mapping[str, Any], root: Path) -> GridScore:
    output = _predict_case(runner, case, root)[case["output"]]
    return score_grids(output, np.load(root / case["reference"]), case["threshold"])


def run_chain(runner: Any, case: Mapping[str, Any], root: Path) -> GridScore:
    features: Dict[str, np.ndarray] = {}
    for step in case["encoder_steps"]:
        inputs = {name: load_input(spec, root) for name, spec in step["inputs"].items()}
        outputs = runner.predict(case["encoder"], inputs)
        for output_name, decoder_input in step["outputs_to"].items():
            features[decoder_input] = np.ascontiguousarray(outputs[output_name], dtype=np.float16)
    grids = runner.predict(case["decoder"], features)[case["output"]]
    return score_grids(grids, np.load(root / case["reference"]), case["threshold"])


def run_features(runner: Any, case: Mapping[str, Any], root: Path) -> Dict[str, Dict[str, float]]:
    outputs = _predict_case(runner, case, root)
    return {name: score_features(outputs[name], np.load(root / path)) for name, path in case["references"].items()}


def run_boxes(runner: Any, case: Mapping[str, Any], root: Path) -> BoxScore:
    output = _predict_case(runner, case, root)[case["output"]]
    return score_boxes(output, np.load(root / case["reference"]), case["threshold"], case["iou"])


Verdict = Tuple[bool, Dict[str, Any]]


def judge_grids(scores: Sequence[GridScore], tolerances: Tolerances) -> Verdict:
    """Pass: the largest heat difference within tolerance, and on every confident frame the
    reference's argmax and offsets within tolerance."""
    merged = merge_grid_scores(scores)
    confident = [i for i, flag in enumerate(merged.confident) if flag]
    ok = (merged.max_heat_diff <= tolerances.heat and all(merged.exact[i] for i in confident)
          and merged.confident_max_offset_diff <= tolerances.offset)
    return ok, merged.summary()


def judge_features(scores: Sequence[Mapping[str, Mapping[str, float]]], tolerances: Tolerances) -> Verdict:
    """Pass: every level's worst relative difference, over every case, within tolerance."""
    worst: Dict[str, Dict[str, float]] = {}
    for result in scores:
        for level, values in result.items():
            entry = worst.setdefault(level, {"max_diff": 0.0, "max_diff_relative": 0.0})
            for key in entry:
                entry[key] = max(entry[key], values[key])
    return all(values["max_diff_relative"] <= tolerances.feature for values in worst.values()), worst


def judge_boxes(scores: Sequence[BoxScore], tolerances: Tolerances) -> Verdict:
    """Pass: every frame finds as many players, and every matched box overlaps within tolerance."""
    merged = merge_box_scores(scores)
    return all(merged.same_count) and merged.min_iou >= tolerances.box_iou, merged.summary()


@dataclass(frozen=True)
class Kind:
    """How one kind of case is run (to a score) and how a group's scores are judged."""
    run: Callable[[Any, Mapping[str, Any], Path], Any]
    judge: Callable[[Sequence[Any], Tolerances], Verdict]


KINDS: Dict[str, Kind] = {
    "grids": Kind(run_grids, judge_grids),
    "stream_chain": Kind(run_chain, judge_grids),
    "features": Kind(run_features, judge_features),
    "boxes": Kind(run_boxes, judge_boxes),
}


def time_predictions(runner: CoreMLRunner, name: str, inputs: Mapping[str, np.ndarray], repeats: int) -> float:
    """The median wall time of `repeats` predictions on this Mac, in milliseconds (not the iPhone's)."""
    runner.predict(name, inputs)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        runner.predict(name, inputs)
        times.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(times))


def evaluate(runner: Any, manifest: Mapping[str, Any], root: Path,
             tolerances: Tolerances = Tolerances()) -> Dict[str, Dict[str, Any]]:
    """Every case run by `runner` (anything with ``predict(name, inputs)``) and judged by group
    (``KINDS``): ``{"pass": bool, "summary": {...}}``."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for case in manifest["cases"]:
        groups.setdefault(case["group"], []).append(case)
    scored: Dict[str, Dict[str, Any]] = {}
    for group, cases in groups.items():
        kinds = {case["kind"] for case in cases}
        if len(kinds) != 1 or not kinds <= set(KINDS):
            raise ValueError(f"group {group!r} has case kinds {sorted(kinds)}; expected one of {sorted(KINDS)}")
        kind = KINDS[kinds.pop()]
        ok, summary = kind.judge([kind.run(runner, case, root) for case in cases], tolerances)
        scored[group] = {"pass": bool(ok), "summary": summary}
    return scored


def validate(manifest: Mapping[str, Any], root: Path, compute_units: str, tolerances: Tolerances,
             repeats: int) -> bool:
    """Every case at `compute_units` on Core ML, printed by group; True when every group passes."""
    runner = CoreMLRunner(manifest, root, compute_units)
    print(f"\n=== compute units {compute_units} ===")
    scored = evaluate(runner, manifest, root, tolerances)
    for group, result in scored.items():
        print(f"{'PASS' if result['pass'] else 'FAIL'}  {group:<18} {json.dumps(result['summary'])}")
    passed = all(result["pass"] for result in scored.values())
    if repeats:
        print("this Mac's median prediction time (not the iPhone's; use Xcode for that):")
        for name, package in manifest["packages"].items():
            inputs = {spec["name"]: np.zeros(spec["shape"], dtype=np.float16) for spec in package["inputs"]}
            print(f"  {name:<24} {time_predictions(runner, name, inputs, repeats):8.2f} ms   "
                  f"(load {runner.load_seconds.get(name, 0.0) * 1000:.0f} ms)")
    return passed


def main(argv: Optional[Sequence[str]] = None) -> int:
    defaults = Tolerances()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--compute-units", nargs="+", default=list(DEFAULT_COMPUTE_UNITS),
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    parser.add_argument("--heat-tolerance", type=float, default=defaults.heat,
                        help=f"the largest heat-map difference that passes (default {defaults.heat})")
    parser.add_argument("--offset-tolerance", type=float, default=defaults.offset,
                        help=f"the largest offset difference on a confident frame that passes (default {defaults.offset})")
    parser.add_argument("--repeats", type=int, default=20, help="predictions timed per package on this Mac (0: none)")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("Core ML predictions run only on macOS; copy this folder to a Mac.")
        return 2
    tolerances = Tolerances(heat=args.heat_tolerance, offset=args.offset_tolerance)
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    print(f"packages exported {manifest['exported']} from {manifest['source_commit']} "
          f"(coremltools {manifest['coremltools']}); references: PyTorch FP32")
    passed = all([validate(manifest, root, units, tolerances, args.repeats) for units in args.compute_units])
    print("\nALL PASS" if passed else "\nSOME CHECKS FAILED (see FAIL lines)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
