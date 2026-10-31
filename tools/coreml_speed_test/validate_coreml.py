"""Check the exported Core ML ball models on a Mac against the PyTorch outputs saved beside them.

Run from this folder (``bash validate.sh`` sets up the environment and runs it):

    python3 validate_coreml.py                       # every package, compute units ALL and CPU_AND_NE
    python3 validate_coreml.py --compute-units CPU_ONLY

``manifest.json`` (written by ``tools/export_coreml.py``) lists the packages, the sample inputs
(``samples/unit*/``: five consecutive same-phase frames of ``system/sample/pomona.mp4`` at 768x432,
RGB in [0, 1], float16) and the PyTorch FP32 reference outputs for them. For every package this
loads it with coremltools, predicts on each sample and prints:

- grids (a unit's 15-channel output): the largest heat-map difference, the largest offset
  difference, and in how many of the unit's frames the heat map's argmax lands on the same cell as
  PyTorch's (exactly, and within one cell), over all frames and over the frames whose PyTorch peak
  reaches the weights' operating threshold;
- features (the stream_s4 encoder): the largest difference, also relative to the reference's range.

``stream_s4 chain`` runs the phone's path: the Core ML encoder on the unit's four transitions, its
outputs fed straight to the Core ML decoder. Exit status 1 when an argmax disagrees on a frame
above the threshold or a heat map differs by more than ``--heat-tolerance``.

Only numpy and coremltools are needed (``requirements-mac.txt``); predictions need macOS 14 or
later (the packages target iOS 17). This file is also imported on the PC, where the metric
functions below score the PyTorch FP16 runs; coremltools is imported only to predict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
FRAMES_PER_UNIT = 5
GRID_PLANES = 3                     # per frame: heat, x offset, y offset
HEAT_CHANNELS = tuple(range(0, FRAMES_PER_UNIT * GRID_PLANES, GRID_PLANES))
OFFSET_CHANNELS = tuple(c for c in range(FRAMES_PER_UNIT * GRID_PLANES) if c not in HEAT_CHANNELS)
DEFAULT_COMPUTE_UNITS = ("ALL", "CPU_AND_NE")
DEFAULT_HEAT_TOLERANCE = 0.05
FEATURE_TOLERANCE = 0.05            # an encoder feature's largest difference over the reference's largest magnitude


@dataclass
class GridScore:
    """How a unit's grids compare with the reference's: the largest heat and offset differences,
    and per frame whether the heat argmax agrees (exact cell, within one cell) and whether the
    reference's peak reaches the threshold."""
    max_heat_diff: float
    max_offset_diff: float
    exact: List[bool]
    near: List[bool]
    confident: List[bool]

    def summary(self) -> Dict[str, Any]:
        confident = [i for i, flag in enumerate(self.confident) if flag]
        return {
            "max_heat_diff": round(self.max_heat_diff, 5),
            "max_offset_diff": round(self.max_offset_diff, 5),
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
    exact, near, confident = [], [], []
    for channel in HEAT_CHANNELS:
        (r0, c0), (r1, c1) = _peak(out[channel]), _peak(ref[channel])
        exact.append((r0, c0) == (r1, c1))
        near.append(abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1)
        confident.append(bool(ref[channel].max() >= threshold))
    return GridScore(float(diff[list(HEAT_CHANNELS)].max()), float(diff[list(OFFSET_CHANNELS)].max()),
                     exact, near, confident)


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


def merge_grid_scores(scores: Sequence[GridScore]) -> GridScore:
    """The scores of several units as one: the largest differences, every frame's flags."""
    return GridScore(max(s.max_heat_diff for s in scores), max(s.max_offset_diff for s in scores),
                     [f for s in scores for f in s.exact], [f for s in scores for f in s.near],
                     [f for s in scores for f in s.confident])


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


def run_case(runner: CoreMLRunner, case: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    """One manifest case scored: a ``grids`` or ``features`` case runs one package; a
    ``stream_chain`` case runs the encoder on the unit's transitions and the decoder on its outputs."""
    kind = case["kind"]
    if kind == "stream_chain":
        features: Dict[str, np.ndarray] = {}
        for step in case["encoder_steps"]:
            inputs = {name: load_input(spec, root) for name, spec in step["inputs"].items()}
            outputs = runner.predict(case["encoder"], inputs)
            for output_name, decoder_input in step["outputs_to"].items():
                features[decoder_input] = np.ascontiguousarray(outputs[output_name], dtype=np.float16)
        grids = runner.predict(case["decoder"], features)[case["output"]]
        score = score_grids(grids, np.load(root / case["reference"]), case["threshold"])
        return {"grid_score": score}
    inputs = {name: load_input(spec, root) for name, spec in case["inputs"].items()}
    outputs = runner.predict(case["package"], inputs)
    if kind == "grids":
        output = outputs[case["output"]]
        return {"grid_score": score_grids(output, np.load(root / case["reference"]), case["threshold"])}
    if kind == "features":
        return {"features": {name: score_features(outputs[name], np.load(root / path))
                             for name, path in case["references"].items()}}
    raise ValueError(f"unknown case kind {kind!r}")


def time_predictions(runner: CoreMLRunner, name: str, inputs: Mapping[str, np.ndarray], repeats: int) -> float:
    """The median wall time of `repeats` predictions on this Mac, in milliseconds (not the iPhone's)."""
    runner.predict(name, inputs)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        runner.predict(name, inputs)
        times.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(times))


def evaluate(runner: Any, manifest: Mapping[str, Any], root: Path, heat_tolerance: float) -> Dict[str, Dict[str, Any]]:
    """Every case run by `runner` (anything with ``predict(name, inputs)``), scored by group:
    ``{"pass": bool, "summary": {...}}``. A grids group passes when its largest heat difference is
    within `heat_tolerance` and every frame above the threshold has the reference's argmax; a
    features group when every level's largest difference is within ``FEATURE_TOLERANCE`` of the
    reference's range."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for case in manifest["cases"]:
        groups.setdefault(case["group"], []).append(case)
    scored: Dict[str, Dict[str, Any]] = {}
    for group, cases in groups.items():
        results = [run_case(runner, case, root) for case in cases]
        if "grid_score" in results[0]:
            merged = merge_grid_scores([r["grid_score"] for r in results])
            ok = merged.max_heat_diff <= heat_tolerance and all(e for e, c in zip(merged.exact, merged.confident) if c)
            scored[group] = {"pass": bool(ok), "summary": merged.summary()}
        else:
            worst: Dict[str, Dict[str, float]] = {}
            for result in results:
                for name, values in result["features"].items():
                    if name not in worst or values["max_diff"] > worst[name]["max_diff"]:
                        worst[name] = values
            ok = all(values["max_diff_relative"] <= FEATURE_TOLERANCE for values in worst.values())
            scored[group] = {"pass": bool(ok), "summary": worst}
    return scored


def validate(manifest: Mapping[str, Any], root: Path, compute_units: str, heat_tolerance: float,
             repeats: int) -> bool:
    """Every case at `compute_units` on Core ML, printed by group; True when every group passes."""
    runner = CoreMLRunner(manifest, root, compute_units)
    print(f"\n=== compute units {compute_units} ===")
    scored = evaluate(runner, manifest, root, heat_tolerance)
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--compute-units", nargs="+", default=list(DEFAULT_COMPUTE_UNITS),
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    parser.add_argument("--heat-tolerance", type=float, default=DEFAULT_HEAT_TOLERANCE,
                        help=f"the largest heat-map difference that passes (default {DEFAULT_HEAT_TOLERANCE})")
    parser.add_argument("--repeats", type=int, default=20, help="predictions timed per package on this Mac (0: none)")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("Core ML predictions run only on macOS; copy this folder to a Mac.")
        return 2
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    print(f"packages exported {manifest['exported']} from {manifest['source_commit']} "
          f"(coremltools {manifest['coremltools']}); references: PyTorch FP32")
    passed = all([validate(manifest, root, units, args.heat_tolerance, args.repeats) for units in args.compute_units])
    print("\nALL PASS" if passed else "\nSOME CHECKS FAILED (see FAIL lines)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
