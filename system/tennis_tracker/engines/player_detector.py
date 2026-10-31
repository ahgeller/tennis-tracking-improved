"""TensorRT player detection with ByteTrack association.

Engine policy: see ``engine_session.open_engine_session``; boxmot follows the
same rule, since the boxes are only useful tracked: with a player engine
configured a missing boxmot raises, and only an unset engine path disables
player tracking.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import Config
from .engine_boxes import POSTPROCESSED_COLUMNS, decode_engine_boxes, engine_output_mode
from .engine_session import TensorRTRuntimeSession, open_engine_session

_PLAYER_CLASS = 0  # the player ("person") class of the player engine


ByteTrackImport = Tuple[Optional[type], Optional[str]]   # (ByteTrack, None) or (None, why not)
_bytetrack: Optional[ByteTrackImport] = None
_bytetrack_import: Optional[threading.Thread] = None


def _import_bytetrack() -> None:
    global _bytetrack
    try:
        from boxmot import ByteTrack
    except ImportError as error:
        _bytetrack = (None, str(error))
    else:
        _bytetrack = (ByteTrack, None)


def prefetch_bytetrack() -> None:
    """Start importing boxmot on a thread.

    The import takes close to three seconds (boxmot's package init pulls in its ReID backbones
    and ultralytics) and is Python work that nothing else needs.  A run that will build a
    ``PlayerDetector`` calls this first, so the import overlaps the CUDA context, the ball
    model and the TensorRT engines being set up; ``_load_bytetrack`` waits for it.
    """
    global _bytetrack_import
    if _bytetrack is None and _bytetrack_import is None:
        _bytetrack_import = threading.Thread(target=_import_bytetrack, name="boxmot-import", daemon=True)
        _bytetrack_import.start()


def _load_bytetrack() -> ByteTrackImport:
    """Returns ``(ByteTrack, None)``, or ``(None, reason)`` when boxmot is not importable;
    waits for a ``prefetch_bytetrack`` import in flight."""
    if _bytetrack_import is not None:
        _bytetrack_import.join()
    if _bytetrack is None:
        _import_bytetrack()
    assert _bytetrack is not None
    return _bytetrack


def effective_player_interval(cfg: Config, fps: Optional[float]) -> int:
    """Every how many frames the player engine and ByteTrack run on a clip of ``fps``:
    ``cfg.player_detect_interval``, and at least every other frame from
    ``cfg.player_every_other_frame_fps`` up (0 turns that off; an unknown rate keeps the interval)."""
    interval = max(1, cfg.player_detect_interval)
    threshold = cfg.player_every_other_frame_fps
    if fps is not None and threshold > 0 and fps >= threshold:
        interval = max(interval, 2)
    return interval


@dataclass
class PlayerPending:
    """A player detection launched for one frame."""

    frame: np.ndarray
    frame_index: int
    width: int
    height: int
    court_kps: Optional[List[float]]
    outputs: Optional[List[torch.Tensor]] = None
    input_tensor: Optional[torch.Tensor] = None  # keeps the engine input alive until wait()
    done: Optional[torch.cuda.Event] = None       # marks this launch's host outputs ready (async sessions)


class PlayerDetector:
    """TensorRT player detection with ByteTrack association (no ReID).

    Engine policy: see ``open_engine_session``; a configured engine without
    boxmot raises too.  Without an engine the detector is disabled with one
    message: ``detect_async_start`` returns None and ``detect_async_finish``
    then returns no boxes.
    """

    def __init__(self, cfg: Config, court_keypoints: Optional[List[float]] = None, *,
                 fps: Optional[float] = None):
        self.cfg = cfg
        self.court_kps = court_keypoints
        self.cached_boxes: List[List[float]] = []
        self._frame_diag = 1.0
        self.session: Optional[TensorRTRuntimeSession] = None
        self.tracker: Optional[Any] = None
        # The engine and ByteTrack run on every Nth frame only (``effective_player_interval``); the
        # frames between get their boxes from ``interpolate_player_boxes`` once pass 1 has them
        # all.  (ByteTrack cannot fill them: an update with no detections marks every track lost
        # and returns no boxes.)
        self._current_interval = effective_player_interval(cfg, fps)
        # ByteTrack's live boxes keyed by track id.
        self.slots: Dict[int, List[float]] = {}

        if not cfg.player_model_path:
            print("[player] No engine configured, player tracking disabled")
            return
        bytetrack, reason = _load_bytetrack()
        if bytetrack is None:
            raise RuntimeError(
                f"[player] player tracking requires boxmot ({reason}): pip install boxmot, "
                "or unset player_model_path to run without players"
            )
        # The engine runs on every frame it tracks (every frame or every other one), the same launch
        # each time, so the launch is replayed from a CUDA graph, captured as the session is built
        # (the pipeline builds this detector before any worker thread starts).
        self.session = open_engine_session("player", cfg.player_model_path, cfg, cuda_graph=cfg.trt_cuda_graph)
        # ByteTrack counts its buffer in updates: a lost track is kept for 60 frames whatever N is.
        self.tracker = bytetrack(
            min_conf=0.1,
            track_thresh=0.3,
            track_buffer=max(1, round(60 / self._current_interval)),
            match_thresh=0.8,
        )
        every = "every frame" if self._current_interval == 1 else f"every {self._current_interval} frames"
        print(f"[player] ByteTrack tracking initialized, {every}")

    @property
    def enabled(self) -> bool:
        return self.session is not None and self.tracker is not None

    def tracks_frame(self, frame_index: int) -> bool:
        """Whether the tracker is updated on ``frame_index``; a disabled detector "tracks" every
        frame (with no boxes), so its frames need no interpolation."""
        return not self.enabled or frame_index % self._current_interval == 0

    def set_court_keypoints(self, court_keypoints: Optional[List[float]]) -> None:
        self.court_kps = court_keypoints

    def _decode_player_boxes(self, pred_t: torch.Tensor, frame_w: int, frame_h: int) -> np.ndarray:
        """Engine output -> ``[x1, y1, x2, y2, conf, cls]`` rows in frame pixels,
        clipped to the frame, for boxmot.  Class ``_PLAYER_CLASS`` only."""
        scale = (frame_w / float(max(self.session.input_w, 1)), frame_h / float(max(self.session.input_h, 1)))
        return decode_engine_boxes(pred_t, scale, self.cfg.player_conf, self.cfg.player_iou,
                                   engine_output_mode(pred_t), cls_filter=[_PLAYER_CLASS],
                                   frame_size=(frame_w, frame_h))

    def detect_async_start(self, frame: np.ndarray, frame_index: int,
                           frame_gpu_chw: Optional[torch.Tensor] = None) -> Optional[PlayerPending]:
        """Launch the engine on source frame ``frame_index`` (a host BGR frame,
        which ByteTrack also needs).  ``frame_gpu_chw`` is the same frame
        already on the GPU as CHW float in [0, 1], used instead when given."""
        if not self.tracks_frame(frame_index) or not self.enabled:
            return None

        h, w = frame.shape[:2]
        self._frame_diag = (w ** 2 + h ** 2) ** 0.5
        pending = PlayerPending(frame, int(frame_index), int(w), int(h), self.court_kps)
        t = None
        if frame_gpu_chw is not None:
            t, _ = self.session.preprocess_cuda_chw_frame(frame_gpu_chw)
        if t is None:
            t, _ = self.session.preprocess_frame(frame)
        outputs = self.session.forward_tensor(t)
        stream = self.session.stream
        if stream is not None:
            # The session's output buffers are reused by the next launch, which may come before
            # this one is finished (pass 1 finishes a frame's players one frame later), so the
            # outputs are copied out on the engine's stream, straight into pinned host memory.  The
            # event marks this launch's copies alone (not everything launched since), and
            # ``detect_async_finish`` waits for it and decodes the host copy: no device copy, and no
            # transfer of its own that would wait for the work queued on the current stream.
            with torch.cuda.stream(stream):
                outputs = [torch.empty(out.shape, dtype=out.dtype, pin_memory=True).copy_(out, non_blocking=True)
                           for out in outputs]
            pending.done = self.session.completion_event()
        pending.outputs = outputs
        pending.input_tensor = t
        return pending

    def _keep_nearest_to_court(self, dets: np.ndarray, court_kps: List[float]) -> np.ndarray:
        """When the engine finds more people than players, keep the ones closest to a court keypoint."""
        scored = []
        for row in dets:
            bx1, by1, bx2, by2 = row[:4]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist = min(((bcx - court_kps[ki]) ** 2 + (bcy - court_kps[ki + 1]) ** 2) ** 0.5
                       for ki in range(0, len(court_kps), 2))
            scored.append((dist, row))
        scored.sort(key=lambda x: x[0])
        return np.array([r for _, r in scored[:self.cfg.num_players]])

    def detect_async_finish(self, pending: Optional[PlayerPending]) -> List[List[float]]:
        if pending is None:
            return []

        assert pending.outputs is not None
        if pending.done is not None:
            pending.done.synchronize()          # the outputs are host copies
        else:
            self.session.wait()
        dets = self._decode_player_boxes(pending.outputs[0], pending.width, pending.height)
        # Drop spectators before the tracker learns them.
        if pending.court_kps and len(dets) > self.cfg.num_players:
            dets = self._keep_nearest_to_court(dets, pending.court_kps)
        if len(dets) == 0:
            dets = np.empty((0, POSTPROCESSED_COLUMNS))

        # tracked_objects rows: [x1, y1, x2, y2, track_id, conf, cls, ind]
        tracked_objects = self.tracker.update(dets, pending.frame)

        boxes: List[List[float]] = []
        self.slots.clear()
        for track in tracked_objects:
            x1, y1, x2, y2, track_id, _conf, _cls, _ = track
            box = [float(x1), float(y1), float(x2), float(y2)]
            boxes.append(box)
            self.slots[int(track_id)] = box

        self.cached_boxes = boxes
        return boxes

    def detect(self, frame: np.ndarray, frame_index: int) -> List[List[float]]:
        return self.detect_async_finish(self.detect_async_start(frame, frame_index))

    def player_boxes_by_slot(self) -> Dict[int, List[float]]:
        """A copy of ByteTrack's live boxes, keyed by track id."""
        return dict(self.slots)


def interpolate_player_boxes(boxes: Sequence[Dict[int, List[float]]],
                             tracked: Sequence[bool]) -> List[Dict[int, List[float]]]:
    """The per-frame player boxes with every frame the tracker skipped (``tracked`` False)
    filled from the tracked frames around it.

    A player tracked on both sides of a gap moves linearly between its two boxes; one tracked on
    one side only keeps the box of the nearer tracked frame that has it (a tie goes to the earlier
    one), so a track that starts or ends inside the gap appears or vanishes half way through it.
    Frames before the first or after the last tracked frame copy that frame.
    """
    keys = [index for index, is_tracked in enumerate(tracked) if is_tracked]
    if not keys:
        return [{} for _ in boxes]
    out = [dict(frame_boxes) for frame_boxes in boxes]
    for index in range(keys[0]):
        out[index] = dict(boxes[keys[0]])
    for index in range(keys[-1] + 1, len(boxes)):
        out[index] = dict(boxes[keys[-1]])
    for a, b in zip(keys, keys[1:]):
        before, after = boxes[a], boxes[b]
        for index in range(a + 1, b):
            t = (index - a) / (b - a)
            nearer = before if t <= 0.5 else after
            filled: Dict[int, List[float]] = {}
            for track_id in list(before) + [key for key in after if key not in before]:
                if track_id in before and track_id in after:
                    filled[track_id] = [p + (q - p) * t for p, q in zip(before[track_id], after[track_id])]
                elif track_id in nearer:
                    filled[track_id] = list(nearer[track_id])
            out[index] = filled
    return out