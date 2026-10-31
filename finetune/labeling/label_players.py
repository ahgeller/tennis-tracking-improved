"""The label tool's player view (``--players``): who each person the tracker followed is.

A pipeline run's player tracks (``tracker_events.player_boxes_by_frame``) are cut into segments:
one track id's boxes, split wherever the track drops out for more than ``SEGMENT_GAP_S``. A
segment is the unit of labelling - the tracker keeps one person under one id for a stretch, so
saying who a segment is once labels every one of its boxes. Each segment is a row of crops taken
evenly through it, under a picture of the whole frame with every box on it; a key says who it is
(P1..P4, or not a player), and the next segment comes up. Where a track jumped from one person to
another, a right click on the first crop of the second person splits the segment there.

What is said is written per frame, one row per box (``player_labels``), into
``labels/<clip>_players.csv`` with the rest of the clip's work. Reopened, each stored label is
matched to the run's box it overlaps on its frame (``player_labels.match_frame``, as the scorer
matches them): a rerun of the tracker numbers its tracks afresh, so a label belongs to its box,
not to the track id it was saved with. The segments are cut again from the run and split wherever
the labels change identity within one, so splits survive. A label no box of this run overlaps is
kept as it is and written back untouched (``PlayerView.orphans``).

The crops come from a decoder of the view's own, reading the clip straight through on a
background thread (a straight read is the one frame order a pipeline run itself trusts), so the
view is up at once and fills in as it goes. The view owns them, by (track, frame), and the reader
fills them under a lock: a split while they load just shows each part the crops of its frames. SPACE takes the suggestion: the identity whose
labelled segments look most like this one (``appearance``: colour histograms of the crops),
among those not on court at the same time as it, with a small preference for the half of the
court the segment stands on.
"""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from finetune.data.player_labels import NOT_PLAYER, PLAYERS, Box, PlayerBoxLabel, match_frame
from finetune.labeling.label_draw import Colour, HudItem, fit_picture, fit_text, middle_ellipsis
from finetune.labeling.label_store import load_players
from finetune.labeling.label_view import ClipView
from finetune.labeling.label_window import KEYS_CONFIRM
from finetune.data.labels import LabelFormatError
from finetune.scoring.tracker_events import (court_keypoints_by_frame, court_side, player_boxes_by_frame, read_run,
                                             run_fps)

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_tool import Tool

SEGMENT_GAP_S = 0.5           # a track missing this long ends its segment
CROPS = 12                    # crops shown per segment, evenly through it
CROP_H = 176                  # crops are kept at this height in pixels, scaled to the row when drawn
CROP_PAD = 0.12               # each crop takes this share of the box's size around it
ROWS_SHOWN = 4                # segment rows on screen at once
OVERVIEW_SHARE = 0.42         # of the view's height, the whole-frame picture at the top
SIDE_BONUS = 0.15             # added to an identity's likeness when its nearest segment is on the same half
LIKE_MIN = 0.35               # below this likeness, the suggestion is the first identity not yet in use
IDENTITY_COLOURS: Dict[Optional[str], Colour] = {
    "P1": (255, 160, 50), "P2": (40, 190, 255), "P3": (100, 215, 80), "P4": (220, 100, 230),
    NOT_PLAYER: (140, 140, 140), None: (70, 70, 70)}


@dataclass
class Segment:
    """One track id's boxes over a stretch of frames, and what is known about who it is."""

    track: str
    frames: List[int]
    boxes: Dict[int, Box]
    side: Optional[str] = None                     # near / far: the half of the court its feet are on most

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    def samples(self) -> List[int]:
        """The frames its crops are taken on: CROPS of them, evenly through it, ends included."""
        if len(self.frames) <= CROPS:
            return list(self.frames)
        step = (len(self.frames) - 1) / (CROPS - 1)
        return sorted({self.frames[int(round(k * step))] for k in range(CROPS)})

    def overlaps(self, other: "Segment") -> bool:
        return self.first <= other.last and other.first <= self.last


def cut_segments(boxes_by_frame: Dict[int, Dict[str, List[float]]], fps: float) -> List[Segment]:
    """The run's tracks cut into segments at every gap over SEGMENT_GAP_S, in order of first frame."""
    gap = max(1, int(round(SEGMENT_GAP_S * fps)))
    by_track: Dict[str, List[int]] = {}
    for frame in sorted(boxes_by_frame):
        for track in boxes_by_frame[frame]:
            by_track.setdefault(track, []).append(frame)
    segments = []
    for track, frames in by_track.items():
        start = 0
        for k in range(1, len(frames) + 1):
            if k == len(frames) or frames[k] - frames[k - 1] > gap:
                part = frames[start:k]
                segments.append(Segment(track, part, {f: tuple(boxes_by_frame[f][track][:4]) for f in part}))  # type: ignore[misc]
                start = k
    return sorted(segments, key=lambda s: (s.first, s.track))


Labels = Dict[Tuple[int, str], PlayerBoxLabel]   # (frame, the run's track) -> the label of that box


def match_labels(stored: Sequence[PlayerBoxLabel], boxes_by_frame: Dict[int, Dict[str, List[float]]]
                 ) -> Tuple[Labels, List[PlayerBoxLabel]]:
    """The stored labels on the run's boxes, frame by frame by overlap (``player_labels.match_frame``):
    (the labels by (frame, the run's track), each now naming that track; the labels no box of the
    run overlaps, as they were, but for a track id this run uses on their frame, which is dropped:
    it named a box of a run that is gone, and a file has one box per track and frame)."""
    by_frame: Dict[int, List[PlayerBoxLabel]] = defaultdict(list)
    for label in stored:
        by_frame[label.frame].append(label)
    matched: Labels = {}
    orphans: List[PlayerBoxLabel] = []
    for frame, labels in sorted(by_frame.items()):
        boxes = boxes_by_frame.get(frame, {})
        pairs = dict(match_frame(labels, boxes))
        for index, label in enumerate(labels):
            if index in pairs:
                matched[(frame, pairs[index])] = label._replace(track=pairs[index])
            else:
                orphans.append(label._replace(track="") if label.track in boxes else label)
    return matched, orphans


def said_at(labels: Labels, track: str, frame: int) -> Optional[str]:
    """Who the box of `track` on `frame` is labelled as; None while it is not."""
    label = labels.get((frame, track))
    return label.player if label is not None else None


def split_at_labels(segments: Sequence[Segment], labels: Labels) -> List[Segment]:
    """Each segment split wherever the labels of its boxes change identity, so a split made in an
    earlier session comes back as it was made."""
    out = []
    for segment in segments:
        start = 0
        for k in range(1, len(segment.frames) + 1):
            if k == len(segment.frames) or (said_at(labels, segment.track, segment.frames[k])
                                            != said_at(labels, segment.track, segment.frames[k - 1])):
                part = segment.frames[start:k]
                out.append(Segment(segment.track, part, {f: segment.boxes[f] for f in part}, segment.side))
                start = k
    return out


def appearance(crops: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """A segment's look: the mean hue-saturation histogram of the middle of its crops (the shirt
    and shorts, not the court around the player), normalised; None without a crop."""
    histograms = []
    for crop in crops:
        h, w = crop.shape[:2]
        middle = crop[int(h * 0.2):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        if middle.size == 0:
            continue
        hsv = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
        histograms.append(cv2.normalize(histogram, None).flatten())
    if not histograms:
        return None
    return np.mean(histograms, axis=0).astype(np.float32)


def crop_box(image: np.ndarray, box: Box) -> Optional[np.ndarray]:
    """The box with CROP_PAD around it, at CROP_H pixels high, the part of it off the picture black
    (so a player at the edge keeps the shape and place of their box); None for a box wholly off
    the picture."""
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    h, w = image.shape[:2]
    left, top, right, bottom = (math.floor(value) for value in (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y))
    in_left, in_top, in_right, in_bottom = max(0, left), max(0, top), min(w, right), min(h, bottom)
    if right - left < 2 or bottom - top < 2 or in_right <= in_left or in_bottom <= in_top:
        return None
    crop = np.zeros((bottom - top, right - left, 3), np.uint8)
    seen = image[in_top:in_bottom, in_left:in_right]
    crop[in_top - top:in_bottom - top, in_left - left:in_right - left] = seen
    scale = CROP_H / crop.shape[0]
    return cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), CROP_H), interpolation=cv2.INTER_AREA)


Wanted = Dict[int, List[Tuple[str, Box]]]        # frame -> the (track, box) to cut a crop of on it


class CropReader:
    """Reads the clip straight through on a thread of its own, cutting the crops `wanted` asks for
    as their frames go by and handing each to `store` (track, frame, crop). The view reads
    ``done`` as it goes."""

    def __init__(self, video: str, wanted: Wanted, store: Callable[[str, int, np.ndarray], None]) -> None:
        self.video, self.wanted, self.store = video, wanted, store
        self.total = len(self.wanted)
        self.done = 0
        self.closing = False
        self.error: Optional[str] = None
        self.thread = threading.Thread(target=self.read, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def read(self) -> None:
        capture = cv2.VideoCapture(self.video)
        try:
            last = max(self.wanted) if self.wanted else -1
            frame = 0
            while frame <= last and not self.closing:
                if frame in self.wanted:
                    ok, image = capture.read()
                    if not ok:
                        self.error = f"the video ended at frame {frame}"
                        return
                    for track, box in self.wanted[frame]:
                        crop = crop_box(image, box)
                        if crop is not None:
                            self.store(track, frame, crop)
                    self.done += 1
                elif not capture.grab():
                    self.error = f"the video ended at frame {frame}"
                    return
                frame += 1
        finally:
            capture.release()

    @property
    def finished(self) -> bool:
        return not self.thread.is_alive()


class PlayerView(ClipView):
    """The segments of the clip at the cursor, one row each, with the whole frame of the selected
    one above them (see the module docstring)."""

    title = "PLAYERS"
    tag = "play"
    help_title = "PLAYERS"
    help = (
        ("1-4", "this segment is P1-P4"), ("0 / v", "not a player"), ("SPACE / ENTER / c", "take the suggestion"),
        ("u", "say nothing of it"), ("a / d, arrows", "previous / next segment"), ("f", "next segment not said"),
        ("click a row", "select it"), ("right click a crop", "split the segment there"),
    ("SHIFT+n", "next clip"), ("TAB", "back to the frames"), ("m", "full screen"), ("s", "save"),
    ("q / ESC", "save and quit"), ("? / F1", "these keys"))

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        super().__init__(session, tool)
        self.segments: List[Segment] = []
        self.labels: Labels = {}                 # (frame, the run's track) -> the label of that box
        self.orphans: List[PlayerBoxLabel] = []  # stored labels no box of the run overlaps: written back as they are
        self.crops: Dict[Tuple[str, int], np.ndarray] = {}   # (track, frame) -> its crop, filled by the reader
        self.crop_lock = threading.Lock()
        self.looks: Dict[tuple, Optional[np.ndarray]] = {}  # appearance() by segment and crops cut, cached
        self.boxes_by_frame: Dict[int, Dict[str, List[float]]] = {}
        self.reader: Optional[CropReader] = None
        self.selected = 0
        self.first_row = 0                       # the segment at the top of the list
        self.fps = 30.0                          # the run's
        self.rows: List[Tuple[int, int, int]] = []               # (segment, y0, y1) of each row drawn
        self.crop_boxes: List[Tuple[int, int, int, int, int, int]] = []   # (segment, frame, x0, y0, x1, y1)

    # ---------------------------------------------------------------- opening a clip
    def load(self) -> None:
        """The clip at the cursor: its segments, its labels, and the reader cutting its crops."""
        self.close_reader()
        source = self.clip
        self.segments, self.labels, self.orphans, self.boxes_by_frame = [], {}, [], {}
        with self.crop_lock:
            self.crops = {}
        self.looks = {}
        self.selected = self.first_row = 0
        if source.run is None or not source.run.is_file():
            self.message = (f"no pipeline run for {source.clip}: open it with --candidates <run folder> "
                            "(evaluate_archive.py --mode pipeline --court --out-dir <run folder>)")
            print(f"[play] {self.message}")
            return
        try:
            if source.players is None:
                source.remember(source.players_path)
                source.players = load_players(source.players_path)
            payload = read_run(source.run)
        except (OSError, ValueError, LabelFormatError) as error:
            self.message = f"{source.clip}: {error}"
            print(f"[play] {self.message}")
            return
        self.boxes_by_frame = player_boxes_by_frame(payload)
        self.labels, self.orphans = match_labels(source.players, self.boxes_by_frame)
        keypoints = court_keypoints_by_frame(payload)
        self.fps = run_fps(payload)
        segments = cut_segments(self.boxes_by_frame, self.fps)
        for segment in segments:
            sides = [court_side(keypoints.get(f), (b[0] + b[2]) / 2.0, b[3]) for f, b in segment.boxes.items()
                     if f in keypoints]
            known = [side for side in sides if side is not None]
            segment.side = max(set(known), key=known.count) if known else None
        self.segments = split_at_labels(segments, self.labels)
        if not self.segments:
            self.message = f"the run tracked nobody in {source.clip}"
            return
        self.selected = self.next_unassigned(-1, 1) or 0
        wanted: Wanted = defaultdict(list)
        for segment in self.segments:
            for frame in segment.samples():
                wanted[frame].append((segment.track, segment.boxes[frame]))
        self.reader = CropReader(str(source.video), dict(wanted), self.store_crop)
        self.reader.start()
        print(f"[play] {source.clip}: {len(self.segments)} segments of {len({s.track for s in self.segments})} "
              f"tracks; {sum(self.identity(s) is not None for s in self.segments)} assigned")

    def store_crop(self, track: str, frame: int, crop: np.ndarray) -> None:
        """The reader's hand-off: one crop, from its thread."""
        with self.crop_lock:
            self.crops[(track, frame)] = crop

    def crop(self, segment: Segment, frame: int) -> Optional[np.ndarray]:
        with self.crop_lock:
            return self.crops.get((segment.track, frame))

    def crop_frames(self, segment: Segment) -> List[int]:
        """The frames of `segment` its row shows a crop of: its own samples before a split, and after
        one, those of the frames the reader cuts (or has cut) that fall in the part."""
        wanted = self.reader.wanted if self.reader is not None else {}
        frames = [frame for frame in segment.frames if any(track == segment.track for track, _ in wanted.get(frame, ()))]
        if len(frames) <= CROPS:
            return frames
        step = (len(frames) - 1) / (CROPS - 1)
        return sorted({frames[int(round(k * step))] for k in range(CROPS)})

    def look(self, segment: Segment) -> Optional[np.ndarray]:
        """The segment's appearance from the crops cut so far, worked out when asked."""
        crops = [crop for crop in (self.crop(segment, frame) for frame in self.crop_frames(segment)) if crop is not None]
        key = (segment.track, segment.first, segment.last, len(crops))
        if key not in self.looks:
            self.looks[key] = appearance(crops)
        return self.looks[key]

    def close(self) -> None:
        self.close_reader()       # leaving the view: no more crops to cut
        super().close()

    def close_reader(self) -> None:
        if self.reader is not None:
            self.reader.closing = True
            self.reader = None

    # ---------------------------------------------------------------- identities
    def identity(self, segment: Segment) -> Optional[str]:
        """Who the segment's boxes are labelled as, None while they are not (split_at_labels keeps a
        segment's labels all the same)."""
        label = self.labels.get((segment.first, segment.track))
        return label.player if label is not None else None

    def assign(self, identity: Optional[str]) -> None:
        """Say who the selected segment is (None: take what was said back), and go to the next one."""
        if not self.segments or self.source is None:
            return
        segment = self.segments[self.selected]
        for frame in segment.frames:
            if identity is None:
                self.labels.pop((frame, segment.track), None)
            else:
                self.labels[(frame, segment.track)] = PlayerBoxLabel(frame, identity, segment.boxes[frame], segment.track)
        self.source.players = list(self.labels.values()) + self.orphans
        self.session.touch("players", source=self.source)
        self.revision += 1
        if identity is not None:
            self.select(self.selected + 1)

    def suggestion(self, index: int) -> Optional[str]:
        """The identity SPACE gives segment `index` (see the module docstring)."""
        segment = self.segments[index]
        busy = {self.identity(other) for other in self.segments if other is not segment and other.overlaps(segment)}
        free = [identity for identity in PLAYERS if identity not in busy]
        if not free:
            return None
        best, best_like = None, -1.0
        mine = self.look(segment)
        if mine is not None:
            for identity in free:
                theirs = [(other, look) for other, look in ((other, self.look(other)) for other in self.segments
                                                            if self.identity(other) == identity) if look is not None]
                if not theirs:
                    continue
                like = max(float(cv2.compareHist(mine, look, cv2.HISTCMP_CORREL)) for _, look in theirs)
                nearest = min((other for other, _ in theirs),
                              key=lambda other: min(abs(other.first - segment.last), abs(segment.first - other.last)))
                if segment.side is not None and nearest.side == segment.side:
                    like += SIDE_BONUS
                if like > best_like:
                    best, best_like = identity, like
        if best is not None and best_like >= LIKE_MIN:
            return best
        used = {self.identity(other) for other in self.segments}
        return next((identity for identity in free if identity not in used), best or free[0])

    def split(self, index: int, frame: int) -> None:
        """Cut segment `index` in two at `frame` (the first frame of the second part)."""
        segment = self.segments[index]
        if frame <= segment.first or frame > segment.last:
            return
        cut = segment.frames.index(frame)
        self.segments[index:index + 1] = [Segment(segment.track, frames, {f: segment.boxes[f] for f in frames}, segment.side)
                                          for frames in (segment.frames[:cut], segment.frames[cut:])]
        self.revision += 1
        print(f"[play] track {segment.track} split at frame {frame}")

    # ---------------------------------------------------------------- moving
    def select(self, index: int) -> None:
        if not self.segments:
            return
        self.selected = max(0, min(index, len(self.segments) - 1))
        if self.selected < self.first_row:
            self.first_row = self.selected
        elif self.selected >= self.first_row + ROWS_SHOWN:
            self.first_row = self.selected - ROWS_SHOWN + 1
        source = self.source
        if source is not None:
            frames = [row.frame for row in source.rows]
            target = self.segments[self.selected].first
            local = min(range(len(frames)), key=lambda k: abs(frames[k] - target))
            self.session.set_cursor(source.start + local)
        self.revision += 1

    def next_unassigned(self, start: int, direction: int) -> Optional[int]:
        index = start + direction
        while 0 <= index < len(self.segments):
            if self.identity(self.segments[index]) is None:
                return index
            index += direction
        return None

    # ---------------------------------------------------------------- input
    def view_key(self, code: int, direction: int) -> None:
        if code in tuple(ord(str(k)) for k in range(1, len(PLAYERS) + 1)):
            self.assign(PLAYERS[code - ord("1")])
        elif code in (ord("0"), ord("v")):
            self.assign(NOT_PLAYER)
        elif code in KEYS_CONFIRM:
            suggested = self.suggestion(self.selected) if self.segments else None
            if suggested is not None:
                self.assign(suggested)
            elif self.segments:       # SPACE takes a suggestion; it never takes a label off (u does)
                print("[play] no identity is free for this segment (all four are on court with it): "
                      "say who it is with 1-4, or 0 for not a player")
        elif code == ord("u"):
            self.assign(None)
        elif code == ord("f"):
            found = self.next_unassigned(self.selected, 1)
            if found is None:
                found = self.next_unassigned(-1, 1)
            if found is None:
                print("[play] every segment is assigned")
            else:
                self.select(found)
        elif direction:
            self.select(self.selected + direction)

    def on_mouse(self, event: int, x: int, y: int, flags: int = 0) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for index, y0, y1 in self.rows:
                if y0 <= y < y1:
                    self.select(index)
                    return
        elif event == cv2.EVENT_RBUTTONDOWN:
            for index, frame, x0, y0, x1, y1 in self.crop_boxes:
                if x0 <= x < x1 and y0 <= y < y1:
                    self.split(index, frame)
                    return

    # ---------------------------------------------------------------- drawing
    def render_key(self) -> tuple:
        reader = self.reader
        return super().render_key() + ((reader.done, reader.finished) if reader is not None else None,)

    def ready(self) -> bool:
        return self.message is None and bool(self.segments)

    def draw(self, canvas: np.ndarray) -> None:
        overview_h = int(self.session.view_h * OVERVIEW_SHARE)
        self.draw_overview(canvas[:overview_h], overview_h)
        self.draw_rows(canvas, overview_h)

    def draw_overview(self, strip: np.ndarray, height: int) -> None:
        """The selected segment's middle frame, whole, with every box the run had on it: the
        segment's own thick in its identity's colour, the rest thin with their track and identity."""
        session, segment = self.session, self.segments[self.selected]
        frame = self.overview_frame(segment)
        picture, scale, off_x, off_y = fit_picture(self.clip.frame(frame), session.view_w, height)
        strip[off_y:off_y + picture.shape[0], off_x:off_x + picture.shape[1]] = picture
        ui, thin = session.ui, max(1, int(round(session.ui)))
        for track, box in sorted(self.boxes_by_frame.get(frame, {}).items(), key=lambda item: item[0] == segment.track):
            label = self.labels.get((frame, track))
            own = track == segment.track
            if own:                       # the selected segment: white until it is said who it is
                colour = IDENTITY_COLOURS[self.identity(segment)] if self.identity(segment) else (255, 255, 255)
            else:
                colour = IDENTITY_COLOURS[label.player if label else None]
                colour = colour if label else (170, 170, 170)
            p1 = (int(box[0] * scale) + off_x, int(box[1] * scale) + off_y)
            p2 = (int(box[2] * scale) + off_x, int(box[3] * scale) + off_y)
            cv2.rectangle(strip, p1, p2, colour, 3 * thin if own else thin, cv2.LINE_AA)
            text = f"{track}" + (f" {label.player}" if label else "")
            cv2.putText(strip, text, (p1[0], max(int(14 * ui), p1[1] - int(5 * ui))), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * ui, colour, thin, cv2.LINE_AA)

    def overview_frame(self, segment: Segment) -> int:
        """The frame the overview shows for `segment`: the label frame nearest its middle that it
        has a box on (the label frames are the ones the clip's decoder is sure of), else its middle."""
        middle = segment.frames[len(segment.frames) // 2]
        label_frames = {row.frame for row in self.clip.rows}
        labelled = [frame for frame in segment.frames if frame in label_frames]
        return min(labelled, key=lambda frame: abs(frame - middle)) if labelled else middle

    def draw_rows(self, canvas: np.ndarray, top: int) -> None:
        """The segment rows under the overview: a caption, then its crops."""
        session = self.session
        ui, thin = session.ui, max(1, int(round(session.ui)))
        row_h = (session.view_h - top) // ROWS_SHOWN
        caption_w = int(190 * ui)
        self.rows, self.crop_boxes = [], []
        for slot, index in enumerate(range(self.first_row, min(len(self.segments), self.first_row + ROWS_SHOWN))):
            segment = self.segments[index]
            y0 = top + slot * row_h
            y1 = y0 + row_h
            self.rows.append((index, y0, y1))
            identity = self.identity(segment)
            if index == self.selected:
                cv2.rectangle(canvas, (0, y0), (session.view_w - 1, y1 - 1), (60, 60, 60), -1)
            chip = IDENTITY_COLOURS[identity]
            pad = int(6 * ui)
            cv2.rectangle(canvas, (pad, y0 + pad), (pad + int(8 * ui), y1 - pad), chip, -1)
            lines = [(identity or "unassigned", chip if identity else (150, 150, 150), 0.7 if identity else 0.55),
                     (f"track {segment.track}", (220, 220, 220), 0.48),
                     (f"{segment.first}-{segment.last}", (180, 180, 180), 0.44),
                     (f"{(segment.last - segment.first + 1) / self.fps:.1f}s {segment.side or ''}", (180, 180, 180), 0.44)]
            y = y0 + int(24 * ui)
            for text, colour, size in lines:
                if y > y1 - pad:
                    break
                cv2.putText(canvas, fit_text(text, caption_w - int(30 * ui), size * ui, thin), (int(22 * ui), y),
                            cv2.FONT_HERSHEY_SIMPLEX, size * ui, colour, thin + (1 if size >= 0.7 else 0), cv2.LINE_AA)
                y += int(20 * ui)
            x = caption_w
            crop_h = row_h - 2 * pad
            for frame in self.crop_frames(segment):
                crop = self.crop(segment, frame)
                width = int(round(crop_h * 0.6)) if crop is None else int(round(crop.shape[1] * crop_h / crop.shape[0]))
                width = min(width, int(crop_h * 1.2))
                if x + width > session.view_w - pad:
                    break
                cy0 = y0 + pad
                if crop is None:
                    cv2.rectangle(canvas, (x, cy0), (x + width - 1, cy0 + crop_h - 1), (45, 45, 45), -1)
                else:
                    shown = cv2.resize(crop, (width, crop_h), interpolation=cv2.INTER_AREA)
                    canvas[cy0:cy0 + crop_h, x:x + width] = shown
                self.crop_boxes.append((index, frame, x, cy0, x + width, cy0 + crop_h))
                x += width + pad
            cv2.line(canvas, (0, y1 - 1), (session.view_w - 1, y1 - 1), (35, 35, 35), 1)

    def hud_lines(self) -> Tuple[List[HudItem], List[HudItem]]:
        segment = self.segments[self.selected]
        assigned = sum(self.identity(s) is not None for s in self.segments)
        reader = self.reader
        loading = ""
        if reader is not None and not reader.finished:
            loading = f"cutting crops {reader.done}/{reader.total}"
        elif reader is not None and reader.error:
            loading = f"crops stopped: {reader.error}"
        suggestion = self.suggestion(self.selected)
        line1: List[HudItem] = [(1, middle_ellipsis(self.clip.clip)), (0, "PLAYERS"),
                                (0, f"segment {self.selected + 1}/{len(self.segments)}"), (2, f"track {segment.track}"),
                                (3, f"frames {segment.first}-{segment.last}"), (0, self.identity(segment) or "unassigned"),
                                (2, f"assigned {assigned}/{len(self.segments)}"), (1, loading)]
        line2: List[HudItem] = [(0, "1-4=P1-P4"), (1, "0/v=not a player"), (0, f"SPACE={suggestion or '-'} (suggested)"),
                                (1, "u=clear"), (0, "a/d=segment"), (2, "f=next unassigned"),
                                (2, "right click a crop=split there"), (2, "SHIFT+n=next clip"), (1, "TAB=frames"),
                                (0, "?=keys")]
        return line1, line2

