"""The label tool's court view (``--court``): the court's keypoints on a few frames of each clip.

The view steps through ``COURT_FRAMES`` label frames spread evenly over the clip, and any frame
already labelled. Each comes up with the pipeline run's court keypoints on it (``court_keypoints``
of the tracking JSON), so only the points the model got wrong need moving: drag one, or nudge the
selected one with i/j/k/l, watching the loupe; SPACE says the frame is right as it stands and
moves on. The net posts (the tops of the posts, which the model does not give) are placed with a
click once selected. A point the camera cannot see is marked not visible (v).

Only what SPACE confirms is a label, written into ``labels/<clip>_court.csv`` (``court_labels``):
a frame shown with the model's points and left without SPACE stays unlabelled, and so does one
taken back with x. A point outside the picture is never written: it cannot be checked.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from finetune.data.court_labels import COURT_POINTS, GROUND_POINTS, CourtPoints, engine_points
from finetune.data.labels import LabelFormatError
from finetune.labeling import label_window as win
from finetune.labeling.label_draw import fit_picture, fit_text, hud_text
from finetune.labeling.label_store import load_court
from finetune.labeling.label_timeline import Timeline
from finetune.labeling.label_window import KEYS_CONFIRM
from finetune.scoring.tracker_events import court_keypoints_by_frame, read_run
from tennis_tracker.court.court_geometry import LINE_PAIRS_14

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_source import Source
    from finetune.labeling.label_tool import Tool

COURT_FRAMES = 5              # frames of each clip offered for labelling, evenly through it
GRAB_PX = 14                  # display pixels: a press this close to a point picks it up
NUDGE_PX, FINE_NUDGE_PX = 1.0, 0.25   # native pixels per i/j/k/l, and with SHIFT
LOUPE = 4                     # the loupe's magnification ...
LOUPE_PX = 240                # ... and its size on screen, at ui 1
POINT_COLOUR = (60, 220, 255)
SELECTED_COLOUR = (255, 255, 255)
LINE_COLOUR = (40, 200, 120)
CONFIRMED_COLOUR = (90, 230, 90)
NET_POSTS = COURT_POINTS[len(GROUND_POINTS):]
SHORT_NAMES = {name: name.replace("_singles", " S").replace("_service", " svc").replace("_", " ")
               for name in COURT_POINTS}


class CourtView:
    """The court keypoints of one frame at a time, over the whole picture (module docstring)."""

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        self.session = session
        self.tool = tool
        self.active = False
        self.source: Optional["Source"] = None
        self.model: Dict[int, CourtPoints] = {}       # frame -> the run's points there
        self.frames: List[int] = []                   # the frames offered, in order
        self.position = 0
        self.drafts: Dict[int, CourtPoints] = {}      # frame -> the points as they stand on screen
        self.history: Dict[int, List[CourtPoints]] = {}   # frame -> its earlier drafts, for u
        self.selected = 0                             # index into COURT_POINTS
        self.dragging = False
        self.revision = 0
        self.message: Optional[str] = None
        self.mapping = (1.0, 0, 0)                    # scale, x and y offset of the picture on screen
        self.cache_key: Optional[tuple] = None
        self.cache: Optional[np.ndarray] = None
        self.timeline = Timeline(session)

    # ---------------------------------------------------------------- opening a clip
    def open(self) -> None:
        self.active = True
        self.tool.grid_view.active = self.tool.sheet_view.active = self.tool.player_view.active = False
        self.tool.follow_ball()
        source = self.source = self.session.source
        self.model, self.drafts, self.history, self.message = {}, {}, {}, None
        self.revision += 1
        try:
            if source.court is None:
                source.court = load_court(source.court_path)
        except LabelFormatError as error:
            self.message = str(error)
            self.frames = []
            print(f"[court] {error}")
            return
        if source.run is not None and source.run.is_file():
            try:
                self.model = {frame: engine_points(kps) for frame, kps in court_keypoints_by_frame(read_run(source.run)).items()}
            except (OSError, ValueError) as error:
                print(f"[court] {source.clip}: the run could not be read ({error}); starting from blank frames")
        else:
            print(f"[court] {source.clip}: no pipeline run (--candidates <run folder>), so no points to start from")
        frames = [row.frame for row in source.rows if row.frame not in source.missing]
        picks = {frames[int(round(k * (len(frames) - 1) / max(1, COURT_FRAMES - 1)))] for k in range(COURT_FRAMES)} if frames else set()
        self.frames = sorted(picks | set(source.court))
        self.position = next((k for k, frame in enumerate(self.frames) if frame not in source.court), 0)
        self.selected = 0
        self.go(self.position)
        print(f"[court] {source.clip}: {len(self.frames)} frames, {len(source.court)} labelled")

    # ---------------------------------------------------------------- the points
    @property
    def frame(self) -> int:
        return self.frames[self.position]

    def points(self) -> CourtPoints:
        """The points of the frame on screen: its draft, else its label, else the model's."""
        frame = self.frame
        if frame not in self.drafts:
            court = self.source.court if self.source and self.source.court is not None else {}
            self.drafts[frame] = dict(court.get(frame) or self.model.get(frame) or {})
        return self.drafts[frame]

    def change(self, points: CourtPoints) -> None:
        """Put `points` on screen for this frame, the old ones kept for u."""
        self.history.setdefault(self.frame, []).append(dict(self.points()))
        self.drafts[self.frame] = points
        self.revision += 1

    def move_point(self, name: str, x: float, y: float) -> None:
        session = self.session
        points = dict(self.points())
        points[name] = (float(min(max(x, 0.0), session.width - 1.0)), float(min(max(y, 0.0), session.height - 1.0)))
        self.change(points)

    def state(self) -> str:
        """What the frame on screen is: labelled as shown, labelled but changed since, or not."""
        court = self.source.court if self.source and self.source.court is not None else {}
        if self.frame in court:
            return "LABELLED" if court[self.frame] == self.points() else "LABELLED, CHANGED (SPACE keeps it)"
        return "not labelled (SPACE confirms)"

    def confirm(self) -> None:
        """SPACE: the points on screen are right; those inside the picture become this frame's label."""
        session, source = self.session, self.source
        if source is None or source.court is None:
            return
        inside = {name: point for name, point in self.points().items()
                  if 0.0 <= point[0] < session.width and 0.0 <= point[1] < session.height}
        if not inside:
            print("[court] no point inside the picture to keep: x unlabels a frame")
            return
        source.court[self.frame] = inside
        self.drafts[self.frame] = dict(inside)
        session.dirty = source.dirty = True
        self.revision += 1
        nxt = next((k for k in range(self.position + 1, len(self.frames)) if self.frames[k] not in source.court), None)
        self.go(nxt if nxt is not None else min(self.position + 1, len(self.frames) - 1))

    def unlabel(self) -> None:
        source = self.source
        if source is not None and source.court is not None and source.court.pop(self.frame, None) is not None:
            self.session.dirty = source.dirty = True
            print(f"[court] frame {self.frame} unlabelled")
        self.revision += 1

    def copy_previous(self) -> None:
        """y: the points of the nearest labelled frame before this one, else after it (a still
        camera's court does not move)."""
        court = self.source.court if self.source and self.source.court is not None else {}
        before = [frame for frame in court if frame < self.frame]
        after = [frame for frame in court if frame > self.frame]
        if not before and not after:
            print("[court] no labelled frame to copy from")
            return
        self.change(dict(court[max(before) if before else min(after)]))

    def reset(self) -> None:
        self.change(dict(self.model.get(self.frame) or {}))

    def undo(self) -> None:
        stack = self.history.get(self.frame)
        if stack:
            self.drafts[self.frame] = stack.pop()
            self.revision += 1

    def hide_point(self) -> None:
        """v: the selected point cannot be seen on this frame."""
        name = COURT_POINTS[self.selected]
        if name in self.points():
            points = dict(self.points())
            del points[name]
            self.change(points)

    def nudge(self, dx: float, dy: float) -> None:
        name = COURT_POINTS[self.selected]
        point = self.points().get(name)
        if point is None:
            print(f"[court] {name} is not placed: click where it is")
            return
        step = FINE_NUDGE_PX if win.shift_held() else NUDGE_PX
        self.move_point(name, point[0] + dx * step, point[1] + dy * step)

    # ---------------------------------------------------------------- moving
    def go(self, position: int) -> None:
        if not self.frames:
            return
        self.position = max(0, min(position, len(self.frames) - 1))
        source = self.source
        if source is not None:
            local = min(range(len(source.rows)), key=lambda k: abs(source.rows[k].frame - self.frame))
            self.session.set_cursor(source.start + local)
        self.revision += 1

    def next_clip(self) -> None:
        session = self.session
        end = session.clip_bounds(session.cursor)[1]
        if end >= len(session.rows):
            print("[court] this is the last clip of the run")
            return
        session.set_cursor(end)
        self.open()

    # ---------------------------------------------------------------- input
    def handle_key(self, code: int, direction: int) -> None:
        if not self.frames:
            if code == ord("c"):
                self.next_clip()
            return
        nudges = {ord("i"): (0, -1), ord("k"): (0, 1), ord("j"): (-1, 0), ord("l"): (1, 0)}
        if code in nudges:
            self.nudge(*nudges[code])
        elif code in (ord(","), ord(".")):
            self.selected = (self.selected + (1 if code == ord(".") else -1)) % len(COURT_POINTS)
            self.revision += 1
        elif code in KEYS_CONFIRM:
            self.confirm()
        elif code == ord("v"):
            self.hide_point()
        elif code == ord("y"):
            self.copy_previous()
        elif code == ord("r"):
            self.reset()
        elif code == ord("x"):
            self.unlabel()
        elif code == ord("u"):
            self.undo()
        elif code == ord("c"):
            self.next_clip()
        elif direction:
            self.go(self.position + direction)

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        scale, off_x, off_y = self.mapping
        return int(round(x * scale + off_x)), int(round(y * scale + off_y))

    def to_frame(self, x: int, y: int) -> Tuple[float, float]:
        scale, off_x, off_y = self.mapping
        return (x + 0.5 - off_x) / scale, (y + 0.5 - off_y) / scale

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        """A press near a point picks it up (and selects it), a drag moves it, a press elsewhere puts
        the selected point there."""
        if not self.frames or y >= self.session.view_h:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            near = [(math.hypot(sx - x, sy - y), name) for name, (sx, sy) in
                    ((name, self.to_screen(*point)) for name, point in self.points().items())]
            close = [item for item in near if item[0] <= GRAB_PX * self.session.ui]
            if close:
                self.selected = COURT_POINTS.index(min(close)[1])
                self.history.setdefault(self.frame, []).append(dict(self.points()))
                self.dragging = True
                self.revision += 1
            else:
                self.move_point(COURT_POINTS[self.selected], *self.to_frame(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and flags & cv2.EVENT_FLAG_LBUTTON:
            points = dict(self.points())
            fx, fy = self.to_frame(x, y)
            session = self.session
            points[COURT_POINTS[self.selected]] = (float(min(max(fx, 0.0), session.width - 1.0)),
                                                   float(min(max(fy, 0.0), session.height - 1.0)))
            self.drafts[self.frame] = points        # one undo step for the whole drag, taken on the press
            self.revision += 1
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    # ---------------------------------------------------------------- drawing
    def render(self) -> np.ndarray:
        session = self.session
        key = (self.revision, session.view_w, session.view_h, session.hud_h, session.ui, session.dirty)
        if key == self.cache_key and self.cache is not None:
            return self.cache
        canvas = np.zeros((session.view_h + session.hud_h, session.view_w, 3), np.uint8)
        ui = session.ui
        if not self.frames or self.source is None:
            cv2.putText(canvas, fit_text(self.message or "no frames to label", session.view_w - int(40 * ui), 0.6 * ui, 1),
                        (int(20 * ui), int(40 * ui)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, (255, 255, 255), 1, cv2.LINE_AA)
            hud_text(canvas[session.view_h + session.timeline_h:], "COURT", "c: next clip   q: save/quit", ui)
        else:
            image = self.source.frame(self.frame)
            picture, scale, off_x, off_y = fit_picture(image, session.view_w, session.view_h)
            self.mapping = (scale, off_x, off_y)
            canvas[off_y:off_y + picture.shape[0], off_x:off_x + picture.shape[1]] = picture
            self.draw_court(canvas[:session.view_h])
            self.draw_loupe(canvas[:session.view_h], image)
            self.timeline.draw(canvas[session.view_h:session.view_h + session.timeline_h], session.view_h)
            hud_text(canvas[session.view_h + session.timeline_h:], *self.hud_lines(), ui)
        self.cache_key, self.cache = key, canvas
        return canvas

    def draw_court(self, display: np.ndarray) -> None:
        """The court's lines between the points placed, then the points, the selected one ringed."""
        ui, thin = self.session.ui, max(1, int(round(self.session.ui)))
        points = self.points()
        for a, b in LINE_PAIRS_14:
            if GROUND_POINTS[a] in points and GROUND_POINTS[b] in points:
                cv2.line(display, self.to_screen(*points[GROUND_POINTS[a]]), self.to_screen(*points[GROUND_POINTS[b]]),
                         LINE_COLOUR, thin, cv2.LINE_AA)
        if all(name in points for name in NET_POSTS):
            cv2.line(display, self.to_screen(*points[NET_POSTS[0]]), self.to_screen(*points[NET_POSTS[1]]),
                     (200, 200, 200), thin, cv2.LINE_AA)
        selected = COURT_POINTS[self.selected]
        for name, point in points.items():
            at = self.to_screen(*point)
            colour = SELECTED_COLOUR if name == selected else POINT_COLOUR
            cv2.circle(display, at, max(3, int(round(4 * ui))), colour, -1 if name != selected else thin, cv2.LINE_AA)
            if name == selected:
                cv2.circle(display, at, int(round(10 * ui)), colour, thin, cv2.LINE_AA)
            cv2.putText(display, str(COURT_POINTS.index(name)), (at[0] + int(6 * ui), at[1] - int(6 * ui)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui, colour, thin, cv2.LINE_AA)

    def draw_loupe(self, display: np.ndarray, image: np.ndarray) -> None:
        """The selected point's neighbourhood at LOUPE times, in the top-right corner, with a
        crosshair on the point (or on the picture's middle while it is not placed)."""
        ui = self.session.ui
        size = int(LOUPE_PX * ui)
        name = COURT_POINTS[self.selected]
        point = self.points().get(name)
        h, w = image.shape[:2]
        cx, cy = point if point is not None else (w / 2.0, h / 2.0)
        # The region is resampled about the point itself, so the crosshair sits on its sub-pixel spot.
        matrix = np.float32([[LOUPE, 0, size / 2.0 - LOUPE * cx], [0, LOUPE, size / 2.0 - LOUPE * cy]])
        loupe = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        on_right = point is not None and self.to_screen(*point)[0] > display.shape[1] // 2
        x0 = int(10 * ui) if on_right else display.shape[1] - size - int(10 * ui)   # away from the point
        y0 = int(10 * ui)
        if x0 < 0 or y0 + size > display.shape[0]:
            return
        display[y0:y0 + size, x0:x0 + size] = loupe
        mid = (x0 + size // 2, y0 + size // 2)
        colour = SELECTED_COLOUR if point is not None else (120, 120, 120)
        gap = int(6 * ui)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(display, (mid[0] + dx * gap, mid[1] + dy * gap), (mid[0] + dx * size // 2, mid[1] + dy * size // 2),
                     colour, 1, cv2.LINE_AA)
        cv2.rectangle(display, (x0, y0), (x0 + size - 1, y0 + size - 1), (200, 200, 200), 1)
        caption = f"{COURT_POINTS.index(name)} {SHORT_NAMES[name]}" + ("" if point is not None else " (not placed)")
        thin = max(1, int(round(ui)))
        cv2.rectangle(display, (x0, y0 + size), (x0 + size - 1, y0 + size + int(22 * ui)), (25, 25, 25), -1)
        cv2.putText(display, fit_text(caption, size - int(10 * ui), 0.48 * ui, thin), (x0 + int(5 * ui), y0 + size + int(16 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ui, (255, 255, 255), thin, cv2.LINE_AA)

    def hud_lines(self) -> Tuple[str, str]:
        source = self.source
        labelled = len(source.court) if source and source.court is not None else 0
        name = COURT_POINTS[self.selected]
        points = self.points()
        missing = [str(COURT_POINTS.index(n)) for n in COURT_POINTS if n not in points]
        line1 = (f"{source.clip if source else ''}  COURT  frame {self.frame} ({self.position + 1}/{len(self.frames)})  "
                 f"{self.state()}  point {COURT_POINTS.index(name)} {name}  labelled frames {labelled}"
                 f"{'  not placed: ' + ','.join(missing) if missing else ''}{'  *unsaved*' if self.session.dirty else ''}")
        line2 = ("drag, or click=place  ,/.=point  ijkl=nudge (SHIFT fine)  v=hidden  SPACE=confirm  y=copy prev  "
                 "r=model  x=unlabel  u=undo  n/p=frame  c=clip  q=quit")
        return line1, line2
