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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from finetune.data.court_labels import COURT_POINTS, GROUND_POINTS, CourtPoints, engine_points
from finetune.data.labels import LabelFormatError
from finetune.labeling import label_window as win
from finetune.labeling.label_draw import HudItem, fit_picture, fit_text, middle_ellipsis
from finetune.labeling.label_store import load_court
from finetune.labeling.label_view import ClipView
from finetune.labeling.label_window import KEY_TAB, KEYS_CONFIRM, next_clip_key
from finetune.scoring.tracker_events import court_keypoints_by_frame, read_run
from tennis_tracker.court.court_geometry import LINE_PAIRS_14

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_tool import Tool

COURT_FRAMES = 5              # frames of each clip offered for labelling, evenly through it
GRAB_PX = 14                  # display pixels: a press this close to a point picks it up
NUDGE_PX, FINE_NUDGE_PX = 1.0, 0.25   # native pixels per i/j/k/l, and with SHIFT
LOUPE = 4                     # the loupe's magnification ...
LOUPE_PX = 240                # ... and its size on screen, at ui 1
POINT_COLOUR = (60, 220, 255)
SELECTED_COLOUR = (255, 255, 255)
LINE_COLOUR = (40, 200, 120)
OUTLINE = (0, 0, 0)           # around every point, number and loupe line, so they read on any court
NET_POSTS = COURT_POINTS[len(GROUND_POINTS):]
SHORT_NAMES = {name: name.replace("_singles", " S").replace("_service", " svc").replace("_", " ")
               for name in COURT_POINTS}
# The keys that throw work away take SHIFT, so a slip of the finger does not: key -> what it does.
SHIFTED = {ord("x"): "unlabel this frame", ord("r"): "go back to the run's points", ord("u"): "undo"}


def point_name(name: str) -> str:
    """How the view names a point everywhere: its number and its short name (3 near right)."""
    return f"{COURT_POINTS.index(name)} {SHORT_NAMES[name]}"


class CourtView(ClipView):
    """The court keypoints of one frame at a time, over the whole picture (module docstring)."""

    title = "COURT"
    tag = "court"
    help_title = "COURT"
    help = (
        ("drag a point", "move it"), ("click", "put the selected point there"), (", / .", "previous / next point"),
        ("i j k l", "nudge 1 px (SHIFT 0.25 px)"), ("v", "the point is hidden here"), ("SPACE / ENTER / c", "confirm the frame"),
        ("y", "copy the nearest labelled frame"), ("SHIFT+r", "back to the run's points"), ("SHIFT+x", "unlabel the frame"),
        ("SHIFT+u", "undo"), ("a / d, arrows", "previous / next frame"),
    ("SHIFT+n", "next clip"), ("TAB", "back to the frames"), ("m", "full screen"), ("s", "save"),
    ("q / ESC", "save and quit"), ("? / F1", "these keys"))

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        super().__init__(session, tool)
        self.model: Dict[int, CourtPoints] = {}       # frame -> the run's points there
        self.frames: List[int] = []                   # the frames offered, in order
        self.position = 0
        self.drafts: Dict[int, CourtPoints] = {}      # frame -> the points as they stand on screen
        self.history: Dict[int, List[CourtPoints]] = {}   # frame -> its earlier drafts, for u
        self.hidden: Dict[int, Set[str]] = {}         # frame -> the points v said cannot be seen there
        self.notice: Optional[str] = None             # a warning on the HUD until the next key
        self.leave_warned: Optional[int] = None       # the frame a move away from was held back once
        self.selected = 0                             # index into COURT_POINTS
        self.dragging = False
        self.hover: Optional[Tuple[float, float]] = None   # the mouse on the picture, for the loupe of a point not placed
        self.mapping = (1.0, 0, 0)                    # scale, x and y offset of the picture on screen
        self.loupe_box: Optional[Tuple[int, int, int]] = None   # (x0, y0, size) of the loupe last drawn

    # ---------------------------------------------------------------- opening a clip
    def load(self) -> None:
        """The clip at the cursor: its labels, the run's points and the frames to offer."""
        source = self.clip
        self.model, self.drafts, self.history, self.hidden = {}, {}, {}, {}
        self.notice = self.leave_warned = None
        try:
            if source.court is None:
                source.remember(source.court_path)
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
        points[name] = session.clamp(x, y)
        self.hidden.get(self.frame, set()).discard(name)
        self.change(points)

    def held_back(self) -> bool:
        """Whether leaving this frame (for another, the frames, or the next clip) is held back: the
        first time, while it has changes SPACE has not confirmed, with a warning; the same move again
        leaves them."""
        if self.unconfirmed() and self.leave_warned != self.frame:
            self.leave_warned = self.frame
            self.notice = "CHANGES NOT CONFIRMED: SPACE keeps them, the same key again leaves them"
            self.revision += 1
            return True
        self.leave_warned = None
        return False

    def handle_key(self, code: int, direction: int) -> None:
        if (code == KEY_TAB or next_clip_key(code)) and self.frames and self.held_back():
            return
        super().handle_key(code, direction)

    def unconfirmed(self) -> bool:
        """Whether this frame's points were changed here and not confirmed (SPACE) since."""
        court = self.source.court if self.source and self.source.court is not None else {}
        return bool(self.history.get(self.frame)) and self.points() != court.get(self.frame)

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
        session.touch("court", source=source)
        self.revision += 1
        nxt = next((k for k in range(self.position + 1, len(self.frames)) if self.frames[k] not in source.court), None)
        self.go(nxt if nxt is not None else min(self.position + 1, len(self.frames) - 1))

    def unlabel(self) -> None:
        source = self.source
        if source is not None and source.court is not None and source.court.pop(self.frame, None) is not None:
            self.session.touch("court", source=source)
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
        self.hidden.setdefault(self.frame, set()).add(name)
        if name in self.points():
            points = dict(self.points())
            del points[name]
            self.change(points)
        else:
            self.revision += 1

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
        """To the frame at `position`; held back once, with a warning, while this frame has changes
        SPACE has not confirmed (the same move again leaves them)."""
        if not self.frames:
            return
        position = max(0, min(position, len(self.frames) - 1))
        if position != self.position and self.held_back():
            return
        self.position = position
        source = self.source
        if source is not None:
            local = min(range(len(source.rows)), key=lambda k: abs(source.rows[k].frame - self.frame))
            self.session.set_cursor(source.start + local)
        self.revision += 1

    # ---------------------------------------------------------------- input
    def view_key(self, code: int, direction: int) -> None:
        if self.notice is not None:
            self.notice = None
            self.revision += 1
        if not self.frames:
            return
        if code in SHIFTED or code in (ord(chr(key).upper()) for key in SHIFTED):
            key = ord(chr(code).lower())
            if code == key and not win.shift_held():
                self.notice = f"SHIFT+{chr(key)} to {SHIFTED[key]}"
                self.revision += 1
            else:
                {ord("x"): self.unlabel, ord("r"): self.reset, ord("u"): self.undo}[key]()
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
        elif direction:
            self.go(self.position + direction)

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """The screen pixel frame point (x, y) falls in (to_frame's inverse: its middle maps back)."""
        scale, off_x, off_y = self.mapping
        return math.floor(x * scale + off_x), math.floor(y * scale + off_y)

    def to_frame(self, x: int, y: int) -> Tuple[float, float]:
        scale, off_x, off_y = self.mapping
        return (x + 0.5 - off_x) / scale, (y + 0.5 - off_y) / scale

    def on_mouse(self, event: int, x: int, y: int, flags: int = 0) -> None:
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
            points[COURT_POINTS[self.selected]] = session.clamp(fx, fy)
            self.hidden.get(self.frame, set()).discard(COURT_POINTS[self.selected])
            self.drafts[self.frame] = points        # one undo step for the whole drag, taken on the press
            self.revision += 1
        elif event == cv2.EVENT_MOUSEMOVE and COURT_POINTS[self.selected] not in self.points():
            self.hover = self.to_frame(x, y)        # the loupe shows where a click would put the point
            self.revision += 1
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    # ---------------------------------------------------------------- drawing
    def ready(self) -> bool:
        return bool(self.frames)

    def draw(self, canvas: np.ndarray) -> None:
        image = self.clip.frame(self.frame)
        picture, scale, off_x, off_y = fit_picture(image, self.session.view_w, self.session.view_h)
        self.mapping = (scale, off_x, off_y)
        canvas[off_y:off_y + picture.shape[0], off_x:off_x + picture.shape[1]] = picture
        self.draw_court(canvas)
        self.draw_loupe(canvas, image)

    def draw_court(self, display: np.ndarray) -> None:
        """The court's lines between the points placed, then the points, the selected one ringed,
        each with its number: outlined in black so they read on any court, and a number that would
        cover another moved off it."""
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
        taken: List[Tuple[int, int, int, int]] = []
        scale = 0.45 * ui
        for name, point in points.items():
            at = self.to_screen(*point)
            colour = SELECTED_COLOUR if name == selected else POINT_COLOUR
            radius = max(3, int(round(4 * ui)))
            cv2.circle(display, at, radius + thin, OUTLINE, thin + 1, cv2.LINE_AA)
            cv2.circle(display, at, radius, colour, -1 if name != selected else thin, cv2.LINE_AA)
            if name == selected:
                cv2.circle(display, at, int(round(10 * ui)), OUTLINE, thin + 2, cv2.LINE_AA)
                cv2.circle(display, at, int(round(10 * ui)), colour, thin, cv2.LINE_AA)
            text = str(COURT_POINTS.index(name))
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thin)
            x, y = at[0] + int(6 * ui), at[1] - int(6 * ui)
            for step in (0, 1, -1, 2, -2, 3):            # off any number already drawn, nearest first
                box = (x, y + step * (text_h + int(4 * ui)) - text_h, x + text_w, y + step * (text_h + int(4 * ui)))
                if not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in taken):
                    break
            taken.append(box)
            cv2.putText(display, text, (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, scale, OUTLINE, thin + 2, cv2.LINE_AA)
            cv2.putText(display, text, (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thin, cv2.LINE_AA)

    def picture_box(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """(left, top, right, bottom) of the picture on screen, the letterbox left out."""
        scale, off_x, off_y = self.mapping
        return off_x, off_y, off_x + int(round(image.shape[1] * scale)), off_y + int(round(image.shape[0] * scale))

    def draw_loupe(self, display: np.ndarray, image: np.ndarray) -> None:
        """The selected point's neighbourhood at LOUPE times, in a top corner of the picture away
        from it, with a crosshair on the point; while it is not placed, on the mouse (where a click
        would put it), or the picture's middle before the mouse has moved. It, its caption and its
        lines all stay on the picture, and scale with ui."""
        ui = self.session.ui
        thin = max(1, int(round(ui)))
        margin, caption_h = int(10 * ui), int(22 * ui)
        left, top, right, bottom = self.picture_box(image)
        size = min(int(LOUPE_PX * ui), right - left - 2 * margin, bottom - top - 2 * margin - caption_h)
        if size < int(60 * ui):
            self.loupe_box = None
            return
        name = COURT_POINTS[self.selected]
        point = self.points().get(name)
        h, w = image.shape[:2]
        cx, cy = point if point is not None else self.hover if self.hover is not None else (w / 2.0, h / 2.0)
        # The region is resampled about the point itself, so the crosshair sits on its sub-pixel spot:
        # frame coordinates put pixel k's centre at k + 0.5, warpAffine at k, and the crosshair is on
        # the loupe's pixel size // 2.
        matrix = np.array([[LOUPE, 0, size // 2 - LOUPE * (cx - 0.5)], [0, LOUPE, size // 2 - LOUPE * (cy - 0.5)]], np.float32)
        loupe = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        on_right = self.to_screen(cx, cy)[0] > (left + right) // 2
        x0 = left + margin if on_right else right - size - margin     # away from the point
        y0 = top + margin
        display[y0:y0 + size, x0:x0 + size] = loupe
        self.loupe_box = (x0, y0, size)
        mid = (x0 + size // 2, y0 + size // 2)
        colour = SELECTED_COLOUR if point is not None else (120, 120, 120)
        gap = int(6 * ui)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            start, end = (mid[0] + dx * gap, mid[1] + dy * gap), (mid[0] + dx * size // 2, mid[1] + dy * size // 2)
            cv2.line(display, start, end, OUTLINE, thin + 2, cv2.LINE_AA)
            cv2.line(display, start, end, colour, thin, cv2.LINE_AA)
        cv2.rectangle(display, (x0, y0), (x0 + size - 1, y0 + size - 1), (200, 200, 200), thin)
        caption = point_name(name) + ("" if point is not None else " (not placed)")
        cv2.rectangle(display, (x0, y0 + size), (x0 + size - 1, y0 + size + caption_h), (25, 25, 25), -1)
        cv2.putText(display, fit_text(caption, size - int(10 * ui), 0.48 * ui, thin), (x0 + int(5 * ui), y0 + size + int(16 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ui, (255, 255, 255), thin, cv2.LINE_AA)

    def hud_lines(self) -> Tuple[List[HudItem], List[HudItem]]:
        source = self.clip
        labelled = len(source.court) if source.court is not None else 0
        points = self.points()
        hidden = self.hidden.get(self.frame, set())
        missing = [str(COURT_POINTS.index(n)) for n in COURT_POINTS if n not in points and n not in hidden]
        unseen = [str(COURT_POINTS.index(n)) for n in COURT_POINTS if n not in points and n in hidden]
        line1: List[HudItem] = [(1, middle_ellipsis(source.clip)), (0, "COURT"),
                                (0, f"frame {self.frame} ({self.position + 1}/{len(self.frames)})"), (0, self.state()),
                                (0, f"point {point_name(COURT_POINTS[self.selected])}"), (3, f"labelled frames {labelled}"),
                                (2, "not placed: " + ",".join(missing) if missing else ""),
                                (2, "hidden: " + ",".join(unseen) if unseen else "")]
        if self.notice:
            line1.insert(0, (0, self.notice))
        line2: List[HudItem] = [(0, "drag, or click=place"), (0, ",/.=point"), (1, "ijkl=nudge (SHIFT fine)"), (1, "v=hidden"),
                                (0, "SPACE=confirm"), (2, "y=copy labelled"), (2, "SHIFT+r=run's points"),
                                (2, "SHIFT+x=unlabel"), (1, "SHIFT+u=undo"), (0, "a/d=frame"), (2, "SHIFT+n=next clip"),
                                (1, "TAB=frames"), (0, "?=keys")]
        return line1, line2
