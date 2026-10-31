"""The label tool's trajectory sheet (w): the whole frame, dimmed, with every label of a stretch
of label frames drawn on it and joined in order, so a label off the ball's flight is plain at a
glance. It only finds frames: a click opens one, and nothing here edits a label. Like the grid
(``label_grid``) it owns its state, picture, clicks and keys, reads the session and moves its
cursor, and asks the tool only to follow the ball.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_draw import HINT_TRUSTED, HINT_UNSURE, HudItem, hud_text, middle_ellipsis
from finetune.labeling.label_store import Row
from finetune.labeling.label_timeline import Timeline
from finetune.labeling.label_view import View
from finetune.labeling.label_window import KEY_TAB

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_tool import Tool

SHEET_ROWS = 60           # label frames on one trajectory sheet (w)


class SheetView(View):
    """The whole frame, dimmed, with every label of a stretch of SHEET_ROWS label frames drawn on
    it, joined in order, and the suggestions as small orange dots. The labels trace the ball's
    flight, so one that is off it - or a run of them following a player - is plain at a glance,
    and the orange path says where the video itself puts the ball. It only finds frames: a click
    on a point opens that frame, and nothing here edits a label.

    A motion trace was tried as the background and dropped: the players' movement swamped it,
    and the ball's streak is a pixel or two wide at this scale, under the labels anyway."""

    help_title = "TRAJECTORY"
    help = (
        ("click a point", "open that frame"), ("a / d, arrows", "the stretch before / after"),
        ("w / right click", "back to the single frame"), ("TAB", "the grid"), ("q / ESC", "save and quit"),
        ("? / F1", "these keys"))

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        super().__init__(session, tool)
        self.start = 0                          # the first row on the sheet
        self.points: List[Tuple[int, int, int]] = []   # (row index, view x, view y) of each label drawn
        self.cache_key: Optional[tuple] = None
        self.cache: Optional[np.ndarray] = None
        self.timeline = Timeline(session)       # the clip's timeline along the top of the HUD

    def open(self) -> None:
        session = self.session
        first, end = session.clip_bounds(session.cursor)
        self.start = max(first, min(session.cursor - SHEET_ROWS // 2, end - SHEET_ROWS))
        self.cache_key = None
        super().open()

    def page(self, direction: int) -> None:
        first, end = self.session.clip_bounds(self.start)
        start = max(first, min(self.start + direction * SHEET_ROWS, end - SHEET_ROWS))
        if start != self.start:
            self.start = start
            self.session.set_cursor(start)   # leaving the sheet lands in the stretch it shows
            self.tool.follow_ball()

    def on_mouse(self, event: int, x: int, y: int, flags: int = 0) -> None:
        if event == cv2.EVENT_RBUTTONDOWN:
            self.close()
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK) and self.points:
            index, px, py = min(self.points, key=lambda p: (p[1] - x) ** 2 + (p[2] - y) ** 2)
            if math.hypot(px - x, py - y) <= 14 * self.session.ui:
                self.session.set_cursor(index)
                self.close()

    def handle_key(self, code: int, direction: int) -> None:
        """The trajectory sheet only finds frames: its own few keys, nothing else."""
        if code == KEY_TAB:
            self.tool.grid_view.open()
        elif code == ord("w"):
            self.close()
        elif direction:
            self.page(direction)

    def render(self) -> np.ndarray:
        session = self.session
        rows = session.rows
        end = session.clip_bounds(self.start)[1]
        indices = list(range(self.start, min(self.start + SHEET_ROWS, end)))
        key = (tuple(indices), Row.revision, session.hints.version, session.view_w, session.view_h, session.hud_h,
               session.cursor, session.ui, session.marks_revision)
        if key == self.cache_key:
            return self.cache
        selected = session.cursor
        base = session.source.frame(rows[selected].frame)
        ui = session.ui
        scale = min(session.view_w / session.width, session.view_h / session.height)
        shown_w, shown_h = max(1, int(round(session.width * scale))), max(1, int(round(session.height * scale)))
        scale = shown_w / session.width                  # what resize really does, for the points
        off_x, off_y = (session.view_w - shown_w) // 2, (session.view_h - shown_h) // 2
        picture = cv2.convertScaleAbs(base, alpha=0.45)
        canvas = np.zeros((session.view_h + session.hud_h, session.view_w, 3), np.uint8)
        canvas[off_y:off_y + shown_h, off_x:off_x + shown_w] = cv2.resize(
            picture, (shown_w, shown_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        display = canvas[:session.view_h]
        to_sheet = lambda x, y: (math.floor(x * scale + off_x), math.floor(y * scale + off_y))
        for index in indices:                            # the suggestions first, under the labels
            hint = session.suggestion_for(index)
            if hint is not None:
                cv2.circle(display, to_sheet(hint["x"], hint["y"]), max(2, int(round(2 * ui))),
                           HINT_TRUSTED if hint["trusted"] else HINT_UNSURE, -1, cv2.LINE_AA)
        self.points = self.trace(display, indices, to_sheet)
        differs = self.draw_points(display, selected)
        shown = [rows[i] for i in indices]
        line1: List[HudItem] = [(1, middle_ellipsis(rows[indices[0]].clip)), (0, "TRAJECTORY"),
                                (0, f"frames {shown[0].frame}-{shown[-1].frame}"), (2, f"{sum(r.visible for r in shown)} with a ball"),
                                (2, f"reviewed {sum(r.reviewed for r in shown)}/{len(shown)}"),
                                (1, f"hint differs {differs} (red)" if differs else "")]
        line2: List[HudItem] = [(0, "click a point=open that frame"), (1, "a/d=stretch before/after"),
                                (0, "w or right click=back"), (2, "TAB=grid"), (0, "?=keys")]
        self.timeline.draw(canvas[session.view_h:session.view_h + session.timeline_h], session.view_h)
        hud_text(canvas[session.view_h + session.timeline_h:], line1, line2, ui, unsaved=session.dirty)
        self.cache_key, self.cache = key, canvas
        return canvas

    def trace(self, display: np.ndarray, indices: List[int], to_sheet) -> List[Tuple[int, int, int]]:
        """Join the labels of `indices` in order with a grey line, broken at every frame with no
        ball; (row index, view x, view y) of each label drawn."""
        points, last = [], None
        thin = max(1, int(round(self.session.ui)))
        for index in indices:
            row = self.session.rows[index]
            if not row.visible:
                last = None
                continue
            point = to_sheet(row.x, row.y)
            if last is not None:
                cv2.line(display, last, point, (200, 200, 200), thin, cv2.LINE_AA)
            last = point
            points.append((index,) + point)
        return points

    def draw_points(self, display: np.ndarray, selected: int) -> int:
        """A dot per label on the trace (green reviewed, yellow not), ringed red where a trusted
        suggestion differs, white on the selected frame, every tenth captioned with its frame
        number; how many differ."""
        session = self.session
        ui, thin = session.ui, max(1, int(round(session.ui)))
        differs = 0
        captions: List[Tuple[int, int, int, int]] = []   # boxes of the frame numbers drawn, never overlapped
        for index, px, py in self.points:
            row = session.rows[index]
            colour = (70, 210, 70) if row.reviewed else (0, 220, 255)
            cv2.circle(display, (px, py), max(2, int(round(3 * ui))), colour, -1, cv2.LINE_AA)
            if session.hint_verdict(index) == "differs":
                differs += 1
                cv2.circle(display, (px, py), int(round(8 * ui)), (0, 0, 255), thin, cv2.LINE_AA)
            if index == selected:
                cv2.circle(display, (px, py), int(round(12 * ui)), (255, 255, 255), thin, cv2.LINE_AA)
            if (index - self.start) % 10 == 0:
                text = str(row.frame)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, thin)
                x0, y1 = px + int(6 * ui), py - int(6 * ui)
                box = (x0, y1 - text_h, x0 + text_w, y1)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in captions):
                    captions.append(box)    # where the flight folds back, a caption over another is skipped
                    cv2.putText(display, text, (x0, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, (255, 255, 255), thin,
                                cv2.LINE_AA)
        return differs
