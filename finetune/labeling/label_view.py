"""What every view of the label tool is: the one the window shows (``Tool.view``) draws the
picture, takes the keys and takes the mouse.

- ``View``: the protocol. ``render`` draws the whole window (picture, timeline and HUD),
  ``handle_key`` takes a key the tool's own guards let through (``label_keys``), ``on_mouse`` a
  mouse event (a double click comes as a press: ``label_mouse``), and ``open`` / ``close`` show it
  or go back to the single frame. ``overview`` says it has no single frame under a parked mouse;
  ``ball_view`` that it works on the ball labels, so leaving the tool records its cursor and zoom
  as the clip's resume point.
- ``ClipView``: a view that works on one clip at a time and keeps labels of its own (the player
  and court views): its clip, why it has nothing to show, its render cache, SHIFT+n to the next
  clip, TAB back to the frames and s to save.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_draw import HudLine, fit_text, hud_text
from finetune.labeling.label_timeline import Timeline
from finetune.labeling.label_window import KEY_SAVE, KEY_TAB, next_clip_key

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_source import Source
    from finetune.labeling.label_tool import Tool


class View:
    """One of the tool's views (module docstring)."""

    overview = True        # no single frame under a parked mouse
    ball_view = True       # works on the ball labels: the cursor and zoom are the clip's resume point
    help_title = ""        # the key list's (? or F1) title ...
    help: Tuple[Tuple[str, str], ...] = ()   # ... and its rows: (key, what it does), the view's own keys

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        self.session = session
        self.tool = tool

    @property
    def active(self) -> bool:
        return self.tool.view is self

    def open(self) -> None:
        self.tool.switch_to(self)

    def close(self) -> None:
        """Back to the single frame, at the cursor."""
        self.tool.switch_to(self.tool.frame_view)

    def toggle(self) -> None:
        if self.active:
            self.close()
        else:
            self.open()

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def handle_key(self, code: int, direction: int) -> None:
        raise NotImplementedError

    def on_mouse(self, event: int, x: int, y: int, flags: int = 0) -> None:
        raise NotImplementedError


class ClipView(View):
    """A view of one clip at a time with labels of its own (module docstring). A subclass says
    what it is (``title``, ``tag``), loads its clip (``load``), says whether it has something to
    show (``ready``), draws it (``draw``) and takes its own keys (``view_key``)."""

    ball_view = False
    title = ""             # on the HUD: PLAYERS, COURT
    tag = ""               # on the console: [play], [court]

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        super().__init__(session, tool)
        self.source: Optional["Source"] = None
        self.message: Optional[str] = None       # why there is nothing to show
        self.revision = 0                        # bumped on every change, for the render cache
        self.cache_key: Optional[tuple] = None
        self.cache: Optional[np.ndarray] = None
        self.timeline = Timeline(session)

    def open(self) -> None:
        """Show the clip at the cursor, loaded afresh; SHIFT+TAB from the frames comes back here."""
        super().open()
        self.tool.home_view = self
        self.source = self.session.source
        self.message = None
        self.revision += 1
        self.load()

    @property
    def clip(self) -> "Source":
        """The clip shown; there is one from the first open on."""
        assert self.source is not None, "the view has not been opened"
        return self.source

    def load(self) -> None:
        raise NotImplementedError

    def ready(self) -> bool:
        raise NotImplementedError

    def draw(self, canvas: np.ndarray) -> None:
        """The picture part of the window (above the timeline)."""
        raise NotImplementedError

    def hud_lines(self) -> Tuple[HudLine, HudLine]:
        raise NotImplementedError

    def view_key(self, code: int, direction: int) -> None:
        raise NotImplementedError

    def next_clip(self) -> None:
        """The next clip of the run, when there is one."""
        session = self.session
        end = session.clip_bounds(session.cursor)[1]
        if end >= len(session.rows):
            print(f"[{self.tag}] this is the last clip of the run")
            return
        session.set_cursor(end)
        self.open()

    def handle_key(self, code: int, direction: int) -> None:
        if next_clip_key(code):         # before n moves on: it is n with SHIFT
            self.next_clip()
        elif code == KEY_SAVE:
            self.tool.save()
        elif code == KEY_TAB:
            self.close()
        elif code == ord("m"):
            self.tool.toggle_fullscreen()
        else:
            self.view_key(code, direction)

    def render_key(self) -> tuple:
        """What the render is drawn from: an unchanged key reuses the last one."""
        session = self.session
        return (self.revision, session.view_w, session.view_h, session.hud_h, session.ui, session.dirty)

    def render(self) -> np.ndarray:
        session = self.session
        key = self.render_key()
        if key == self.cache_key and self.cache is not None:
            return self.cache
        canvas = np.zeros((session.view_h + session.hud_h, session.view_w, 3), np.uint8)
        hud = canvas[session.view_h + session.timeline_h:]
        if self.source is None or not self.ready():
            ui = session.ui
            text = self.message or f"nothing to label in {self.source.clip if self.source else 'this clip'}"
            cv2.putText(canvas, fit_text(text, session.view_w - int(40 * ui), 0.6 * ui, 1), (int(20 * ui), int(40 * ui)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, (255, 255, 255), 1, cv2.LINE_AA)
            hud_text(hud, self.title, [(0, "SHIFT+n=next clip"), (0, "TAB=frames"), (1, "q=save/quit")], ui,
                     unsaved=session.dirty)
        else:
            self.draw(canvas[:session.view_h])
            self.timeline.draw(canvas[session.view_h:session.view_h + session.timeline_h], session.view_h)
            hud_text(hud, *self.hud_lines(), session.ui, unsaved=session.dirty)
        self.cache_key, self.cache = key, canvas
        return canvas
