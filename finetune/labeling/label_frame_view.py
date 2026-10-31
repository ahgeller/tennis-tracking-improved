"""The label tool's single-frame view: the frame warped to the view's zoom and origin with the
trail, the mark, the badges and the overview inset drawn on it, and the HUD bar under it: the
clip's timeline (``label_timeline``) and the two text lines with their clickable chips.

``FrameView`` draws from the session's rows, cursor, view geometry and suggestions (the tool's
``render`` decides when a redraw is due, see ``Tool.render_state``) and asks the tool for the
toggles a chip click names. The chip and inset boxes it records while drawing are what a click on
the bar or the inset is matched against.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_candidates import CANDIDATE_KEYS, CANDIDATE_TEXT
from finetune.labeling.label_draw import (FILTERS, GHOST, HIT, LANDING, RALLY, SEPARATOR, ChipRow, HudItem, draw_badge,
                                          draw_candidate_point, draw_ghost, draw_landing_point, draw_mark, draw_outline_badge,
                                          draw_ticks, enhance, fit_text, hit, hit_badge, hud_text, label_mark_colour,
                                          landing_badge, mark_plan, middle_ellipsis, review_badge)
from finetune.labeling.label_store import Row
from finetune.labeling.label_timeline import KIND_COLOURS, Timeline
from finetune.labeling.label_view import View
from finetune.refiner.refine import Suggestion
from finetune.data.detector_metrics import REFERENCE_HEIGHT

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_tool import Tool

INSET_SHARE = 0.22        # width of the overview inset as a share of the view
THUMBNAILS_KEPT = 64      # small copies of frames kept for the inset


class FrameView(View):
    """The single-frame view's picture and HUD, drawn for the tool's current frame."""

    overview = False
    help_title = "FRAME"
    help = (
        ("SPACE / ENTER / c", "the mark is good: next frame"), ("click", "the ball is here"),
        ("right click", "centre the view here"), ("wheel / + -", "zoom"), ("y", "take the hint"),
        ("x", "keep the draft, not the hint"), ("v", "no ball on this frame"), ("r", "accept the run's sure frames"),
        ("u", "undo on this frame"), ("i j k l", "nudge 1 px (SHIFT 0.25 px)"), ("a / d, arrows", "previous / next frame"),
        ("1 2 4 6 8", "frames a / d move"), ("f / b", "next / previous uncertain or flagged"),
        ("e", "next frame the hint disagrees with"), ("/", "a landing here (again: not)"), (".", "a racket hit here"),
        (",", "who hit it: near, far, not said"), ("[ / ]", "a rally starts / ends here"),
        ("; / SHIFT+;", "next / previous open candidate"), ("\\", "the candidates here are not events"),
        ("TAB", "the grid of crops"), ("w", "the trajectory sheet"), ("z", "the grid's size"),
        ("SHIFT+TAB", "back to the players or court view, if one was open"),
    ("t", "trail"), ("g", "auto-centre on / off"), ("h", "picture filter"), ("o", "inset"),
    ("m", "full screen"), ("s", "save"), ("q / ESC", "save and quit"), ("? / F1", "these keys"))

    def render(self) -> np.ndarray:
        return self.tool.render_frame()

    def handle_key(self, code: int, direction: int) -> None:
        keys = self.tool.keys
        if not keys.plain_key(code, direction):
            keys.common_key(code)

    def on_mouse(self, event: int, x: int, y: int, flags: int = 0) -> None:
        self.tool.pointer.frame_event(event, x, y, flags)

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        super().__init__(session, tool)
        self.inset_box = None         # where the inset is on screen, so a click in it can re-centre there
        self.inset_scale = 1.0        # frame px -> inset px
        self.centre_button = None     # where the HUD chip that toggles auto-centring is, in display pixels
        self.step_buttons = []        # ... and the x1..x8 ones, as (box, frames) pairs
        self.filter_button = None     # ... and the filter and inset chips
        self.inset_button = None
        self.thumbs: Dict[tuple, np.ndarray] = {}   # (clip, frame, width) -> small copy, for the inset
        self.timeline = Timeline(session)           # the strip along the top of the HUD

    # ---------------------------------------------------------------- the picture
    def draw(self, image: np.ndarray, before: Optional[np.ndarray], mode: str, row: Row,
             hint: Optional[Suggestion], offer: Optional[Suggestion]) -> np.ndarray:
        """A fresh canvas: the warped frame with its marks, the inset, the badges and the HUD."""
        canvas = self.warp_picture(image, before, mode)
        display = canvas[:self.session.view_h]
        self.draw_marks(display, row, hint, offer)
        self.draw_inset(display, image, row, self.session.ui)
        self.draw_badges(display, row)
        self.draw_candidate_banner(display)
        self.draw_hud(canvas, row, hint, offer)
        return canvas

    def draw_candidate_banner(self, display: np.ndarray) -> None:
        """While a click would confirm the tracker's candidate between two frames (; came to it,
        session.awaiting_candidate), a band across the whole top of the picture in the event's
        colour says so: the click places an event, not the ball label it places everywhere else."""
        event = self.session.awaiting_candidate
        if event is None:
            return
        ui = self.session.ui
        text_h = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, 2)[0][1]
        height = int(24 * ui) + text_h + int(round(6 * ui))   # over the badges along the top (draw_badge), wholly
        colour = KIND_COLOURS[event.kind]
        cv2.rectangle(display, (0, 0), (display.shape[1] - 1, height), colour, -1)
        what = "WHERE THE BALL BOUNCED" if event.kind == "landing" else "WHERE THE RACKET MET THE BALL"
        text = fit_text(f"CLICK {what}: CONFIRMS THE TRACKER'S {CANDIDATE_TEXT[event.kind][:-1]} BETWEEN FRAMES"
                        "   any key: no click", display.shape[1] - int(24 * ui), 0.6 * ui, 2)
        cv2.putText(display, text, (int(12 * ui), (height + text_h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui,
                    (0, 0, 0), 2, cv2.LINE_AA)

    def warp_picture(self, image: np.ndarray, before: Optional[np.ndarray], mode: str) -> np.ndarray:
        """A fresh canvas for picture and HUD with the frame drawn into its top at the view's
        scale and origin, filtered by `mode` (the motion filter diffs against `before`)."""
        session = self.session
        x0, y0 = session.origin
        # Resize with exact sub-pixel origin handling via warpAffine. warpAffine puts pixel centres on
        # whole numbers while frame coordinates put them at k + 0.5 (see Tool.to_view): the half-pixel
        # terms line the drawn picture up with where a click is stored.
        s = session.scale
        matrix = np.array([[s, 0.0, s * (0.5 - x0) - 0.5], [0.0, s, s * (0.5 - y0) - 0.5]], dtype=np.float64)
        interpolation = cv2.INTER_LINEAR if session.zoom < 8 else cv2.INTER_NEAREST
        # One canvas for picture and HUD, the picture warped straight into its top: building them
        # apart and stacking them copied the whole view once more per redraw. A fresh canvas each
        # time, so an image handed out earlier (the cache, a caller's reference) is never drawn over.
        canvas = np.empty((session.view_h + session.hud_h, session.view_w, 3), np.uint8)
        display = canvas[:session.view_h]
        cv2.warpAffine(image, matrix, (session.view_w, session.view_h), dst=display, flags=interpolation)
        if mode != "off":
            previous = None
            if before is not None:
                previous = cv2.warpAffine(before, matrix, (session.view_w, session.view_h), flags=interpolation)
            display[:] = enhance(display, mode, previous)
        return canvas

    def draw_marks(self, display: np.ndarray, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> None:
        """The trail, the mark (the suggestion on offer or the label, with the other ticked beside
        it) or, on a frame with no ball and no offer, the dashed guess; and the landing point, where
        the frame's landing has one (the click a landing with no ball label under it asks for)."""
        session = self.session
        landing = session.landing_for(session.cursor)
        touched = session.source.landings.get(landing) if landing is not None else None
        if touched is not None:
            draw_landing_point(display, *session.to_view(*touched), session.ui)
        worn_hit = session.hit_for(session.cursor)
        contact = session.source.hits[worn_hit].point if worn_hit is not None else None
        if contact is not None:
            draw_landing_point(display, *session.to_view(*contact), session.ui, HIT)
        for event in session.open_candidates(session.cursor):
            if event.point is not None:
                draw_candidate_point(display, *session.to_view(*event.point), session.ui, KIND_COLOURS[event.kind])
        ui = session.ui
        thin = max(1, int(round(ui)))
        if session.trail:
            for previous in session.rows[max(0, session.cursor - 12):session.cursor]:
                if previous.visible and previous.clip == row.clip:
                    cv2.circle(display, session.to_view(previous.x, previous.y), max(3, int(3 * ui)),
                               (200, 200, 60), thin, cv2.LINE_AA)
        radius = int(max(8, 6 * session.zoom) * ui)
        # The suggestion on offer is the mark SPACE takes, drawn as the label would be, in orange -
        # bright when it is sure, dull when only a guess; the draft it replaces is ticked in grey.
        plan = mark_plan(row, hint, offer, label_mark_colour(row))
        for point, colour in plan.ticks:
            draw_ticks(display, *session.to_view(*point), colour, ui)
        if plan.mark is not None:
            draw_mark(display, *session.to_view(*plan.mark), radius, plan.colour, ui)
        elif session.ghost is not None:
            draw_ghost(display, *session.to_view(*session.ghost), radius, ui)

    def thumbnail(self, frame: int, image: np.ndarray, width: int, height: int) -> np.ndarray:
        key = (self.session.source.clip, frame, width)
        thumb = self.thumbs.get(key)
        if thumb is None:
            thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if len(self.thumbs) >= THUMBNAILS_KEPT:
                self.thumbs.pop(next(iter(self.thumbs)))
            self.thumbs[key] = thumb
        return thumb

    def draw_inset(self, display: np.ndarray, image: np.ndarray, row: Row, ui: float) -> None:
        """A small copy of the whole frame in the bottom-right corner while zoomed in, with the part on
        screen boxed and the ball dotted. When the detector's guess is nowhere near, this is where the
        real ball is found without zooming out: a click on it brings that spot into the zoomed view."""
        session = self.session
        self.inset_box = None
        if not session.inset or session.zoom < 1.5:
            return
        width = int(session.view_w * INSET_SHARE)
        height = max(1, int(round(width * session.height / session.width)))
        margin = int(10 * ui)
        left, top = session.view_w - width - margin, session.view_h - height - margin
        if left < 0 or top < 0:
            return
        panel = self.thumbnail(row.frame, image, width, height).copy()
        scale = width / session.width
        self.inset_scale = scale
        x0, y0 = session.origin
        x1, y1 = x0 + session.view_w / session.scale, y0 + session.view_h / session.scale
        cv2.rectangle(panel, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale) - 1, int(y1 * scale) - 1),
                      (255, 255, 255), 1)
        point = session.target_point()
        if point is not None:
            cv2.circle(panel, (int(point[0] * scale), int(point[1] * scale)), max(2, int(2 * ui)),
                       (0, 255, 255) if row.visible else GHOST, -1, cv2.LINE_AA)
        display[top:top + height, left:left + width] = panel
        cv2.rectangle(display, (left - 1, top - 1), (left + width, top + height), (110, 110, 110), 1)
        self.inset_box = (left, top, left + width - 1, top + height - 1)

    def draw_badges(self, display: np.ndarray, row: Row) -> None:
        """The row's review badge at the top right; to its left the LANDING badge on a frame marked
        as one (/, LANDING 1/2 on either side of a landing between two frames), the HIT badge (.,
        with its hitter), RALLY START or RALLY END ([ and ]), and an outlined badge for each of the
        tracker's candidates on the frame nobody has answered yet; the frames-left counter at the
        top left."""
        session = self.session
        ui = session.ui
        edge = session.view_w
        badge = review_badge(row)
        if badge is not None:
            edge = draw_badge(display, badge[0], badge[1], (0, 0, 0), ui, right_edge=edge)[0]
        landing = landing_badge(session.landing_for(session.cursor), row.frame)
        if landing is not None:
            edge = draw_badge(display, landing, LANDING, (0, 0, 0), ui, right_edge=edge)[0]
        worn_hit = session.hit_for(session.cursor)
        hit_text = hit_badge(worn_hit, row.frame, session.source.hits[worn_hit].hitter if worn_hit is not None else None)
        if hit_text is not None:
            edge = draw_badge(display, hit_text, HIT, (0, 0, 0), ui, right_edge=edge)[0]
        rally = session.rally_mark(session.cursor)
        if rally is not None:
            edge = draw_badge(display, f"RALLY {rally.upper()}", RALLY, (0, 0, 0), ui, right_edge=edge)[0]
        for event in session.open_candidates(session.cursor):
            text = CANDIDATE_TEXT[event.kind] + (f" {event.hitter.upper()}" if event.hitter else "")
            edge = draw_outline_badge(display, text, KIND_COLOURS[event.kind], ui, right_edge=edge)[0]
        remaining = len(session.rows) - sum(r.reviewed for r in session.rows)
        counter = "ALL DONE" if remaining == 0 else f"{remaining} LEFT"
        fill, text_colour = ((70, 190, 70), (0, 0, 0)) if remaining == 0 else ((50, 50, 50), (255, 255, 255))
        draw_badge(display, counter, fill, text_colour, session.ui)

    # ---------------------------------------------------------------- the HUD
    def place_items(self) -> List[HudItem]:
        """Where in the clip, the run and the uncertain walk the cursor is."""
        session = self.session
        source = session.source
        items: List[HudItem] = [(1, f"{middle_ellipsis(source.clip)} [{session.cursor - source.start + 1}/{len(source.rows)}]")]
        if len(session.sources) > 1:
            items += [(5, f"clip {session.sources.index(source) + 1}/{len(session.sources)}"),
                      (6, f"run {session.cursor + 1}/{len(session.rows)}")]
        if session.walk:
            left = sum(1 for i in session.walk if not session.rows[i].reviewed)
            items.append((2, f"UNCERTAIN ONLY {session.walk_pos + 1}/{len(session.walk)} ({left} left)"))
        return items

    def hud_place(self) -> str:
        return SEPARATOR.join(text for _, text in self.place_items())

    def hud_state(self, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> str:
        """The label, and what the suggestion says about it."""
        session = self.session
        state = "NOT VISIBLE" if not row.visible else f"({row.x:.1f}, {row.y:.1f})"
        if offer is not None:
            sure = "sure" if offer["trusted"] else "UNSURE, look closely"
            if row.visible:
                gap = math.hypot(offer["x"] - row.x, offer["y"] - row.y) * REFERENCE_HEIGHT / session.height
                return f"HINT ({offer['x']:.1f}, {offer['y']:.1f}) {sure}  grey draft {gap:.1f}px off"
            return f"HINT ({offer['x']:.1f}, {offer['y']:.1f}) {sure}  draft: no ball"
        if hint is not None and not row.visible:
            return state + "  hint: the video shows a ball here (y takes it)"
        if hint is not None:
            gap = math.hypot(hint["x"] - row.x, hint["y"] - row.y) * REFERENCE_HEIGHT / session.height
            state += f"  hint {gap:.1f}px" if gap >= 0.05 else "  hint: here"
            verdict = session.hint_verdict(session.cursor)
            if verdict is not None:
                state += "  hint ok" if verdict == "agrees" else "  hint differs"
            else:
                state += " (unsure)"
        return state

    def hud_items(self, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]
                  ) -> Tuple[List[HudItem], List[HudItem]]:
        """The two lines of the HUD as items (label_draw.fit_items drops the least important whole):
        this frame (where it is, its label, where the label came from while it is unreviewed, its
        landing, hit and rally marks, the clip's counts of them); then what the frame waits for or
        the keys that matter, and the run's counts."""
        session = self.session
        source = session.source
        line1: List[HudItem] = self.place_items() + [(0, f"frame {row.frame}"), (1, self.hud_state(row, hint, offer))]
        if not row.reviewed and row.source != "none":   # "none": no guess came with the frame, nothing to say
            line1.append((4, f"from {row.source} " + (f"{row.conf:.2f}" if row.source == "det" else "conf ?")))
        line1.append((2, "EDITED" if row.edited else "ok" if row.reviewed else "unreviewed"))
        worn_hit = session.hit_for(session.cursor)
        rally = session.rally_at(session.cursor)
        line1 += [(2, landing_badge(session.landing_for(session.cursor), row.frame) or ""),
                  (2, hit_badge(worn_hit, row.frame, source.hits[worn_hit].hitter if worn_hit is not None else None) or ""),
                  (2, f"rally {rally[0]}/{rally[1]}" if rally is not None else ""),
                  (5, f"landings {len(source.landings)}")]
        if source.hits or source.rally_starts or source.rally_ends:
            line1 += [(5, f"hits {len(source.hits)}"), (5, f"rallies {len(source.rally_starts)}")]
        line1.append((6, f"zoom x{session.zoom:.1f}"))
        return line1, self.key_items(offer) + self.count_items()

    def key_items(self, offer: Optional[Suggestion]) -> List[HudItem]:
        """What the frame waits for first, so a narrow bar cuts the counts, not the ask: a landing's
        or a hit's click, a candidate's, the candidates on the frame; else the keys that matter."""
        session = self.session
        open_here = session.open_candidates(session.cursor)
        if session.awaiting_landing is not None:
            return [(0, "CLICK WHERE THE BALL BOUNCED"), (1, "/=unmark"), (1, "other key=no point")]
        if session.awaiting_hit is not None:
            return [(0, "CLICK WHERE THE RACKET MET THE BALL"), (1, ".=unmark"), (1, "other key=no point")]
        if session.awaiting_candidate is not None:
            event = session.awaiting_candidate
            where = "BALL BOUNCED" if event.kind == "landing" else "RACKET MET IT"
            return [(0, f"{CANDIDATE_TEXT[event.kind]} BETWEEN FRAMES: CLICK WHERE THE {where}"), (1, "\\=no"), (1, ";=next")]
        if session.candidate_stop == session.cursor and any(
                session.candidate_between(session.cursor, event.kind) is not None for event in open_here):
            return [(0, " ".join(CANDIDATE_TEXT[event.kind] for event in open_here) + " BETWEEN FRAMES"),
                    (0, ";=answer it here (then click)"), (1, "\\=no")]
        if open_here:
            return ([(0, f"{CANDIDATE_TEXT[event.kind]} {CANDIDATE_KEYS[event.kind]}=yes") for event in open_here]
                    + [(1, "\\=no"), (1, ";=next"), (2, "?=keys")])
        if offer is not None:
            keys = [(0, "SPACE=take hint"), (1, "x=keep draft")]
        else:
            keys = [(0, "SPACE=good"), (1, "y=take hint")]
        return keys + [(1, "v=no ball"), (2, "/=landing"), (2, ".=hit"), (0, "?=keys")]

    def count_items(self) -> List[HudItem]:
        """How much of the run is reviewed, uncertain, flagged and disputed, and how many of the
        tracker's candidates are left."""
        session = self.session
        rows = session.rows
        disputed = session.disputed_count()
        candidates = session.open_candidate_count()
        flagged = sum(1 for r in rows if r.suspect)
        return [(3, f"reviewed {sum(r.reviewed for r in rows)}/{len(rows)}"),
                (4, f"uncertain left {sum(1 for r in rows if r.uncertain and not r.reviewed)}"),
                (3, f"flagged left {flagged}" if flagged else ""),
                (3, f"hint differs {disputed} (e)" if disputed else ""),
                (3, f"candidates left {candidates} (;)" if candidates else "")]

    def hud_lines(self, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> Tuple[str, str]:
        """The HUD's two lines in full, before any item is dropped to fit."""
        line1, line2 = self.hud_items(row, hint, offer)
        return SEPARATOR.join(t for _, t in line1 if t), SEPARATOR.join(t for _, t in line2 if t)

    def draw_hud(self, canvas: np.ndarray, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> None:
        """The timeline, and under it the two text lines and the clickable chips, along the bottom
        of the single-frame view."""
        session = self.session
        ui = session.ui
        self.timeline.draw(canvas[session.view_h:session.view_h + session.timeline_h], session.view_h)
        hud = canvas[session.view_h + session.timeline_h:]
        hud[:] = 0
        line1, line2 = self.hud_items(row, hint, offer)
        # Buttons along the right end of line 2, laid out right to left and measured before the text,
        # so a narrow window drops items of line 2 instead of running under them; line 1 runs over
        # them, its width its own. Every box is kept for click_hud: they are all clickable.
        chips = ChipRow(hud, session.hud_h - session.timeline_h, session.view_h + session.timeline_h, ui)
        pad = chips.pad
        # Lit green means on; the words stay short so the text lines keep their room
        edge, self.centre_button = chips.chip(session.view_w - pad, "g  CENTRE", session.follow)
        edge, self.inset_button = chips.chip(edge - pad, "o  INSET", session.inset)
        edge, self.filter_button = chips.chip(edge - pad, f"h  {FILTERS[session.filter].upper() if session.filter else 'FILTER'}",
                                              session.filter != 0)
        self.step_buttons = []
        for jump in (8, 6, 4, 2, 1):                     # right to left, so x1 comes out leftmost
            edge, box = chips.chip(edge - pad, f"x{jump}", jump == session.step)
            self.step_buttons.append((box, jump))
        edge = chips.caption(edge, "a/d step")
        hud_text(hud, line1, line2, ui, unsaved=session.dirty, line2_right=edge)

    # ---------------------------------------------------------------- clicks on what was drawn
    def click_hud(self, x: int, y: int) -> None:
        """A left click on the HUD bar: the frame under it on the timeline, or the chip under it."""
        if y < self.session.view_h + self.session.timeline_h:
            frame = self.timeline.frame_at(x)
            if frame is not None:
                self.tool.go_to_frame(frame)
            return
        if hit(self.centre_button, x, y):
            self.tool.toggle_follow()
        elif hit(self.filter_button, x, y):
            self.tool.cycle_filter()
        elif hit(self.inset_button, x, y):
            self.tool.toggle_inset()
        for box, jump in self.step_buttons:
            if hit(box, x, y):
                self.session.step = jump

    def inset_hit(self, x: int, y: int) -> bool:
        return hit(self.inset_box, x, y)

    def click_inset(self, x: int, y: int) -> None:
        """Either button in the overview: bring that part of the frame into the zoomed view. The
        left button does not place a label from a thumbnail - nothing there is precise enough."""
        session = self.session
        left, top = self.inset_box[0], self.inset_box[1]
        session.centre_on((x + 0.5 - left) / self.inset_scale, (y + 0.5 - top) / self.inset_scale)
