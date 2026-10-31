"""The clip's timeline: a strip along the top of the HUD, the clip's first label frame at its left
end and its last at its right, with the rallies drawn as bars, the landings and hits as ticks, the
tracker's candidates nobody has answered yet as short dim ticks along the bottom and the frame on
screen as a white line. A click on it goes to the frame under the pointer (``FrameView.click_hud``).
Every view draws it the same way (the single frame, the grid, the sheet); only the single frame
takes clicks on it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

from finetune.data.event_labels import pair_rally_marks
from finetune.labeling.label_draw import HIT, LANDING, RALLY, Colour
from finetune.labeling.label_candidates import is_answered

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession

BASE = (48, 48, 48)               # the strip where nothing is marked
RALLY_OPEN = (110, 120, 60)       # a rally with only its start or only its end marked, dimmer than RALLY
CANDIDATE_DIM = 0.55              # an open candidate's tick, at this share of its kind's colour
KIND_COLOURS = {"landing": LANDING, "hit": HIT, "rally_start": RALLY, "rally_end": RALLY}


class Timeline:
    """Draws the strip and maps a click on it back to a frame."""

    def __init__(self, session: "LabelSession") -> None:
        self.session = session
        self.box: Optional[Tuple[int, int, int, int]] = None   # where it was drawn, in display pixels
        self.span: Tuple[int, int] = (0, 1)                    # the frames its two ends stand for

    def draw(self, strip: np.ndarray, y0: int, selected: Optional[int] = None) -> None:
        """Draw the timeline of the clip row `selected` (the cursor's by default) across `strip`,
        whose top sits `y0` display pixels down."""
        session = self.session
        index = session.cursor if selected is None else selected
        source = session.by_clip[session.rows[index].clip]
        height, width = strip.shape[:2]
        strip[:] = 0
        pad = max(1, int(round(2 * session.ui)))
        left, right = pad, max(pad + 1, width - pad)
        top, bottom = pad, max(pad + 1, height - pad)
        first, last = source.rows[0].frame, source.rows[-1].frame
        self.span = (first, max(first + 1, last))
        self.box = (left, y0 + top, right - 1, y0 + bottom - 1)
        to_x = self.to_x
        cv2.rectangle(strip, (left, top), (right - 1, bottom - 1), BASE, -1)
        for rally in pair_rally_marks(source.rally_starts, source.rally_ends):
            marked = [frame for frame in rally if frame is not None]    # one end at least (pair_rally_marks)
            colour = RALLY if rally.complete else RALLY_OPEN
            cv2.rectangle(strip, (to_x(marked[0]), top), (max(to_x(marked[0]) + 1, to_x(marked[-1])), bottom - 1),
                          colour, -1)
        tick = max(1, int(round(session.ui)))
        for frames, colour in ((source.landings, LANDING), (source.hits, HIT)):
            for frame in frames:
                x = to_x(frame)
                cv2.rectangle(strip, (x - tick // 2, top), (x - tick // 2 + tick - 1, bottom - 1), colour, -1)
        for x, colour in self.open_candidates(source):
            low = top + (bottom - top) * 2 // 3
            cv2.rectangle(strip, (x - tick // 2, low), (x - tick // 2 + tick - 1, bottom - 1), colour, -1)
        x = to_x(session.rows[index].frame)
        cv2.rectangle(strip, (x - tick, 0), (x + tick, height - 1), (255, 255, 255), -1)

    def open_candidates(self, source) -> List[Tuple[int, Colour]]:
        """(strip x, colour) of each candidate of the clip nobody has answered yet."""
        if source.candidates is None:
            return []
        labels = self.session.label_frames(source)
        out = []
        for event in source.candidates.events:
            if not is_answered(event, source.candidates.fps, labels, source.rejected):
                colour = KIND_COLOURS[event.kind]
                out.append((self.to_x(event.frame), (int(colour[0] * CANDIDATE_DIM), int(colour[1] * CANDIDATE_DIM),
                                                     int(colour[2] * CANDIDATE_DIM))))
        return out

    def to_x(self, frame: float) -> int:
        """The strip x of source frame `frame` (box-relative to the strip, not to the display)."""
        if self.box is None:
            return 0
        left, _, right, _ = self.box
        first, last = self.span
        share = min(max((frame - first) / (last - first), 0.0), 1.0)
        return int(round(left + share * (right - left)))

    def frame_at(self, x: int) -> Optional[float]:
        """The source frame under display x on the strip, None when the strip was not drawn."""
        if self.box is None:
            return None
        left, _, right, _ = self.box
        first, last = self.span
        share = min(max((x - left) / max(1, right - left), 0.0), 1.0)
        return first + share * (last - first)
