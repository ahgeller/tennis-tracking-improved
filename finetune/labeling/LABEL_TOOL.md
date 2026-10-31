# The label tool

`finetune/labeling/label_tool.py` corrects the detector's draft labels as fast as possible.

```powershell
python finetune/labeling/label_tool.py <clip>                        # videos/<clip>.mp4 + labels/<clip>_ball.csv
python finetune/labeling/label_tool.py video22..video53             # a whole run, one clip after another
python finetune/labeling/label_tool.py video22..video53 --uncertain # only the frames worth a look
python finetune/labeling/label_tool.py <clip> --grid                # start on the grid of crops
python finetune/labeling/label_tool.py <clip> --players             # who each tracked person is
python finetune/labeling/label_tool.py <clip> --court               # the court's keypoints
```

`ft.py label <clip>` runs the same thing with the paths filled in, and pre-labels
first when there is no draft yet.

Besides the ball, the tool labels everything the court-event stages are scored
on: landings (`/`), racket hits (`.`), rallies (`[` and `]`), who each player is
(`--players`) and the court (`--court`). A pipeline run's own answers are offered
as candidates to confirm or reject (see *Candidates*), and every label type has a
scorecard (see *Scorecards*).

## Where the labels go

`--labels` defaults to the *label home* (`workspace.label_home()`): the main
checkout's `finetune/labels`, even when the tool is run from a git worktree
(`phases/<name>`, whose `finetune/labels` is only a copy made by
`tools/link_worktree_data.py`). An edit made in a copy would never reach the labels
everything else reads, so a worktree's tool writes past its copy to the real ones.
The tool prints the folder it uses as `[labels] <path>` when it starts. With the
workspace relocated (`TENNIS_WORKSPACE`) the label home is that workspace's
`labels/`; `--labels <folder>` overrides it either way.

Several clips run as one session: the last frame of one leads straight into the
first of the next, `n` and `p` cross the join, zoom carries over, and every edit is
written back to the clip the frame came from. The HUD names the clip you are in.

## A frame

Every label frame (every 2nd frame of a 60 FPS clip, every frame at 30 FPS) comes
up with the mark drawn and, on Windows, the mouse cursor already sitting on it. If
the mark is on the ball, press SPACE (or ENTER / `c`). If it is not, click where the
ball is. Both move to the next frame. The view keeps the ball in the middle and the
mouse goes with it, so the pointer is always sitting on the next mark. The mark is
dead centre on every frame, even one at the very edge of the picture: the view runs
off the frame there and that strip is drawn black; zoom stays where you set it. The
window is resizable (drag it, maximise it, or press `m` for full screen) and the
picture is redrawn at whatever size you give it.

A frame with no guess at all (the detector missed) gets a dashed "guess" ring where
the labels either side say the ball should be; the view centres there and the mouse
is parked on it, so if it is right a click is all it takes. Nothing is written until
you do click.

## Keys

| key | what it does |
|---|---|
| left click | the ball is here: save and go to the next frame |
| SPACE / ENTER / `c` | the mark is good: next frame. On an unreviewed frame with a hint on offer, the mark *is* the hint and SPACE moves the label onto it |
| `v` | the ball is not visible on this frame: next frame |
| `x` | the draft is right, not the hint on offer: confirm the label where it is and go on (on a reviewed frame the same as SPACE) |
| `y` | take the suggestion: move the point to the orange mark and go on, like a click there (see *Suggestions*) |
| `r` | accept the whole run of confident guesses up to the next iffy one |
| `n` / `d` / right arrow | next frame |
| `p` / `a` / left arrow | previous frame |
| `1` `2` `4` `6` `8` | how many frames those four move at a time; the x1..x8 buttons on the bar do the same. Holding `d` or `a` at x8 covers ground eight times as fast; SPACE, `v` and a click always move exactly one frame |
| `f` / `b` | jump to the next / previous unreviewed uncertain frame, and to any frame an audit (`ft.py check --audit --mark`) has flagged, drawn in magenta, even one you had reviewed before the audit. SPACE answers a flag as well as a click does (it is good where it is), and a later audit does not raise it again unless the label moves |
| `e` / SHIFT+`e` | jump to the next / previous frame whose label the suggestion disagrees with ("hint differs" on the bar), reviewed or not |
| `i` / `j` / `k` / `l` | nudge the point one pixel up / left / down / right (stays on the frame); with SHIFT held, a quarter of a pixel |
| `u` | undo the last change on this frame's label (not a landing, hit or rally mark: their keys take them off again) |
| `/` | this frame is a landing: the ball is on the ground here. Pressed again, it is not. The frame wears a LANDING badge (its grid tile too) and the bar counts the clip's landings. Pressed on the frame after (or before) one already marked, the ball bounced between the two: the two marks become one landing between them and both frames wear LANDING 1/2. When the landing has no ball label under it (one between two frames, or one on a frame labelled "no ball") the bar asks for a click where the ball touched the ground. Saved with the labels to `labels/<clip>_bounces.csv` (see *Landings*) |
| `.` | this frame is a racket hit (serves included), exactly as `/` is a landing: pressed again it is not; on the frame after (or before) one already marked, the contact was between the two (HIT 1/2); with no ball label under it the bar asks for a click where the racket met the ball. Saved to `labels/<clip>_hits.csv` (see *Hits*) |
| `,` | who hit this frame's hit: near (the player on the half nearer the camera), far, not said, round again. Never filled in from the tracker |
| `[` / `]` | a rally starts / ends on this frame; pressed again, it does not. A frame is never both. Saved to `labels/<clip>_rallies.csv` (see *Rallies*) |
| `;` / SHIFT+`;` (or `:`) | go to the next / previous frame wearing a candidate nobody has answered yet (see *Candidates*) |
| backslash | the candidates on this frame are wrong: reject them and go to the next one. On a frame whose candidates are all answered, it reopens the ones rejected there |
| click on the timeline | go to that frame (the strip along the top of the bar, see *The timeline*) |
| mouse wheel, `+` / `-` | smooth zoom in / out about the mouse / the ball (kept across frames) |
| right click | re-centre the view here |
| `g` / click the badge | auto-centre on / off. On (the default) every frame puts the mark in the middle of the screen with the mouse on it; off leaves the view and the mouse alone, for looking round a frame |
| `h` / click the badge | cycle the view filter: OFF, DIM WHITE (court lines, white kit and sky pulled down to a dull blue-grey so a yellow ball stops hiding on them), BALL POP (only tennis-ball yellows keep their colour), MOTION (what moved since the previous label frame lights up magenta). Remembered between sessions in `labels/label_tool.settings.json` |
| `o` / click the badge | overview inset on / off: while zoomed in, a small copy of the whole frame sits bottom-right with the part on screen boxed and the ball dotted. Click anywhere in it to bring that spot into the zoomed view, the quick way to reach a ball the detector put somewhere else entirely |
| TAB | toggle the grid: a page of crops around each ball (see *The grid*) |
| `w` | the trajectory sheet (see *The trajectory sheet*) |
| `m` | full screen on / off (the window also maximises normally) |
| `t` | toggle the trail of the previous labels |
| `s` | save |
| `q` / ESC | save and quit |

## The grid

TAB (or `--grid`) shows a page of crops around each ball, 8 or 16 to a page (`z`
switches). Click a crop to correct it without moving the crop; right click opens the
full frame; `1`-`9` or `,` / `.` pick the tile the keys act on; `y` takes its
suggestion and moves on, `x` keeps its draft instead; `v` marks it absent; `i` / `j`
/ `k` / `l` nudge it; `/` marks it as a landing; SPACE confirms the displayed page,
hints on offer (orange crosses) included; `a` / `d` change pages; `+` / `-` zoom the
crops. Hits, rallies and candidates are marked in the single-frame view only (`,`
and `.` pick tiles here).

## The trajectory sheet

`w` shows the whole frame with every label of 60 label frames drawn on it, joined
in order, and the suggestions as orange dots: a label off the ball's path stands
out at once. Click a point to open that frame; `a` / `d` move to the stretch before
/ after; `w` or right click goes back. It only finds frames, never edits.

## Suggestions

While you work, the tool measures each label frame's ball from the video itself:
the centre of what moved between that frame and its neighbours, looked for around
the current label and around where the labels either side put the ball
(`finetune/refiner/refine.py`). A small network trained on the frames you have reviewed
(`finetune/refiner/cnn_refiner.py`; "hint model" in the finetune menu retrains it) then puts
it where you would click and says whether it is sure. Without a trained version the
measurement is shown as it is.

On a frame you have not reviewed yet the suggestion is the mark: it is drawn as the
ball circle in orange, bright when it is sure, dull when only a guess; the mouse
parks on it, and SPACE (or a click without moving) takes it; the draft label it
replaces is ticked in grey beside it. So an unreviewed frame asks one thing, "is the
mark on the ball?": SPACE if it is, click where the ball is if not, `x` if the grey
draft was right after all. A sure suggestion is always offered; an unsure one is
offered only when it agrees with the draft within 3 px, where it is still the better
mark, and the dull colour says to look closely. The measurements behind those rules
are in `Suggestions.offer_for` (`finetune/labeling/label_hints.py`).

Once you have reviewed or edited a frame your label is the mark and the suggestion
is a hint beside it, four small ticks with its distance on the bar, and `y` moves the
label there. A label taken from a suggestion is recorded as source "refined". The
clip you are in is re-measured a moment after you change or confirm a label,
steered by the labels you have reviewed, so fixing a frame helps its neighbours. A
frame labelled "no ball" gets a suggestion only when the video clearly shows a ball
on its path. Where a trusted suggestion sits beside a label, the bar says whether
they agree: "hint ok" (within 3 px) or "hint differs" (further), and `e` walks
those. Agreeing never skips a frame or marks it reviewed; that is still your SPACE.
`r` (accept run) confirms the detector's own guesses, not the suggestions: it is for
frames you are not looking at.

## Landings

The court events a `--court` run exports are scored against the frames where the
ball hit the ground (`score_court_events.py`), and those frames are labelled here.
Step to the frame where the ball is on the ground and press `/`: the frame wears a
LANDING badge, the bar says `LANDING` beside its state and counts the clip's
landings, and `/` again unmarks it. Mark every landing of every rally, serves and
faults included; on a 60 FPS clip the label frame nearest the landing is close
enough (the scorer allows 0.1 s). The marks are saved with the labels to
`labels/<clip>_bounces.csv` with blank coordinates: the landing point is the ball
label of that frame (`finetune/data/bounce_labels.py`), so it follows the label if
you move it. A clip with no landings marked has no landing file, and the scorer
leaves it out; a file written by hand with coordinates keeps them.

The ball often lands between two frames: in the air on one, already up on the
next. Press `/` on both (the frame already marked, then `/` on the next or the
previous one): the two marks become one landing between them, saved as
`frame_0412.5` (on a 60 FPS clip labelled every other frame, the unlabelled frame
between the two), and both frames wear LANDING 1/2. `/` on either of the two frames
removes the landing.

A landing with no ball label under it - one between two frames, or one on a frame
labelled "no ball" (the ball hidden behind a player, say) - has no label to take its
point from, so the bar asks you to click where the ball touched the ground: the
scuff, or the spot under the ball. That click is the landing point, drawn as a
small cross and saved as the row's coordinates, and the frame stays up. Any other
key instead leaves the landing without a point, and the scorer takes the ball
labels either side of it: the midpoint of the ball labels on the two frames a
between landing lies between, or on the frame before and the frame after a landing
on a frame without a ball label, or the one of them there is (`resolve_points` in
`finetune/data/bounce_labels.py`). From the grid, `/` marks and joins the same way
but asks for no click.

## Hits

The racket hits are labelled as the landings are. Step to the frame where the
racket meets the ball and press `.`: the frame wears a HIT badge and the bar counts
the clip's hits. When the contact falls between two frames (the ball still coming in
on one, already going out on the next), press `.` on both: one hit between them, HIT
1/2 on both, saved as `frame_0412.5`. A hit with no ball label under it (between two
frames, or on a frame labelled "no ball") waits for a click where the racket met the
ball, drawn as a small red cross; any other key leaves it without a point and the
scorer takes the ball labels either side, as for a landing. Mark serves and every
stroke of the rally. `,` says who hit it: `near`, `far` or not said, shown on the
badge as HIT NEAR or HIT FAR. Only `,` says it: a hit is never given the tracker's
hitter, since the hit scorer measures the tracker's hitter against yours. A hit that
becomes one between two frames keeps the hitter it had.

## Rallies

`[` marks the frame a rally starts on: the server's toss leaving the hand (for a
serve with no toss in view, the serve's hit). `]` marks the frame it is over: the
ball is dead (its second bounce, the net, the fence, or the frame it leaves play for
good). A fault is a rally of its own: it ends where the faulted serve is dead, and
the second serve starts the next one, as the tracker counts them (every serve
starts a rally). Each start pairs with the next end; the bar says which rally a
frame is in (`rally 2/5`) and the timeline draws each rally as a bar, dimmer while
one end is still missing. The scorer leaves a rally missing an end out, and does
not count what the tracker found over it against it.

## Candidates

With a pipeline run for the clip (`evaluate_archive.py --archive finetune --mode
pipeline --court --out-dir <run folder>`; the tool reads `--candidates <run
folder>`, by default `output/label_candidates`), the run's own landings, hits and
rally starts and ends are offered on the label frames nearest them: an outlined
badge (`LANDING?`, `HIT? FAR`, `RALLY START?`, `RALLY END?`) and, for a landing or a
hit, a dashed ring where the tracker put it. The bar says what is asked
(`LANDING? /=yes  \=no  ;=next`) and counts the candidates left.

A candidate is only a suggestion: nothing about it is saved as a label, and the
scorers never read it. To confirm one, mark the event with its own key (`/`, `.`,
`[` or `]`): the label answers it. A landing or hit candidate that falls between two
label frames (on a 60 FPS clip labelled every other frame, the unlabelled frame
between them) comes up, when `;` brings you to it, waiting for the click where it
happened, and a band across the top of the picture in the event's colour says so:
the click places the landing (or the hit) between the two frames with that point,
exactly as `/` (or `.`) on both frames and a click would. Any other key, and the
click is the ball label's again. `/` and `.` themselves work on such a frame as they
always do. A backslash rejects the candidates on the frame (kept in
`labels/<clip>_events.review.json`, so a rerun of the tracker that moves an event by
a frame or two does not bring it back) and goes to the next, but never waits for a
click there: on a candidate between two frames the bar offers `;` to answer it, and
only then does a click place it. An event the tracker missed is marked with its key
like any other.

`;` walks the candidates nobody has answered (with a label of its kind within 0.1 s,
within a few seconds for a rally mark, or rejected), across the clips of a run.

## The timeline

Every view has a strip along the top of its bar: the clip from its first label
frame at the left to its last at the right, the rallies as bars, landings (blue) and
hits (red) as ticks, the candidates still open as short dim ticks along its bottom,
and the frame on screen as a white line. A click on it in the single-frame view goes
to that frame.

## Player identity (`--players`)

`--players` opens the player view on the first clip given: the tracker's person
tracks from the run folder (`--candidates`), cut into *segments*, one track id until
it drops out for over half a second. Each segment is a row of twelve crops taken
evenly through it, under the whole frame with every tracked box drawn (the selected
segment's thick, the others with their track id and identity). Say who it is and the
next segment comes up:

| key | what it does |
|---|---|
| `1` - `4` | the segment is player P1 - P4 (number the clip's players however you like, the same way all through the clip) |
| `0` / `v` | the segment is not a player (a spectator, a ball kid, someone on the next court, the umpire) |
| SPACE / ENTER / `c` | take the suggestion on the bar: the identity whose segments look most like this one, among those not on court at the same time, with a small preference for the same half; the first identity not yet used when nothing looks alike |
| `u` | take back what was said about this segment |
| `n` / `p` / arrows, a click on a row | the next / previous segment, or that one |
| `f` | the next segment nobody has said anything about |
| right click on a crop | split the segment there: the track jumped from one person to another |
| SHIFT+`n` | the next clip of the run |
| `s` | save |
| `q` / ESC | save and quit |

The crops are cut by reading the clip straight through in the background; the bar
counts them in. What is said is saved per frame and box to
`labels/<clip>_players.csv`, so it outlives the run it was made on; reopened, the
segments are cut again and split wherever the labels change identity.

## The court (`--court`)

`--court` opens the court view: five frames spread over the clip (and any already
labelled), each with the run's court keypoints drawn on it with the court's lines,
so only the wrong ones need moving. The loupe in a top corner (the one away from the
point) shows the selected point enlarged.

| key | what it does |
|---|---|
| drag a point | move it (and select it) |
| click elsewhere | put the selected point there: how the two net posts, which the model does not give, are placed |
| `,` / `.` | select the previous / next point (0 - 13 the engine's, 14 / 15 the left / right net post tops) |
| `i` / `j` / `k` / `l` | nudge the selected point a pixel; with SHIFT, a quarter |
| `v` | the selected point cannot be seen on this frame |
| SPACE / ENTER / `c` | the frame is right: its points become its label, and the next frame not yet labelled comes up |
| `y` | copy the nearest labelled frame's points (a still camera's court does not move) |
| `r` | back to the model's points |
| `x` | unlabel this frame |
| `u` | undo the last change on this frame |
| `n` / `p` / arrows | the next / previous frame, without confirming |
| SHIFT+`n` | the next clip of the run |
| `s` | save |
| `q` / ESC | save and quit |

Only what SPACE confirms is a label; a point outside the picture is never written.
Saved to `labels/<clip>_court.csv`.

## Scorecards

```powershell
python score_labels.py --out-dir output/label_candidates                    # every type, every labelled clip
python score_labels.py --out-dir output/label_candidates --holdout mixed_v1 # the held-out set only
python -m finetune.scoring.hits_score --out-dir <run> --compare before.log  # one type, against an earlier log
```

`score_labels.py` runs the landing (`score_court_events.py`), hit
(`finetune.scoring.hits_score`), rally (`rallies_score`), player identity
(`players_score`) and court (`court_score`) scorers on one pipeline run folder, each
on the clips with labels of its type, and says so for a type nobody has labelled
yet. Every scorer takes `--clips`, `--holdout`, `--include-excluded`, `--models` and
`--compare`, and reads its labels by one rule (`finetune/scoring/label_scoring.py`):
the label home unless `--archive` says otherwise, every clip with that type's file
except those in `exclude.txt`. Whether a clip's ball labels have been reviewed only
decides whether a landing or hit without its own point may take one from them (it
is still scored for its timing); it never holds the clip's other labels back.

- hits: precision, recall, timing error in frames (median and p90), hitter
  accuracy, contact point error in pixels at 1080p;
- rallies: mean overlap per labelled rally, recall and precision at an overlap of
  0.5, start and end error in seconds;
- player identity: coverage of the labelled player boxes, the share of them on the
  track given that player (each track one player, each player one track, over the
  whole clip), identity switches, spectators taken for players;
- court: keypoint error in pixels at 1080p and on the court plane in metres, the
  metres only on frames whose labelled points pin the plane down (`no_plane`
  counts the others, and a note under the clip says why).

## `--uncertain`

`--uncertain` walks only the frames that want attention (no confident guess, or
flagged by `ft.py check --audit --mark`) and skips confident predictions, which may
still be wrong. `n` / `p` / SPACE step between those frames; the rest of the labels
are untouched and still saved. Imported CSVs without confidence are treated as
unknown, not as perfect detections. Use the regular or `--grid` view to check small
centering errors in every imported label.

## Output

`labels/<clip>_ball.csv` (`frame,ball_x,ball_y`). Review progress lives in
`labels/<clip>_ball.review.json` so you can quit and resume. Beside them, one file
per label type, each written only when something is marked and removed when the
last mark is taken off:

| file | columns | what a row is |
|---|---|---|
| `<clip>_bounces.csv` | `frame,ball_x,ball_y` | a landing: `frame_###`, or `frame_###.5` between two frames; the point where it touched the ground, blank for the ball label's (see *Landings*) |
| `<clip>_hits.csv` | `frame,ball_x,ball_y,hitter` | a racket hit, as a landing is, and `near`, `far` or blank (see *Hits*) |
| `<clip>_rallies.csv` | `start,end` | a rally: `frame_###` cells, either blank while that end is not marked |
| `<clip>_players.csv` | `frame,player,x1,y1,x2,y2,track` | one tracked box: `P1`-`P4` or `none`, the box in native pixels, the run's track id it came from |
| `<clip>_court.csv` | `frame,point,x,y` | one court point on a labelled frame: the engine's fourteen in its order (`far_left`, `far_right`, `near_left`, `near_right`, `far_left_singles`, `near_left_singles`, `far_right_singles`, `near_right_singles`, `far_service_left`, `far_service_right`, `near_service_left`, `near_service_right`, `far_centre`, `near_centre`), then `net_post_left`, `net_post_right` |
| `<clip>_events.review.json` | `{"rejected": {kind: [frames]}}` | the candidates rejected, by kind (`landing`, `hit`, `rally_start`, `rally_end`) |

Everything is autosaved
every few seconds and on any exit; a frame you already dealt with shows a REVIEWED
(space) or EDITED (click / `v` / nudge) badge in the top-right corner when you come
back to it, and the top-left corner counts how many frames are left to do.
