# Test B: stricter gap-filling rules on the e60_0.42 run

Verdict: **KILL for switching any fill off; KEEP (d), tag fills as inferred.** No fill rule
removes false alarms faster than it loses hits in a way the product would want: dropping the
interpolation fills has the best ratio (33 FA removed per 7 hits lost) but every one of the 33
removed alarms is bridged (on the ball's line between the flanking labels, a hidden ball), and
none of the 16 off-line false alarms is an interpolation row. Dropping the motion fills removes
12 of the 16 off-line alarms but costs 83 hits.

All numbers measured (`run.py`, CPU only, over the recorded e60_0.42 tracking JSONs). The
baseline reproduces the gate: 8111 hits = recall 0.953, FA 95 = 79 bridged / 16 false; the
script asserts it.

## How the rules were applied

Every output row carries its `source` (`det`, `interp`, `motion`). A rule is applied by turning
those rows into absent frames; no selector or replay code was changed. The refit question (fills
nudge neighbouring detections at weight 0.15 / 0.30) was checked by replaying the selector over
the recorded detections with `_fill_internal_gaps` and `_close_single_frame_holes` patched to
no-ops: the patched replay and the post-filtered replay score identically (8023 hits, 26 FA =
22 / 4), so removing rows is exact. (The replay has no motion masks, so its own motion rows are
missing in both: replay chain 0.945 / 64 against the control's 0.953 / 95; the rules were
therefore measured on the recorded run, which has the real motion rows.)

## Results (pooled, 8515 visible / 518 invisible)

| rule | recall | hits | hits lost | FA (bridged / false) | FA removed | FA removed per hit lost |
|---|---|---|---|---|---|---|
| baseline | 0.953 | 8111 | - | 95 (79 / 16) | - | - |
| (a) no interpolation fill | 0.952 | 8104 | 7 | 62 (46 / 16) | 33 (33 / 0) | 4.7 |
| (b) no motion fill | 0.943 | 8028 | 83 | 59 (55 / 4) | 36 (24 / 12) | 0.43 |
| (c) neither | 0.942 | 8021 | 90 | 26 (22 / 4) | 69 (57 / 12) | 0.77 |

Per source group in `fill_rules.csv`. Wrong outputs: 10 -> 6 (a) -> 6 (b) -> 2 (c).

## (d) fills kept and tagged as inferred: what each source contributes

| source | hits | loc (10-30 px) | wrong (> 30 px) | FA bridged | FA false | rows on unlabelled frames |
|---|---|---|---|---|---|---|
| det | 8021 | 54 | 2 | 22 | 4 | 3652 |
| interp | 7 | 3 | 4 | 33 | 0 | 20 |
| motion | 83 | 13 | 4 | 24 | 12 | 46 |

So the scorecard with and without inferred rows:

- det only (observed): recall 0.942, FA 26 (22 bridged / 4 false)
- det + inferred (shipped): recall 0.953, FA 95 (79 bridged / 16 false)

Of the 69 rows the rules remove from invisible labels, 57 are bridged: within 10 px of the line
between the flanking labels, inside a run of at most 3 invisible labels. Those are the ball
behind a player, the net post or a racket, which the labeller could not place; the product
almost certainly wants them drawn, as inferred. The 12 off-line alarms are all motion rows (a
blob on a player or line pulled the fill off the ball's path): 12 motion rows against 83 motion
hits, so the motion fill is still worth 7 hits per off-line alarm.

## Reading

- Interpolation fills are already tight: 47 labelled rows, 7 hits, 33 bridged, 0 off-line.
  Their only cost is 4 wrong rows (a fill through a hit where the ball changed direction).
- The remaining 4 off-line alarms and 2 wrong rows in det-only output are the chooser's, not
  the fills'. The raw detector at this threshold (replay `raw`) makes 21 off-line alarms and 41
  wrong rows, so the chooser removes 17 off-line alarms and 39 wrong rows while the fills add
  back 12 off-line alarms; the net pipeline is still ahead of the raw detector on both.
- The honest gate metric for the fills is therefore the one the scorer already prints: false
  alarms split bridged / false. A rule that scores bridged alarms as inferred rather than as
  errors changes the headline from 95 to 16 false alarms with no code change.

## Files

- `run.py`: `python experiments/tier0/B_fill_rules/run.py` (about 2 min; `--no-replay` for 20 s).
- `fill_rules.csv`, `source_contribution.json`, `replay_crosscheck.csv`.
