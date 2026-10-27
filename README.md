# Triton Ball Tennis: training branch

This branch holds everything the tracker on `main` does not need to run: the
labelling workspace, the trainer, the Colab notebook and the scorers. This
README is the training playbook; the workspace commands (import, label, check,
audit) are in `finetune/README.md`.

The tracker itself is `system/`: `main`'s whole tree, byte for byte, documented
by `system/README.md`. Everything outside it is training and labelling:
`finetune/` (the workspace, the trainer, the scorers), the root scripts that front
them, `tests/` (their suites) and `audit/` (their ledgers). Changes to the tracker
are made in `system/` and ported to `main`, or made on `main` and ported here;
`AGENTS.md` has the recipe.

Everything below runs in the notebook `finetune/GridTrackNet_FineTune_Colab.ipynb`
on a Colab GPU (A100 if you can get one) with the data zip on Google Drive. The PC
is used to build the zips and to score the results.

## One-time setup

1. On the PC, from the repository root:

   ```powershell
   & 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/ft.py colab-zip
   & 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/ft.py colab-zip --code-only
   ```

   The first writes `output/tennis_finetune_colab.zip` (videos, labels, records,
   bundled weights, code; about 870 MB). The second writes the small
   `output/tennis_finetune_code.zip` (code and records only).
2. In Google Drive, make a folder `MyDrive/tennis_finetune/` and put both zips in
   it. Every run's outputs land in `MyDrive/tennis_finetune/runs/<TAG>/`.
3. Open the notebook in Colab, Runtime > Change runtime type > GPU.

After a code change on the PC, rebuild only the code zip and replace it on Drive;
the notebook lays it over the unpacked copy on every run. After new labels,
rebuild the big zip.

## The settings that matter

| setting | what it is |
|---|---|
| `ARCH` | `gridtracknet` (the tracker's current net, `.npz`) or `heatmap_s4` (the stride-4 heat-map net, `.pt`) |
| `MODE` | `scratch` trains a new network; `finetune` continues from `START_WEIGHTS` |
| `START_WEIGHTS` | a file in the zip's `models/` folder or a path under `runs/` (say `heat1/best.pt`). A scratch run reads it only for the hard-negative prepass |
| `HOLDOUT` | `mixed_v1`: 24 whole recordings never trained on, the honest test |
| `SELECTION_RUN` | set to a finished run's `TAG` to train on every clip following that run's recipe |
| `EPOCHS` | a ceiling; the best epoch is kept by the held-out metric. On a resume, raising it trains longer |
| `LR` | Adadelta step. 1.0 from scratch, 0.3 for a fine-tune. May be lowered on a resume, never raised |
| `AUGMENT` | `basic translate` is the recipe; `strong` is harsher colour and blur |
| `TAG` | the run's folder. Keeping a `TAG` and pressing Run all resumes that run |

The trainer refuses to resume a run whose other settings changed, and says which.

## Stage 1: the selection run (held out, from scratch)

This is the run that finds the best network and the best epoch, on recordings it
has never seen.

```
ARCH = "heatmap_s4"        (or "gridtracknet" to repeat the old recipe)
MODE = "scratch"
START_WEIGHTS = "ball_detector.pt"
HOLDOUT = "mixed_v1"
SELECTION_RUN = ""
EPOCHS = 40
LR = 1.0
AUGMENT = "basic translate"
SCORE_EVERY = 2
TAG = "heat1"              (any new name)
```

Run all. Every scored epoch prints one table row: train loss, held-out recall,
wrong, false alarm, median px, and which epoch is best so far.

Reading the table:

- Recall still climbing by a point every few epochs: let it run.
- Recall flat for 10 epochs (within about 0.02) while false alarm swings:
  the step is too big. Go to stage 2 now; the rest of the epochs are wasted.
- `EPOCHS` reached while recall is still climbing: raise `EPOCHS`, keep the
  `TAG`, Run all.
- Recall swinging between scored rows, or falling towards 0, while the train
  loss keeps falling: the network's normalisation statistics are too large for
  float16 (heat1 overflowed from epoch 18 at `LR` 1.0; heat3 swung from epoch
  12). The trainer now holds the normalised convolutions at a fixed size after
  every step, so this should not recur; it still warns when a running variance
  passes a tenth of the ceiling and stops the run, naming the layer, if one
  overflows. If it does, start a new `TAG` from the last good epoch's file in
  `epochs/` with `MODE = "finetune"` and `LR = 0.1`, and report it.

The first heat runs (heat1 and heat3, October 2026) reached 0.897 and 0.883
recall at epoch 10 before their statistics grew; the old network from scratch
plateaued at 0.85.

## Stage 2: the rate drop (resume the same run)

Interrupt the cell, then change only these and Run all:

```
LR = 0.1
EPOCHS = (current epoch + 20)
TAG = unchanged
```

The settings cell sees the checkpoint, sees that the rate went down, and adds
`--lr-drop`. The log says `[train] learning rate 1 -> 0.1 from epoch N`. Expect the
false-alarm column to settle within 4 to 6 epochs and recall to hold or gain a
point. When the table is flat for 8 epochs, the run is done; `best.*` in the run
folder is the winner.

## Stage 3: judge the winner on the PC

Download `best.pt` (or `best.npz`) from `MyDrive/tennis_finetune/runs/<TAG>/` and
score it on the held-out set, per source, against the bundled weights:

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluate_archive.py --archive finetune --mode raw --holdout mixed_v1 --gridtracknet-weights <best.pt> > output/<TAG>_raw.log
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluate_archive.py --archive finetune --mode raw --holdout mixed_v1 > output/bundled_raw.log
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/scoring/compare_eval_logs.py output/bundled_raw.log output/<TAG>_raw.log
```

The log has one `source=` line each for custom, prof and grid. The bundled
weights (ctrl1, since 2026-10-26) trained on the clips outside `mixed_v1`, so
every held-out line compares fairly. The bar for a new architecture: at least 2 points
more recall at equal or lower false alarm, and the same commands with
`--mode pipeline` (the whole tracker, selector included) not worse.

To run the tracker with the new weights on any video:

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' system/clean_tracker.py -i <video.mp4> --weights <best.pt>
```

Add `--showtracks` to see what the selector chose from: every selector track is drawn on
the tracking video in its own colour, chosen or not, under the chosen track's green trail
and marker (`--court` adds the court events and minimap, `--motion-diagnostics` the
per-frame motion diagnostics in the tracking JSON).

## Stage 4: the deployable weights (every clip)

The winner from stage 1 never saw the 24 held-out clips. For the weights you
actually run, carry it on over all 86 clips. Fine-tuning the winner is cheaper
than a full retrain and keeps what it learned:

```
ARCH = same as the selection run
MODE = "finetune"
START_WEIGHTS = "<selection TAG>/best.pt"
HOLDOUT = ""
SELECTION_RUN = "<selection TAG>"
EPOCHS = 10
LR = 0.3
TAG = "<selection TAG>_all"
```

Run all. The result is `winner.pt` in the new run folder. These weights saw every
labelled clip, so nothing can measure them any more; the only test left is the
tracker on unlabelled footage.

If you would rather retrain on everything from nothing (same recipe, best epoch
count, no held-out set), set `MODE = "scratch"`, `EPOCHS = 0` and the rest as
above. It takes as long as stage 1 did.

## Stage 5: new footage later

When you label new clips, do not fold them in blindly:

1. Score the current deployable weights on the new clips first, with
   `evaluate_archive.py --archive finetune --mode raw --clips <new clips>`. That is
   the honest number for footage the model never saw.
2. Rebuild the big zip (`ft.py colab-zip`) and replace it on Drive.
3. Fine-tune on everything: stage 4 with `START_WEIGHTS` set to the current
   deployable file and a new `TAG`.
4. A full retrain (stage 1 then 4) is only needed when the new footage changes
   the picture, say a new camera or many new matches.

Keep `finetune/holdouts.json` as it is; `mixed_v1` must stay unseen by every
selection run so the stage 3 numbers remain comparable over time.

## Keeping one network

Architectures exist so that one can be chosen; the ones that lose are deleted,
not kept behind a flag. Two were removed on 2026-10-26 after the held-out
comparison: `memory_s4` (HeatmapNet with a convolutional GRU carrying a state
between consecutive units of one phase) and `heatmap_s2` (HeatmapNet with one
more decoder block, answering at stride 2). The heat control matched the memory
network and the stride-2 curve equalled the control's, so neither earned its
weight; the runs and their numbers are in
`audit/records/gridtracknet_recipe_2026-10.md`, which keeps their history. A
`.pt` still names its architecture inside, so a record of either is refused by
name (`ball_models.arch_of`), and a `config.json` of their day may carry
`chain`, which the trainer no longer reads or compares.

Two remain, and for each the whole of it is:

- `gridtracknet`: `system/tennis_tracker/gridtracknet.py` stays (its loader, resize and
  decoder serve every architecture); its entry in `system/tennis_tracker/ball_models.py`
  (`GRIDTRACKNET` in `ARCHITECTURES`, `MODEL_CLASSES`, `WEIGHTS_SUFFIX`, `_MODULES`,
  the `.npz` branch of `arch_of`), the `GridTrackNet` class and its NPZ `load_model`,
  `gridtracknet_loss` in `finetune/training/loop.py` with the `.npz` branch of
  `save_weights` and the Kaiming branch of `fresh_model`, the one-hot branch of
  `dataset.make_target`, the `.npz` template handling of `--weights` in the trainer
  (`resolve_arch`), the `ARCH` option in the notebook, `ft.py promote`, and
  `system/tests/test_gridtracknet.py` with the GridTrackNet cases of the trainer tests.
- `heatmap_s4`: `system/tennis_tracker/heatmapnet.py` (the `HeatmapNet` class and the
  `.pt` record functions), its entry in `ball_models` (`HEATMAP_S4` in
  `ARCHITECTURES`, `MODEL_CLASSES`, `WEIGHTS_SUFFIX`, `_MODULES`, the record branch
  of `arch_of`), `heatmap_loss` and `HEATMAP_S4` in `loop.LOSSES` with the `.pt`
  branch of `save_weights`, the Gaussian branch of `dataset.make_target`, the
  trainer's `--arch` choice and `HEAT_RECIPE`, the notebook `ARCH` option and its
  parent-start check, and `system/tests/test_heatmapnet.py` with the heatmap cases of
  the ball-models, trainer and loop tests.

With one `.pt` architecture left, `read_record` with `arch=None` and `arch_of`'s
record read stay only so that a stale record is refused by name; if that is not
wanted, `arch_of` can go back to the suffix alone.

The winner's weights are promoted by copying them over the default path in
`system/tennis_tracker/paths.py` (`BALL_WEIGHTS`, `system/models/ball_detector.pt`, what
`Config.model_path` defaults to), which `ft.py promote` does with a backup. On
2026-10-26 that file became the heat-map control run's winner (ctrl1, the ledger
has the numbers); the GridTrackNet `.npz` files moved to `system/models/legacy/`
(`gridtracknet_finetuned.npz`, the previous default, and `gridtracknet_upstream.npz`,
the export it was fine-tuned from) beside the old YOLO `ball.engine`, and
`system/clean_tracker.py --weights` still runs any of them. Record every promotion in
`audit/records/gridtracknet_recipe_2026-10.md` with its held-out numbers.

## Where things are

- `finetune/README.md`: importing, labelling, checking and auditing clips.
- `finetune/training/train_gridtracknet.py`: the trainer; `--help` lists every flag.
- `audit/records/gridtracknet_recipe_2026-10.md`: the ledger of runs and what each
  one changed.
- `AGENTS.md`: the map of the workspace code, the layout and the port recipe;
  `system/AGENTS.md`: the map of the tracker's.
