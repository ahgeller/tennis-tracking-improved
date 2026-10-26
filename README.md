# Triton Ball Tennis: training branch

This branch holds everything the tracker on `main` does not need to run: the
labelling workspace, the trainer, the Colab notebook and the scorers. This
README is the training playbook. The tracker itself is documented on `main`;
the workspace commands (import, label, check, audit) are in `finetune/README.md`.

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
START_WEIGHTS = "gridtracknet_weights_torch.npz"
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
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluation/compare_eval_logs.py output/bundled_raw.log output/<TAG>_raw.log
```

The log has one `source=` line each for custom, prof and grid. The bundled
weights trained on the custom clips, so compare on prof and grid; those are the
only unseen numbers for both. The bar for a new architecture: at least 2 points
more recall at equal or lower false alarm, and the same commands with
`--mode pipeline` (the whole tracker, selector included) not worse.

To run the tracker with the new weights on any video:

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py -i <video.mp4> --weights <best.pt>
```

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

## Where things are

- `finetune/README.md`: importing, labelling, checking and auditing clips.
- `finetune/training/train_gridtracknet.py`: the trainer; `--help` lists every flag.
- `audit/records/gridtracknet_recipe_2026-10.md`: the ledger of runs and what each
  one changed.
- `AGENTS.md`: the map of the code.
