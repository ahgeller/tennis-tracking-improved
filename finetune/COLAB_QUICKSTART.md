# Colab quickstart: train the ball detector

One file goes to Google Drive, one notebook runs it.

## 1. On the PC: build the zip

```
python finetune/ft.py colab-zip
```

This writes `output/tennis_finetune_colab.zip`. It holds the workspace videos and labels, the records (`clips.csv`,
`review_status.json`, `holdouts.json`, `exclude.txt`), the tracker's model (`models/ball_detector.pt`, the default start
and teacher), the pseudo-labelled clips if any (step 4), and the trainer's code. The code list is derived from the
trainer's imports and written to `code/code_manifest.json`. The notebook checks it.

A code-only change does not need the videos again. `python finetune/ft.py colab-zip --code-only` writes the small
`output/tennis_finetune_code.zip`. Upload it beside the big one and the notebook lays it over the big zip's code.

## 2. On Drive and Colab

1. Upload `tennis_finetune_colab.zip` to the folder `MyDrive/tennis_finetune/` (make the folder if it is not there).
2. Open `finetune/Optimization_Train_Colab.ipynb` in Colab (File -> Upload notebook, or from GitHub).
3. Runtime -> Change runtime type -> GPU (T4 works; L4 or A100 is several times faster).
4. Runtime -> Run all.

Everything a run writes goes to `MyDrive/tennis_finetune/runs/<TAG>/`: the log, `results.csv`, `checkpoint.pt` and the
weights. If Colab recycles the runtime, reopen the notebook and Run all with the same `TAG`. The run resumes from its
last epoch.

## 3. The settings cell

| Setting | Default | What it does |
|---|---|---|
| `ARCH` | `heatmap_s4` | The network. The cell checks it against the staged `ball_models.py`. |
| `MODE` | `scratch` | `scratch` trains a new network; `finetune` starts from `START_WEIGHTS`, which must be that architecture's file. |
| `DISTILL`, `TEACHER` | on, `ball_detector.pt` | The student also learns the teacher's soft heat maps and offsets. |
| `DISTILL_LOSS`, `DISTILL_WEIGHT` | `qfl`, 1.0 | The soft heat-map loss (`qfl` or `mse`) and its weight. |
| `TEACHER_MAPS` | `online` | `online` runs the teacher on every augmented batch. `cache` precomputes its answers once. |
| `MGD_WEIGHT` | 0 | Feature distillation (MGD). Off by default; 2e-5 is the paper's detection setting. |
| `PSEUDO_SHARE` | 0.3 | The pseudo-labelled clips' share of every epoch, when the zip holds them. 0 leaves them out. |
| `AUGMENT` | `basic translate copypaste` | `copypaste` pastes a second, motion-blurred real ball along a real path. |
| `HOLDOUT` | `mixed_v1` | The held-out set the epochs are scored on. |
| `SELECTION_RUN` | empty | Set it to a finished run's `TAG` for the all-clips run that makes the deployable weights. |
| `LR` | 1.0 | Adadelta. It may be lowered on a resume (for example to 0.3); the trainer records the schedule. |
| `EMA`, `EMA_DECAY` | on, 0.999 | An exponential moving average of the weights, written every epoch (`last_ema.pt`) and scored beside the live weights on every scored epoch (`best_ema.pt`, the `ema_*` columns of `results.csv`). The winner stays the live network's; the run compares the two. Scoring takes twice as long. |
| `VIS_HEAD`, `VIS_WEIGHT` | off, 0.1 | `stream_s4` only: trains the per-frame visibility head ("a ball is visible") at this weight of the heat loss, against the visible and absent labels. `results.csv` gains its held-out AUC (`vis_auc`) and balanced accuracy (`vis_bacc`). The 15-channel output is unchanged. |
| `VIS_POOL`, `VIS_GAP_WEIGHT` | `global`, 1.0 | With `VIS_HEAD`: the head's variant (`global`, the s4_fg1 head, a mean over the whole unit; `peak`, a per-frame answer read at each frame's own heat peak, `streamnet.PeakHead`) and the loss weight of the gap frames, the absent labels inside a rally (runs of at most 3 flanked by visible labels) that the selector bridges; 1 is off. `results.csv` gains `vis_gap_auc`, the head's AUC on those frames alone, the number that must move off 0.5 for the gate to work. |
| `VIS_ONLY` | off | With `VIS_HEAD` and `MODE` finetune: train the head alone on `START_WEIGHTS` (`--vis-only`). Every other weight and BatchNorm statistic stays as loaded, so the detector's held-out numbers are the start's to the bit and `vis_gap_auc` is the number to read. Needs `DISTILL` and `EMA` off. The s4_fg1_peak recipe: `ARCH` stream_s4, `START_WEIGHTS` `s4_fg1/best_lr03_e46.pt` (uploaded under `runs/`), `VIS_POOL` peak, `VIS_GAP_WEIGHT` 4, `VIS_WEIGHT` 1.0, `LR` 1.0, `EPOCHS` 10, `SCORE_EVERY` 10. |

The evidence for each default is in `audit/records/opt_training.md`; `EMA` and `VIS_HEAD` are the experiments F and G of
the stream_s4 research of 2026-10-31 (`finetune/training/loop.py` has the reasoning), first run in `TAG` `s4_fg1`.

## 4. Optional: pseudo-labelled data

Unlabelled tennis video becomes training data through the teacher pipeline. Only the frames it is sure of become
labels, and the rest are ignored. On the PC (the pipeline needs the GPU):

```
python finetune/ft.py pseudo add D:\tennis_unlabelled\*.mp4    # 30 or 60 FPS; others are refused with an ffmpeg line
python finetune/ft.py pseudo label                             # runs the tracker's detector and selector on each new video
python finetune/ft.py pseudo status
python finetune/ft.py colab-zip                                # packs finetune/pseudo/ into the zip
```

Two public sources (also in `finetune/DATASETS.md` section 2.5):

- **Mendeley "Tennis Shot Side-View and Top-View"** (CC BY 4.0, 2.42 GB): https://data.mendeley.com/datasets/75m8vz7jr2 ->
  "Download All". It holds 472 GoPro 1080p60 clips (236 side view and 236 top view, clay and an indoor court). Unzip it
  and `pseudo add` the clips of the view you want.
- **CalTennis** (CC BY-NC 4.0, about 104 GB in all): https://huggingface.co/datasets/demalenk/caltennis. Fetch one
  session at a time:

  ```
  pip install -U huggingface_hub
  hf download demalenk/caltennis --repo-type dataset --include "09_18_2025_16_15_court5/*" --local-dir D:\caltennis
  ```

  Each session holds per-camera phone videos (`N`, `SE`, `WSW`, ...) and a large `_overhead.mp4`. Phone video can
  have a variable frame rate. If `pseudo add` refuses a clip, re-encode it to a constant 60 FPS first, using the
  `ffmpeg -i <in> -vf fps=60 -c:v libx264 -crf 14 <out>.mp4` line it prints.

## 5. After the run

Download `best.pt` from `MyDrive/tennis_finetune/runs/<TAG>/` and score it on the PC:
`evaluate_archive.py --archive finetune --mode raw --holdout mixed_v1 --gridtracknet-weights best.pt`.
With `EMA`, `best_ema.pt` sits beside it and `winner.json` sets its metrics against the live best epoch's (`ema.beats_live`);
a record with the visibility head loads and scores like any other (the head is ignored by the tracker).
For the deployable weights, set `SELECTION_RUN` to that `TAG` with a new `TAG` and `EPOCHS` 0, then Run all.
`python finetune/ft.py promote --weights winner.pt` makes the winner the tracker's model (with a backup).
