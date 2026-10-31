# Models

This folder holds the model weights the tracker runs. They are not in the
repository: the weights trained for this project are proprietary. Put your own
files here under these names (`tennis_tracker/paths.py` is where they are spelled):

| file | what it is | needed for |
|---|---|---|
| `candidates/stream_s4_e60.pt` | the default ball detector (a `stream_s4` record, epoch 60; run at 0.42) | every run, unless `--weights` names another |
| `ball_detector.pt` | the heat-map ball detector (a `heatmap_s4` record; run at 0.40), the training workspace's model | `--weights models/ball_detector.pt` |
| `player.engine` | TensorRT player detector | every run |
| `courtdetection.engine` | TensorRT court keypoint detector | every run |
| `ctb_regr_bounce.cbm` | CatBoost bounce model | `--court` only |

`clean_tracker.py --weights <file>` runs any other ball detector, `.pt` or
GridTrackNet `.npz`, without copying it here. Without `--conf` a run takes the
threshold the weights were gated at (`tennis_tracker/ball_weights.py`: the two files
above; 0.40 for any other). That table is keyed by path, not by what is in the file:
weights elsewhere (a retrained stream_s4 in `finetune/models/`) run at 0.40, which the
run prints, and weights copied over one of these names (`ft.py promote`) take that
name's threshold, which `promote` warns about. Gate new weights and give them their
own entry in `ball_weights.py`, or pass `--conf`.

## Train your own ball detector

The `train` branch has the labelling workspace and the trainer. Its `README.md` is
the full playbook; in short:

1. Check out `train`, import or label clips with `finetune/ft.py` (the workspace
   commands are in `finetune/README.md`).
2. Train the heat-map network from scratch, on a Colab GPU with the notebook
   `finetune/GridTrackNet_FineTune_Colab.ipynb` (`ARCH = "heatmap_s4"`,
   `MODE = "scratch"`), or locally:

   ```powershell
   python finetune/train_gridtracknet.py --arch heatmap_s4 --from-scratch --holdout mixed_v1
   ```

3. Score the result on the held-out clips, then copy the winner here as
   `ball_detector.pt` (`finetune/ft.py promote` does it with a backup).

## Or start from open-source models

None of these are this project's weights, and each has its own license.

- **Ball:** GridTrackNet ([VKorpelshoek/GridTrackNet](https://github.com/VKorpelshoek/GridTrackNet),
  MIT) ships `model_weights.h5`. The tracker reads it as an `.npz` of the Keras
  layer weights (the names `gridtracknet.load_model` reads); run it with
  `--weights <file>.npz`.
- **Players:** any Ultralytics YOLO detection model trained on COCO
  ([ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0); class 0,
  person, is the player. Save it as `models/player.pt` and build the engine with
  `python -m tennis_tracker.engines.build_engines`.
- **Court:** the court engine is a YOLO pose model over the court keypoints. There
  is no drop-in open release; [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
  has a 14-keypoint court dataset and its own pretrained model to train one from.
  Save it as `models/courtdetection.pt` and build the engine as above.
- **Bounces:** the CatBoost bounce model linked from the README of
  [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject), saved
  as `ctb_regr_bounce.cbm`.
