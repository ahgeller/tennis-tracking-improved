# Synthetic training data for the ball detector: 200k frames? (2026-10-28)

The question: should we make about 200,000 frames of synthetic training data for the five-frame
ball detector (`heatmap_s4` and the Phase B students)? Can we, and would it help? This record
gives the decision, the evidence, the options with their costs, a small local pilot and the plan.
"Measured" means measured here. Everything else is cited or marked as opinion.

## 0. Decision: go, but differently

**Do not render a 200k-frame synthetic set, and do not store one.** Three things support this:

- The recipe already makes about that many synthetic units *online*. `--augment copypaste` pastes
  a real ball along a real path into half of all units drawn. A 30-epoch run of 11,864 units
  per epoch therefore trains on about 178k freshly pasted units, at no cost on disk.
- The literature finds that volume saturates early: around 10k to 50k images in the
  driving-detection studies (section 2). What pays is *diversity*: new scenes, cameras and
  conditions.
- Our weak group (`grid`: amateur phone, broadcast, night and gravel) is a scene problem. A
  renderer or our own camera's plates cannot cheaply supply those scenes.

What to do instead, in order of expected value:

1. **Pseudo-labelled real video first. It needs no hand labelling.**
   - Run the measured filter (`pseudo_labels.py`, precision 0.9989) over:
     - the two Playsight match videos already on this PC (about 800k frames);
     - Mendeley 75m8vz7jr2;
     - CalTennis.
   - Copy-paste plus self-training is additive in the literature (Simple Copy-Paste: 48.5 to
     50.0 for either alone, 51.4 for both).
   - The GridTrackNet matches not yet imported are mostly professional matches, which is the
     `prof` domain rather than the weak amateur one. Their labels would also need the same
     frame-by-frame correction as ours, so they come later, if at all.
2. **Keep synthetic data online and target it at specific failure modes**, not volume:
   - occlusion by players and the net (TOTNet);
   - track dropouts and partial tracks;
   - physically integrated blur;
   - hard negatives on the background plates.
3. **If you want synthetic clips on new backgrounds, use the generator added here**
   (`finetune/training/synthetic_clips.py`). It writes real ball crops along real rally tracks
   onto teacher-vetted, ball-free stretches of unlabelled match video, as H.264 clips that the
   trainer reads through `--pseudo`. The pilot (section 5) did not show it helping. Cap the volume
   at 10 to 20k units unless a 2-seed run at full length shows a gain on `mixed_v1`.
4. **Full 3D rendering (Blender, Kubric, Unreal, NVIDIA Isaac Sim with Replicator): no-go for
   now.**
   - There is no published evidence that rendered pixels help a TrackNet or WASB style ball
     detector on real video.
   - The realism factors that matter here are the ball of a few pixels, codec artefacts, blur
     and real clutter. A renderer gets these least right, and copy-paste onto real video gets
     them most right.
   - Setup cost is weeks.

## 1. What we have (measured from `finetune/clips.csv` and the frame cache)

| | clips | label frames | visible |
|---|---:|---:|---:|
| custom (own camera, fixed) | 12 + 1 moving | 6,243 | 5,666 |
| grid (GridTrackNet partial: amateur + broadcast) | 22 fixed + 10 moving | 17,479 | 16,187 |
| prof (TrackNet tennis broadcast) | 40 + 1 moving | 9,292 | 9,068 |
| **total** | **86** | **33,014** | **30,921** |

- **Training split (outside `mixed_v1`):** 11,862 five-frame units in 62 clips (grid 7,797,
  prof 2,837, custom 1,228).
- **Held out (`mixed_v1`):** 24 clips with 8,515 visible and 518 absent labels.
- **The bar (ctrl1, raw 0.50):** recall 0.929, wrong 0.006, false alarm 0.083. By group: own
  camera 0.946, prof 0.953, grid 0.842.
- **What makes grid weak:**
  - The held-out grid clips are a moving broadcast camera (`grid_match8`), daylight gravel
    (`grid_match16`) and night (`grid_match22`).
  - `opt_roi.md` found that a crop on the ball lifts grid by 3.8 points because it hides the
    distractors. Grid's losses are clutter and scene losses, not a shortage of ball examples.
- **Frame cache:** 14 GB for the 33k cached frames, about 430 KB per 768x432 PNG. A
  synthetic frame cached the same way measured 460 KB.
- **Unlabelled video on this PC:**
  - `HoldsSomeTennis/input_videos/PomonaPitzer Women vs. UCSD.mp4`: 563,609 frames, 2.6 h,
    1080p60. It is the same Playsight installation as the Pomona training clip.
  - `VolyAvw.mp4`: 250,062 frames, 1080p60, a Playsight camera at another venue (Arizona).
  - Both are behind-baseline fixed cameras: plates and pseudo-label sources for the own-camera
    domain only.

## 2. Evidence (literature)

A research subagent fetched and checked the sources in this section. Each paper is marked as
**measured** (numbers from its own tables) or as **opinion**.

| source | task | synthetic data | measured result | ratio / curriculum | relevance |
|---|---|---|---|---|---|
| [WASB, BMVC'23](https://arxiv.org/abs/2311.05237), [BlurBall](https://arxiv.org/abs/2509.18387) | ball detection (5 sports); table-tennis blur | none, real only | BlurBall: labelling the streak midpoint instead of the front lifts WASB F1 from 95.58 to 96.00 | n/a | The SOTA for our model family trains on real data only. Our labels and pastes already use the midpoint. |
| [TOTNet](https://arxiv.org/abs/2508.09650) | occluded ball tracking (TT, tennis, badminton) | synthetic occlusion: ball region mean-filled, noise patches | full-occlusion RMSE 24.43 to 12.31, accuracy 0.61 to 0.74, **only with weighted BCE**; without it, no gain | online augmentation | A targeted synthetic corruption, not volume. Directly applicable. |
| [TT spin from 2D tracks](https://arxiv.org/abs/2504.19863) | spin classification and 3D from tracks | 50k MuJoCo trajectories, given as coordinates | real-test spin accuracy 74% to 92% with all augmentations; "sudden end" (track dropout) alone 96% | synthetic only | Modelling the right corruption closes the gap |
| [Where Is The Ball](https://arxiv.org/abs/2506.05763) | 2D to 3D lifting | Unity PhysX trajectories, about 5k | NRMSE: real only 0.29, synthetic only 0.17, **synthetic then real 0.08** | pretrain, then fine-tune | Trajectories, not pixels |
| [3DGS ball](https://arxiv.org/abs/2506.05092) | robot-soccer ball, YOLOv8 | Unreal, simple vs Gaussian-splat ball | mAP50: simple 0.962, 3DGS 0.990, real 0.994 | synthetic only | Large static ball: says little about a blurred 3 px ball |
| [Cut, Paste and Learn](https://arxiv.org/abs/1708.01642) | instance detection | about 6k copy-paste images | real 86.3, synthetic 76.2, **both 88.8**. Blend modes: none 65.9, mixed 72.4. 10% real 15.8, 10% real + synthetic 43.2 | mixed | A mix of blend modes beats any single photoreal one. Large gains only when real data is scarce. |
| [Simple Copy-Paste](https://arxiv.org/abs/2012.07177) | COCO | real instances pasted online | +6.9 AP at 10% of COCO. **Additive with self-training:** 48.5, 50.0 (ST), 50.0 (CP), 51.4 (both) | online | Supports our copy-paste + pseudo-label recipe |
| [Driving in the Matrix](https://arxiv.org/abs/1610.01983) | KITTI cars | GTA renders, 10k / 50k / 200k | moderate AP 0.383 / 0.501 / **0.526**; easy AP 50k 0.686, 200k 0.680 | synthetic only | **50k to 200k adds about 2.5 points or nothing** |
| [Domain randomisation](https://arxiv.org/abs/1804.06516) | KITTI cars | 100k DR renders | AP50 78.1 (DR) vs 79.7 (VKITTI); after fine-tuning on 6k real, 98.5 (+2.1 over real only) | pretrain, then fine-tune | Saturation: about 10k images with ImageNet init, about 50k from scratch (read off a plot) |
| [Nowruzi et al.](https://arxiv.org/abs/1907.07061) | car / person, SSD-MobileNet | 15k renders per engine | synthetic pretrain + 10% real ≥ full real on car recall; **fine-tuning beat mixing and was more stable** | 2.5 to 10% real | Small model, like ours |
| [Mayer et al. IJCV'18](https://arxiv.org/abs/1801.06397) | optical flow | Chairs, Things, 2D shapes | a staged schedule (simple, then complex) beats mixing; Chairs (5.73 EPE) beat Things (6.64) | curriculum | Opinion with partial support: realism of *camera flaws* matters, photorealism does not |
| [He et al. ICLR'23](https://arxiv.org/abs/2210.07574), [Fan et al. CVPR'24](https://arxiv.org/abs/2312.04567) | classification on diffusion images | GLIDE / SD / Imagen | CIFAR-10 zero-shot: 2k images per class 80.06, 4k 80.08 (saturated). Supervised models scale much worse on synthetic data than on real. | | Generative data plateaus on diversity; no fit for 3 px blurred balls over 5 frames |
| [UAV-Sim](https://arxiv.org/abs/2310.16255) | drone small objects | NeRF renders | YOLOv8n 0.234 to 0.263 mAP50; **YOLOv8s 0.256 to 0.255 (none)** | mixed | Gains vanish as capacity grows. Authors' opinion: synthetic data does not replace real. |
| [Brooks & Barron](https://arxiv.org/abs/1811.11745) | motion-blur synthesis | 300k triplets | PSNR on real blur: averaging 28.06, theirs 34.14 | | Physically integrated blur (many sub-frames) is the realistic reference |
| [Kubric](https://arxiv.org/abs/2203.03570), Isaac Sim Replicator ([object SDG](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html)) | generators | Blender + PyBullet; Omniverse RTX path tracing | no sim-to-real number for small fast objects in either | | Tools, not evidence |

**Not found.** No paper trains a pixel-level multi-frame ball heat-map detector on rendered video
and measures the effect on real footage, for any racket or ball sport. The sports papers use
synthetic *trajectories* (lifting, spin) or synthetic *corruptions* (occlusion), not rendered
pixels.

**What this means for us (opinion, from the table):**

- We are past the low-data regime where the large synthetic gains were measured: 10% of real
  data, few-shot settings, or tiny models. We have 31k real visible balls, a teacher to distil
  from, and pseudo-label sources.
- Expect single-digit gains at most, and only on conditions the real data lacks.
- Fine-tuning on real data after synthetic pretraining beats a flat mix, and the run must
  always finish on real data.

### NVIDIA Isaac Sim

What it is: NVIDIA's robotics simulator on Omniverse (USD scenes, PhysX, RTX path tracing).
Replicator is its synthetic-data module. It randomises assets, poses, lights and materials and
writes labelled images. It can render motion blur by path-tracing sub-frames.

Can it run here?

- [Its requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
  list Windows 11 and an RTX 5080 (16 GB) as the "good" tier, with 32 to 64 GB RAM, a
  50 to 500 GB SSD and driver 595.97.
- Laptop GPUs are not mentioned.
- A forum report describes a Blackwell-laptop GUI crash with 5.1 that was fixed by changing the
  driver
  ([thread](https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74/365335)).
- So it is feasible, but it would pin a driver on a machine whose TensorRT and NVDEC work is
  tuned to its current driver.

Would it help?

- Not for this problem. Its strength is photoreal scenes with known geometry for robots.
- Our gap is real amateur and broadcast clutter, codec artefacts and 3 px blurred balls.
- We would have to build tennis courts, crowds, players with rackets and broadcast camera
  moves as 3D assets. That is weeks of work.
- The outcome would still be a domain gap that the cited studies close only by fine-tuning on
  real data, which we already have.
- Reconsider it only for something real video cannot give: exact 3D ground truth for
  `flight3d` (spin, bounce height, depth), where rendered *trajectories* have measured value
  (Where Is The Ball).

## 3. What fits a five-frame temporal model

The network reads motion across five frames, so the synthetic data has to be temporally
consistent.

- **One ball identity across the frames.** Use the same crop and the same exposure on all
  frames. Both `copy_paste` and the generator do this.
- **Real trajectories with real discontinuities: hits, bounces, net cords, out of frame.** Both
  use real labelled paths rather than parabola fits. The generator uses whole 60-frame rally
  tracks, including the absent frames.
- **Blur as a streak along the velocity, centred on the label.** BlurBall's convention, which
  we already use. The current streak is a line kernel. Integrating many sub-frames (Brooks &
  Barron) would add realism, but **this is untested here**.
- **Codec artefacts on the ball.** An online paste lands on already-decoded frames, so the
  pasted ball is cleaner than a real one. The generator's H.264 pass fixes that; the pilot did
  not separate its effect.
- **Static background with a moving ball.** Plates from real video give the right camera
  noise, players and scoreboard. Rendered or diffusion frames give the wrong temporal noise
  (flicker), and the network can learn to key on it.
- **Hard negatives.** These are ball-like things (heads, shoes, line marks, spare balls, the
  ball in a server's hand) on frames labelled "none". Ball-free plates supply them for free.
  Most false alarms in `mixed_v1` are "bridged" or ball-like objects next to play.
- **Occlusion.** A TOTNet-style mean-fill over the ball, or a paste *behind* a player mask, is
  a cheap augmentation. Untested here.

## 4. Options and costs (one RTX 5080 laptop, Windows; Colab for long runs)

| option | what it gives | cost to make | disk for 200k units | fit to the weak group | verdict |
|---|---|---|---|---|---|
| (a) online copy-paste, as now | new paste every draw, same 62 scenes | 0.5 ms/unit on the CPU | 0 | low (same scenes) | keep; it already "is" the 200k |
| (b) plates from unlabelled video + real crops on real tracks (`synthetic_clips.py`) | new backgrounds of the plate cameras, codec-consistent ball, hard negatives | measured: 5.6 s per 60-frame clip (87% of plates rejected as possibly holding a ball); 200k units = about 7,150 clips = **about 11 h** | MP4 store 0.21 MB/clip = **1.5 GB**; trainer frame cache 460 KB/frame x about 430k frames = **about 200 GB** | only as good as the plate sources: Playsight-only today, so the own-camera group | go, small (10 to 20k units), on scene-diverse plates |
| (c) full 3D render (Blender/Kubric/Unreal/Isaac Sim) | perfect labels, any camera | weeks of asset work; minutes per frame for path-traced blur | as (b) once cached | unknown, no precedent | no-go |
| (d) generative (diffusion/video models) | new scenes | GPU-hours per clip; temporal flicker; ball not controllable at 3 px | as (b) | unknown; the evidence plateaus at about 2k images per class | no-go |
| pseudo-labels: the two Playsight videos here, Mendeley (472 clips, 2.4 GB), CalTennis (104 GB) | real frames, 74% of balls kept at 0.9989 precision, no hand labelling | one teacher pass | cache per kept unit | medium (own-camera-like / court-level); low for grid | **go first** |
| real: GridTrackNet matches not imported (68, 48 GB) | mostly professional matches (the user's note), so the `prof` domain | download, import, and the frame-by-frame label correction our other clips needed | about 30 GB of cache | low for grid | later, if at all. Untested shortcut: keep a frame only where its shipped label and the teacher agree within 3 px, ignore the rest, the way `pseudo_labels` ignores uncertain frames, so nothing is hand-corrected |

**How the trainer consumes it.**

- No trainer change is needed. The generator writes `videos/<clip>.mp4` and
  `labels/<clip>_pseudo.csv`, with `ball` and `none` states at 30 FPS.
- `--pseudo DIR --pseudo-share S` mixes them in. The frame cache extracts the needed frames as
  PNGs under `cache/data/pseudo/`.
- Online distillation reads them like any unit.
- **At 200k units the PNG cache is the binding cost (about 200 GB).** A run of that size should
  read frames straight from the MP4s, or cache them as JPEG. That is a trainer change, and
  only worth making if a smaller run shows a gain.

## 5. Pilot (local, heatmap_s4 from scratch, QFL distillation, `basic translate copypaste`, batch 4)

**Synthetic set.**

- 160 clips of 60 frames from the two Playsight videos: 9,600 frames, 4,480 units, 32 MB of MP4.
- Every plate was vetted ball-free by the teacher (max heat < 0.3 over all frames).
- Tracks: 216 real 60-frame windows from the custom and prof training clips. Crops: 119 own-camera
  crops of the copy-paste bank.

**Generator fixes made after looking at the frames:**

1. Tracks of other cameras were shifted anywhere in the frame and flew over the stands. They now
   stay within 80 px sideways (`MAX_DX_PX`).
2. Crops cut on clay or grass courts kept a rim of that court in their matte and showed as red
   or green balls on blue courts. Only own-camera crops are used now (`CROP_SOURCES`).

Zoomed samples of the synthetic clips are in the scratch run.

**Arms**, both seed 11, 3 epochs of 11,864 units (2,966 steps each), scored every epoch on the
24 `mixed_v1` clips:

- A: the recipe as it stands.
- B: the same, with the synthetic units as 30% of the draws (`--pseudo-share 0.3`) and the same
  number of steps.

**Per epoch** (the trainer's score, raw, threshold 0.50):

| arm | epoch | val focal | recall | wrong | false alarm | median px | minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| A control | 1 | 0.817 | 0.655 | 0.013 | 0.027 | 1.13 | 11.7 |
| A control | 2 | 0.596 | 0.672 | 0.010 | 0.014 | 0.83 | 10.1 |
| A control | 3 | 0.554 | **0.817** | 0.010 | 0.029 | 0.90 | 10.4 |
| B + synthetic 30% | 1 | 0.696 | 0.653 | 0.011 | 0.023 | 0.92 | 12.0 |
| B + synthetic 30% | 2 | 0.634 | 0.670 | 0.006 | 0.015 | 0.88 | 13.2 |
| B + synthetic 30% | 3 | **0.526** | 0.776 | 0.008 | 0.027 | 0.83 | 14.8 |

**Epoch-3 weights through `evaluate_archive.py --archive finetune --mode raw --holdout mixed_v1`**
(pooled threshold sweep):

| threshold | A recall / false alarm | B recall / false alarm |
|---|---|---|
| 0.30 | 0.907 / 0.143 | 0.891 / 0.116 |
| 0.40 | 0.869 / 0.079 | 0.848 / 0.056 |
| 0.50 | 0.817 / 0.029 | 0.777 / 0.027 |
| 0.60 | 0.733 / 0.017 | 0.650 / 0.012 |

**Per source** (that tool's per-clip table at 0.40, recall / false alarm):

| source | A | B |
|---|---|---|
| custom | 0.874 / 0.135 | 0.842 / 0.092 |
| grid | 0.802 / 0.020 | 0.781 / 0.027 |
| prof | 0.896 / 0.062 | 0.885 / 0.037 |

Pilot GPU time: about 100 minutes, which covers three training runs, two scorings and
generating 310 clips. The A and B runs went through `tools/gpu_lock.py`.

**Seed noise.** The same recipe at epoch 1 gave recall 0.393 with seed 7 (this session) and
0.655 with seed 11. Section 5.2 of `opt_training.md` had 0.640 with seed 7. A one-epoch,
one-seed comparison therefore cannot rank anything; the previous record's 2.6-point copy-paste
gain is inside this noise. Only multi-seed runs at full length can settle a few-point effect.

**Reading.** This is one seed and 3 of a full run's epochs, so it can show a large effect and
cannot show a small one.

- **At the fixed 0.50 threshold, synthetic data cost recall:** 0.776 against 0.817 overall, and
  every source lower at 0.40.
- **At equal false alarm the two arms are level.** Interpolating A's sweep to B's 0.056 false
  alarm gives A about 0.845 recall, against B's 0.848.
- B's validation focal loss is lower (0.526 against 0.554). The synthetic share made the network
  less confident: 30% of the draws became one-ball plates of only two venues. It did not make it
  find more balls.
- **Grid did not improve** (0.781 against 0.802), as expected: the plates are Playsight
  own-camera footage.
- Custom false alarms fell (0.092 against 0.135), which fits the hard-negative effect of
  ball-free plates. But custom recall fell by as much.

**Verdict on the core claim.** Adding about 4.5k synthetic units (about 38% of the real unit
count) as 30% of the draws did not improve held-out detection here. That matches the literature:
in a regime with plenty of real data, synthetic data mostly shifts calibration unless it adds
scenes the real data lacks.

Not tested, and the next things to try if this route is pursued:

- 2 seeds at full length;
- a lower share (10 to 15%);
- synthetic-then-real staging;
- plates from venues unlike any training clip.

## 6. Recommended plan and data volume

1. **Pseudo-labels (go first, no hand labelling).**
   - The two Playsight videos already on this PC (about 800k frames): `ft.py pseudo add`, then
     `pseudo label`.
   - Mendeley 75m8vz7jr2 (2.4 GB) locally.
   - CalTennis per-session on Colab.
   - `--pseudo-share` 0.3 as recorded. The kept frames are real pixels, which beats any
     synthetic plate of the same video.
2. **The weak grid group needs amateur, night, gravel and moving-camera footage**, not
   synthetic balls.
   - Short of labelling more such clips, the cheapest real source is unlabelled amateur
     YouTube or phone rallies run through the same pseudo-label filter. It needs no manual
     labels.
   - The GridTrackNet matches not yet imported are mostly professional. They would only be
     worth importing with the untested label-and-teacher agreement filter in section 4, not
     with frame-by-frame correction.
3. **Targeted online augmentations** (each one a flag, each measured with 2 seeds at full
   length):
   - an occlusion mean-fill (TOTNet; needs the loss weighting it reports);
   - pastes behind player masks;
   - multi-sub-frame blur;
   - several blend modes (Cut-Paste-Learn);
   - the copy-paste rate (`PASTE_P` is untested at 0.5).
4. **Synthetic clips (`synthetic_clips.py`), only on new scenes, capped.**
   - Plates from scene-diverse unlabelled video, not the venues already in training.
   - 10 to 20k units, at most about 30% of draws, or as a first stage before fine-tuning on
     real data (Nowruzi, Mayer).
   - Scale towards 50k only if a 2-seed full-length run beats the control on `mixed_v1` and on
     the group it targets. Past 50k the evidence says the return is about zero.
5. **No 200k set, no 3D render, no diffusion data**, unless step 4 shows a gain that keeps
   growing with volume.

**Run protocol for any of these.**

- At least 2 seeds per arm at full training length.
- The `mixed_v1` gates of the plan: raw 0.50, `epoch_score`, per-source, the curve at equal
  false alarm.
- `evaluate_archive.py --archive finetune --mode raw --holdout mixed_v1` for the
  per-source lines.

## 7. Files

- `finetune/training/synthetic_clips.py`: the generator. It is in `mypy.ini`; tests are in
  `tests/test_synthetic_clips.py`.

  ```
  python finetune/training/synthetic_clips.py --bank RUN/copypaste_bank.npz --out DIR \
      --videos "<60 FPS fixed-camera match>.mp4" ... --clips 150 --seed 1
  python finetune/training/train_gridtracknet.py ... --pseudo DIR --pseudo-share 0.3 --epoch-units 11864
  ```

  `--epoch-units` keeps the step count equal to a run without synthetic data; without it,
  `--pseudo` *adds* the synthetic share to every epoch.
- Pilot outputs (not committed): the scratch run directories `runA_s7`, `A3_s11`, `B3_s11` and
  `synthB`. Their cached frames in `finetune/cache/data/pseudo/` were deleted afterwards.
