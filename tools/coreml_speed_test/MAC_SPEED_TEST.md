# Timing the tracker's models on an iPhone

This folder holds the tennis tracker's ball and player detectors as Core ML models, ready for
Xcode's performance report: Xcode copies a model to a connected iPhone, runs it many times and
reports how long one prediction takes and which chip (Neural Engine, GPU or CPU) ran each layer.
No app and no code are needed. The whole session takes about 30 to 45 minutes, most of it the
one-time setup.

| file | what it is |
|---|---|
| `models/stream_s4_encoder.mlpackage` | **the phone model, part 1.** Runs once per video frame: the new frame and the one before it in, its features out. |
| `models/stream_s4_decoder.mlpackage` | **the phone model, part 2.** Runs once per video frame: the last four encoder results in, the ball heat map out. |
| `models/stream_s4_unit.mlpackage` | the same network as one model over five frames, for comparison only |
| `models/heatmap_s4_unit.mlpackage` | the heavy ball network, for comparison only |
| `models/player_detector.mlpackage` | **the player detector** (YOLO11n): one frame squeezed to 640x640 in, candidate player boxes out |
| `validate.sh`, `validate_coreml.py`, `requirements-mac.txt` | an optional check that the models give the same answers as the PC |
| `samples/` | five-frame clips and single frames from the Pomona video and the PC's answers on them, for that check |
| `manifest.json` | the details: tensor shapes, source weights and their md5, the PC's checks |

The ball models take fixed 768x432 frames and the player detector a fixed 640x640 picture; all
compute in 16-bit floating point and need iOS 17 or later on the iPhone. The court model is not
here: its source weights could not be matched to the one the tracker ships.

## 1. What you need

- **A Mac with Xcode.** Install Xcode from the Mac App Store. The current version, Xcode 27,
  needs macOS Tahoe 26.6 or later ([Xcode support](https://developer.apple.com/support/xcode/)).
  Xcode 15 or later works (the models target iOS 17); older versions name some windows differently (noted below).
- **An iPhone on iOS 17 or later** and a cable that connects it to the Mac. Xcode 27 supports
  iPhones on iOS 17 or later ([Xcode support](https://developer.apple.com/support/xcode/)).
- **A free Apple Account** (the Apple ID you already use is fine; no paid developer membership).

## 2. One-time setup

1. **Open Xcode once.** Accept the license and let it install its components. If it offers
   iOS platform support, install it (later: Xcode > Settings > Components).
2. **Sign in with your Apple Account:** Xcode > Settings > Apple Accounts, click +, sign in
   (in older Xcode: Xcode > Settings > Accounts). Apple's device-setup steps start here
   ([Running your app on devices](https://developer.apple.com/documentation/xcode/running-your-app-on-simulated-or-physical-devices#Set-up-code-signing-for-physical-devices)).
3. **Connect the iPhone with the cable and unlock it.** When the iPhone asks
   **Trust This Computer?**, tap **Trust** and enter the iPhone's passcode.
4. **Pair it in Xcode.** Open Device Hub: Xcode > Open Developer Tool > Device Hub (in older
   Xcode: Window > Devices and Simulators). Select the iPhone in the sidebar and follow what
   the window says; click **Pair** if it appears
   ([Device Hub](https://developer.apple.com/documentation/xcode/managing-your-simulated-and-physical-devices-in-device-hub#Pair-devices-using-a-cable)).
5. **Turn on Developer Mode on the iPhone.** The switch appears only after step 4 has started
   pairing. On the iPhone: Settings > Privacy & Security > Developer Mode (under Security), turn
   it on, tap **Restart**. After the restart, swipe up, tap **Enable** and enter the passcode
   ([Enabling Developer Mode](https://developer.apple.com/documentation/xcode/enabling-developer-mode-on-a-device)).
6. Back in Device Hub, wait until the iPhone is listed as available (the first time this can
   take a few minutes; keep it unlocked and connected).

## 3. Optional: check the models give the PC's answers (5 minutes)

Open Terminal (Applications > Utilities > Terminal), type `bash ` (with a space after it),
drag `validate.sh` from this folder into the Terminal window, and press Return. That is:

```sh
bash /path/to/coreml_speed_test/validate.sh
```

The first run sets up a private Python environment in `.venv` here (it needs the internet once)
and then runs every model on this Mac, first with all compute units and then with CPU and
Neural Engine. Each line says PASS or FAIL. It ends with **ALL PASS** when every ball model's
heat map stays within 0.05 of the PC's and finds the ball in the same grid cell on every frame
where the PC sees a ball, and the player detector finds the same players with boxes overlapping
the PC's by at least 90%. Keep the output (copy the Terminal text) and send it back. The
times it prints are this Mac's, not the iPhone's.

## 4. Time each model on the iPhone

Before you start: close the other apps on the iPhone, plug it in, keep it unlocked with the
screen on (Settings > Display & Brightness > Auto-Lock > Never, for the session; set it back
after), and do not use it while a report runs.

For **each** of the five models, in this order: `stream_s4_encoder`, `stream_s4_decoder`,
`player_detector`, `stream_s4_unit`, `heatmap_s4_unit` (the first three are the ones that
matter most; do them first if time is short):

1. In Finder, open `models/` and double-click the `.mlpackage` (or in Xcode: File > Open). Xcode
   opens the model viewer; no project is needed.
2. Click the **Performance** tab at the top of the viewer.
3. Click **+** in the lower-left corner.
4. Choose the **iPhone** (not "My Mac") and click **Next**.
5. Choose compute unit **All** and click **Run Test**. Wait; nothing shows until it finishes
   (from a few seconds to a couple of minutes; the first run of a model takes longest).
6. Take the screenshots in the next section.
7. Click **+** again, choose the iPhone, choose **CPU and Neural Engine**, **Run Test**, and take
   the same screenshots.

That is 10 reports in all. Xcode keeps every report in the model's Performance tab.
Apple's guide to these reports:
[Analyzing a Core ML model's performance in Xcode](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode).

## 5. What to screenshot (per report)

Press Shift-Command-4, then Space, then click the Xcode window to capture it.

1. **The top of the report:** the three tiles **Prediction**, **Load** and **Compilation** in
   milliseconds, with "Compute Units Selected" above them. They show the **median** by default;
   leave it on Median. Then open the menu on the **Load** tile, choose **Uncached**, and take a
   second screenshot (that is the first-ever load on the phone; the median is a cached load).
2. **The compute unit mapping:** the summary line below the tiles (for example
   "All: 120, CPU: 2, GPU: 0, Neural Engine: 118") and the table under it. A filled diamond
   marks the chip that ran a layer. If any rows sit on CPU or GPU, scroll to them and capture
   those rows; hovering over a diamond shows more detail, capture that too.

At the end, select all reports of one model in the Performance tab, click the table icon next
to **Open in Instruments**, and click **Export to CSV**; send the CSV files with the screenshots.

## 6. Reading the result against the budget

At 60 frames a second the phone has **16.7 ms per frame for everything**: the ball model, the
player and court models, the tracking logic and drawing. The ball model should take a small part
of that; as a rough guide under about 5 ms leaves room for the rest, 5 to 10 ms is tight, and
above 16.7 ms it cannot keep up with 60 FPS at all.

- **stream_s4 (the phone model):** the tracker runs the encoder once and the decoder once for
  every frame, so its cost per frame is
  **encoder median Prediction + decoder median Prediction**.
  Example: encoder 1.4 ms + decoder 1.1 ms = 2.5 ms per frame, 15% of the budget.
- **stream_s4_unit** does the same work as four encoder runs plus one decoder run in a single
  call. It is not how the tracker runs; it shows what the split saves.
- **heatmap_s4_unit** runs once per frame: its median Prediction is its per-frame cost. It is
  about ten times the work of stream_s4 per frame and was never expected to fit; it is the
  yardstick.
- Compare **All** against **CPU and Neural Engine**: if "All" is faster, the GPU helps; if the
  numbers match, the Neural Engine is doing the work. Layers that fall back to the CPU are the
  first thing to fix for speed (in the encoder, watch the first layers: `sub`, `relu`, `concat`
  and `pixel_unshuffle`, which turn the two frames into the network's input).
- **player_detector**: at 60 FPS the tracker runs it on every other frame (and fills the frames
  between), so its cost per frame is **half its median Prediction**. The model stops before
  the last step, removing duplicate boxes (non-maximum suppression); the app does that on the
  few boxes above the score threshold, which costs microseconds.
- **The court model** (not in this folder) runs once every 400 frames, about every 7 seconds;
  its cost is a short stall now and then, not a per-frame share.
- **Load** is paid once when the app starts the model, not per frame.

The work per call, counted on the PC (billions of multiply-accumulates, GMAC):

| model | GMAC per call | calls per frame | GMAC per frame |
|---|---:|---:|---:|
| stream_s4_encoder | 1.29 | 1 | 1.29 |
| stream_s4_decoder | 1.35 | 1 | 1.35 |
| stream_s4 (encoder + decoder) | | | **2.64** |
| stream_s4_unit | 6.51 | (comparison) | |
| heatmap_s4_unit | 24.75 | 1 | 24.75 |
| player_detector | 3.16 | 1/2 | 1.58 |

Please send back: the 10 reports' screenshots, the CSV exports, the iPhone model and iOS version
(Settings > General > About), the Xcode version (Xcode > About Xcode), and the `validate.sh`
output if you ran it. A filled-in line like this is the summary we need:

| iPhone / iOS | units | encoder ms | decoder ms | encoder + decoder | player ms | stream_s4_unit ms | heatmap_s4_unit ms |
|---|---|---:|---:|---:|---:|---:|---:|
| | All | | | | | | |
| | CPU and Neural Engine | | | | | | |

## 7. If something goes wrong

- **The iPhone is not in the device list:** unlock it, reconnect the cable, check Developer Mode
  is on, and look at Device Hub for a message. After an iOS update the iPhone must be paired again.
- **"The performance report has crashed on device"** with All or CPU and Neural Engine: other
  developers have reported this ([Apple Developer Forums](https://developer.apple.com/forums/thread/767412)).
  Restart the iPhone and run it again; if it persists, run **CPU only** once to confirm the model
  loads, screenshot the error and send it back.
- **The report takes very long:** the first report of a model compiles it for the Neural Engine;
  later runs are quicker.
- **`validate.sh` says no Python was found:** run `xcode-select --install`, then run it again.

## Sources

- Apple, [Analyzing a Core ML model's performance in Xcode](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode): opening a model, the Performance tab, + and device and compute-unit choice, the Prediction/Load/Compilation tiles and their statistics, the compute-unit mapping, comparison and CSV export.
- Apple, [Enabling Developer Mode on a device](https://developer.apple.com/documentation/xcode/enabling-developer-mode-on-a-device).
- Apple, [Managing your simulated and physical devices in Device Hub](https://developer.apple.com/documentation/xcode/managing-your-simulated-and-physical-devices-in-device-hub): pairing by cable, Trust This Computer.
- Apple, [Running your app on simulated or physical devices](https://developer.apple.com/documentation/xcode/running-your-app-on-simulated-or-physical-devices): Xcode > Settings > Apple Accounts.
- Apple, [Xcode support](https://developer.apple.com/support/xcode/): Xcode 27 requires macOS Tahoe 26.6+, supports devices on iOS 17+.
- Apple, WWDC22 [Optimize your Core ML usage](https://developer.apple.com/videos/play/wwdc2022/10027/): the performance report's workflow on a connected device.
- coremltools, [Installing Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/installing-coremltools.html) and [PyPI](https://pypi.org/project/coremltools/#files): coremltools 9.0, macOS and Linux only; predictions need macOS.
