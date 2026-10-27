"""The labelling and GridTrackNet fine-tuning workspace.

``ft.py`` is the front door: the launchers and the commands that run another script;
``menu.py`` is its interactive menu, ``commands.py`` holds the commands that work in this process
and ``interpreter.py`` finds the python with cv2 and torch to hand over to. The modules behind
them are grouped by what they own:

- ``finetune.data``: the label files and their sidecars, the workspace layout, the review
  policy, the audit, the raw prepass over a clip and the dataset ingest (stdlib + numpy; nothing
  here needs cv2 or torch at import, which is what lets ``ft.py`` start under any interpreter);
- ``finetune.labeling``: the label tool (cv2);
- ``finetune.refiner``: its suggestion model, the network that places suggestions and that
  network's training (cv2, torch);
- ``finetune.training``: the trainer and the draft writer;
- ``finetune.scoring``: the workspace scorers (the detector and pipeline evaluator, the landing
  scorer, the selector replay, the track-break buckets and the log comparison).

The tracker itself lives in ``system/`` (``SYSTEM``), a copy of the ``main`` branch, whose
packages (``tennis_tracker``, ``ball_in_play_selector``, ``evaluation``) are imported here as
top-level packages. Importing this package puts ``system/`` on ``sys.path``; that is the one
place it is done, and every way in passes through it, because every module here and every
script at the repository root imports a ``finetune.*`` module. ``tennis_tracker.paths.ROOT`` is
therefore ``system/`` (the models are the system's); ``REPOSITORY`` is the repository root,
which holds ``finetune/`` and ``output/``.

Modules import each other as ``finetune.<package>.<module>``. The scripts among them
(``labeling/label_tool.py``, ``refiner/hints.py``, ``training/pretrack.py``,
``training/train_gridtracknet.py``, ``data/realign.py``, ``data/import_data.py``) can also be run
as ``python finetune/<package>/<script>.py`` from the repository root: each one puts the
repository root on ``sys.path`` and imports this package only when it is the program being run,
never as a side effect of being imported.
"""
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SYSTEM = REPOSITORY / "system"

if str(SYSTEM) not in sys.path:
    sys.path.insert(0, str(SYSTEM))
