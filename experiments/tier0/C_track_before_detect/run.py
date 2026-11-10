"""Test C end to end: dump the top-5 heat peaks of stream_s4 e60 over the mixed_v1 holdout
(GPU, waits until the shared GPU has under 6000 MiB in use), then score them (CPU).

    python experiments/tier0/C_track_before_detect/run.py            # both steps
    python experiments/tier0/C_track_before_detect/run.py --score-only

Python: C:\\Users\\Andrew\\miniforge3\\envs\\tennis-opt\\python.exe with PYTHONIOENCODING=utf-8.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_BUSY_MIB = 6000


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return max(int(line.strip()) for line in out.splitlines() if line.strip())


def wait_for_gpu() -> None:
    while (used := gpu_used_mib()) >= GPU_BUSY_MIB:
        print(f"[run] GPU has {used} MiB in use, waiting", flush=True)
        time.sleep(15)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-only", action="store_true", help="Reuse the peaks/ dump, skip the GPU step")
    args = parser.parse_args(argv)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if not args.score_only:
        wait_for_gpu()
        with open(HERE / "dump.log", "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(HERE / "dump_peaks.py")], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    with open(HERE / "score.log", "w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(HERE / "score.py")], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        log.write(result.stdout)
    print(result.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
