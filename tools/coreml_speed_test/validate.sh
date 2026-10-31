#!/usr/bin/env bash
# Check the Core ML packages in this folder against the saved PyTorch outputs, on this Mac:
#
#     bash validate.sh
#
# The first run makes ./.venv with a Python coremltools 9.0 supports (3.9 to 3.13; Xcode's
# /usr/bin/python3 is one) and installs requirements-mac.txt into it; later runs reuse it. The
# venv is rebuilt when it cannot import coremltools and numpy, or when requirements-mac.txt has
# changed since it was installed (the stamp .venv/requirements.sha256 is written only after a
# complete install), so an interrupted or failed install is retried on the next run.
# Extra arguments go to validate_coreml.py (for example: bash validate.sh --compute-units CPU_ONLY).
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "Core ML predictions run only on macOS." >&2
  exit 2
fi

supported() {
  "$1" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null
}

wanted="$(shasum -a 256 requirements-mac.txt | cut -d ' ' -f 1)"
venv_ready() {
  [[ -x .venv/bin/python ]] &&
    [[ "$(cat .venv/requirements.sha256 2>/dev/null)" == "$wanted" ]] &&
    .venv/bin/python -c 'import coremltools, numpy' 2>/dev/null
}

if ! venv_ready; then
  rm -rf .venv
  PYTHON=""
  for candidate in python3.13 python3.12 python3.11 python3.10 /usr/bin/python3 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && supported "$(command -v "$candidate")"; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  done
  if [[ -z "$PYTHON" ]]; then
    echo "No Python 3.9 to 3.13 found. Install Xcode's command line tools (xcode-select --install)" >&2
    echo "or Python 3.12 from python.org, then run this again." >&2
    exit 2
  fi
  echo "Setting up .venv with $PYTHON ($("$PYTHON" --version))"
  "$PYTHON" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements-mac.txt
  .venv/bin/python -c 'import coremltools, numpy'
  echo "$wanted" > .venv/requirements.sha256
fi

exec .venv/bin/python validate_coreml.py "$@"
