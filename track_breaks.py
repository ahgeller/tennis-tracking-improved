"""Where a pipeline run loses the ball, from its tracking JSONs and the clips' labels (``finetune.scoring.track_breaks``).

    python track_breaks.py output/fix1_no_court/pipeline --compare output/baseline/track_breaks.log
"""
from finetune.scoring.track_breaks import main

if __name__ == "__main__":
    raise SystemExit(main())
