"""Every label type's scorecard for one pipeline run (``finetune.scoring.scorecard``).

    python score_labels.py --out-dir output/label_candidates [--holdout mixed_v1] [--only hits rallies]
"""
from finetune.scoring.scorecard import main

if __name__ == "__main__":
    raise SystemExit(main())
