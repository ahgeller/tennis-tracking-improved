"""Score a pipeline run's exported landings against the hand-labelled ones (``finetune.scoring.court_events_score``).

    python score_court_events.py --archive finetune --out-dir output/<run>/pipeline
"""
from finetune.scoring.court_events_score import main

if __name__ == "__main__":
    raise SystemExit(main())
