"""tennis_tracker.paths: the model assets are spelled once and every consumer agrees with it."""
from pathlib import Path

from tennis_tracker import paths
from tennis_tracker.config import Config


def test_assets_sit_under_the_repository_models_folder():
    assert paths.ROOT == Path(__file__).resolve().parents[1]
    assert paths.MODELS == paths.ROOT / "models"
    for asset in (paths.HEATMAP_WEIGHTS, paths.PLAYER_ENGINE, paths.COURT_ENGINE, paths.BOUNCE_MODEL):
        assert asset.parent == paths.MODELS and asset.is_absolute()
    assert paths.STREAM_WEIGHTS.parent == paths.MODELS / "candidates" and paths.STREAM_WEIGHTS.is_absolute()
    assert paths.BALL_WEIGHTS == paths.STREAM_WEIGHTS                   # the default ball detector


def test_config_defaults_use_the_same_files():
    cfg = Config()
    assert (cfg.model_path, cfg.player_model_path, cfg.court_model_path) == (
        str(paths.BALL_WEIGHTS), str(paths.PLAYER_ENGINE), str(paths.COURT_ENGINE))


def test_the_default_threshold_is_the_default_detectors():
    from tennis_tracker.ball_weights import STREAM_S4, operating_conf
    assert Config().conf == operating_conf(paths.BALL_WEIGHTS) == STREAM_S4.conf == 0.42
