from __future__ import annotations

from copy import deepcopy
import json

from src.config import load_config
from src.pipeline import run_data_pipeline, run_feature_label_pipeline, run_training_pipeline


def test_smoke_data_features_train_pipeline(tmp_path) -> None:
    cfg = load_config("configs")
    cfg = deepcopy(cfg)
    cfg["data"]["download_days"] = 3
    cfg["training"]["num_boost_round"] = 20
    cfg["walk_forward"]["val_days"] = 1
    cfg["walk_forward"]["test_days"] = 1
    cfg["timeframes"] = ["5m", "15m", "1h"]
    cfg["horizons"] = ["5m", "15m", "1h"]
    cfg["horizons_map"] = {
        "5m": {"timeframe": "5m", "minutes": 5},
        "15m": {"timeframe": "15m", "minutes": 15},
        "1h": {"timeframe": "1h", "minutes": 60},
    }

    run_data_pipeline(cfg, asset="BTC", root=tmp_path)
    fl = run_feature_label_pipeline(cfg, asset="BTC", root=tmp_path)
    assert "5m" in fl

    train = run_training_pipeline(cfg, asset="BTC", root=tmp_path)
    assert isinstance(train, dict)
    assert len(train.get("trained", {})) >= 1

    data_run = json.loads((tmp_path / "artifacts" / "runs" / "data_btc.json").read_text(encoding="utf-8"))
    features_run = json.loads((tmp_path / "artifacts" / "runs" / "features_labels_btc.json").read_text(encoding="utf-8"))
    training_run = json.loads((tmp_path / "artifacts" / "runs" / "training_btc.json").read_text(encoding="utf-8"))
    for payload in (data_run, features_run, training_run):
        assert "meta" in payload
        assert "git_commit" in payload["meta"]
        assert "config_hash" in payload["meta"]
        assert "dataset_hash" in payload["meta"]
