from __future__ import annotations

from copy import deepcopy

from src.config import load_config
from src.pipeline import run_doctor


class _Resp:
    ok = True
    status_code = 200


def test_doctor_preflight_contract(tmp_path, monkeypatch) -> None:
    cfg = deepcopy(load_config("configs"))
    cfg["horizons_map"] = {"5m": {"timeframe": "5m", "minutes": 5}}
    cfg["assets"] = {"BTC": {"symbol": "BTC/USDT", "market_symbol": "BTCUSDT", "status": "enabled"}}
    monkeypatch.setattr("src.pipeline.requests.get", lambda *args, **kwargs: _Resp())

    result = run_doctor(cfg, asset="BTC", root=tmp_path)
    assert isinstance(result, dict)
    assert "ok" in result
    assert "checks" in result
    assert "meta" in result
    assert {"git_commit", "config_hash", "dataset_hash"} <= set(result["meta"].keys())

    names = {item["name"] for item in result["checks"]}
    assert "config.asset" in names
    assert "config.horizons" in names
    assert "models.present" in names
    assert "binance.ping" in names
