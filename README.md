# UlyanAI MVP

AI-сервис прогнозирования диапазонов цен для BTC/ETH/SOL (MVP) по ТЗ для Polymarket.

## Что реализовано
- ETL: Binance OHLCV 1m -> clean -> resampled (5m..1w), хранение в Parquet.
- QC: дубли, шаг времени, пропуски, OHLC, volume, UTC, future-data checks.
- Features/Labels: feature_set v1, целевая `r_H(t)=log(P(t+H)/P(t))`.
- Training: LightGBM quantile models (q10/q50/q90), калибровка интервала.
- Evaluation: walk-forward отчеты.
- Simulation: paper-симулятор с edge/risk-limit логикой.
- Inference API (FastAPI): `/v1/health`, `/v1/status`, `/v1/predict`, `/v1/predict_batch`, `/v1/metrics`.
- UI: `web/index.html` (Dashboard + Model/Data Status) в терминах price ranges.

## Ключевой контракт
Внешний ответ API только в языке цены:
- `mode: "price_ranges"`
- `price_levels: {p10,p50,p90}`
- `price_range: {low,high,nominal}`
- `median_price`

## Запуск локально
```powershell
python -m pip install -r requirements.txt
python -m src.cli --asset BTC pipeline-all
python -m src.cli --asset ETH pipeline-all
python -m src.cli --asset SOL pipeline-all
python -m src.cli service-run
```

или

```powershell
scripts/run_all.ps1
scripts/run_service.ps1
```

Запрос прогноза:
```powershell
curl -H "X-API-Key: dev-key" "http://localhost:8000/v1/predict?asset=BTC&horizon=1h"
```

UI: открой `web/index.html` и укажи API key (`dev-key` по умолчанию).

## Запуск в Docker
```powershell
docker compose up --build
```

## Конфиги
- `configs/base.yaml`
- `configs/assets.yaml`
- `configs/horizons.yaml`
- `configs/feature_sets/v1.yaml`

## Тесты
```powershell
pytest -q
```
