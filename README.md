# UlyanAI

UlyanAI — сервис прогнозирования ценовых диапазонов криптоактивов (BTC/ETH/SOL) с Web GUI для управления событиями и анализа качества модели.

Проект включает полный цикл:
- загрузка и подготовка данных;
- обучение quantile-моделей;
- walk-forward оценка;
- paper simulation;
- inference API;
- веб-панель для работы с событиями.

## Кратко о том, как это работает

1. Берем свечи Binance (`1m`), чистим и ресемплим в нужные горизонты (`5m..1w`).
2. Строим features и labels.
3. Обучаем LightGBM quantile модели (`q10/q50/q90`).
4. Отдаем прогноз через API в формате `price_ranges`.
5. В Event Dashboard можно создать событие, мониторить его до экспирации, получить факт и метрики (`hit/miss`, ошибки, ширина диапазона).

Если модельный артефакт не найден, сервис использует fallback-возвраты для горизонта.

## Основной API-контракт

Ключевые поля ответа прогноза:
- `mode: "price_ranges"`
- `price_levels: { p10, p50, p90 }`
- `price_range: { low, high, nominal }`
- `median_price`
- `model_version`
- `stale_data`

## Стек

- Python, FastAPI, Uvicorn
- Pandas, NumPy, PyArrow
- LightGBM, scikit-learn
- SQLite (хранилище событий)
- HTML/CSS/JS (Web GUI)

## Структура проекта

- `src/` — код пайплайнов, моделей, API, симуляции.
- `configs/` — конфиги сервиса, активов, горизонтов, feature set.
- `data/` — parquet-данные (raw/clean/resampled/features/labels).
- `artifacts/` — артефакты обучения, отчеты, run-метаданные, sqlite БД событий.
- `web/index.html` — интерфейс Dashboard.
- `tests/` — тесты.

## Быстрый старт (локально)

```powershell
python -m pip install -r requirements.txt
python -m src.cli --asset BTC pipeline-all
python -m src.cli service-run
```

После запуска:
- API и GUI: `http://localhost:8000/`
- API key по умолчанию: `dev-key`
- Header: `X-API-Key`

Проверка:
```powershell
curl -H "X-API-Key: dev-key" "http://localhost:8000/v1/predict?asset=BTC&horizon=1h"
```

## Команды CLI

- `data-sync` — загрузка/очистка/ресемпл/QC.
- `features-build` — генерация features + labels.
- `train-run` — обучение моделей.
- `eval-walk-forward` — walk-forward оценка.
- `sim-run` — paper simulation.
- `pipeline-all` — полный цикл.
- `service-run` — запуск FastAPI сервиса.

Примеры:
```powershell
python -m src.cli --asset BTC data-sync
python -m src.cli --asset BTC train-run
python -m src.cli service-run --reload
```

## PowerShell скрипты

- `scripts/run_all.ps1 -Asset BTC`
- `scripts/run_service.ps1`
- `scripts/run_service.ps1 -Reload`

## Web GUI

GUI открывается по `/` и предназначен для:
- создания и отмены Event;
- мониторинга активных/завершенных событий;
- просмотра деталей события (overview/chart/live/payload/result/model);
- фильтрации и анализа истории;
- экспорта в JSON/CSV/PNG/PDF.

## Основные API endpoints

`v1`:
- `GET /v1/health`
- `GET /v1/status`
- `GET /v1/predict`
- `POST /v1/predict_batch`
- `GET /v1/metrics`

`events`:
- `POST /api/events`
- `GET /api/events`
- `GET /api/events/{event_id}`
- `GET /api/events/{event_id}/prices`
- `POST /api/events/{event_id}/cancel`
- `GET /api/models`
- `GET /api/models/production`
- `GET /api/metrics/summary`
- `GET /api/alerts`
- `GET /api/stream/events` (SSE)

## Конфигурация

Главные файлы:
- `configs/base.yaml`
- `configs/assets.yaml`
- `configs/horizons.yaml`
- `configs/feature_sets/v1.yaml`

В `base.yaml` задаются:
- сервис (`host`, `port`, `api_key`, rate limit);
- источники данных и TTL кэша;
- параметры тренировки, walk-forward и simulation;
- пути хранения (`data`, `artifacts`).

## Docker

```powershell
docker compose up --build
```

Сервис будет доступен на `http://localhost:8000`.

## Тесты

```powershell
pytest -q
```

## Важные заметки

- Все пути в проекте относительные к корню репозитория.
- Временные метки хранятся в UTC.
- Поддерживаемые горизонты: `5m`, `15m`, `1h`, `4h`, `1d`, `1w`.
