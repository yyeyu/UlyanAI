(() => {
      "use strict";
      const HORIZONS = ["5m", "15m", "1h", "4h", "1d", "1w"];
      const TRAINING_BUDGET_PRESETS = {
        F0: {
          label: "F0",
          horizons: ["5m"],
          quantile_strategy: "interval_only",
          selected_quantiles: ["q10", "q50", "q90"],
          train_window_days: 120,
          val_days: 14,
          test_days: 14,
          num_boost_round: 60,
          early_stopping_rounds: 10,
          scale_grid: [0.8, 1.0, 1.2],
        },
        F1: {
          label: "F1",
          horizons: ["5m", "15m"],
          quantile_strategy: "selected_set",
          selected_quantiles: ["q10", "q25", "q50", "q75", "q90"],
          train_window_days: 365,
          val_days: 30,
          test_days: 30,
          num_boost_round: 120,
          early_stopping_rounds: 20,
          scale_grid: [0.75, 1.0, 1.2, 1.5, 2.0],
        },
        F2: {
          label: "F2",
          horizons: ["5m", "15m", "1h"],
          quantile_strategy: "full_grid",
          selected_quantiles: ["q10", "q50", "q90"],
          train_window_days: 540,
          val_days: 45,
          test_days: 45,
          num_boost_round: 240,
          early_stopping_rounds: 30,
          scale_grid: [0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0],
        },
      };
      const SCALE_SELECTION_RULE_OPTIONS = [
        { value: "min_abs_coverage_gap_then_width", label: "Gap then width" },
        { value: "min_abs_coverage_gap_then_min_width", label: "Gap then min width" },
        { value: "min_abs_coverage_gap", label: "Coverage gap only" },
        { value: "min_score", label: "Min score" },
      ];
      const SWEEP_AXIS_DEFS = (() => {
        const rows = [
          { group: "Target", path: "horizons", label: "Horizons", kind: "enum_list", options: HORIZONS },
          { group: "Target", path: "base_timeframe_mode", label: "Base timeframe mode", kind: "enum", options: ["legacy", "base_1m"] },
          { group: "Split", path: "train_window_mode", label: "Train window mode", kind: "enum", options: ["expanding", "rolling"] },
          { group: "Split", path: "train_window_days", label: "Train window days", kind: "int", allowRange: true, min: 1, max: 3650 },
          { group: "Split", path: "val_days", label: "Validation days", kind: "int", allowRange: true, min: 1, max: 3650 },
          { group: "Split", path: "test_days", label: "Test days", kind: "int", allowRange: true, min: 1, max: 3650 },
          { group: "Walk-forward", path: "walk_forward_enabled", label: "Walk-forward enabled", kind: "bool" },
          { group: "Walk-forward", path: "wf_train_days", label: "WF train days", kind: "int", allowRange: true, min: 1, max: 3650 },
          { group: "Walk-forward", path: "wf_val_days", label: "WF validation days", kind: "int", allowRange: true, min: 1, max: 3650 },
          { group: "Walk-forward", path: "wf_step_days", label: "WF step days", kind: "int", allowRange: true, min: 1, max: 3650 },
          { group: "Walk-forward", path: "wf_folds", label: "WF folds", kind: "int", allowRange: true, min: 1, max: 100 },
          { group: "Interval", path: "target_coverage_percent", label: "Target coverage %", kind: "int", allowRange: true, min: 1, max: 99 },
          { group: "Interval", path: "interval_mode", label: "Interval mode", kind: "enum", options: ["symmetric", "custom", "multi_pack"] },
          { group: "Interval", path: "custom_q_low_percent", label: "Custom q low %", kind: "int", allowRange: true, min: 1, max: 98 },
          { group: "Interval", path: "custom_q_high_percent", label: "Custom q high %", kind: "int", allowRange: true, min: 2, max: 99 },
          { group: "Interval", path: "multi_interval_coverages", label: "Multi interval coverages", kind: "int_list", min: 1, max: 99 },
          { group: "Interval", path: "quantile_strategy", label: "Q strategy", kind: "enum", options: ["interval_only", "selected_set", "full_grid"] },
          { group: "Interval", path: "selected_quantiles", label: "Selected q-set", kind: "quantile_list" },
          { group: "Features", path: "feature_set_version", label: "Feature set", kind: "enum", options: ["feat_v1", "feat_v2"] },
          { group: "Calibration", path: "scale_grid", label: "Scale grid", kind: "float_list" },
          { group: "Calibration", path: "scale_selection_rule", label: "Scale selection rule", kind: "enum", options: SCALE_SELECTION_RULE_OPTIONS.map((item) => item.value) },
          { group: "Calibration", path: "calibration_method", label: "Calibration method", kind: "enum", options: ["grid_scale"] },
          { group: "Feature Overrides", path: "feature_overrides.return_windows", label: "Return windows", kind: "int_list", min: 1, max: 10000 },
          { group: "Feature Overrides", path: "feature_overrides.vol_windows", label: "Volatility windows", kind: "int_list", min: 2, max: 10000 },
          { group: "Feature Overrides", path: "feature_overrides.volume_windows", label: "Volume windows", kind: "int_list", min: 2, max: 10000 },
          { group: "Feature Overrides", path: "feature_overrides.trend_windows", label: "Trend windows", kind: "int_list", min: 2, max: 10000 },
          { group: "Feature Overrides", path: "feature_overrides.rsi_window", label: "RSI window", kind: "int", allowRange: true, min: 2, max: 1000 },
          { group: "Feature Overrides", path: "feature_overrides.atr_window", label: "ATR window", kind: "int", allowRange: true, min: 2, max: 1000 },
          { group: "Feature Overrides", path: "feature_overrides.macd.fast", label: "MACD fast", kind: "int", allowRange: true, min: 2, max: 1000 },
          { group: "Feature Overrides", path: "feature_overrides.macd.slow", label: "MACD slow", kind: "int", allowRange: true, min: 2, max: 1000 },
          { group: "Feature Overrides", path: "feature_overrides.macd.signal", label: "MACD signal", kind: "int", allowRange: true, min: 2, max: 1000 },
        ];
        HORIZONS.forEach((horizon) => {
          rows.push({ group: "Target", path: `steps_overrides.${horizon}`, label: `steps_ahead ${horizon}`, kind: "int", allowRange: true, min: 1, max: 10080 });
        });
        ["returns", "volatility", "volume", "trend", "rsi", "atr", "macd"].forEach((name) => {
          rows.push({ group: "Feature Groups", path: `feature_groups.${name}`, label: `Feature group ${name}`, kind: "bool" });
        });
        [
          ["learning_rate", "float", 0.0001, 10],
          ["num_leaves", "int", 2, 131072],
          ["max_depth", "int", -1, 4096],
          ["min_data_in_leaf", "int", 1, 1000000],
          ["feature_fraction", "float", 0, 1],
          ["bagging_fraction", "float", 0, 1],
          ["bagging_freq", "int", 0, 100000],
          ["lambda_l1", "float", 0, 1000000],
          ["lambda_l2", "float", 0, 1000000],
          ["num_boost_round", "int", 1, 1000000],
          ["early_stopping_rounds", "int", 0, 1000000],
          ["seed", "int", -2147483648, 2147483647],
        ].forEach(([name, kind, min, max]) => {
          rows.push({ group: "Hyperparams", path: `hyperparams.${name}`, label: name, kind, allowRange: true, min, max });
        });
        return rows;
      })();
      const SWEEP_AXIS_BY_PATH = Object.fromEntries(SWEEP_AXIS_DEFS.map((item) => [item.path, item]));
      const defaultSettings = {
        apiBase: location.protocol === "file:" ? "http://localhost:8000" : location.origin,
        apiHeader: "X-API-Key",
        apiKey: "dev-key",
        refreshSec: 5,
        autoRefresh: "on",
      };
      const state = {
        settings: { ...defaultSettings },
        activeTab: "events",
        activeLabTab: "compare",
        statusInfo: null,
        quote: null,
        events: { items: [], total: 0, page: 1, page_size: 10 },
        markets: { items: [], total: 0, fetched_at: null, active: true, closed: false },
        marketsLoadInFlight: false,
        marketsLastLoadedMs: 0,
        selectedEvent: null,
        selectedSamples: [],
        detailMarketSamples: [],
        detailLiveSamples: [],
        detailChartTimer: null,
        detailScale: null,
        detailMarketIntervalSec: 15,
        detailWindowStartMs: null,
        detailWindowEndMs: null,
        detailChartSeq: 0,
        detailTv: null,
        models: [],
        productionModels: [],
        analysis: {
          summary: null,
          scoreboard: null,
          chartSeries: null,
          histSeries: null,
          tournamentSummary: null,
          deepEvents: [],
          deepTotal: 0,
        },
        lab: {
          validation: null,
          jobs: [],
          jobLogs: [],
          selectedJobId: null,
          recentModels: [],
          selectedModelId: null,
          validateTimer: null,
          jobStatusById: {},
          budgetPresetInitialized: false,
          sweepAxes: [],
          sweepAxisSeq: 1,
        },
        pollTimer: null,
        tickerTimer: null,
        quoteLiveTimer: null,
        quoteFetchInFlight: false,
        searchTimer: null,
        refreshInFlight: false,
        eventsLoadedOnce: false,
        seenEventIds: new Set(),
      };
      const el = (id) => document.getElementById(id);
      const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "-" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d });
      const price = (v) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "-" : Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      const pct = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "-" : `${(Number(v) * 100).toFixed(d)}%`;
      const pctRaw = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "-" : `${Number(v).toFixed(d)}%`;
      const dt = (iso) => { if (!iso) return "-"; try { return new Date(iso).toISOString().replace("T", " ").replace("Z", "+00:00"); } catch { return iso; } };
      const esc = (v) => String(v ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll("\"", "&quot;").replaceAll("'", "&#39;");
      const cycleText = (item) => item?.cycle_id ? `${String(item.cycle_id).slice(0, 8)} ${Number(item.cycle_seq || 0)}/${Number(item.cycle_total || 0)}` : "-";
      const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
      function flashNode(node, className = "flash-update") {
        if (!node) return;
        node.classList.remove(className);
        void node.offsetWidth;
        node.classList.add(className);
        setTimeout(() => node.classList.remove(className), 240);
      }
      function setNodeText(nodeOrId, value, { flash = false } = {}) {
        const node = typeof nodeOrId === "string" ? el(nodeOrId) : nodeOrId;
        if (!node) return false;
        const next = String(value ?? "");
        const changed = node.textContent !== next;
        node.textContent = next;
        if (flash && changed) flashNode(node);
        return changed;
      }
      function triggerScanSweep(target) {
        if (!target) return;
        const nodes = Array.isArray(target) ? target : [target];
        nodes.forEach((node) => {
          if (!node) return;
          node.classList.remove("is-scanning");
          void node.offsetWidth;
          node.classList.add("is-scanning");
          setTimeout(() => node.classList.remove("is-scanning"), 340);
        });
      }
      function parseHorizonMs(horizon) {
        const raw = String(horizon || "").trim().toLowerCase();
        const match = raw.match(/^(\d+)\s*([mhdw])$/);
        if (!match) return NaN;
        const n = Number(match[1]);
        if (!Number.isFinite(n) || n <= 0) return NaN;
        const unit = match[2];
        if (unit === "m") return n * 60 * 1000;
        if (unit === "h") return n * 60 * 60 * 1000;
        if (unit === "d") return n * 24 * 60 * 60 * 1000;
        if (unit === "w") return n * 7 * 24 * 60 * 60 * 1000;
        return NaN;
      }
      function detailPreStartMs(ev) {
        const hMs = parseHorizonMs(ev?.horizon);
        if (!Number.isFinite(hMs)) return 10 * 60 * 1000;
        return clamp(Math.round(hMs * 0.2), 5 * 60 * 1000, 30 * 60 * 1000);
      }
      function parseBinanceIntervalSec(interval) {
        const raw = String(interval || "").trim().toLowerCase();
        const m = raw.match(/^(\d+)([mhdw])$/);
        if (!m) return NaN;
        const n = Number(m[1]);
        if (!Number.isFinite(n) || n <= 0) return NaN;
        const unit = m[2];
        if (unit === "m") return n * 60;
        if (unit === "h") return n * 60 * 60;
        if (unit === "d") return n * 24 * 60 * 60;
        if (unit === "w") return n * 7 * 24 * 60 * 60;
        return NaN;
      }
      function intervalFromHorizonSec(horizon) {
        return 15;
      }
      function computeDetailWindowMs(ev, nowMs = Date.now()) {
        const frameMs = 60 * 60 * 1000;
        const createdMs = Date.parse(ev?.created_at);
        const expiresMs = Date.parse(ev?.expires_at);
        const parsedHorizonMs = parseHorizonMs(ev?.horizon);
        const eventStartMs = Number.isFinite(createdMs) ? createdMs : nowMs;
        const fallbackEndMs = eventStartMs + (Number.isFinite(parsedHorizonMs) ? parsedHorizonMs : (5 * 60 * 1000));
        const eventEndMs = (Number.isFinite(expiresMs) && expiresMs > eventStartMs) ? expiresMs : fallbackEndMs;
        const eventSpanMs = Math.max(60 * 1000, eventEndMs - eventStartMs);
        if (eventSpanMs >= frameMs) {
          const startMs = eventStartMs;
          const endMs = startMs + frameMs;
          return {
            startMs,
            endMs,
            fetchStartMs: startMs,
            fetchEndMs: Math.min(nowMs, endMs),
          };
        }
        const padMs = Math.max(0, Math.floor((frameMs - eventSpanMs) / 2));
        const startMs = eventStartMs - padMs;
        const endMs = eventEndMs + padMs;
        return {
          startMs,
          endMs,
          fetchStartMs: startMs,
          fetchEndMs: Math.min(nowMs, endMs),
        };
      }
      function resolveDetailChartIntervalSec(ev, candles = null) {
        if (Number.isFinite(state.detailMarketIntervalSec) && state.detailMarketIntervalSec > 0) return state.detailMarketIntervalSec;
        if (Array.isArray(candles) && candles.length >= 2) {
          let minDiff = Infinity;
          for (let i = 1; i < candles.length; i++) {
            const diff = Number(candles[i].time) - Number(candles[i - 1].time);
            if (Number.isFinite(diff) && diff > 0 && diff < minDiff) minDiff = diff;
          }
          if (Number.isFinite(minDiff) && minDiff > 0) return minDiff;
        }
        return intervalFromHorizonSec(ev?.horizon);
      }
      function setDetailChartEmptyState(show, message = "No market data for this event.") {
        const node = el("detailChartEmpty");
        if (!node) return;
        node.textContent = message;
        node.classList.toggle("show", Boolean(show));
      }
      function formatUsd(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return "-";
        return `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      }
      function updateDetailRangeWidthLabel(ev) {
        const node = el("detailChartWidth");
        if (!node) return;
        const low = Number(ev?.pred_low);
        const high = Number(ev?.pred_high);
        if (!Number.isFinite(low) || !Number.isFinite(high)) {
          node.textContent = "Width: -";
          return;
        }
        node.textContent = `Width: ${formatUsd(Math.abs(high - low))}`;
      }
      function setDetailRangeOverlayHidden() {
        const node = el("detailChartRange");
        if (!node) return;
        node.classList.remove("show");
        node.textContent = "";
      }
      function destroyDetailTvChart() {
        const runtime = state.detailTv;
        if (!runtime) return;
        if (runtime.overlayRaf) {
          try { cancelAnimationFrame(runtime.overlayRaf); } catch {}
          runtime.overlayRaf = null;
        }
        if (runtime.logicalRangeHandler) {
          try { runtime.chart.timeScale().unsubscribeVisibleLogicalRangeChange(runtime.logicalRangeHandler); } catch {}
        }
        if (runtime.visibleRangeHandler) {
          try { runtime.chart.timeScale().unsubscribeVisibleTimeRangeChange(runtime.visibleRangeHandler); } catch {}
        }
        if (runtime.crosshairHandler) {
          try { runtime.chart.unsubscribeCrosshairMove(runtime.crosshairHandler); } catch {}
        }
        if (runtime.hostWheelHandler) runtime.host.removeEventListener("wheel", runtime.hostWheelHandler);
        if (runtime.hostPointerHandler) runtime.host.removeEventListener("pointermove", runtime.hostPointerHandler);
        if (runtime.hostMouseDownHandler) runtime.host.removeEventListener("mousedown", runtime.hostMouseDownHandler);
        if (runtime.hostTouchMoveHandler) runtime.host.removeEventListener("touchmove", runtime.hostTouchMoveHandler);
        if (runtime.resizeObserver) {
          try { runtime.resizeObserver.disconnect(); } catch {}
        }
        if (runtime.resizeHandler) window.removeEventListener("resize", runtime.resizeHandler);
        try { runtime.chart.remove(); } catch {}
        state.detailTv = null;
      }
      function ensureDetailTvChart() {
        const host = el("detailTvChart");
        if (!host) return null;
        const lc = window.LightweightCharts;
        if (!lc || typeof lc.createChart !== "function") {
          setDetailChartEmptyState(true, "TradingView chart library is unavailable.");
          return null;
        }
        if (state.detailTv && state.detailTv.host === host) return state.detailTv;
        destroyDetailTvChart();
        const width = Math.max(40, host.clientWidth || 760);
        const height = Math.max(40, host.clientHeight || 280);
        const chart = lc.createChart(host, {
          width,
          height,
          layout: {
            background: { type: "solid", color: "#0a0a0a" },
            textColor: "#aab6ae",
            fontFamily: "\"IBM Plex Mono\", \"JetBrains Mono\", monospace",
            fontSize: 11,
          },
          grid: {
            vertLines: { color: "rgba(120,120,120,.12)" },
            horzLines: { color: "rgba(120,120,120,.12)" },
          },
          crosshair: {
            mode: lc.CrosshairMode?.Normal ?? 0,
            vertLine: {
              width: 1,
              color: "rgba(102,255,170,.32)",
              style: lc.LineStyle?.Solid ?? 0,
              labelBackgroundColor: "#070707",
            },
            horzLine: {
              width: 1,
              color: "rgba(102,255,170,.26)",
              style: lc.LineStyle?.Solid ?? 0,
              labelBackgroundColor: "#070707",
            },
          },
          rightPriceScale: {
            borderColor: "#1d1d1d",
            scaleMargins: { top: 0.1, bottom: 0.1 },
          },
          timeScale: {
            borderColor: "#1d1d1d",
            rightOffset: 3,
            barSpacing: 8,
            minBarSpacing: 0.4,
            timeVisible: true,
            secondsVisible: false,
          },
          localization: {
            priceFormatter: (v) => Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
          },
          handleScroll: { mouseWheel: true, pressedMouseMove: true, vertTouchDrag: false, horzTouchDrag: true },
          handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
        });
        const candleSeries = chart.addCandlestickSeries({
          upColor: "#00ff88",
          downColor: "#ff5d7f",
          borderVisible: true,
          borderUpColor: "#00ff88",
          borderDownColor: "#ff5d7f",
          wickUpColor: "#66ffaa",
          wickDownColor: "#ff8ea6",
          priceLineVisible: true,
          lastValueVisible: true,
        });
        const runtime = {
          host,
          chart,
          candleSeries,
          priceLines: [],
          candles: [],
          intervalSec: resolveDetailChartIntervalSec(state.selectedEvent),
          resizeObserver: null,
          resizeHandler: null,
          overlayRaf: null,
          scheduleOverlaySync: null,
          logicalRangeHandler: null,
          visibleRangeHandler: null,
          crosshairHandler: null,
          hostWheelHandler: null,
          hostPointerHandler: null,
          hostMouseDownHandler: null,
          hostTouchMoveHandler: null,
        };
        runtime.scheduleOverlaySync = () => {
          if (runtime.overlayRaf) return;
          runtime.overlayRaf = requestAnimationFrame(() => {
            runtime.overlayRaf = null;
            updateDetailRangeOverlay(state.selectedEvent, runtime);
          });
        };
        const resize = () => {
          const w = Math.max(40, host.clientWidth || width);
          const h = Math.max(40, host.clientHeight || height);
          chart.applyOptions({ width: w, height: h });
          runtime.scheduleOverlaySync();
        };
        if (typeof ResizeObserver !== "undefined") {
          runtime.resizeObserver = new ResizeObserver(resize);
          runtime.resizeObserver.observe(host);
        } else {
          runtime.resizeHandler = resize;
          window.addEventListener("resize", resize);
        }
        runtime.logicalRangeHandler = () => runtime.scheduleOverlaySync();
        runtime.visibleRangeHandler = () => runtime.scheduleOverlaySync();
        runtime.crosshairHandler = () => runtime.scheduleOverlaySync();
        try { chart.timeScale().subscribeVisibleLogicalRangeChange(runtime.logicalRangeHandler); } catch {}
        try { chart.timeScale().subscribeVisibleTimeRangeChange(runtime.visibleRangeHandler); } catch {}
        try { chart.subscribeCrosshairMove(runtime.crosshairHandler); } catch {}
        runtime.hostWheelHandler = () => runtime.scheduleOverlaySync();
        runtime.hostPointerHandler = () => runtime.scheduleOverlaySync();
        runtime.hostMouseDownHandler = () => runtime.scheduleOverlaySync();
        runtime.hostTouchMoveHandler = () => runtime.scheduleOverlaySync();
        host.addEventListener("wheel", runtime.hostWheelHandler, { passive: true });
        host.addEventListener("pointermove", runtime.hostPointerHandler, { passive: true });
        host.addEventListener("mousedown", runtime.hostMouseDownHandler);
        host.addEventListener("touchmove", runtime.hostTouchMoveHandler, { passive: true });
        state.detailTv = runtime;
        return runtime;
      }
      function clearDetailPriceLines(runtime) {
        if (!runtime?.candleSeries || !Array.isArray(runtime.priceLines)) return;
        runtime.priceLines.forEach((line) => {
          try { runtime.candleSeries.removePriceLine(line); } catch {}
        });
        runtime.priceLines = [];
      }
      function addDetailPriceLine(runtime, options) {
        if (!runtime?.candleSeries) return;
        const line = runtime.candleSeries.createPriceLine(options);
        runtime.priceLines.push(line);
      }
      function fmtClock(ms, withSec = true) {
        if (!Number.isFinite(ms)) return "-";
        const d = new Date(ms);
        const hh = String(d.getHours()).padStart(2, "0");
        const mm = String(d.getMinutes()).padStart(2, "0");
        if (!withSec) return `${hh}:${mm}`;
        const ss = String(d.getSeconds()).padStart(2, "0");
        return `${hh}:${mm}:${ss}`;
      }
      function updateDetailChartLegend(ev, candles) {
        const legend = el("detailChartLegend");
        if (!legend) return;
        if (!ev || !candles.length) {
          legend.textContent = "-";
          return;
        }
        const isActive = ev.status === "active";
        const createdMs = Date.parse(ev.created_at);
        const expiresMs = Date.parse(ev.expires_at);
        const updatedMs = Date.parse(ev.updated_at);
        const closeMs = isActive
          ? Date.now()
          : (Number.isFinite(updatedMs) ? updatedMs : (Number.isFinite(expiresMs) ? expiresMs : Date.now()));
        const closePx = Number(ev.actual_price ?? ev.current_price);
        const last = candles[candles.length - 1];
        const rows = [
          `<strong>open</strong> ${price(ev.price_t0)} @ ${fmtClock(createdMs, true)}`,
          `<strong>model</strong> [${price(ev.pred_low)}; ${price(ev.pred_high)}], med ${price(ev.pred_mid)}`,
          (!isActive && Number.isFinite(closePx))
            ? `<strong>close</strong> ${price(closePx)} @ ${fmtClock(closeMs, true)}`
            : `<strong>last</strong> ${price(last.close)} @ ${fmtClock(last.time * 1000, true)}`
        ];
        legend.innerHTML = rows.join("<br>");
      }
      function snapMarkerTime(candles, targetSec, mode = "nearest") {
        if (!candles.length || !Number.isFinite(targetSec)) return null;
        if (mode === "ceil") {
          for (let i = 0; i < candles.length; i++) {
            if (candles[i].time >= targetSec) return candles[i].time;
          }
          return candles[candles.length - 1].time;
        }
        if (mode === "floor") {
          for (let i = candles.length - 1; i >= 0; i--) {
            if (candles[i].time <= targetSec) return candles[i].time;
          }
          return candles[0].time;
        }
        let best = candles[0].time;
        let bestDiff = Math.abs(best - targetSec);
        for (let i = 1; i < candles.length; i++) {
          const t = candles[i].time;
          const diff = Math.abs(t - targetSec);
          if (diff < bestDiff) {
            best = t;
            bestDiff = diff;
          }
        }
        return best;
      }
      function buildDetailMarkers(ev, candles) {
        const markers = [];
        if (!ev || !candles.length) return markers;
        const openPx = Number(ev.price_t0);
        const closePx = Number(ev.actual_price ?? ev.current_price);
        const createdSec = Math.floor(Date.parse(ev.created_at) / 1000);
        const openMarkerTime = snapMarkerTime(candles, createdSec, "ceil");
        if (Number.isFinite(openPx) && Number.isFinite(openMarkerTime)) {
          markers.push({
            time: openMarkerTime,
            position: "belowBar",
            color: "#00ff66",
            shape: "arrowUp",
            text: `open ${price(openPx)}`
          });
        }
        if (ev.status === "active") {
          const last = candles[candles.length - 1];
          markers.push({
            time: last.time,
            position: "aboveBar",
            color: "#66ffaa",
            shape: "circle",
            text: `last ${price(last.close)}`
          });
        } else if (Number.isFinite(closePx)) {
          const expiresSec = Math.floor(Date.parse(ev.expires_at) / 1000);
          const updatedSec = Math.floor(Date.parse(ev.updated_at) / 1000);
          const closeSec = Number.isFinite(updatedSec) ? updatedSec : expiresSec;
          const closeMarkerTime = snapMarkerTime(candles, closeSec, "floor");
          if (Number.isFinite(closeMarkerTime)) {
            markers.push({
              time: closeMarkerTime,
              position: "aboveBar",
              color: "#66ffaa",
              shape: "arrowDown",
              text: `close ${price(closePx)}`
            });
          }
        }
        return markers;
      }
      function updateDetailRangeOverlay(ev, runtime) {
        const node = el("detailChartRange");
        if (!node || !runtime?.candleSeries || !runtime?.host || !ev) {
          setDetailRangeOverlayHidden();
          return;
        }
        const chartNode = runtime.host.parentElement;
        if (!chartNode) {
          setDetailRangeOverlayHidden();
          return;
        }
        const low = Number(ev.pred_low);
        const high = Number(ev.pred_high);
        if (!Number.isFinite(low) || !Number.isFinite(high)) {
          setDetailRangeOverlayHidden();
          return;
        }
        const yLow = runtime.candleSeries.priceToCoordinate(low);
        const yHigh = runtime.candleSeries.priceToCoordinate(high);
        if (!Number.isFinite(yLow) || !Number.isFinite(yHigh)) {
          setDetailRangeOverlayHidden();
          return;
        }
        const hostRect = runtime.host.getBoundingClientRect();
        const chartRect = chartNode.getBoundingClientRect();
        const offsetLeft = Math.max(0, hostRect.left - chartRect.left);
        const offsetTop = Math.max(0, hostRect.top - chartRect.top);
        const hostWidth = Math.max(1, runtime.host.clientWidth || Math.round(hostRect.width) || 0);
        const hostHeight = Math.max(1, runtime.host.clientHeight || Math.round(hostRect.height) || 0);
        const topRaw = Math.min(yLow, yHigh);
        const bottomRaw = Math.max(yLow, yHigh);
        const top = clamp(topRaw, 0, hostHeight);
        const bottom = clamp(bottomRaw, 0, hostHeight);
        const height = Math.max(2, bottom - top);
        node.style.left = `${offsetLeft}px`;
        node.style.width = `${hostWidth}px`;
        node.style.top = `${offsetTop + top}px`;
        node.style.height = `${height}px`;
        node.textContent = "";
        node.classList.add("show");
      }
      function rebuildDetailChartDecorations(ev, candles) {
        const runtime = state.detailTv;
        if (!runtime || !ev || !candles.length) return;
        const lc = window.LightweightCharts || {};
        const dashed = lc.LineStyle?.Dashed ?? 2;
        const dotted = lc.LineStyle?.Dotted ?? 1;
        clearDetailPriceLines(runtime);
        const low = Number(ev.pred_low);
        const high = Number(ev.pred_high);
        const mid = Number(ev.pred_mid);
        const closePx = Number(ev.actual_price ?? ev.current_price);
        if (Number.isFinite(low)) {
          addDetailPriceLine(runtime, { price: low, color: "rgba(0,255,102,.36)", lineWidth: 1, lineStyle: dotted, axisLabelVisible: false, title: "model low" });
        }
        if (Number.isFinite(high)) {
          addDetailPriceLine(runtime, { price: high, color: "rgba(0,255,102,.36)", lineWidth: 1, lineStyle: dotted, axisLabelVisible: false, title: "model high" });
        }
        if (Number.isFinite(mid)) {
          addDetailPriceLine(runtime, { price: mid, color: "rgba(102,255,170,.76)", lineWidth: 1, lineStyle: dashed, axisLabelVisible: false, title: "median" });
        }
        if (ev.status !== "active" && Number.isFinite(closePx)) {
          addDetailPriceLine(runtime, { price: closePx, color: "rgba(102,255,170,.92)", lineWidth: 1, lineStyle: lc.LineStyle?.Solid ?? 0, axisLabelVisible: false, title: "close" });
        }
        runtime.candleSeries.setMarkers(buildDetailMarkers(ev, candles));
        updateDetailRangeOverlay(ev, runtime);
      }

      function showBanner(message) {
        const node = el("globalBanner");
        if (!message) { node.classList.remove("show"); node.textContent = ""; return; }
        node.classList.add("show");
        node.textContent = message;
      }
      function showToast(message) {
        const node = document.createElement("div");
        node.className = "toast";
        node.textContent = message;
        el("toastWrap").appendChild(node);
        setTimeout(() => node.remove(), 3000);
      }
      function setConnection(ok, text) { el("connDot").classList.toggle("bad", !ok); el("connText").textContent = text; }
      function normalizeBaseUrl(value) {
        const raw = String(value ?? "").trim();
        if (!raw) return defaultSettings.apiBase;
        if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
        return `http://${raw}`;
      }
      function loadSettings() {
        try {
          const raw = localStorage.getItem("ulyanai_gui_settings");
          if (!raw) return;
          const parsed = JSON.parse(raw);
          state.settings = { ...defaultSettings, ...parsed, apiBase: normalizeBaseUrl(parsed.apiBase), refreshSec: Math.max(3, Number(parsed.refreshSec || 5)), autoRefresh: parsed.autoRefresh === "off" ? "off" : "on" };
        } catch { state.settings = { ...defaultSettings }; }
      }
      function saveSettings() { localStorage.setItem("ulyanai_gui_settings", JSON.stringify(state.settings)); }
      function renderSettings() { el("settingsApiBase").value = state.settings.apiBase; el("settingsApiHeader").value = state.settings.apiHeader; el("settingsApiKey").value = state.settings.apiKey; el("settingsRefreshSec").value = String(state.settings.refreshSec); el("settingsAutoRefresh").value = state.settings.autoRefresh; }
      function readSettings() {
        state.settings.apiBase = normalizeBaseUrl(el("settingsApiBase").value);
        state.settings.apiHeader = (el("settingsApiHeader").value || "").trim() || "X-API-Key";
        state.settings.apiKey = (el("settingsApiKey").value || "").trim();
        state.settings.refreshSec = Math.max(3, Number(el("settingsRefreshSec").value || "5"));
        state.settings.autoRefresh = el("settingsAutoRefresh").value === "off" ? "off" : "on";
      }
      function normalizeEventsPageSize(value) {
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return 10;
        return Math.max(1, Math.min(500, Math.floor(parsed)));
      }
      function renderEventPageSize() {
        const input = el("eventsPageSizeInput");
        if (!input) return;
        const current = normalizeEventsPageSize(state.events.page_size || 10);
        state.events.page_size = current;
        input.value = String(current);
      }
      function headers() { const out = {}; const key = String(state.settings.apiKey || "").trim(); const head = String(state.settings.apiHeader || "").trim(); if (key && head) out[head] = key; return out; }
      async function api(path, { method = "GET", query = null, body = null, text = false } = {}) {
        const url = new URL(path, normalizeBaseUrl(state.settings.apiBase));
        if (query) Object.entries(query).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v); });
        const hs = headers();
        if (body !== null) hs["Content-Type"] = "application/json";
        const res = await fetch(url.toString(), { method, headers: hs, body: body === null ? undefined : JSON.stringify(body) });
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
        return text ? res.text() : res.json();
      }
      async function apiDownload(path, { query = null, fallbackName = "download.bin" } = {}) {
        const url = new URL(path, normalizeBaseUrl(state.settings.apiBase));
        if (query) Object.entries(query).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v); });
        const res = await fetch(url.toString(), { method: "GET", headers: headers() });
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
        const blob = await res.blob();
        let fileName = fallbackName;
        const disposition = String(res.headers.get("content-disposition") || "");
        const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
        const plainMatch = disposition.match(/filename=\"?([^\";]+)\"?/i);
        const rawName = utf8Match?.[1] || plainMatch?.[1];
        if (rawName) {
          try { fileName = decodeURIComponent(rawName); } catch { fileName = rawName; }
        }
        const href = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = href;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(href), 1200);
      }
      function statusTag(status) {
        if (status === "completed") return `<span class="tag completed">completed</span>`;
        if (status === "cancelled") return `<span class="tag cancelled">cancelled</span>`;
        return `<span class="tag active">active</span>`;
      }
      function resultTag(hit) { if (hit === true) return `<span class="tag hit">hit</span>`; if (hit === false) return `<span class="tag miss">miss</span>`; return "-"; }
      function timerText(item, now = Date.now()) {
        if (!item?.expires_at) return "-";
        if (item.status === "cancelled") return "отменен";
        if (item.status === "completed") return "завершен";
        const left = Math.floor((Date.parse(item.expires_at) - now) / 1000);
        if (left <= 0) return "ожидаем фиксацию";
        const d = Math.floor(left / 86400), h = Math.floor((left % 86400) / 3600), m = Math.floor((left % 3600) / 60), s = left % 60;
        if (d > 0) return `${d}d ${h}h ${m}m`;
        if (h > 0) return `${h}h ${m}m ${s}s`;
        return `${m}m ${s}s`;
      }
      function currentValue(item, now = Date.now()) {
        if (Date.parse(item.expires_at) > now) return "-";
        if (item.actual_price !== null && item.actual_price !== undefined) return price(item.actual_price);
        if (item.current_price !== null && item.current_price !== undefined) return price(item.current_price);
        return "-";
      }
      function detailLastTickSample() {
        if (state.detailLiveSamples.length) return state.detailLiveSamples[state.detailLiveSamples.length - 1];
        if (state.detailMarketSamples.length) return state.detailMarketSamples[state.detailMarketSamples.length - 1];
        if (state.selectedSamples.length) return state.selectedSamples[state.selectedSamples.length - 1];
        return null;
      }
      function detailLastTickText() {
        const sample = detailLastTickSample();
        if (!sample) return "нет данных";
        return `${price(sample.price)} @ ${dt(sample.ts)}`;
      }
      function pctWithUsd(pctValue, usdValue) {
        const pctText = pctRaw(pctValue);
        if (pctText === "-") return "-";
        const usd = Number(usdValue);
        if (!Number.isFinite(usd)) return pctText;
        return `${pctText} (${formatUsd(Math.abs(usd))})`;
      }
      function refreshDetailLiveBlock(ev = state.selectedEvent) {
        if (!ev || !state.selectedEvent || state.selectedEvent.event_id !== ev.event_id) return;
        setNodeText("detailTimerValue", timerText(ev, Date.now()));
        setNodeText("detailLastTickValue", detailLastTickText());
      }
      function collectEventFilters() { return { status: el("eventsStatusFilter").value, horizon: el("eventsHorizonFilter").value, cycle_id: el("eventsCycleFilter").value.trim(), result: el("eventsResultFilter").value, q: el("eventsSearchInput").value.trim() }; }
      function renderEventsTable() {
        const body = el("eventsTableBody");
        const items = state.events.items || [];
        const now = Date.now();
        const page = Number(state.events.page || 1);
        const allowNewPulse = state.eventsLoadedOnce && page === 1;
        if (!items.length) body.innerHTML = `<tr><td colspan="13" class="mut">События не найдены.</td></tr>`;
        else body.innerHTML = items.map((item) => {
          const m = item.metrics || {};
          const eventId = String(item.event_id || "");
          const isNew = allowNewPulse && eventId && !state.seenEventIds.has(eventId);
          return `<tr class="row-clickable${isNew ? " row-new" : ""}" data-event-id="${esc(eventId)}">
            <td>${statusTag(item.status)}</td><td>${esc(cycleText(item))}</td><td>${esc(dt(item.created_at))}</td><td>${esc(item.horizon)}</td><td>${esc(price(item.price_t0))}</td>
            <td>[${esc(price(item.pred_low))}; ${esc(price(item.pred_high))}]</td>
            <td><span class="timer" data-event-id="${esc(eventId)}">${esc(timerText(item, now))}</span></td>
            <td>${esc(currentValue(item, now))}</td><td>${esc(item.model_id || "-")}</td><td>${resultTag(m.hit)}</td>
            <td>${esc(price(m.abs_error))}</td><td>${esc(pctRaw(m.rel_error))}</td><td>${esc(pctRaw(m.width_pct))}</td></tr>`;
        }).join("");
        items.forEach((item) => { if (item?.event_id) state.seenEventIds.add(String(item.event_id)); });
        state.eventsLoadedOnce = true;
        const total = Number(state.events.total || 0), pageSize = Number(state.events.page_size || 10), pages = Math.max(1, Math.ceil(total / pageSize));
        setNodeText("eventsMeta", `Загружено ${items.length} из ${total}`);
        setNodeText("eventsPageInfo", `Страница ${page}/${pages}, всего ${total}`);
        el("eventsPrevBtn").disabled = page <= 1;
        el("eventsNextBtn").disabled = page >= pages;
        triggerScanSweep(body.closest(".panel"));
      }
      function updateTimers() {
        updateMarketTimers();
        if (state.selectedEvent) return;
        if (state.activeTab !== "events") return;
        const map = new Map((state.events.items || []).map((it) => [it.event_id, it]));
        document.querySelectorAll(".timer[data-event-id]").forEach((node) => {
          const id = node.getAttribute("data-event-id");
          const item = map.get(id);
          if (!item) return;
          node.textContent = timerText(item, Date.now());
          const row = node.closest("tr");
          if (row && row.children[7]) row.children[7].textContent = currentValue(item, Date.now());
        });
      }
      async function loadEvents(pageOverride = null) {
        const pageSize = normalizeEventsPageSize(state.events.page_size || 10);
        state.events.page_size = pageSize;
        const query = { ...collectEventFilters(), page: pageOverride ?? state.events.page ?? 1, page_size: pageSize, sort_by: "created_at", sort_dir: "desc" };
        state.events = await api("/api/events", { query });
        renderEventPageSize();
        renderEventsTable();
      }
      function parseClockMinutes(value) {
        const match = String(value || "").trim().match(/^(\d{1,2})(?::(\d{2}))?([AP]M)$/i);
        if (!match) return NaN;
        let hour = Number(match[1]);
        const minute = Number(match[2] || 0);
        const ap = match[3].toUpperCase();
        if (!Number.isFinite(hour) || !Number.isFinite(minute)) return NaN;
        if (hour === 12) hour = 0;
        if (ap === "PM") hour += 12;
        return (hour * 60) + minute;
      }
      function detectBtcUpDownTf(title) {
        const raw = String(title || "").trim();
        if (!raw || !/^bitcoin up or down/i.test(raw)) return null;
        if (/^bitcoin up or down this week\?$/i.test(raw)) return "1w";
        if (/^bitcoin up or down on .+\?$/i.test(raw)) return "1d";
        const range = raw.match(/ - .+?,\s*(\d{1,2}(?::\d{2})?[AP]M)-(\d{1,2}(?::\d{2})?[AP]M) ET$/i);
        if (range) {
          const start = parseClockMinutes(range[1]);
          const end = parseClockMinutes(range[2]);
          if (Number.isFinite(start) && Number.isFinite(end)) {
            const duration = (end - start + (24 * 60)) % (24 * 60);
            if (duration === 5) return "5m";
            if (duration === 15) return "15m";
            if (duration === 240) return "4h";
          }
          return null;
        }
        if (/ - .+?,\s*\d{1,2}(?::\d{2})?[AP]M ET$/i.test(raw)) return "1h";
        return null;
      }
      function todayMonthDayKeyEt() {
        const parts = new Intl.DateTimeFormat("en-US", {
          timeZone: "America/New_York",
          month: "long",
          day: "numeric",
        }).formatToParts(new Date());
        const month = parts.find((part) => part.type === "month")?.value || "";
        const day = parts.find((part) => part.type === "day")?.value || "";
        return `${month} ${day}`.trim().toLowerCase();
      }
      function eventMonthDayKey(title) {
        const raw = String(title || "").trim();
        if (!raw) return null;
        let match = raw.match(/ - ([A-Za-z]+ \d{1,2}), /i);
        if (match) return String(match[1]).trim().toLowerCase();
        match = raw.match(/ on ([A-Za-z]+ \d{1,2})\?/i);
        if (match) return String(match[1]).trim().toLowerCase();
        return null;
      }
      function marketExpiresInMsText(endMs, nowMs = Date.now()) {
        if (!Number.isFinite(endMs)) return "-";
        const leftSec = Math.floor((endMs - nowMs) / 1000);
        if (leftSec <= 0) return "expired";
        const d = Math.floor(leftSec / 86400);
        const h = Math.floor((leftSec % 86400) / 3600);
        const m = Math.floor((leftSec % 3600) / 60);
        const s = leftSec % 60;
        if (d > 0) return `${d}d ${h}h ${m}m`;
        if (h > 0) return `${h}h ${m}m ${s}s`;
        return `${m}m ${s}s`;
      }
      function marketExpiresInText(endDate, nowMs = Date.now()) {
        const endMs = Date.parse(String(endDate || ""));
        return marketExpiresInMsText(endMs, nowMs);
      }
      function hasAllRequiredMarketTfs(items) {
        const required = new Set(["5m", "15m", "1h", "4h", "1d"]);
        (items || []).forEach((item) => {
          const tf = String(item?.market_tf || "");
          if (required.has(tf)) required.delete(tf);
        });
        return required.size === 0;
      }
      function updateMarketTimers() {
        if (state.activeTab !== "markets") return;
        const now = Date.now();
        let hasExpired = false;
        document.querySelectorAll(".market-expiry[data-end-ms]").forEach((node) => {
          const endMs = Number(node.getAttribute("data-end-ms"));
          if (!Number.isFinite(endMs)) {
            node.textContent = "-";
            return;
          }
          if (endMs <= now) hasExpired = true;
          node.textContent = marketExpiresInMsText(endMs, now);
        });
        if (!hasExpired) return;
        if (state.marketsLoadInFlight) return;
        if ((now - Number(state.marketsLastLoadedMs || 0)) < 1500) return;
        loadMarkets().catch(onError);
      }
      function pickTargetBtcEvents(items) {
        const tfOrder = ["5m", "15m", "1h", "4h", "1d", "1w"];
        const order = new Map(tfOrder.map((tf, idx) => [tf, idx]));
        const todayKey = todayMonthDayKeyEt();
        const candidates = [];
        (items || []).forEach((item) => {
          const tf = detectBtcUpDownTf(item?.title);
          if (!tf) return;
          const eventKey = eventMonthDayKey(item?.title);
          if (!eventKey || eventKey !== todayKey) return;
          candidates.push({ ...item, market_tf: tf });
        });
        const byTf = new Map();
        candidates.forEach((item) => {
          const tf = String(item.market_tf || "");
          if (!byTf.has(tf)) byTf.set(tf, []);
          byTf.get(tf).push(item);
        });
        const nowMs = Date.now();
        const out = [];
        tfOrder.forEach((tf) => {
          const rows = byTf.get(tf) || [];
          if (!rows.length) return;
          rows.sort((a, b) => {
            const ta = Date.parse(a.end_date || "");
            const tb = Date.parse(b.end_date || "");
            const aValid = Number.isFinite(ta);
            const bValid = Number.isFinite(tb);
            if (aValid !== bValid) return aValid ? -1 : 1;
            if (!aValid && !bValid) return String(a.title || "").localeCompare(String(b.title || ""));
            const aFuture = ta >= nowMs;
            const bFuture = tb >= nowMs;
            if (aFuture !== bFuture) return aFuture ? -1 : 1;
            if (aFuture && bFuture) return ta - tb;
            return tb - ta;
          });
          out.push(rows[0]);
        });
        out.sort((a, b) => (order.get(a.market_tf) ?? 99) - (order.get(b.market_tf) ?? 99));
        return out;
      }
      function marketStatusTag(item) {
        if (item?.closed === true) return `<span class="tag completed">closed</span>`;
        if (item?.active === true) return `<span class="tag active">active</span>`;
        if (item?.active === false) return `<span class="tag cancelled">inactive</span>`;
        return "-";
      }
      function marketUsd(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return "-";
        return `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      }
      function toOptionalNumber(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
      }
      function parseMaybeJsonArray(value) {
        if (Array.isArray(value)) return value;
        if (typeof value !== "string") return [];
        const raw = value.trim();
        if (!raw) return [];
        try {
          const parsed = JSON.parse(raw);
          return Array.isArray(parsed) ? parsed : [];
        } catch {
          return [];
        }
      }
      function parseTokenIds(value) {
        if (Array.isArray(value)) return value.map((item) => String(item ?? "")).filter(Boolean);
        if (typeof value === "string") {
          const fromJson = parseMaybeJsonArray(value).map((item) => String(item ?? "")).filter(Boolean);
          if (fromJson.length) return fromJson;
          const raw = value.trim();
          return raw ? [raw] : [];
        }
        return [];
      }
      function normalizePolymarketMarketRaw(market) {
        const outcomes = parseMaybeJsonArray(market?.outcomes).map((item) => String(item ?? "")).filter(Boolean);
        const outcomePrices = parseMaybeJsonArray(market?.outcomePrices).map((item) => toOptionalNumber(item)).filter((item) => item !== null);
        const tokenIds = parseTokenIds(market?.clobTokenIds);
        return {
          market_id: market?.id !== undefined && market?.id !== null ? String(market.id) : null,
          condition_id: market?.conditionId !== undefined && market?.conditionId !== null ? String(market.conditionId) : null,
          slug: market?.slug !== undefined && market?.slug !== null ? String(market.slug) : null,
          question: market?.question !== undefined && market?.question !== null ? String(market.question) : null,
          active: typeof market?.active === "boolean" ? market.active : null,
          closed: typeof market?.closed === "boolean" ? market.closed : null,
          end_date: market?.endDate !== undefined && market?.endDate !== null ? String(market.endDate) : null,
          volume: toOptionalNumber(market?.volumeNum ?? market?.volume),
          volume_clob: toOptionalNumber(market?.volumeClob),
          best_ask: toOptionalNumber(market?.bestAsk),
          last_trade_price: toOptionalNumber(market?.lastTradePrice),
          outcomes,
          outcome_prices: outcomePrices,
          yes_token_id: tokenIds[0] || null,
          no_token_id: tokenIds[1] || null,
        };
      }
      function normalizePolymarketEventRaw(event) {
        const rawMarkets = Array.isArray(event?.markets) ? event.markets : [];
        const topMarket = rawMarkets.find((item) => item && typeof item === "object");
        const rawTags = Array.isArray(event?.tags) ? event.tags : [];
        const tags = rawTags.map((tag) => {
          if (!tag || typeof tag !== "object") return "";
          return String(tag.label ?? tag.slug ?? "").trim();
        }).filter(Boolean);
        return {
          event_id: event?.id !== undefined && event?.id !== null ? String(event.id) : "",
          slug: event?.slug !== undefined && event?.slug !== null ? String(event.slug) : null,
          title: event?.title !== undefined && event?.title !== null ? String(event.title) : null,
          description: event?.description !== undefined && event?.description !== null ? String(event.description) : null,
          active: typeof event?.active === "boolean" ? event.active : null,
          closed: typeof event?.closed === "boolean" ? event.closed : null,
          start_date: event?.startDate !== undefined && event?.startDate !== null ? String(event.startDate) : null,
          end_date: event?.endDate !== undefined && event?.endDate !== null ? String(event.endDate) : null,
          volume: toOptionalNumber(event?.volume),
          volume_24hr: toOptionalNumber(event?.volume24hr),
          liquidity: toOptionalNumber(event?.liquidityClob ?? event?.liquidity),
          open_interest: toOptionalNumber(event?.openInterest),
          icon: event?.icon !== undefined && event?.icon !== null ? String(event.icon) : null,
          markets_count: rawMarkets.length,
          markets: topMarket ? [normalizePolymarketMarketRaw(topMarket)] : [],
          tags,
        };
      }
      async function fetchPolymarketCryptoDirect({ active = true, closed = false } = {}) {
        const limit = 200;
        const maxPages = 3;
        const items = [];
        const seenIds = new Set();
        for (let page = 0; page < maxPages; page += 1) {
          const offset = page * limit;
          const url = new URL("https://gamma-api.polymarket.com/events");
          url.searchParams.set("tag_slug", "crypto");
          url.searchParams.set("order", "endDate");
          url.searchParams.set("limit", String(limit));
          url.searchParams.set("offset", String(offset));
          if (active !== null && active !== undefined) url.searchParams.set("active", String(Boolean(active)));
          if (closed !== null && closed !== undefined) url.searchParams.set("closed", String(Boolean(closed)));
          const response = await fetch(url.toString(), { cache: "no-store" });
          if (!response.ok) throw new Error(`Gamma HTTP ${response.status}`);
          const payload = await response.json();
          if (!Array.isArray(payload)) throw new Error("Gamma invalid payload");
          if (!payload.length) break;
          for (const row of payload) {
            if (!row || typeof row !== "object") continue;
            const eventId = row.id !== undefined && row.id !== null ? String(row.id) : "";
            if (eventId && seenIds.has(eventId)) continue;
            if (eventId) seenIds.add(eventId);
            items.push(normalizePolymarketEventRaw(row));
          }
          if (hasAllRequiredMarketTfs(pickTargetBtcEvents(items))) break;
          if (payload.length < limit) break;
        }
        return {
          source: "polymarket_gamma_direct",
          tag_slug: "crypto",
          active: active ?? null,
          closed: closed ?? null,
          total: items.length,
          fetched_at: new Date().toISOString(),
          items,
        };
      }
      function renderMarketsTable() {
        const body = el("marketsTableBody");
        if (!body) return;
        const items = state.markets.items || [];
        if (!items.length) {
          body.innerHTML = `<tr><td colspan="11" class="mut">No matching BTC Up/Down events found.</td></tr>`;
        } else {
          body.innerHTML = items.map((item) => {
            const top = (Array.isArray(item.markets) && item.markets.length) ? item.markets[0] : null;
            const href = item.slug ? `https://polymarket.com/event/${encodeURIComponent(item.slug)}` : "";
            const endMs = Date.parse(item.end_date || "");
            return `<tr>
              <td>${marketStatusTag(item)}</td>
              <td><span class="tag active">${esc(item.market_tf || "-")}</span></td>
              <td><div class="markets-title">${esc(item.title || "-")}</div><div class="mut">${esc(item.slug || "-")}</div></td>
              <td><div>${esc(top?.question || "-")}</div><div class="mut">${esc(top?.slug || "-")}</div></td>
              <td>${esc(dt(item.end_date))}</td>
              <td><span class="market-expiry" data-end-ms="${Number.isFinite(endMs) ? esc(String(endMs)) : ""}">${esc(marketExpiresInText(item.end_date))}</span></td>
              <td>${esc(marketUsd(item.volume))}</td>
              <td>${esc(marketUsd(item.liquidity))}</td>
              <td>${esc(marketUsd(item.open_interest))}</td>
              <td>${esc(num(item.markets_count, 0))}</td>
              <td>${href ? `<a class="markets-link" href="${esc(href)}" target="_blank" rel="noreferrer">open</a>` : "-"}</td>
            </tr>`;
          }).join("");
        }
        updateMarketTimers();
        const shown = Number(items.length);
        const counts = { "5m": 0, "15m": 0, "1h": 0, "4h": 0, "1d": 0, "1w": 0 };
        items.forEach((item) => {
          const tf = String(item?.market_tf || "");
          if (tf in counts) counts[tf] += 1;
        });
        const byTf = `5m:${counts["5m"]} | 15m:${counts["15m"]} | 1h:${counts["1h"]} | 4h:${counts["4h"]} | 1d:${counts["1d"]} | 1w:${counts["1w"]}`;
        const fetchedAt = state.markets.fetched_at ? dt(state.markets.fetched_at) : "-";
        setNodeText("marketsMeta", `BTC Up/Down today (ET), nearest expiry per TF: ${shown} events (${byTf}). fetched_at: ${fetchedAt}`);
        triggerScanSweep(body.closest(".panel"));
      }
      async function loadMarkets() {
        if (state.marketsLoadInFlight) return;
        state.marketsLoadInFlight = true;
        try {
          const query = { active: true, closed: false };
          let payload;
          try {
            payload = await fetchPolymarketCryptoDirect(query);
          } catch (error) {
            payload = await api("/api/markets/polymarket/crypto", { query });
          }
          const selectedItems = pickTargetBtcEvents(payload.items || []);
          state.markets = {
            ...state.markets,
            items: selectedItems,
            total: Number(selectedItems.length),
            fetched_at: payload.fetched_at || null,
            active: payload.active,
            closed: payload.closed,
          };
          state.marketsLastLoadedMs = Date.now();
          renderMarketsTable();
        } finally {
          state.marketsLoadInFlight = false;
        }
      }
      async function loadStatus() { state.statusInfo = await api("/v1/status"); setConnection(true, "Connected"); return state.statusInfo; }
      async function loadQuoteFromBinance() {
        const asset = "BTC";
        const symbol = asset === "BTC" ? "BTCUSDT" : `${asset}USDT`;
        const response = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${encodeURIComponent(symbol)}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`Binance HTTP ${response.status}`);
        const payload = await response.json();
        const spot = Number(payload.price);
        if (!Number.isFinite(spot)) throw new Error("Binance invalid price");
        const nowIso = new Date().toISOString();
        state.quote = { asset, price_spot: spot, as_of: nowIso, source: "binance_spot" };
        setNodeText("headerQuotePrice", price(spot), { flash: true });
        setNodeText("headerQuoteLive", "LIVE", { flash: true });
        el("headerQuoteLive")?.classList.remove("fallback");
        triggerScanSweep(el("headerQuotePrice")?.closest(".app-header"));
      }
      async function loadQuoteFromApiFallback() {
        const payload = await api("/v1/predict", { query: { asset: "BTC", horizon: "5m" } });
        state.quote = payload;
        setNodeText("headerQuotePrice", price(payload.price_spot), { flash: true });
        setNodeText("headerQuoteLive", payload.stale_data ? "LIVE*" : "LIVE", { flash: true });
        el("headerQuoteLive")?.classList.toggle("fallback", Boolean(payload.stale_data));
        triggerScanSweep(el("headerQuotePrice")?.closest(".app-header"));
      }
      async function loadQuote({ allowFallback = true } = {}) {
        try {
          await loadQuoteFromBinance();
        } catch (error) {
          if (!allowFallback) throw error;
          await loadQuoteFromApiFallback();
        }
      }
      async function loadModels() {
        const [all, prod] = await Promise.all([api("/api/models", { query: { asset: "BTC" } }), api("/api/models/production", { query: { asset: "BTC" } })]);
        state.models = all.items || [];
        state.productionModels = prod.items || [];
        renderCreateModelOptions();
        renderCompareModelOptions();
      }
      function orderedCreateModels() {
        const horizon = el("createHorizonSelect")?.value || "";
        const prodSet = new Set((state.productionModels || []).map((m) => m.model_id));
        const production = (state.productionModels || []).filter((m) => !horizon || m.horizon === horizon);
        const others = (state.models || []).filter((m) => (!horizon || m.horizon === horizon) && !prodSet.has(m.model_id));
        return [...production, ...others];
      }
      function selectedCreateTournamentModelIds() {
        const select = el("createModelMultiSelect");
        if (!select) return [];
        return Array.from(select.selectedOptions || []).map((option) => option.value).filter(Boolean);
      }
      function syncCreateModeUi() {
        const mode = el("createModeSelect")?.value || "single";
        const isTournament = mode === "tournament";
        const isSingle = mode === "single";
        const runsField = el("createRunsField");
        const singleField = el("createModelSingleField");
        const multiField = el("createModelMultiField");
        if (runsField) runsField.style.display = isSingle ? "none" : "grid";
        if (singleField) singleField.style.display = isTournament ? "none" : "grid";
        if (multiField) multiField.style.display = isTournament ? "grid" : "none";
        if (el("createRunsInput")) {
          el("createRunsInput").disabled = isSingle;
          if (isSingle) el("createRunsInput").value = "1";
        }
        if (el("createModelSelect")) el("createModelSelect").disabled = isTournament;
        if (el("createModelMultiSelect")) el("createModelMultiSelect").disabled = !isTournament;
        const runs = Math.max(1, Number(el("createRunsInput")?.value || "1"));
        const hint = el("createPlanHint");
        if (!hint) return;
        if (isTournament) {
          const modelCount = selectedCreateTournamentModelIds().length;
          const totalEvents = modelCount * runs;
          hint.textContent = modelCount >= 2
            ? `Tournament: ${modelCount} models x ${runs} runs = ${totalEvents} events`
            : "Tournament requires at least 2 models for the selected horizon.";
          return;
        }
        if (mode === "cycle") {
          hint.textContent = `Cycle: ${runs} sequential run${runs === 1 ? "" : "s"} on one model.`;
          return;
        }
        hint.textContent = "Single mode creates exactly one event.";
      }
      function renderCreateModelOptions() {
        const single = el("createModelSelect");
        const multi = el("createModelMultiSelect");
        if (!single || !multi) return;
        const previousSingle = single.value;
        const previousMulti = new Set(selectedCreateTournamentModelIds());
        const ordered = orderedCreateModels();
        const prodIds = new Set((state.productionModels || []).map((m) => m.model_id));
        single.innerHTML = ordered.length
          ? [`<option value="">auto (production)</option>`, ...ordered.map((m) => `<option value="${esc(m.model_id)}">${esc(m.model_id)}${prodIds.has(m.model_id) ? " [prod]" : ""}</option>`)].join("")
          : `<option value="">auto (production)</option>`;
        if (previousSingle && Array.from(single.options).some((option) => option.value === previousSingle)) {
          single.value = previousSingle;
        }
        multi.innerHTML = ordered
          .map((m) => `<option value="${esc(m.model_id)}">${esc(m.model_id)}${prodIds.has(m.model_id) ? " [prod]" : ""}</option>`)
          .join("");
        Array.from(multi.options).forEach((option) => {
          option.selected = previousMulti.has(option.value);
        });
        syncCreateModeUi();
      }
      function orderedCompareModels() {
        const horizon = el("analysisHorizonFilter")?.value || "";
        const prodSet = new Set((state.productionModels || []).map((m) => m.model_id));
        const production = (state.productionModels || []).filter((m) => !horizon || m.horizon === horizon);
        const others = (state.models || []).filter((m) => (!horizon || m.horizon === horizon) && !prodSet.has(m.model_id));
        return [...production, ...others];
      }
      function selectedCompareModelIds() {
        const select = el("compareModelsSelect");
        if (!select) return [];
        return Array.from(select.selectedOptions || []).map((option) => option.value).filter(Boolean);
      }
      function setCompareModelSelection(modelIds, { append = true } = {}) {
        const select = el("compareModelsSelect");
        if (!select) return [];
        const wanted = new Set(append ? selectedCompareModelIds() : []);
        (modelIds || []).forEach((modelId) => {
          const value = String(modelId || "").trim();
          if (value) wanted.add(value);
        });
        renderCompareModelOptions();
        Array.from(select.options || []).forEach((option) => {
          option.selected = wanted.has(option.value);
        });
        const baseline = el("compareBaselineSelect");
        if (baseline && baseline.value && !wanted.has(baseline.value)) {
          baseline.value = "";
        }
        return Array.from(wanted);
      }
      function renderCompareModelOptions(preferredModelId = "") {
        const select = el("compareModelsSelect");
        const baseline = el("compareBaselineSelect");
        if (!select) return;
        const previous = new Set(selectedCompareModelIds());
        if (preferredModelId) previous.add(preferredModelId);
        const search = String(el("compareModelSearch")?.value || "").trim().toLowerCase();
        const ordered = orderedCompareModels().filter((m) => !search || String(m.model_id || "").toLowerCase().includes(search));
        const prodIds = new Set((state.productionModels || []).map((m) => m.model_id));
        select.innerHTML = ordered
          .map((m) => `<option value="${esc(m.model_id)}">${esc(m.model_id)}${prodIds.has(m.model_id) ? " [prod]" : ""}</option>`)
          .join("");
        Array.from(select.options || []).forEach((option) => {
          option.selected = previous.has(option.value);
        });
        if (baseline) {
          const previousBaseline = baseline.value;
          baseline.innerHTML = [`<option value="">None</option>`, ...ordered.map((m) => `<option value="${esc(m.model_id)}">${esc(m.model_id)}</option>`)].join("");
          if (previousBaseline && Array.from(baseline.options).some((option) => option.value === previousBaseline)) {
            baseline.value = previousBaseline;
          }
        }
      }
      function selectedCompareColumns() {
        const columnAliases = {
          q_count: "quantiles_count",
        };
        return Array.from(document.querySelectorAll("#compareColumnPicker input[data-col]:checked"))
          .map((node) => columnAliases[node.getAttribute("data-col")] || node.getAttribute("data-col"))
          .filter(Boolean);
      }
      function compareQueryBase() {
        return {
          days: Number(el("analysisDaysFilter")?.value || "30"),
          asset: "BTC",
          horizon: el("analysisHorizonFilter")?.value || "",
          cycle_id: el("analysisCycleFilter")?.value.trim() || "",
          mode: el("compareModeSelect")?.value || "simple",
          model_ids: selectedCompareModelIds().join(","),
          exclude_stale: el("compareExcludeStale")?.checked ? "true" : "",
        };
      }
      function applyTournamentComparePreset() {
        if (el("compareModeSelect")) el("compareModeSelect").value = "matched";
        if (el("compareExcludeStale")) el("compareExcludeStale").checked = true;
        if (el("analysisCycleFilter") && !el("analysisCycleFilter").value.trim()) {
          el("analysisCycleFilter").value = el("eventsCycleFilter")?.value.trim() || el("deepCycleFilter")?.value.trim() || "";
        }
        loadAnalysis().catch(onError);
      }
      function showBuilderBanner(message, isError = false) {
        const node = el("builderBanner");
        if (!node) return;
        node.textContent = String(message || "");
        node.classList.toggle("show", Boolean(message));
        node.style.borderColor = isError ? "rgba(255,93,127,.55)" : "rgba(0,255,102,.34)";
        node.style.background = isError ? "rgba(255,93,127,.1)" : "rgba(0,255,102,.08)";
        node.style.color = isError ? "#ffe3ea" : "#e8fff2";
      }
      function asNumber(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
      }
      function parseCsvStrings(value) {
        const seen = new Set();
        const out = [];
        String(value || "").split(/[\r\n,;]+/).forEach((item) => {
          const text = String(item || "").trim();
          if (!text || seen.has(text)) return;
          seen.add(text);
          out.push(text);
        });
        return out;
      }
      function parseCsvInts(value, { minimum = 1, maximum = 99999 } = {}) {
        return String(value || "")
          .split(",")
          .map((item) => Number(item.trim()))
          .filter((item) => Number.isInteger(item) && item >= minimum && item <= maximum);
      }
      function parseSweepIntSpec(value, { minimum = 1, maximum = 99999 } = {}) {
        const seen = new Set();
        const out = [];
        const push = (raw) => {
          const item = Number(raw);
          if (!Number.isInteger(item) || item < minimum || item > maximum || seen.has(item)) return;
          seen.add(item);
          out.push(item);
        };
        String(value || "").split(/[\r\n,;]+/).forEach((token) => {
          const text = String(token || "").trim();
          if (!text) return;
          const rangeMatch = text.match(/^(\d+)\s*\.\.\s*(\d+)(?:\s+step\s*=\s*(\d+))?$/i);
          if (rangeMatch) {
            const start = Number(rangeMatch[1]);
            const end = Number(rangeMatch[2]);
            const rawStep = Number(rangeMatch[3] || "1");
            const step = Math.max(1, Math.abs(rawStep));
            if (!Number.isInteger(start) || !Number.isInteger(end)) return;
            if (start <= end) {
              for (let item = start; item <= end; item += step) push(item);
            } else {
              for (let item = start; item >= end; item -= step) push(item);
            }
            return;
          }
          push(text);
        });
        return out;
      }
      function parseSweepNumberSpec(value, { minimum = Number.NEGATIVE_INFINITY, maximum = Number.POSITIVE_INFINITY, integerOnly = false } = {}) {
        const seen = new Set();
        const out = [];
        const normalizeNumber = (raw) => {
          const numeric = Number(raw);
          if (!Number.isFinite(numeric)) return null;
          if (integerOnly && !Number.isInteger(numeric)) return null;
          const bounded = integerOnly ? Math.trunc(numeric) : Number(numeric.toFixed(10));
          if (bounded < minimum || bounded > maximum) return null;
          return bounded;
        };
        const push = (raw) => {
          const normalized = normalizeNumber(raw);
          if (normalized === null) return;
          const signature = integerOnly ? String(normalized) : normalized.toFixed(10);
          if (seen.has(signature)) return;
          seen.add(signature);
          out.push(normalized);
        };
        String(value || "").split(/[\r\n,;]+/).forEach((token) => {
          const text = String(token || "").trim();
          if (!text) return;
          const rangeMatch = text.match(/^(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)(?:\s+step\s*=\s*(-?\d+(?:\.\d+)?))?$/i);
          if (!rangeMatch) {
            push(text);
            return;
          }
          const start = normalizeNumber(rangeMatch[1]);
          const end = normalizeNumber(rangeMatch[2]);
          const rawStep = normalizeNumber(rangeMatch[3] || (integerOnly ? "1" : "0.01"));
          if (start === null || end === null || rawStep === null) return;
          const step = Math.abs(Number(rawStep));
          if (!(step > 0)) return;
          const direction = start <= end ? 1 : -1;
          const guard = 10000;
          let iterations = 0;
          for (let current = start; direction > 0 ? current <= end + 1e-9 : current >= end - 1e-9; current += direction * step) {
            push(current);
            iterations += 1;
            if (iterations >= guard) break;
          }
        });
        return out;
      }
      function parseSweepHyperparamsSpec(value) {
        const specs = {};
        const integerKeys = new Set(["num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq", "num_boost_round", "early_stopping_rounds", "seed"]);
        const constrainedKeys = {
          feature_fraction: { minimum: 0, maximum: 1 },
          bagging_fraction: { minimum: 0, maximum: 1 },
        };
        String(value || "").split(/[\r\n;]+/).forEach((line) => {
          const text = String(line || "").trim();
          if (!text) return;
          const parts = text.split("=");
          if (parts.length < 2) return;
          const key = String(parts.shift() || "").trim();
          const spec = parts.join("=").trim();
          if (!key || !spec) return;
          const config = constrainedKeys[key] || {};
          const values = parseSweepNumberSpec(spec, {
            minimum: config.minimum ?? Number.NEGATIVE_INFINITY,
            maximum: config.maximum ?? Number.POSITIVE_INFINITY,
            integerOnly: integerKeys.has(key),
          });
          if (!values.length) return;
          specs[key] = values;
        });
        return specs;
      }
      function parseCsvFloats(value) {
        return String(value || "")
          .split(",")
          .map((item) => Number(item.trim()))
          .filter((item) => Number.isFinite(item));
      }
      function parseTagText(value) {
        const out = {};
        String(value || "").split(/\r?\n/).forEach((line) => {
          const raw = line.trim();
          if (!raw || !raw.includes(":")) return;
          const idx = raw.indexOf(":");
          const key = raw.slice(0, idx).trim();
          if (!key) return;
          out[key] = raw.slice(idx + 1).trim();
        });
        return out;
      }
      function tagsToText(tags) {
        return Object.entries(tags || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
      }
      function sweepAxisSpec(path) {
        return SWEEP_AXIS_BY_PATH[String(path || "").trim()] || null;
      }
      function sweepAxisModeOptions(spec) {
        const options = [{ value: "fixed", label: "Fixed" }, { value: "list", label: "Sweep list" }];
        if (spec && spec.allowRange && (spec.kind === "int" || spec.kind === "float")) {
          options.push({ value: "range", label: "Sweep range" });
        }
        return options;
      }
      function nextSweepAxisId() {
        const next = Number(state.lab.sweepAxisSeq || 1);
        state.lab.sweepAxisSeq = next + 1;
        return `axis_${next}`;
      }
      function baseValueForSweepPath(path) {
        const clean = String(path || "").trim();
        if (!clean) return null;
        if (clean === "horizons") return builderSelectedHorizons();
        if (clean === "base_timeframe_mode") return el("builderBaseMode")?.value || "legacy";
        if (clean === "train_window_mode") return el("builderTrainWindowMode")?.value || "expanding";
        if (clean === "train_window_days") return Number(el("builderTrainWindowDays")?.value || "365");
        if (clean === "val_days") return Number(el("builderValDays")?.value || "30");
        if (clean === "test_days") return Number(el("builderTestDays")?.value || "30");
        if (clean === "walk_forward_enabled") return Boolean(el("builderWalkForwardEnabled")?.checked);
        if (clean === "wf_train_days") return Number(el("builderWfTrainDays")?.value || "180");
        if (clean === "wf_val_days") return Number(el("builderWfValDays")?.value || "30");
        if (clean === "wf_step_days") return Number(el("builderWfStepDays")?.value || "30");
        if (clean === "wf_folds") return Number(el("builderWfFolds")?.value || "3");
        if (clean === "target_coverage_percent") return Number(el("builderCoverage")?.value || "80");
        if (clean === "interval_mode") return el("builderIntervalMode")?.value || "symmetric";
        if (clean === "custom_q_low_percent") return Number(el("builderCustomLow")?.value || "10");
        if (clean === "custom_q_high_percent") return Number(el("builderCustomHigh")?.value || "90");
        if (clean === "multi_interval_coverages") return parseCsvInts(el("builderMultiCoverages")?.value || "", { minimum: 1, maximum: 99 });
        if (clean === "quantile_strategy") return el("builderQStrategy")?.value || "interval_only";
        if (clean === "selected_quantiles") return builderSelectedQuantiles();
        if (clean === "feature_set_version") return el("builderFeatureSetVersion")?.value || "feat_v1";
        if (clean === "scale_grid") return parseCsvFloats(el("builderScaleGrid")?.value || "");
        if (clean === "scale_selection_rule") return el("builderScaleRule")?.value || "min_abs_coverage_gap_then_width";
        if (clean === "calibration_method") return el("builderCalibrationMethod")?.value || "grid_scale";
        if (clean.startsWith("steps_overrides.")) {
          const horizon = clean.split(".")[1];
          return Number(el(`builderStep${horizon}`)?.value || "1");
        }
        if (clean.startsWith("feature_groups.")) {
          const suffix = clean.split(".")[1];
          const map = {
            returns: "builderFeatureReturns",
            volatility: "builderFeatureVolatility",
            volume: "builderFeatureVolume",
            trend: "builderFeatureTrend",
            rsi: "builderFeatureRsi",
            atr: "builderFeatureAtr",
            macd: "builderFeatureMacd",
          };
          return Boolean(el(map[suffix])?.checked);
        }
        if (clean.startsWith("feature_overrides.")) {
          const suffix = clean.replace("feature_overrides.", "");
          if (suffix === "return_windows") return parseCsvInts(el("builderReturnWindows")?.value || "", { minimum: 1, maximum: 10000 });
          if (suffix === "vol_windows") return parseCsvInts(el("builderVolWindows")?.value || "", { minimum: 2, maximum: 10000 });
          if (suffix === "volume_windows") return parseCsvInts(el("builderVolumeWindows")?.value || "", { minimum: 2, maximum: 10000 });
          if (suffix === "trend_windows") return parseCsvInts(el("builderTrendWindows")?.value || "", { minimum: 2, maximum: 10000 });
          if (suffix === "rsi_window") return Number(el("builderRsiWindow")?.value || "14");
          if (suffix === "atr_window") return Number(el("builderAtrWindow")?.value || "14");
          if (suffix === "macd.fast") return Number(el("builderMacdFast")?.value || "12");
          if (suffix === "macd.slow") return Number(el("builderMacdSlow")?.value || "26");
          if (suffix === "macd.signal") return Number(el("builderMacdSignal")?.value || "9");
        }
        if (clean.startsWith("hyperparams.")) {
          const suffix = clean.replace("hyperparams.", "");
          const map = {
            learning_rate: "builderLearningRate",
            num_leaves: "builderNumLeaves",
            max_depth: "builderMaxDepth",
            min_data_in_leaf: "builderMinDataLeaf",
            feature_fraction: "builderFeatureFraction",
            bagging_fraction: "builderBaggingFraction",
            bagging_freq: "builderBaggingFreq",
            lambda_l1: "builderLambdaL1",
            lambda_l2: "builderLambdaL2",
            num_boost_round: "builderBoostRounds",
            early_stopping_rounds: "builderEarlyStopping",
            seed: "builderSeed",
          };
          return asNumber(el(map[suffix])?.value);
        }
        return null;
      }
      function buildQuantileOptions(selectedValues) {
        const selected = new Set((selectedValues || []).map((item) => String(item)));
        const rows = [];
        for (let idx = 1; idx <= 99; idx += 1) {
          const key = `q${String(idx).padStart(2, "0")}`;
          rows.push(`<option value="${key}"${selected.has(key) ? " selected" : ""}>${key}</option>`);
        }
        return rows.join("");
      }
      function cloneAxisValue(value) {
        if (Array.isArray(value)) return value.map((item) => cloneAxisValue(item));
        return value;
      }
      function seedSweepAxisValue(path) {
        const spec = sweepAxisSpec(path);
        const baseValue = cloneAxisValue(baseValueForSweepPath(path));
        if (!spec) {
          return { mode: "list", values: [baseValue], start: null, end: null, step: null };
        }
        if (spec.kind === "enum_list" || spec.kind === "quantile_list" || spec.kind === "int_list" || spec.kind === "float_list") {
          const candidate = Array.isArray(baseValue) ? baseValue : [];
          return { mode: "list", values: [candidate], start: null, end: null, step: null };
        }
        const scalar = baseValue !== null && baseValue !== undefined ? baseValue : (spec.kind === "bool" ? false : "");
        return {
          mode: "list",
          values: [scalar],
          start: spec.allowRange ? scalar : null,
          end: spec.allowRange ? scalar : null,
          step: spec.kind === "int" ? 1 : 0.1,
        };
      }
      function createSweepAxis(path = "target_coverage_percent") {
        const seeded = seedSweepAxisValue(path);
        return {
          id: nextSweepAxisId(),
          path,
          mode: seeded.mode,
          values: seeded.values,
          start: seeded.start,
          end: seeded.end,
          step: seeded.step,
        };
      }
      function renderSweepScalarInput(axisId, valueIndex, spec, value) {
        if (spec.kind === "enum") {
          const options = (spec.options || []).map((option) => `<option value="${esc(option)}"${String(value) === String(option) ? " selected" : ""}>${esc(option)}</option>`).join("");
          return `<select data-role="axis-value" data-axis-id="${esc(axisId)}" data-index="${valueIndex}">${options}</select>`;
        }
        if (spec.kind === "bool") {
          return `<select data-role="axis-value" data-axis-id="${esc(axisId)}" data-index="${valueIndex}"><option value="true"${value === true ? " selected" : ""}>true</option><option value="false"${value === false ? " selected" : ""}>false</option></select>`;
        }
        if (spec.kind === "int") {
          return `<input data-role="axis-value" data-axis-id="${esc(axisId)}" data-index="${valueIndex}" type="number" step="1" value="${esc(value ?? "")}" />`;
        }
        if (spec.kind === "float") {
          return `<input data-role="axis-value" data-axis-id="${esc(axisId)}" data-index="${valueIndex}" type="number" step="0.0001" value="${esc(value ?? "")}" />`;
        }
        return `<input data-role="axis-value" data-axis-id="${esc(axisId)}" data-index="${valueIndex}" value="${esc(value ?? "")}" />`;
      }
      function renderSweepListCandidate(axis, spec, candidate, valueIndex) {
        if (spec.kind === "enum_list") {
          const selected = new Set(Array.isArray(candidate) ? candidate.map((item) => String(item)) : []);
          return `<div class="sweep-chip-grid">${(spec.options || []).map((option) => `<label class="choice-pill"><input data-role="axis-enum-list-item" data-axis-id="${esc(axis.id)}" data-index="${valueIndex}" value="${esc(option)}" type="checkbox"${selected.has(option) ? " checked" : ""} />${esc(option)}</label>`).join("")}</div>`;
        }
        if (spec.kind === "quantile_list") {
          return `<select multiple size="8" data-role="axis-multi-select" data-axis-id="${esc(axis.id)}" data-index="${valueIndex}" class="compare-model-select">${buildQuantileOptions(Array.isArray(candidate) ? candidate : [])}</select>`;
        }
        const nested = Array.isArray(candidate) ? candidate : [];
        const step = spec.kind === "int_list" ? "1" : "0.01";
        const type = "number";
        return `<div class="sweep-nested-list">${nested.map((item, nestedIndex) => `<div class="sweep-inline-row"><input data-role="axis-nested-item" data-axis-id="${esc(axis.id)}" data-index="${valueIndex}" data-nested-index="${nestedIndex}" type="${type}" step="${step}" value="${esc(item)}" /><button type="button" data-action="remove-axis-nested-item" data-axis-id="${esc(axis.id)}" data-index="${valueIndex}" data-nested-index="${nestedIndex}">Remove</button></div>`).join("")}<button type="button" data-action="add-axis-nested-item" data-axis-id="${esc(axis.id)}" data-index="${valueIndex}">Add item</button></div>`;
      }
      function renderSweepAxisEditor(axis, spec) {
        if (!spec) return `<div class="mut">Unsupported axis.</div>`;
        if (axis.mode === "fixed") {
          return `<div class="mut">Using the fixed base-form value for ${esc(spec.label)}.</div>`;
        }
        if (axis.mode === "range") {
          return `<div class="sweep-inline-row"><input data-role="axis-range-start" data-axis-id="${esc(axis.id)}" type="number" step="${spec.kind === "int" ? "1" : "0.0001"}" value="${esc(axis.start ?? "")}" placeholder="start" /><input data-role="axis-range-end" data-axis-id="${esc(axis.id)}" type="number" step="${spec.kind === "int" ? "1" : "0.0001"}" value="${esc(axis.end ?? "")}" placeholder="end" /><input data-role="axis-range-step" data-axis-id="${esc(axis.id)}" type="number" step="${spec.kind === "int" ? "1" : "0.0001"}" value="${esc(axis.step ?? "")}" placeholder="step" /></div>`;
        }
        return `<div class="sweep-values-stack">${(axis.values || []).map((value, valueIndex) => `<div class="sweep-value-card">${renderSweepScalarOrListValue(axis, spec, value, valueIndex)}<div class="btn-row field-offset-sm"><button type="button" data-action="remove-axis-value" data-axis-id="${esc(axis.id)}" data-index="${valueIndex}">Remove value</button></div></div>`).join("")}<button type="button" data-action="add-axis-value" data-axis-id="${esc(axis.id)}">Add value</button></div>`;
      }
      function renderSweepScalarOrListValue(axis, spec, value, valueIndex) {
        if (spec.kind === "enum_list" || spec.kind === "quantile_list" || spec.kind === "int_list" || spec.kind === "float_list") {
          return renderSweepListCandidate(axis, spec, value, valueIndex);
        }
        return renderSweepScalarInput(axis.id, valueIndex, spec, value);
      }
      function renderSweepAxisRow(axis) {
        const spec = sweepAxisSpec(axis.path);
        const modeOptions = sweepAxisModeOptions(spec).map((item) => `<option value="${esc(item.value)}"${axis.mode === item.value ? " selected" : ""}>${esc(item.label)}</option>`).join("");
        const groupedOptions = [];
        const groups = [...new Set(SWEEP_AXIS_DEFS.map((item) => item.group))];
        groups.forEach((group) => {
          const options = SWEEP_AXIS_DEFS.filter((item) => item.group === group).map((item) => `<option value="${esc(item.path)}"${item.path === axis.path ? " selected" : ""}>${esc(item.label)}</option>`).join("");
          groupedOptions.push(`<optgroup label="${esc(group)}">${options}</optgroup>`);
        });
        return `<article class="sweep-axis-card" data-axis-id="${esc(axis.id)}"><div class="sweep-axis-top"><div class="field"><label>Parameter</label><select data-role="axis-path" data-axis-id="${esc(axis.id)}">${groupedOptions.join("")}</select></div><div class="field"><label>Mode</label><select data-role="axis-mode" data-axis-id="${esc(axis.id)}">${modeOptions}</select></div><div class="field"><label>Base value</label><div class="mut sweep-base-value">${esc(JSON.stringify(baseValueForSweepPath(axis.path)))}</div></div><div class="field"><label>&nbsp;</label><button type="button" data-action="remove-axis" data-axis-id="${esc(axis.id)}">Remove axis</button></div></div>${renderSweepAxisEditor(axis, spec)}</article>`;
      }
      function renderSweepAxisBuilder() {
        const host = el("builderSweepAxesList");
        const empty = el("builderSweepAxesEmpty");
        if (!host || !empty) return;
        const axes = state.lab.sweepAxes || [];
        empty.style.display = axes.length ? "none" : "block";
        host.innerHTML = axes.map((axis) => renderSweepAxisRow(axis)).join("");
      }
      function addSweepAxis(path = "target_coverage_percent") {
        state.lab.sweepAxes = [...(state.lab.sweepAxes || []), createSweepAxis(path)];
        renderSweepAxisBuilder();
        scheduleBuilderValidate();
      }
      function updateSweepAxis(axisId, updater) {
        state.lab.sweepAxes = (state.lab.sweepAxes || []).map((axis) => axis.id === axisId ? { ...axis, ...updater(axis) } : axis);
        renderSweepAxisBuilder();
        scheduleBuilderValidate();
      }
      function removeSweepAxis(axisId) {
        state.lab.sweepAxes = (state.lab.sweepAxes || []).filter((axis) => axis.id !== axisId);
        renderSweepAxisBuilder();
        scheduleBuilderValidate();
      }
      function appendSweepAxisValue(axisId) {
        updateSweepAxis(axisId, (axis) => {
          const spec = sweepAxisSpec(axis.path);
          const baseValue = cloneAxisValue(baseValueForSweepPath(axis.path));
          let nextValue = baseValue;
          if (spec && (spec.kind === "enum_list" || spec.kind === "quantile_list" || spec.kind === "int_list" || spec.kind === "float_list")) nextValue = Array.isArray(baseValue) ? baseValue : [];
          if (spec && spec.kind === "bool") nextValue = false;
          return { values: [...(axis.values || []), cloneAxisValue(nextValue)] };
        });
      }
      function setAxisPath(axisId, path) {
        const next = createSweepAxis(path);
        updateSweepAxis(axisId, () => ({ path: next.path, mode: next.mode, values: next.values, start: next.start, end: next.end, step: next.step }));
      }
      function setAxisMode(axisId, mode) {
        updateSweepAxis(axisId, (axis) => {
          const spec = sweepAxisSpec(axis.path);
          if (mode === "range" && !(spec && spec.allowRange)) return {};
          return { mode };
        });
      }
      function collectSweepAxesPayload() {
        return (state.lab.sweepAxes || [])
          .filter((axis) => axis.mode === "list" || axis.mode === "range")
          .map((axis) => ({
            path: axis.path,
            mode: axis.mode,
            values: axis.mode === "list" ? cloneAxisValue(axis.values || []) : [],
            start: axis.mode === "range" ? asNumber(axis.start) : null,
            end: axis.mode === "range" ? asNumber(axis.end) : null,
            step: axis.mode === "range" ? asNumber(axis.step) : null,
          }));
      }
      function renderSweepPreview(preview) {
        const payload = preview || {};
        setNodeText("builderSweepRequested", String(payload.requested_total ?? 1));
        setNodeText("builderSweepEffective", String(payload.effective_total ?? 1));
        setNodeText("builderSweepDuplicates", String(payload.duplicate_count ?? 0));
        setNodeText("builderSweepInvalid", String(payload.invalid_count ?? 0));
        setNodeText("builderSweepModels", String(payload.estimated_model_count ?? 0));
        setNodeText("builderSweepHeavy", payload.resource_heavy ? "high" : "normal");
        const body = el("builderSweepPreviewBody");
        if (body) {
          const axes = Array.isArray(payload.axes) ? payload.axes : [];
          body.innerHTML = axes.length
            ? axes.map((item) => `<tr><td>${esc(item.label || item.path)}</td><td>${esc(item.mode || "-")}</td><td>${esc(String(item.requested_count ?? 0))}</td><td>${esc(String(item.effective_count ?? 0))}</td><td>${esc(String(item.dropped_count ?? 0))}</td><td>${esc(((item.warnings || []).concat(item.errors || [])).join(" | ") || "-")}</td></tr>`).join("")
            : `<tr><td colspan="6" class="mut">No sweep axes. Base form remains fixed.</td></tr>`;
        }
        const lines = [];
        (payload.heavy_reasons || []).forEach((item) => lines.push(`heavy: ${item}`));
        (payload.duplicate_reasons || []).forEach((item) => lines.push(`dedupe: ${item}`));
        setNodeText("builderSweepPreviewJson", lines.length ? lines.join("\n") : "Sweep preview is clean.");
      }
      function modelStatusTag(status) {
        const value = String(status || "active");
        if (value === "archived") return `<span class="tag cancelled">archived</span>`;
        if (value === "deleted") return `<span class="tag miss">deleted</span>`;
        return `<span class="tag active">active</span>`;
      }
      function jobStatusTag(status) {
        const value = String(status || "pending").trim().toLowerCase();
        if (value === "succeeded") return `<span class="tag hit">succeeded</span>`;
        if (value === "failed") return `<span class="tag miss">failed</span>`;
        if (value === "canceled") return `<span class="tag cancelled">canceled</span>`;
        if (value === "running") return `<span class="tag active">running</span>`;
        return `<span class="tag active">pending</span>`;
      }
      function setBuilderInputValue(id, value) {
        const node = el(id);
        if (!node) return;
        node.value = String(value);
      }
      function setBuilderHorizons(horizons) {
        const selected = new Set((horizons || []).map((item) => String(item)));
        HORIZONS.forEach((horizon) => {
          const node = el(`builderHorizon${horizon}`);
          if (!node) return;
          node.checked = selected.has(horizon);
        });
      }
      function applyTrainingBudgetPreset(presetName, { scheduleValidate = true } = {}) {
        const presetKey = String(presetName || "").trim().toUpperCase();
        const preset = TRAINING_BUDGET_PRESETS[presetKey];
        if (!preset) return false;
        if (el("builderBudgetPreset")) el("builderBudgetPreset").value = presetKey;
        setBuilderHorizons(preset.horizons);
        setBuilderInputValue("builderQStrategy", preset.quantile_strategy);
        setBuilderInputValue("builderTrainWindowDays", preset.train_window_days);
        setBuilderInputValue("builderValDays", preset.val_days);
        setBuilderInputValue("builderTestDays", preset.test_days);
        setBuilderInputValue("builderBoostRounds", preset.num_boost_round);
        setBuilderInputValue("builderEarlyStopping", preset.early_stopping_rounds);
        setBuilderInputValue("builderScaleGrid", (preset.scale_grid || []).join(","));
        if (preset.quantile_strategy === "selected_set") {
          setBuilderQuantileSelection(preset.selected_quantiles || []);
        } else if (preset.quantile_strategy === "interval_only") {
          setBuilderQuantileSelection(builderPrimaryQuantiles());
        } else {
          setBuilderQuantileSelection(preset.selected_quantiles || ["q10", "q50", "q90"]);
        }
        if (el("builderQuantileConfirm")) el("builderQuantileConfirm").checked = false;
        syncBuilderModeStates();
        renderSweepAxisBuilder();
        if (scheduleValidate) scheduleBuilderValidate();
        return true;
      }
      function builderQuantileSoftLimit() {
        return clamp(Number(el("builderQSoftLimit")?.value || state.lab.validation?.quantile_soft_limit || "11"), 1, 99);
      }
      function builderPrimaryQuantiles() {
        const coverage = clamp(Number(el("builderCoverage")?.value || "80"), 1, 99);
        const qLowPercent = clamp(Math.round((100 - coverage) / 2), 1, 99);
        const qHighPercent = clamp(100 - qLowPercent, 1, 99);
        const fallbackLow = `q${String(qLowPercent).padStart(2, "0")}`;
        const fallbackHigh = `q${String(qHighPercent).padStart(2, "0")}`;
        const low = String(state.lab.validation?.q_low || fallbackLow);
        const high = String(state.lab.validation?.q_high || fallbackHigh);
        return [low, "q50", high].filter((value, index, items) => items.indexOf(value) === index);
      }
      function syncBuilderModeStates() {
        const baseMode = el("builderBaseMode")?.value || "legacy";
        const intervalMode = el("builderIntervalMode")?.value || "symmetric";
        const quantileStrategy = el("builderQStrategy")?.value || "interval_only";
        const stepInputsEnabled = baseMode === "base_1m";
        HORIZONS.map((horizon) => `builderStep${horizon}`).forEach((id) => {
          const node = el(id);
          if (node) node.disabled = !stepInputsEnabled;
        });
        ["builderCustomLow", "builderCustomHigh"].forEach((id) => {
          const node = el(id);
          if (node) node.disabled = intervalMode !== "custom";
        });
        if (el("builderMultiCoverages")) el("builderMultiCoverages").disabled = intervalMode !== "multi_pack";
        const manualQuantiles = quantileStrategy === "selected_set";
        document.querySelectorAll("#builderQuantileGrid input[type='checkbox']").forEach((node) => {
          node.disabled = !manualQuantiles;
        });
        ["builderQSearch", "builderAddQ50Btn", "builderSelectPrimaryBtn", "builderSelectDecilesBtn", "builderSelectRangeBtn", "builderClearQBtn", "builderQRangeStart", "builderQRangeEnd"].forEach((id) => {
          const node = el(id);
          if (node) node.disabled = !manualQuantiles;
        });
        const addQ50Btn = el("builderAddQ50Btn");
        const needsQ50 = manualQuantiles && !builderSelectedQuantiles().includes("q50");
        if (addQ50Btn) {
          addQ50Btn.textContent = needsQ50 ? "Add q50 (Recommended)" : "Add q50";
          addQ50Btn.classList.toggle("btn-primary", needsQ50);
        }
        const estimatedQuantileCount = new Set([
          ...builderSelectedQuantiles(),
          ...builderPrimaryQuantiles(),
        ]).size;
        const softLimit = Number(state.lab.validation?.quantile_soft_limit || builderQuantileSoftLimit());
        const requiresConfirmation = quantileStrategy === "full_grid"
          || (quantileStrategy === "selected_set" && (estimatedQuantileCount > softLimit || Boolean(state.lab.validation?.requires_confirmation)));
        if (el("builderQuantileConfirm")) {
          el("builderQuantileConfirm").disabled = !requiresConfirmation;
          if (!requiresConfirmation) el("builderQuantileConfirm").checked = false;
        }
      }
      function builderSelectedHorizons() {
        return HORIZONS.filter((h) => el(`builderHorizon${h}`)?.checked);
      }
      function builderSelectedQuantiles() {
        return [...document.querySelectorAll("#builderQuantileGrid input[type='checkbox']:checked")]
          .map((node) => String(node.value || ""))
          .filter(Boolean);
      }
      function setBuilderQuantileSelection(keys) {
        const wanted = new Set((keys || []).map((item) => String(item || "").toLowerCase()));
        document.querySelectorAll("#builderQuantileGrid input[type='checkbox']").forEach((node) => {
          node.checked = wanted.has(String(node.value || "").toLowerCase());
        });
        refreshBuilderQuantileVisuals();
      }
      function refreshBuilderQuantileVisuals() {
        const search = String(el("builderQSearch")?.value || "").trim().toLowerCase();
        document.querySelectorAll(".quantile-option").forEach((node) => {
          const input = node.querySelector("input");
          const value = String(input?.value || "").toLowerCase();
          node.classList.toggle("is-selected", Boolean(input?.checked));
          const hidden = search && !value.includes(search) && !value.replace("q", "").includes(search.replace("q", ""));
          node.classList.toggle("is-hidden", Boolean(hidden));
        });
        const selected = builderSelectedQuantiles();
        setNodeText("builderQuantileCount", String(selected.length || 0));
        setNodeText("builderQuantileList", selected.length ? selected.join(", ") : "No custom quantiles selected.");
        syncBuilderModeStates();
      }
      function renderBuilderQuantileGrid() {
        const host = el("builderQuantileGrid");
        if (!host || host.childElementCount) return;
        const rows = [];
        for (let idx = 1; idx <= 99; idx += 1) {
          const key = `q${String(idx).padStart(2, "0")}`;
          const checked = key === "q10" || key === "q50" || key === "q90";
          rows.push(`<label class="quantile-option${checked ? " is-selected" : ""}"><input type="checkbox" value="${key}"${checked ? " checked" : ""} />${key}</label>`);
        }
        host.innerHTML = rows.join("");
        host.addEventListener("change", () => {
          refreshBuilderQuantileVisuals();
          scheduleBuilderValidate();
        });
        refreshBuilderQuantileVisuals();
        syncBuilderModeStates();
      }
      function selectBuilderQuantileRange() {
        let start = clamp(Number(el("builderQRangeStart")?.value || "1"), 1, 99);
        let end = clamp(Number(el("builderQRangeEnd")?.value || "99"), 1, 99);
        if (start > end) [start, end] = [end, start];
        if (el("builderQRangeStart")) el("builderQRangeStart").value = String(start);
        if (el("builderQRangeEnd")) el("builderQRangeEnd").value = String(end);
        const keys = [];
        for (let idx = start; idx <= end; idx += 1) keys.push(`q${String(idx).padStart(2, "0")}`);
        setBuilderQuantileSelection(keys);
      }
      function collectBuilderFeatureGroups() {
        return {
          returns: Boolean(el("builderFeatureReturns")?.checked),
          volatility: Boolean(el("builderFeatureVolatility")?.checked),
          volume: Boolean(el("builderFeatureVolume")?.checked),
          trend: Boolean(el("builderFeatureTrend")?.checked),
          rsi: Boolean(el("builderFeatureRsi")?.checked),
          atr: Boolean(el("builderFeatureAtr")?.checked),
          macd: Boolean(el("builderFeatureMacd")?.checked),
        };
      }
      function collectBuilderPayload() {
          return {
            asset: el("builderAsset").value || "BTC",
            training_budget_preset: el("builderBudgetPreset")?.value || "custom",
            base_timeframe_mode: el("builderBaseMode").value || "legacy",
            horizons: builderSelectedHorizons(),
          steps_overrides: {
            "5m": Number(el("builderStep5m").value || "5"),
            "15m": Number(el("builderStep15m").value || "15"),
            "1h": Number(el("builderStep1h").value || "60"),
            "4h": Number(el("builderStep4h").value || "240"),
            "1d": Number(el("builderStep1d").value || "1440"),
            "1w": Number(el("builderStep1w").value || "10080"),
          },
          target_coverage_percent: Number(el("builderCoverage").value || "80"),
          interval_mode: el("builderIntervalMode").value || "symmetric",
          custom_q_low_percent: Number(el("builderCustomLow").value || "10"),
          custom_q_high_percent: Number(el("builderCustomHigh").value || "90"),
          multi_interval_coverages: parseCsvInts(el("builderMultiCoverages").value || "", { minimum: 1, maximum: 99 }),
          quantile_strategy: el("builderQStrategy").value || "interval_only",
          selected_quantiles: builderSelectedQuantiles(),
          quantile_soft_limit: builderQuantileSoftLimit(),
          confirm_resource_heavy: Boolean(el("builderQuantileConfirm").checked),
          train_window_mode: el("builderTrainWindowMode").value || "expanding",
          train_window_days: Number(el("builderTrainWindowDays").value || "365"),
          val_days: Number(el("builderValDays").value || "30"),
          test_days: Number(el("builderTestDays").value || "30"),
          walk_forward_enabled: Boolean(el("builderWalkForwardEnabled").checked),
          wf_train_days: Number(el("builderWfTrainDays").value || "180"),
          wf_val_days: Number(el("builderWfValDays").value || "30"),
          wf_step_days: Number(el("builderWfStepDays").value || "30"),
          wf_folds: Number(el("builderWfFolds").value || "3"),
          feature_set_version: el("builderFeatureSetVersion").value || "feat_v1",
          feature_groups: collectBuilderFeatureGroups(),
          feature_overrides: {
            return_windows: parseCsvInts(el("builderReturnWindows").value || "", { minimum: 1, maximum: 10000 }),
            vol_windows: parseCsvInts(el("builderVolWindows").value || "", { minimum: 2, maximum: 10000 }),
            volume_windows: parseCsvInts(el("builderVolumeWindows").value || "", { minimum: 2, maximum: 10000 }),
            trend_windows: parseCsvInts(el("builderTrendWindows").value || "", { minimum: 2, maximum: 10000 }),
            rsi_window: Number(el("builderRsiWindow").value || "14"),
            atr_window: Number(el("builderAtrWindow").value || "14"),
            macd: {
              fast: Number(el("builderMacdFast").value || "12"),
              slow: Number(el("builderMacdSlow").value || "26"),
              signal: Number(el("builderMacdSignal").value || "9"),
            },
          },
          hyperparams: {
            learning_rate: asNumber(el("builderLearningRate").value),
            num_leaves: asNumber(el("builderNumLeaves").value),
            max_depth: asNumber(el("builderMaxDepth").value),
            min_data_in_leaf: asNumber(el("builderMinDataLeaf").value),
            feature_fraction: asNumber(el("builderFeatureFraction").value),
            bagging_fraction: asNumber(el("builderBaggingFraction").value),
            bagging_freq: asNumber(el("builderBaggingFreq").value),
            lambda_l1: asNumber(el("builderLambdaL1").value),
            lambda_l2: asNumber(el("builderLambdaL2").value),
            num_boost_round: asNumber(el("builderBoostRounds").value),
            early_stopping_rounds: asNumber(el("builderEarlyStopping").value),
            seed: asNumber(el("builderSeed").value),
          },
            calibration_method: el("builderCalibrationMethod").value || "grid_scale",
            scale_grid: parseCsvFloats(el("builderScaleGrid").value || ""),
            scale_selection_rule: el("builderScaleRule").value.trim() || "min_abs_coverage_gap_then_width",
            sweep_axes: collectSweepAxesPayload(),
            sweep_target_coverages: [],
            sweep_train_window_days: [],
            sweep_feature_set_versions: [],
            sweep_calibration_methods: [],
            sweep_hyperparams: {},
            experiment_id: el("builderExperimentId").value.trim() || null,
            parent_model_id: el("builderParentModelId").value.trim() || null,
            notes: el("builderNotes").value.trim() || null,
            tags: parseTagText(el("builderTags").value),
          };
      }
        function applyBuilderValidation(result, { showBanner = true } = {}) {
          state.lab.validation = result;
          setNodeText("builderCoverageValue", `${num((Number(result?.requested_coverage || 0) * 100), 0)}%`);
          setNodeText("builderRequestedCoverage", pct(result?.requested_coverage));
          setNodeText("builderDerivedLow", result?.q_low || "-");
          setNodeText("builderDerivedHigh", result?.q_high || "-");
          setNodeText("builderEffectiveCoverage", pct(result?.effective_coverage));
          setNodeText("builderImpliedCoverage", pct(result?.implied_coverage));
          renderSweepPreview(result?.sweep_preview || {});
          if (el("builderQSoftLimit")) el("builderQSoftLimit").value = String(result?.quantile_soft_limit || 11);
        if (Array.isArray(result?.quantiles_final) && result.quantiles_final.length) {
          if (String(result?.quantile_strategy || "") !== "selected_set") {
            setBuilderQuantileSelection(result.quantiles_final);
          }
          setNodeText("builderQuantileList", result.quantiles_final.join(", "));
          setNodeText("builderQuantileCount", String(result.quantiles_final.length));
        }
        const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
        const errors = Array.isArray(result?.errors) ? result.errors : [];
        const softLimit = Number(result?.quantile_soft_limit || 11);
        setNodeText("builderQuantileWarning", result?.requires_confirmation ? `resource heavy: confirmation required (> ${softLimit} q)` : `q soft-limit: ${softLimit}`);
        syncBuilderModeStates();
        if (showBanner) {
          const confirmationReasons = Array.isArray(result?.confirmation_reasons) ? result.confirmation_reasons : [];
          const text = errors.length ? errors.join(" | ") : [...warnings, ...confirmationReasons].join(" | ");
          showBuilderBanner(text, errors.length > 0);
        }
      }
        async function validateBuilder({ showBanner = true, jobType = null } = {}) {
          const body = collectBuilderPayload();
          const activeJobType = jobType || (body.sweep_axes?.length ? "sweep_train" : "train_model");
          const result = await api("/api/lab/validate", { method: "POST", query: { job_type: activeJobType }, body });
          applyBuilderValidation(result, { showBanner });
          return result;
        }
        function scheduleBuilderValidate() {
          if (state.lab.validateTimer) clearTimeout(state.lab.validateTimer);
          state.lab.validateTimer = setTimeout(() => { validateBuilder({ showBanner: false }).catch(() => {}); }, 220);
        }
        function selectedJobModelIds() {
          const current = (state.lab.jobs || []).find((item) => item.job_id === state.lab.selectedJobId);
          if (!current || !Array.isArray(current?.result?.models)) return [];
          return current.result.models
            .map((row) => String(row?.model_version || ""))
            .filter(Boolean);
        }
        function syncSelectedJobActions() {
          const button = el("builderAddJobModelsToCompareBtn");
          if (!button) return;
          button.disabled = selectedJobModelIds().length === 0;
        }
        function renderLabJobs() {
          const body = el("builderJobsBody");
          if (!body) return;
          const items = state.lab.jobs || [];
          if (!items.length) {
            body.innerHTML = `<tr><td colspan="10" class="mut">No jobs.</td></tr>`;
            syncSelectedJobActions();
            return;
          }
          body.innerHTML = items.map((item) => `<tr class="row-clickable${state.lab.selectedJobId === item.job_id ? " row-new" : ""}" data-job-id="${esc(item.job_id)}"><td>${jobStatusTag(item.status)}</td><td>${esc(item.job_kind)}</td><td>${esc(item.asset)}</td><td>${esc((item.horizons || []).join(", "))}</td><td>${esc(item.stage || "-")}</td><td>${esc(`${num(item.progress, 0)}%`)}</td><td>${esc(dt(item.created_at))}</td><td>${esc(dt(item.started_at))}</td><td>${esc(dt(item.finished_at))}</td><td><button data-action="view" data-job-id="${esc(item.job_id)}">Logs</button> <button data-action="cancel" data-job-id="${esc(item.job_id)}"${(item.status !== "pending" && item.status !== "running") ? " disabled" : ""}>Cancel</button> <button data-action="retry" data-job-id="${esc(item.job_id)}">Retry</button></td></tr>`).join("");
          syncSelectedJobActions();
        }
        async function loadLabJobLogs(jobId) {
          if (!jobId) return;
          const payload = await api(`/api/lab/jobs/${encodeURIComponent(jobId)}/logs`);
          state.lab.selectedJobId = jobId;
        state.lab.jobLogs = payload.logs || [];
        const job = payload.job || {};
        const metaLines = [
          `job_id: ${job.job_id || "-"}`,
            `status: ${job.status || "-"}`,
            `created: ${dt(job.created_at)}`,
            `started: ${dt(job.started_at)}`,
            `finished: ${dt(job.finished_at)}`,
            `models: ${num(Array.isArray(job?.result?.models) ? job.result.models.length : 0, 0)}`,
          ];
          const lines = [
            ...metaLines,
            "",
            ...(state.lab.jobLogs || []).map((row) => `[${dt(row.ts)}] ${String(row.level || "info").toUpperCase()}${row.stage ? `/${row.stage}` : ""} ${row.message}`),
          ];
          setNodeText("builderJobLogs", lines.length ? lines.join("\n") : "No logs.");
          renderLabJobs();
        }
      async function loadLabJobs() {
        const statuses = el("builderJobStatusFilter").value.trim();
        const payload = await api("/api/lab/jobs", { query: { statuses, limit: 120 } });
        const previousStatuses = { ...(state.lab.jobStatusById || {}) };
        const items = payload.items || [];
        const nextStatuses = {};
        const newlySucceededModels = [];
        items.forEach((item) => {
          nextStatuses[item.job_id] = item.status;
          if (item.status === "succeeded" && previousStatuses[item.job_id] !== "succeeded") {
            const modelIds = Array.isArray(item?.result?.models)
              ? item.result.models.map((row) => String(row?.model_version || "")).filter(Boolean)
              : [];
            newlySucceededModels.push(...modelIds);
          }
        });
        state.lab.jobStatusById = nextStatuses;
        state.lab.jobs = items;
        renderLabJobs();
        if (newlySucceededModels.length) {
          await Promise.all([loadLabModels(), loadModels()]);
          const preferredModelId = newlySucceededModels.find((modelId) => (state.lab.recentModels || []).some((item) => item.model_id === modelId));
          if (preferredModelId) {
            selectLabModel(preferredModelId);
            setEventModelSelection(preferredModelId);
          }
        }
        if (!state.lab.selectedJobId) {
          if (!(state.lab.jobLogs || []).length) setNodeText("builderJobLogs", "Select a job to view logs.");
          return;
        }
        const exists = state.lab.jobs.some((item) => item.job_id === state.lab.selectedJobId);
        if (!exists) {
          state.lab.selectedJobId = null;
          state.lab.jobLogs = [];
          setNodeText("builderJobLogs", "Select a job to view logs.");
          return;
        }
        await loadLabJobLogs(state.lab.selectedJobId);
      }
      function selectLabModel(modelId) {
        state.lab.selectedModelId = modelId || null;
        const item = (state.lab.recentModels || []).find((row) => row.model_id === state.lab.selectedModelId) || null;
        setNodeText("builderSelectedModelId", item?.model_id || "-");
        setNodeText("builderSelectedModelStatus", item?.status || "-");
        el("builderModelNotes").value = item?.notes || "";
        el("builderModelTags").value = tagsToText(item?.tags || {});
        ["builderUseModelForEventsBtn", "builderUseModelForCompareBtn", "builderSaveModelMetaBtn", "builderPromoteModelBtn", "builderArchiveModelBtn", "builderUnarchiveModelBtn", "builderSoftDeleteModelBtn", "builderPurgeModelBtn"].forEach((id) => { el(id).disabled = !item; });
        if (!item) return;
        el("builderPromoteModelBtn").disabled = item.status !== "active";
        el("builderArchiveModelBtn").disabled = item.status !== "active";
        el("builderUnarchiveModelBtn").disabled = item.status !== "archived";
        el("builderPurgeModelBtn").disabled = Number(item.event_refs || 0) > 0;
      }
      function renderLabModels() {
        const body = el("builderModelsBody");
        if (!body) return;
        const items = state.lab.recentModels || [];
        if (!items.length) {
          body.innerHTML = `<tr><td colspan="6" class="mut">No models.</td></tr>`;
          selectLabModel(null);
          return;
        }
        body.innerHTML = items.map((item) => `<tr class="row-clickable${state.lab.selectedModelId === item.model_id ? " row-new" : ""}" data-model-id="${esc(item.model_id)}"><td>${modelStatusTag(item.status)}</td><td>${esc(item.model_id)}</td><td>${esc(item.asset)}</td><td>${esc(item.horizon)}</td><td>${esc(dt(item.created_at))}</td><td>${esc(num(item.event_refs, 0))}</td></tr>`).join("");
        if (!state.lab.selectedModelId || !items.some((item) => item.model_id === state.lab.selectedModelId)) {
          selectLabModel(items[0]?.model_id || null);
        } else {
          selectLabModel(state.lab.selectedModelId);
        }
      }
      async function loadLabModels() {
        const includeArchived = el("builderShowArchived").checked ? "true" : "";
        const includeDeleted = el("builderShowDeleted").checked ? "true" : "";
        const payload = await api("/api/models", { query: { asset: "BTC", include_archived: includeArchived, include_deleted: includeDeleted, limit: 80 } });
        state.lab.recentModels = payload.items || [];
        renderLabModels();
      }
      function setEventModelSelection(modelId) {
        const select = el("createModelSelect");
        if (!select || !modelId) return false;
        const exists = [...select.options].some((option) => option.value === modelId);
        if (!exists) return false;
        select.value = modelId;
        return true;
      }
      function useSelectedModelForEvents() {
        const modelId = state.lab.selectedModelId;
        if (!modelId) return;
        const applied = setEventModelSelection(modelId);
        showToast(applied ? `Event model selected: ${modelId}` : "Model is not available in Event selector yet.");
      }
        function useSelectedModelForCompare() {
          const modelId = state.lab.selectedModelId;
          if (!modelId) return;
          const model = (state.lab.recentModels || []).find((item) => item.model_id === modelId) || (state.models || []).find((item) => item.model_id === modelId);
          if (model?.horizon && el("analysisHorizonFilter")) {
          el("analysisHorizonFilter").value = model.horizon;
        }
        renderCompareModelOptions(modelId);
        const filter = el("deepModelFilter");
        if (filter) filter.value = "";
        if (state.activeTab === "lab" && state.activeLabTab === "compare") {
          Promise.all([loadCompareSummary(), loadCompareScoreboard(), loadDeep()]).then(() => loadCompareChart()).then(() => showToast(`Compare filter set: ${modelId}`)).catch(onError);
          return;
        }
          state.activeLabTab = "compare";
          setTab("lab");
          showToast(`Compare filter set: ${modelId}`);
        }
        async function addSelectedJobModelsToCompare() {
          const modelIds = selectedJobModelIds();
          if (!modelIds.length) {
            showToast("Selected job has no models yet.");
            return;
          }
          await loadModels();
          const known = modelIds.filter((modelId) => (state.models || []).some((item) => item.model_id === modelId) || (state.productionModels || []).some((item) => item.model_id === modelId));
          if (!known.length) {
            showToast("Models are not available in Compare selector yet.");
            return;
          }
          const firstModel = (state.models || []).find((item) => item.model_id === known[0]) || (state.productionModels || []).find((item) => item.model_id === known[0]);
          if (firstModel?.horizon && el("analysisHorizonFilter")) {
            el("analysisHorizonFilter").value = firstModel.horizon;
          }
          setCompareModelSelection(known, { append: true });
          const filter = el("deepModelFilter");
          if (filter) filter.value = "";
          if (state.activeTab === "lab" && state.activeLabTab === "compare") {
            await Promise.all([loadCompareSummary(), loadCompareScoreboard(), loadDeep()]);
            await loadCompareChart();
            showToast(`Added ${known.length} model(s) to Compare.`);
            return;
          }
          state.activeLabTab = "compare";
          setTab("lab");
          showToast(`Added ${known.length} model(s) to Compare.`);
        }
        async function saveSelectedModelMeta() {
          const modelId = state.lab.selectedModelId;
          if (!modelId) return;
          await api(`/api/models/${encodeURIComponent(modelId)}/metadata`, {
            method: "POST",
          body: {
            notes: el("builderModelNotes").value.trim() || null,
            tags: parseTagText(el("builderModelTags").value),
            parent_model_id: null,
            experiment_id: null,
          },
        });
        showToast("Model metadata saved.");
        await loadLabModels();
      }
      async function runSelectedModelAction(action) {
        const modelId = state.lab.selectedModelId;
        if (!modelId) return;
        if (action === "purge") {
          if (!window.confirm(`Purge model ${modelId}? This removes the registry row.`)) return;
          if (!window.confirm("Confirm purge again.")) return;
          await api(`/api/models/${encodeURIComponent(modelId)}/purge`, { method: "POST", query: { confirm: "true" } });
        } else {
          await api(`/api/models/${encodeURIComponent(modelId)}/${action}`, { method: "POST" });
        }
        showToast(`Model action applied: ${action}`);
        await Promise.all([loadLabModels(), loadModels()]);
      }
        async function downloadExperimentReport(format = "pdf") {
          const experimentId = String(el("builderExperimentId")?.value || "").trim();
          if (!experimentId) {
            showToast("Set experiment_id first.");
            return;
          }
          const cycleIdRaw = String(el("builderReportCycleId")?.value || "").trim();
          const cycleId = cycleIdRaw || null;
          const payload = await api("/api/reports/generate", {
            method: "POST",
            body: {
              experiment_id: experimentId,
              cycle_id: cycleId,
            },
          });
          const formatClean = String(format || "pdf").trim().toLowerCase() === "txt" ? "txt" : "pdf";
          const route = formatClean === "txt"
            ? payload?.report_txt_url || `/api/reports/${encodeURIComponent(experimentId)}/download`
            : payload?.report_pdf_url || `/api/reports/${encodeURIComponent(experimentId)}/download`;
          const query = route.includes("?") ? null : { format: formatClean };
          await apiDownload(route, { query, fallbackName: `${experimentId}_report.${formatClean}` });
          showToast(`Report downloaded: ${formatClean.toUpperCase()}.`);
        }
        async function startBuilderJob(kind) {
          const route = kind === "sweep"
            ? "/api/lab/jobs/sweep"
            : (kind === "wf_eval" ? "/api/lab/jobs/wf_eval" : "/api/lab/jobs/train");
          const jobType = kind === "sweep" ? "sweep_train" : (kind === "wf_eval" ? "wf_eval" : "train_model");
          const validation = await validateBuilder({ showBanner: true, jobType });
          if (!validation?.ok) throw new Error((validation?.errors || []).join(" | ") || "Builder validation failed");
          const response = await api(route, { method: "POST", body: collectBuilderPayload() });
          const count = Number((response.items || []).length || 0);
          state.lab.selectedJobId = response.items?.[0]?.job_id || state.lab.selectedJobId;
          if (kind === "sweep") {
            showToast("Sweep job created.");
          } else if (kind === "wf_eval") {
            showToast("Walk-forward eval job created.");
          } else {
            showToast(count > 1 ? `${count} jobs created.` : "Training job created.");
          }
          await loadLabJobs();
        }
      async function loadLabCreateData() {
        renderBuilderQuantileGrid();
        renderSweepAxisBuilder();
        if (!state.lab.budgetPresetInitialized) {
          const preset = String(el("builderBudgetPreset")?.value || "custom").trim();
          if (preset && preset.toLowerCase() !== "custom") {
            applyTrainingBudgetPreset(preset, { scheduleValidate: false });
          }
          state.lab.budgetPresetInitialized = true;
        }
        syncBuilderModeStates();
        await Promise.all([loadLabJobs(), loadLabModels(), loadModels()]);
        await validateBuilder({ showBanner: false });
      }
      function renderDetailContent() {
        const panel = el("eventDetailsPanel"), content = el("eventDetailsContent"), title = el("eventDetailsTitle"), cancelBtn = el("cancelEventBtn");
        if (!state.selectedEvent) { panel.classList.remove("open"); document.body.classList.remove("details-open"); content.innerHTML = ""; return; }
        panel.classList.add("open");
        document.body.classList.add("details-open");
        const ev = state.selectedEvent, m = ev.metrics || {};
        title.textContent = `Детали события ${ev.event_id}`;
        cancelBtn.disabled = ev.status !== "active";
        const meta = ev.model_meta || {};
        const toRows = (rows) => rows.map(([k, v]) => `<div class="k">${k}</div><div>${v}</div>`).join("");
        const rangeWidthUsd = Math.abs(Number(ev.pred_high) - Number(ev.pred_low));
        const marketRows = [
          ["event_id", esc(ev.event_id)],
          ["status", statusTag(ev.status)],
          ["asset / horizon", `${esc(ev.asset)} / ${esc(ev.horizon)}`],
          ["cycle", esc(cycleText(ev))],
          ["created_at", esc(dt(ev.created_at))],
          ["expires_at", esc(dt(ev.expires_at))],
          ["BTC open (t0)", esc(price(ev.price_t0))],
          ["Прогноз диапазон", `[${esc(price(ev.pred_low))}; ${esc(price(ev.pred_high))}]`],
          ["Ширина диапазона", esc(formatUsd(rangeWidthUsd))],
          ["Медиана прогноза", esc(price(ev.pred_mid))],
          ["Таймер", `<span id="detailTimerValue">${esc(timerText(ev, Date.now()))}</span>`],
          ["Последний тик", `<span id="detailLastTickValue">${esc(detailLastTickText())}</span>`],
          ["note", esc(ev.note || "-")],
        ];
        const modelResultRows = [
          ["Result", resultTag(m.hit)],
          ["Abs Error", esc(formatUsd(m.abs_error))],
          ["Rel Error", esc(pctRaw(m.rel_error))],
          ["Width", esc(pctWithUsd(m.width_pct, rangeWidthUsd))],
          ["Latency (ms)", esc(num(m.latency_ms, 0))],
          ["model_id", esc(ev.model_id || "-")],
          ["git_commit", esc(meta.git_commit || "unknown")],
          ["train_time", esc(meta.train_time || "unknown")],
          ["dataset_hash", esc(meta.dataset_hash || "unknown")],
          ["config_hash", esc(meta.config_hash || "unknown")],
        ];
        content.innerHTML = `<div class="detail-stack">
          <section class="detail-section full">
            <h3 class="detail-section-title">Chart</h3>
            <div class="chart" style="height:300px"><div id="detailTvChart" class="detail-tv-chart"></div><div id="detailChartRange" class="detail-chart-range"></div><div id="detailChartEmpty" class="detail-chart-empty">No market data for this event.</div></div>
          </section>
          <div class="detail-grid">
            <section class="detail-section">
              <h3 class="detail-section-title">Обзор и ход рынка</h3>
              <div class="detail-kv">${toRows(marketRows)}</div>
            </section>
            <section class="detail-section">
              <h3 class="detail-section-title">Результат и модель</h3>
              <div class="detail-kv">${toRows(modelResultRows)}</div>
            </section>
            <section class="detail-section full">
              <details class="detail-accordion">
                <summary>Payload</summary>
                <div class="detail-accordion-body">
                  <div class="detail-accordion-body-inner">
                    <pre class="detail-json">${esc(JSON.stringify(ev.prediction || {}, null, 2))}</pre>
                  </div>
                </div>
              </details>
            </section>
          </div>
        </div>`;
        triggerScanSweep(content);
        updateDetailRangeWidthLabel(ev);
        const seq = ++state.detailChartSeq;
        const eventId = ev.event_id;
        requestAnimationFrame(async () => {
          if (seq !== state.detailChartSeq || !state.selectedEvent || state.selectedEvent.event_id !== eventId) return;
          try {
            if (!state.detailMarketSamples.length) await loadEventMarketSeries(ev);
          } catch {}
          if (seq !== state.detailChartSeq || !state.selectedEvent || state.selectedEvent.event_id !== eventId) return;
          refreshDetailLiveBlock(ev);
          drawDetailTvChart();
          startDetailChartTicker();
        });
      }
      async function loadEventMarketSeries(ev) {
        state.detailMarketSamples = [];
        state.detailLiveSamples = [];
        state.detailScale = null;
        state.detailMarketIntervalSec = 15;
        state.detailWindowStartMs = null;
        state.detailWindowEndMs = null;
        if (!ev || String(ev.asset || "").toUpperCase() !== "BTC") return;
        const symbol = "BTCUSDT";
        const nowMs = Date.now();
        const win = computeDetailWindowMs(ev, nowMs);
        state.detailWindowStartMs = win.startMs;
        state.detailWindowEndMs = win.endMs;
        const startMs = Math.max(0, Math.floor(win.fetchStartMs));
        const endMs = Math.max(startMs, Math.floor(win.fetchEndMs));
        const bucketSec = 15;
        const bucketMs = bucketSec * 1000;
        const sourceInterval = "1s";
        const sourceStepMs = 1000;
        const maxRowsPerReq = 1000;
        const allRows = [];
        let cursorMs = startMs;
        let guard = 0;
        while (cursorMs <= endMs && guard < 64) {
          const reqEndMs = Math.min(endMs, cursorMs + sourceStepMs * (maxRowsPerReq - 1));
          const url = new URL("https://api.binance.com/api/v3/klines");
          url.searchParams.set("symbol", symbol);
          url.searchParams.set("interval", sourceInterval);
          url.searchParams.set("startTime", String(cursorMs));
          url.searchParams.set("endTime", String(reqEndMs));
          url.searchParams.set("limit", String(maxRowsPerReq));
          const response = await fetch(url.toString(), { cache: "no-store" });
          if (!response.ok) throw new Error(`Binance klines HTTP ${response.status}`);
          const rows = await response.json();
          if (!Array.isArray(rows) || !rows.length) break;
          allRows.push(...rows);
          const lastOpenMs = Number(rows[rows.length - 1]?.[0]);
          if (!Number.isFinite(lastOpenMs)) break;
          const nextCursorMs = lastOpenMs + sourceStepMs;
          if (nextCursorMs <= cursorMs) break;
          cursorMs = nextCursorMs;
          guard += 1;
          if (rows.length < maxRowsPerReq && reqEndMs >= endMs) break;
        }
        const sourceSamples = allRows
          .map((row) => {
            const ts = Number(row[0]);
            const open = Number(row[1]);
            const high = Number(row[2]);
            const low = Number(row[3]);
            const close = Number(row[4]);
            if (!Number.isFinite(ts) || !Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) return null;
            return { ts, open, high, low, close };
          })
          .filter(Boolean)
          .filter((row) => row.ts >= startMs && row.ts <= endMs)
          .sort((a, b) => a.ts - b.ts);
        const byBucket = new Map();
        for (let i = 0; i < sourceSamples.length; i++) {
          const src = sourceSamples[i];
          const tSec = Math.floor(src.ts / bucketMs) * bucketSec;
          const existing = byBucket.get(tSec);
          if (!existing) {
            byBucket.set(tSec, {
              time: tSec,
              open: src.open,
              high: src.high,
              low: src.low,
              close: src.close,
            });
            continue;
          }
          existing.high = Math.max(existing.high, src.high);
          existing.low = Math.min(existing.low, src.low);
          existing.close = src.close;
        }
        const firstBucketSec = Math.floor(startMs / bucketMs) * bucketSec;
        const lastBucketSec = Math.floor(endMs / bucketMs) * bucketSec;
        let prevClose = Number(ev.price_t0);
        if (!Number.isFinite(prevClose) && sourceSamples.length) prevClose = sourceSamples[0].close;
        const candles = [];
        for (let tSec = firstBucketSec; tSec <= lastBucketSec; tSec += bucketSec) {
          const row = byBucket.get(tSec);
          if (row) {
            candles.push(row);
            prevClose = row.close;
            continue;
          }
          if (Number.isFinite(prevClose)) {
            candles.push({ time: tSec, open: prevClose, high: prevClose, low: prevClose, close: prevClose });
          }
        }
        state.detailMarketSamples = candles.map((row) => ({ ...row, ts: new Date(row.time * 1000).toISOString(), price: row.close }));
        refreshDetailLiveBlock(ev);
      }
      function upsertDetailLiveSample(tsMs, priceValue) {
        const p = Number(priceValue);
        const t = Math.floor(Number(tsMs) / 1000) * 1000;
        if (!Number.isFinite(p) || !Number.isFinite(t)) return null;
        const iso = new Date(t).toISOString();
        const items = state.detailLiveSamples;
        const last = items.length ? items[items.length - 1] : null;
        if (last && Date.parse(last.ts) === t) {
          last.price = p;
        } else {
          items.push({ ts: iso, price: p });
        }
        if (items.length > 7200) state.detailLiveSamples = items.slice(-7200);
        return { ts: iso, tSec: Math.floor(t / 1000), price: p };
      }
      async function pullDetailLiveTicker() {
        const ev = state.selectedEvent;
        if (!ev || String(ev.asset || "").toUpperCase() !== "BTC") return null;
        const response = await fetch("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", { cache: "no-store" });
        if (!response.ok) throw new Error(`Binance ticker HTTP ${response.status}`);
        const payload = await response.json();
        const px = Number(payload.price);
        if (!Number.isFinite(px)) return null;
        return upsertDetailLiveSample(Date.now(), px);
      }
      function applyPriceToCandles(candles, sampleSec, priceValue, intervalSec) {
        const p = Number(priceValue);
        const rawSec = Math.floor(Number(sampleSec));
        const step = Math.max(1, Math.floor(Number(intervalSec) || 5));
        if (!Number.isFinite(p) || !Number.isFinite(rawSec)) return null;
        const bucket = Math.floor(rawSec / step) * step;
        if (!candles.length) {
          const first = { time: bucket, open: p, high: p, low: p, close: p };
          candles.push(first);
          return { candle: first, inserted: 1 };
        }
        const last = candles[candles.length - 1];
        if (bucket === last.time) {
          last.high = Math.max(Number(last.high), p);
          last.low = Math.min(Number(last.low), p);
          last.close = p;
          return { candle: last, inserted: 0 };
        }
        if (bucket > last.time) {
          const gapBars = Math.floor((bucket - last.time) / step) - 1;
          let prevClose = Number(last.close);
          if (!Number.isFinite(prevClose)) prevClose = p;
          for (let i = 0; i < gapBars; i++) {
            const t = last.time + step * (i + 1);
            candles.push({ time: t, open: prevClose, high: prevClose, low: prevClose, close: prevClose });
          }
          const open = prevClose;
          const appended = { time: bucket, open, high: Math.max(open, p), low: Math.min(open, p), close: p };
          candles.push(appended);
          return { candle: appended, inserted: gapBars + 1 };
        }
        for (let i = candles.length - 1; i >= 0; i--) {
          const row = candles[i];
          if (row.time === bucket) {
            row.high = Math.max(Number(row.high), p);
            row.low = Math.min(Number(row.low), p);
            row.close = p;
            return { candle: row, inserted: 0 };
          }
          if (row.time < bucket) {
            const prevClose = Number(row.close);
            const open = Number.isFinite(prevClose) ? prevClose : p;
            const inserted = { time: bucket, open, high: Math.max(open, p), low: Math.min(open, p), close: p };
            candles.splice(i + 1, 0, inserted);
            return { candle: inserted, inserted: 1 };
          }
        }
        const first = candles[0];
        const open = Number.isFinite(Number(first.open)) ? Number(first.open) : p;
        const inserted = { time: bucket, open, high: Math.max(open, p), low: Math.min(open, p), close: p };
        candles.unshift(inserted);
        return { candle: inserted, inserted: 1 };
      }
      function buildCandlesFromPriceSamples(samples, intervalSec) {
        const candles = [];
        const sorted = [...(samples || [])]
          .map((row) => ({ ts: Date.parse(row.ts), price: Number(row.price) }))
          .filter((row) => Number.isFinite(row.ts) && Number.isFinite(row.price))
          .sort((a, b) => a.ts - b.ts);
        sorted.forEach((row) => applyPriceToCandles(candles, Math.floor(row.ts / 1000), row.price, intervalSec));
        return candles;
      }
      function buildDetailCandles(ev, maxCandles = 12000) {
        const intervalSec = Math.max(1, Math.floor(resolveDetailChartIntervalSec(ev) || 15));
        let candles = [];
        if (state.detailMarketSamples.length) {
          candles = state.detailMarketSamples
            .map((row) => ({
              time: Number(row.time),
              open: Number(row.open),
              high: Number(row.high),
              low: Number(row.low),
              close: Number(row.close),
            }))
            .filter((row) => Number.isFinite(row.time) && Number.isFinite(row.open) && Number.isFinite(row.high) && Number.isFinite(row.low) && Number.isFinite(row.close))
            .sort((a, b) => a.time - b.time);
        } else {
          candles = buildCandlesFromPriceSamples(state.selectedSamples, intervalSec);
        }
        if (state.detailLiveSamples.length) {
          const live = [...state.detailLiveSamples]
            .map((s) => ({ tsSec: Math.floor(Date.parse(s.ts) / 1000), price: Number(s.price) }))
            .filter((s) => Number.isFinite(s.tsSec) && Number.isFinite(s.price))
            .sort((a, b) => a.tsSec - b.tsSec);
          live.forEach((s) => applyPriceToCandles(candles, s.tsSec, s.price, intervalSec));
        }
        if (!ev || !candles.length) return { candles, intervalSec };
        const isActive = ev.status === "active";
        const nowMs = Date.now();
        const win = Number.isFinite(state.detailWindowStartMs) && Number.isFinite(state.detailWindowEndMs)
          ? { startMs: state.detailWindowStartMs, endMs: state.detailWindowEndMs }
          : computeDetailWindowMs(ev, nowMs);
        const rightEdgeMs = isActive ? Math.min(nowMs, win.endMs) : win.endMs;
        const closePx = Number(ev.actual_price ?? ev.current_price);
        if (!isActive && Number.isFinite(closePx) && Number.isFinite(rightEdgeMs)) {
          applyPriceToCandles(candles, Math.floor(rightEdgeMs / 1000), closePx, intervalSec);
        }
        candles = candles.filter((c) => {
          const tsMs = c.time * 1000;
          return tsMs >= win.startMs && tsMs <= (rightEdgeMs + intervalSec * 1000);
        });
        if (candles.length > maxCandles) candles = candles.slice(candles.length - maxCandles);
        return { candles, intervalSec };
      }
      function updateDetailTvWithSample(sample) {
        const runtime = state.detailTv;
        const ev = state.selectedEvent;
        if (!runtime || !ev || !sample) return;
        const intervalSec = Math.max(1, Math.floor(runtime.intervalSec || resolveDetailChartIntervalSec(ev) || 15));
        if (Number.isFinite(state.detailWindowStartMs) && (sample.tSec * 1000) < state.detailWindowStartMs) return;
        if (Number.isFinite(state.detailWindowEndMs) && (sample.tSec * 1000) > state.detailWindowEndMs) return;
        const updated = applyPriceToCandles(runtime.candles, sample.tSec, sample.price, intervalSec);
        if (!updated || !updated.candle) return;
        if (updated.inserted > 1 || runtime.candles.length <= 2) {
          runtime.candleSeries.setData(runtime.candles);
        } else {
          runtime.candleSeries.update(updated.candle);
        }
        setDetailChartEmptyState(false);
        runtime.candleSeries.setMarkers(buildDetailMarkers(ev, runtime.candles));
        updateDetailRangeOverlay(ev, runtime);
        updateDetailChartLegend(ev, runtime.candles);
        refreshDetailLiveBlock(ev);
      }
      function stopDetailChartTicker() {
        if (state.detailChartTimer) clearInterval(state.detailChartTimer);
        state.detailChartTimer = null;
      }
      function startDetailChartTicker() {
        stopDetailChartTicker();
        const activeEventId = state.selectedEvent?.event_id;
        if (!activeEventId) return;
        if (state.selectedEvent.status !== "active") return;
        if (!state.detailMarketSamples.length) {
          loadEventMarketSeries(state.selectedEvent).then(() => {
            if (state.selectedEvent?.event_id === activeEventId) drawDetailTvChart();
          }).catch(() => {});
        }
        const tick = async () => {
          if (!state.selectedEvent || state.selectedEvent.event_id !== activeEventId) return;
          refreshDetailLiveBlock(state.selectedEvent);
          let sample = null;
          try {
            sample = await pullDetailLiveTicker();
          } catch {}
          if (!sample) {
            const fallback = state.detailLiveSamples.length
              ? state.detailLiveSamples[state.detailLiveSamples.length - 1]
              : (state.detailMarketSamples.length
                ? state.detailMarketSamples[state.detailMarketSamples.length - 1]
                : (state.selectedSamples.length ? state.selectedSamples[state.selectedSamples.length - 1] : null));
            if (fallback && Number.isFinite(Number(fallback.price))) sample = upsertDetailLiveSample(Date.now(), Number(fallback.price));
          }
          if (!state.selectedEvent || state.selectedEvent.event_id !== activeEventId) return;
          if (sample && state.detailTv) updateDetailTvWithSample(sample);
          else drawDetailTvChart();
          refreshDetailLiveBlock(state.selectedEvent);
        };
        tick().catch(() => {});
        state.detailChartTimer = setInterval(() => { tick().catch(() => {}); }, 1000);
      }
      function drawDetailTvChart() {
        const ev = state.selectedEvent;
        if (!ev) return;
        refreshDetailLiveBlock(ev);
        updateDetailRangeWidthLabel(ev);
        const runtime = ensureDetailTvChart();
        if (!runtime) return;
        const built = buildDetailCandles(ev);
        runtime.intervalSec = built.intervalSec;
        runtime.candles = built.candles;
        if (!runtime.candles.length) {
          runtime.candleSeries.setData([]);
          setDetailChartEmptyState(true, "No Binance data for this event.");
          setDetailRangeOverlayHidden();
          clearDetailPriceLines(runtime);
          updateDetailChartLegend(ev, []);
          return;
        }
        setDetailChartEmptyState(false);
        runtime.candleSeries.setData(runtime.candles);
        rebuildDetailChartDecorations(ev, runtime.candles);
        updateDetailChartLegend(ev, runtime.candles);
        const nowMs = Date.now();
        const win = Number.isFinite(state.detailWindowStartMs) && Number.isFinite(state.detailWindowEndMs)
          ? { startMs: state.detailWindowStartMs, endMs: state.detailWindowEndMs }
          : computeDetailWindowMs(ev, nowMs);
        try {
          runtime.chart.timeScale().setVisibleRange({
            from: Math.floor(win.startMs / 1000),
            to: Math.floor(win.endMs / 1000),
          });
        } catch {
          runtime.chart.timeScale().fitContent();
        }
        requestAnimationFrame(() => updateDetailRangeOverlay(ev, runtime));
      }
      async function openEvent(eventId, resetTab = false) {
        const id = encodeURIComponent(eventId);
        const [eventPayload, prices] = await Promise.all([api(`/api/events/${id}`), api(`/api/events/${id}/prices`, { query: { limit: 500 } })]);
        stopDetailChartTicker();
        destroyDetailTvChart();
        state.selectedEvent = eventPayload;
        state.selectedSamples = prices.samples || [];
        state.detailMarketSamples = [];
        state.detailLiveSamples = [];
        state.detailScale = null;
        state.detailMarketIntervalSec = intervalFromHorizonSec(eventPayload?.horizon);
        {
          const win = computeDetailWindowMs(eventPayload, Date.now());
          state.detailWindowStartMs = win.startMs;
          state.detailWindowEndMs = win.endMs;
        }
        state.detailChartSeq += 1;
        if (!resetTab) {
          try { await loadEventMarketSeries(eventPayload); } catch {}
        }
        renderDetailContent();
      }
      function closeEvent() { stopDetailChartTicker(); destroyDetailTvChart(); state.selectedEvent = null; state.selectedSamples = []; state.detailMarketSamples = []; state.detailLiveSamples = []; state.detailScale = null; state.detailMarketIntervalSec = 15; state.detailWindowStartMs = null; state.detailWindowEndMs = null; state.detailChartSeq += 1; renderDetailContent(); }
      function drawAnalysisChart(series) {
        const canvas = el("analysisChart"), ctx = canvas.getContext("2d"), w = canvas.clientWidth || 900, h = canvas.clientHeight || 280, dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h);
        if (!series.length) { ctx.fillStyle = "#8a968e"; ctx.font = "12px IBM Plex Mono"; ctx.fillText("Нет данных для выбранного периода.", 16, 24); return; }
        const pad = 36, pw = w - pad * 2, ph = h - pad * 2, maxWidth = Math.max(1, ...series.map((s) => Number(s.width_pct || 0)));
        const x = (i) => pad + (pw * i) / Math.max(1, series.length - 1), yCov = (v) => pad + (1 - Math.max(0, Math.min(1, Number(v || 0)))) * ph, yWidth = (v) => pad + (1 - Math.max(0, Math.min(1, Number(v || 0) / maxWidth))) * ph;
        ctx.strokeStyle = "rgba(148,148,148,.24)"; for (let i = 0; i <= 4; i++) { const yy = pad + (ph * i) / 4; ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(w - pad, yy); ctx.stroke(); }
        ctx.strokeStyle = "#00ff66"; ctx.lineWidth = 2; ctx.beginPath(); series.forEach((s, i) => { const xx = x(i), yy = yCov(s.coverage); if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); }); ctx.stroke();
        ctx.strokeStyle = "#66ffaa"; ctx.lineWidth = 2; ctx.beginPath(); series.forEach((s, i) => { const xx = x(i), yy = yWidth(s.width_pct); if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); }); ctx.stroke();
      }
      function renderByHorizon(map) {
        const body = el("horizonBreakdownBody"), rows = Object.entries(map || {});
        if (!rows.length) { body.innerHTML = `<tr><td colspan="5" class="mut">Нет данных.</td></tr>`; return; }
        body.innerHTML = rows.sort((a, b) => HORIZONS.indexOf(a[0]) - HORIZONS.indexOf(b[0])).map(([h, d]) => `<tr><td>${esc(h)}</td><td>${esc(num(d.count, 0))}</td><td>${esc(pct(d.coverage))}</td><td>${esc(pctRaw(d.avg_width_pct))}</td><td>${esc(price(d.avg_abs_error))}</td></tr>`).join("");
      }
      function renderSummary(summary) {
        if (!summary) { ["kpiCoverage", "kpiCalibration", "kpiBestHorizon", "kpiCompleted", "kpiActive", "kpiFreshness"].forEach((id) => { setNodeText(id, "-"); }); renderByHorizon({}); drawAnalysisChart([]); return; }
        setNodeText("kpiCoverage", `${pct(summary.actual_coverage)} / ${pct(summary.target_coverage)}`, { flash: true });
        const delta = Number(summary.coverage_delta || 0);
        setNodeText("kpiCalibration", delta < -0.03 ? "ниже target" : delta > 0.03 ? "выше target" : "в норме", { flash: true });
        setNodeText("kpiBestHorizon", summary.best_horizon || "-", { flash: true });
        setNodeText("kpiCompleted", num(summary.completed_events, 0), { flash: true });
        setNodeText("kpiActive", num(summary.active_events, 0), { flash: true });
        const fresh = Number(state.statusInfo?.data_freshness_seconds ?? NaN);
        setNodeText("kpiFreshness", Number.isFinite(fresh) ? `${fresh}s` : "-", { flash: true });
        renderByHorizon(summary.by_horizon || {});
        drawAnalysisChart(summary.series || []);
        triggerScanSweep([el("analysisChart")?.closest(".chart"), el("kpiCoverage")?.closest(".panel")]);
      }
      async function loadSummary() {
        const days = Number(el("analysisDaysFilter").value || "30"), horizon = el("analysisHorizonFilter").value, cycle_id = el("analysisCycleFilter").value.trim();
        state.analysis.summary = await api("/api/metrics/summary", { query: { days, horizon, cycle_id } });
        renderSummary(state.analysis.summary);
      }
      function collectDeep() {
        const from = el("deepFromFilter").value, to = el("deepToFilter").value;
        return { status: el("deepStatusFilter").value, horizon: el("deepHorizonFilter").value, cycle_id: el("deepCycleFilter").value.trim(), result: el("deepResultFilter").value, model_id: el("deepModelFilter").value.trim(), created_from: from ? new Date(from).toISOString() : "", created_to: to ? new Date(to).toISOString() : "", q: el("deepSearchFilter").value.trim(), page: 1, page_size: Number(el("deepPageSizeFilter").value || "300"), sort_by: "created_at", sort_dir: "desc" };
      }
      function drawCompareChart(items) {
        const canvas = el("analysisChart"), ctx = canvas.getContext("2d"), w = canvas.clientWidth || 900, h = canvas.clientHeight || 280, dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h);
        const rows = Array.isArray(items) ? items.filter(Boolean) : [];
        if (!rows.length) { ctx.fillStyle = "#8a968e"; ctx.font = "12px IBM Plex Mono"; ctx.fillText("No compare data for the selected filter.", 16, 24); return; }
        const xMetric = el("compareScatterXMetric")?.value || "coverage";
        const yMetric = el("compareScatterYMetric")?.value || "avg_abs_error";
        const metricValue = (item, metric) => {
          const raw = item?.[metric];
          const numeric = Number(raw);
          return Number.isFinite(numeric) ? numeric : null;
        };
        const points = rows
          .map((item) => ({ model_id: item.model_id, x: metricValue(item, xMetric), y: metricValue(item, yMetric) }))
          .filter((item) => item.x !== null && item.y !== null);
        if (!points.length) { ctx.fillStyle = "#8a968e"; ctx.font = "12px IBM Plex Mono"; ctx.fillText("No numeric points for scatter.", 16, 24); return; }
        const pad = 42, pw = w - pad * 2, ph = h - pad * 2;
        const minX = Math.min(...points.map((item) => Number(item.x)));
        const maxX = Math.max(...points.map((item) => Number(item.x)));
        const minY = Math.min(...points.map((item) => Number(item.y)));
        const maxY = Math.max(...points.map((item) => Number(item.y)));
        const scaleX = (value) => pad + ((Number(value) - minX) / Math.max(1e-9, maxX - minX || 1)) * pw;
        const scaleY = (value) => pad + ph - ((Number(value) - minY) / Math.max(1e-9, maxY - minY || 1)) * ph;
        ctx.strokeStyle = "rgba(148,148,148,.24)";
        for (let i = 0; i <= 4; i++) {
          const xx = pad + (pw * i) / 4;
          const yy = pad + (ph * i) / 4;
          ctx.beginPath(); ctx.moveTo(xx, pad); ctx.lineTo(xx, h - pad); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(w - pad, yy); ctx.stroke();
        }
        ctx.fillStyle = "#9eb8a8";
        ctx.font = "11px IBM Plex Mono";
        ctx.fillText(`X: ${xMetric}`, pad, 18);
        ctx.fillText(`Y: ${yMetric}`, w - pad - 90, 18);
        points.forEach((point) => {
          const px = scaleX(point.x);
          const py = scaleY(point.y);
          ctx.fillStyle = "rgba(0,255,102,.8)";
          ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#dfffea";
          ctx.font = "10px IBM Plex Mono";
          ctx.fillText(String(point.model_id || "-").slice(0, 10), px + 7, py - 7);
        });
      }
      function renderCompareRows(scoreboard) {
        const head = el("compareScoreboardHead");
        const body = el("horizonBreakdownBody");
        const payload = scoreboard || state.analysis.scoreboard || {};
        const rows = Array.isArray(payload.items) ? payload.items : [];
        const dynamicColumns = Array.isArray(payload.columns) && payload.columns.length ? payload.columns : ["completed_events", "coverage", "avg_width_pct", "avg_abs_error"];
        const labelMap = {
          asset: "Asset",
          horizon: "Horizon",
          base_timeframe: "Base TF",
          steps_ahead: "Steps",
          quantiles_count: "Q Count",
          target_coverage_effective: "Eff Coverage",
          group_key: "Group",
          status: "Status",
          completed_events: "Events",
          coverage: "Coverage",
          avg_width_pct: "Avg Width %",
          avg_abs_error: "Avg Abs Error",
          avg_rel_error: "Avg Rel Error",
          avg_interval_score: "Avg Interval Score",
          avg_wis: "Avg WIS",
          matched_groups: "Groups",
          group_wins: "Wins",
          delta_coverage: "Δ Coverage",
          delta_avg_width_pct: "Δ Width",
          delta_avg_abs_error: "Δ Abs",
          delta_avg_rel_error: "Δ Rel",
          delta_avg_interval_score: "Δ Interval Score",
          delta_avg_wis: "Δ WIS",
        };
        const baseline = String(payload.baseline_model_id || "").trim();
        const visibleColumns = [...dynamicColumns];
        if (baseline) {
          if (dynamicColumns.includes("coverage")) visibleColumns.push("delta_coverage");
          if (dynamicColumns.includes("avg_width_pct")) visibleColumns.push("delta_avg_width_pct");
          if (dynamicColumns.includes("avg_abs_error")) visibleColumns.push("delta_avg_abs_error");
          if (dynamicColumns.includes("avg_rel_error")) visibleColumns.push("delta_avg_rel_error");
          if (dynamicColumns.includes("avg_interval_score")) visibleColumns.push("delta_avg_interval_score");
          if (dynamicColumns.includes("avg_wis")) visibleColumns.push("delta_avg_wis");
        }
        if (head) {
          head.innerHTML = `<tr><th>Model</th>${visibleColumns.map((col) => `<th>${esc(labelMap[col] || col)}</th>`).join("")}<th>Open</th></tr>`;
        }
        const formatCell = (item, col) => {
          const value = item?.[col];
          if (col === "coverage" || col === "delta_coverage" || col === "target_coverage_effective") return value === null || value === undefined ? "-" : pct(value);
          if (col === "avg_width_pct" || col === "delta_avg_width_pct" || col === "avg_rel_error" || col === "delta_avg_rel_error") return value === null || value === undefined ? "-" : pctRaw(value);
          if (col === "avg_abs_error" || col === "delta_avg_abs_error" || col === "avg_interval_score" || col === "delta_avg_interval_score" || col === "avg_wis" || col === "delta_avg_wis") return value === null || value === undefined ? "-" : price(value);
          if (col === "completed_events" || col === "matched_groups" || col === "group_wins" || col === "steps_ahead" || col === "quantiles_count") return num(value, 0);
          if (col === "status") return esc(value || "-");
          return esc(value || "-");
        };
        if (!rows.length) { body.innerHTML = `<tr><td colspan="${visibleColumns.length + 2}" class="mut">No compare rows.</td></tr>`; return; }
        body.innerHTML = rows.map((item) => `<tr class="row-clickable" data-model-id="${esc(item.model_id)}"><td>${esc(item.model_id)}</td>${visibleColumns.map((col) => `<td>${formatCell(item, col)}</td>`).join("")}<td><button type="button" data-open-model="${esc(item.model_id)}">Events</button></td></tr>`).join("");
      }
      function renderCompareGroups(groups) {
        const body = el("compareGroupBreakdownBody");
        if (!body) return;
        const rows = Array.isArray(groups) ? groups : [];
        if (!rows.length) { body.innerHTML = `<tr><td colspan="7" class="mut">No matched groups.</td></tr>`; return; }
        body.innerHTML = rows.map((item) => `<tr><td>${esc(item.key)}</td><td>${esc(dt(item.created_at))}</td><td>${esc(item.cycle_id || "-")}</td><td>${esc((item.model_ids || []).join(", "))}</td><td>${esc(item.best_model_id || "-")}</td><td>${esc(item.best_abs_error === null || item.best_abs_error === undefined ? "-" : price(item.best_abs_error))}</td><td>${esc((item.hit_models || []).join(", ") || "-")}</td></tr>`).join("");
      }
      function renderCompareSummary(summary) {
        const hint = el("compareSummaryHint");
        if (!summary) {
          ["kpiCoverage", "kpiCalibration", "kpiBestHorizon", "kpiCompleted", "kpiActive", "kpiFreshness"].forEach((id) => { setNodeText(id, "-"); });
          if (hint) hint.textContent = "";
          renderCompareGroups([]);
          return;
        }
        const items = Array.isArray(summary.items) ? summary.items : [];
        const best = items.length ? items[0] : null;
        const avgCoverage = items.length ? items.reduce((sum, item) => sum + Number(item.coverage || 0), 0) / items.length : null;
        setNodeText("kpiCoverage", `${num((summary.compared_model_ids || []).length, 0)} models`, { flash: true });
        setNodeText("kpiCalibration", String(summary.mode || "simple").toUpperCase(), { flash: true });
        setNodeText("kpiBestHorizon", best?.model_id || "-", { flash: true });
        setNodeText("kpiCompleted", num(summary.total_completed, 0), { flash: true });
        setNodeText("kpiActive", num(summary.total_groups, 0), { flash: true });
        setNodeText("kpiFreshness", avgCoverage === null ? "-" : pct(avgCoverage), { flash: true });
        if (hint) {
          hint.innerHTML = [
            `<span>Mode: ${esc(String(summary.mode || "simple").toUpperCase())}</span>`,
            `<span>Models: ${esc((summary.compared_model_ids || []).join(", ") || "auto")}</span>`,
            `<span>Events: ${esc(num(summary.total_events, 0))}</span>`,
            `<span>Groups: ${esc(num(summary.total_groups, 0))}</span>`,
          ].join("");
        }
        renderCompareGroups(summary.groups || []);
        triggerScanSweep([el("compareGroupBreakdownBody")?.closest(".panel"), el("kpiCoverage")?.closest(".panel")]);
      }
      function renderTournamentSummaryJson(summary) {
        const node = el("tournamentSummaryJson");
        if (!node) return;
        if (!summary) {
          node.textContent = "Tournament summary will appear here.";
          return;
        }
        node.textContent = JSON.stringify(summary, null, 2);
        triggerScanSweep(node.closest(".panel"));
      }
      async function loadTournamentCycleSummary() {
        const cycleId = String(el("analysisCycleFilter")?.value || "").trim();
        if (!cycleId) {
          showToast("Set Cycle ID first.");
          return null;
        }
        const payload = await api(`/api/cycles/${encodeURIComponent(cycleId)}/summary`);
        state.analysis.tournamentSummary = payload;
        renderTournamentSummaryJson(payload);
        const hint = el("compareSummaryHint");
        if (hint) {
          const recommendation = String(payload.recommendation || "continue").toUpperCase();
          const winner = payload.winner_model_id ? String(payload.winner_model_id) : "-";
          const reason = payload.stop_reason ? String(payload.stop_reason) : "-";
          hint.innerHTML = [
            `<span>Tournament: ${esc(recommendation)}</span>`,
            `<span>Winner: ${esc(winner)}</span>`,
            `<span>Paired runs: ${esc(num(payload.paired_runs, 0))}</span>`,
            `<span>Reason: ${esc(reason)}</span>`,
          ].join("");
        }
        return payload;
      }
      async function loadCompareSummary() {
        const days = Number(el("analysisDaysFilter").value || "30");
        const horizon = el("analysisHorizonFilter").value;
        const cycle_id = el("analysisCycleFilter").value.trim();
        const mode = el("compareModeSelect")?.value || "simple";
        const model_ids = selectedCompareModelIds().join(",");
        const exclude_stale = el("compareExcludeStale")?.checked ? "true" : "";
        state.analysis.summary = await api("/api/metrics/compare", { query: { days, asset: "BTC", horizon, cycle_id, mode, model_ids, exclude_stale } });
        renderCompareSummary(state.analysis.summary);
      }
      async function loadCompareScoreboard() {
        const query = compareQueryBase();
        const baseline_model_id = el("compareBaselineSelect")?.value || "";
        const columns = selectedCompareColumns().join(",");
        state.analysis.scoreboard = await api("/api/compare/scoreboard", { query: { ...query, baseline_model_id, columns } });
        renderCompareRows(state.analysis.scoreboard);
        return state.analysis.scoreboard;
      }
      function drawCompareLineSeries(payload) {
        const canvas = el("analysisChart"), ctx = canvas.getContext("2d"), w = canvas.clientWidth || 900, h = canvas.clientHeight || 280, dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h);
        const series = Array.isArray(payload?.series) ? payload.series.filter((item) => Array.isArray(item.points) && item.points.length) : [];
        if (!series.length) { ctx.fillStyle = "#8a968e"; ctx.font = "12px IBM Plex Mono"; ctx.fillText("No time-series points.", 16, 24); return; }
        const pad = 36, pw = w - pad * 2, ph = h - pad * 2;
        const flat = series.flatMap((item) => item.points.map((point) => ({ x: Date.parse(`${point.date}T00:00:00Z`), y: Number(point.value) }))).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
        const minX = Math.min(...flat.map((point) => point.x));
        const maxX = Math.max(...flat.map((point) => point.x));
        const minY = Math.min(...flat.map((point) => point.y));
        const maxY = Math.max(...flat.map((point) => point.y));
        const scaleX = (value) => pad + ((value - minX) / Math.max(1e-9, maxX - minX || 1)) * pw;
        const scaleY = (value) => pad + ph - ((value - minY) / Math.max(1e-9, maxY - minY || 1)) * ph;
        ctx.strokeStyle = "rgba(148,148,148,.24)";
        for (let i = 0; i <= 4; i++) { const yy = pad + (ph * i) / 4; ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(w - pad, yy); ctx.stroke(); }
        const palette = ["#00ff66", "#66ffaa", "#ffcc66", "#66ccff", "#ff7ab6", "#b6ff66"];
        series.forEach((item, index) => {
          ctx.strokeStyle = palette[index % palette.length];
          ctx.lineWidth = 2;
          ctx.beginPath();
          item.points.forEach((point, pointIndex) => {
            const xx = scaleX(Date.parse(`${point.date}T00:00:00Z`));
            const yy = scaleY(Number(point.value));
            if (pointIndex === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
          });
          ctx.stroke();
        });
      }
      function drawCompareHistogram(payload) {
        const canvas = el("analysisChart"), ctx = canvas.getContext("2d"), w = canvas.clientWidth || 900, h = canvas.clientHeight || 280, dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h);
        const series = Array.isArray(payload?.series) ? payload.series.filter((item) => Array.isArray(item.bins) && item.bins.length) : [];
        const bins = series.length ? series[0].bins : [];
        if (!bins.length) { ctx.fillStyle = "#8a968e"; ctx.font = "12px IBM Plex Mono"; ctx.fillText("No histogram bins.", 16, 24); return; }
        const pad = 36, pw = w - pad * 2, ph = h - pad * 2;
        const maxCount = Math.max(1, ...series.flatMap((item) => item.bins.map((bin) => Number(bin.count || 0))));
        const slot = pw / Math.max(1, bins.length);
        const palette = ["rgba(0,255,102,.68)", "rgba(102,255,170,.48)", "rgba(255,204,102,.48)", "rgba(102,204,255,.48)"];
        ctx.strokeStyle = "rgba(148,148,148,.24)";
        for (let i = 0; i <= 4; i++) { const yy = pad + (ph * i) / 4; ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(w - pad, yy); ctx.stroke(); }
        bins.forEach((_, index) => {
          const subSlot = Math.max(2, (slot - 4) / Math.max(1, series.length));
          series.forEach((item, seriesIndex) => {
            const bin = item.bins[index];
            const barH = (Number(bin?.count || 0) / maxCount) * ph;
            const x = pad + slot * index + 2 + subSlot * seriesIndex;
            ctx.fillStyle = palette[seriesIndex % palette.length];
            ctx.fillRect(x, pad + ph - barH, Math.max(2, subSlot - 2), barH);
          });
        });
      }
      async function loadCompareChart() {
        const chartType = el("compareChartTypeSelect")?.value || "scatter";
        if (chartType === "scatter") {
          if (!state.analysis.scoreboard) await loadCompareScoreboard();
          drawCompareChart(state.analysis.scoreboard?.items || []);
          return;
        }
        const query = compareQueryBase();
        if (chartType === "series") {
          const metric = el("compareSeriesMetric")?.value || "coverage";
          const rolling_days = Number(el("compareRollingDays")?.value || "7");
          state.analysis.chartSeries = await api("/api/compare/series", { query: { ...query, metric, rolling_days } });
          drawCompareLineSeries(state.analysis.chartSeries);
          return;
        }
        const metric = el("compareHistMetric")?.value || "abs_error";
        const bins = Number(el("compareHistBins")?.value || "10");
        state.analysis.histSeries = await api("/api/compare/hist", { query: { ...query, metric, bins } });
        drawCompareHistogram(state.analysis.histSeries);
      }
      function collectDeepFilters() {
        const from = el("deepFromFilter").value, to = el("deepToFilter").value;
        const explicitModelId = el("deepModelFilter").value.trim();
        const compareModelIds = explicitModelId ? [] : selectedCompareModelIds();
        return { status: el("deepStatusFilter").value, horizon: el("deepHorizonFilter").value, cycle_id: el("deepCycleFilter").value.trim(), result: el("deepResultFilter").value, model_id: explicitModelId, model_ids: compareModelIds.join(","), exclude_stale: el("compareExcludeStale")?.checked ? "true" : "", created_from: from ? new Date(from).toISOString() : "", created_to: to ? new Date(to).toISOString() : "", q: el("deepSearchFilter").value.trim(), page: 1, page_size: Number(el("deepPageSizeFilter").value || "300"), sort_by: "created_at", sort_dir: "desc" };
      }
      function renderDeep() {
        const items = state.analysis.deepEvents || [], body = el("deepEventsBody");
        if (!items.length) body.innerHTML = `<tr><td colspan="10" class="mut">Нет данных для выбранного фильтра.</td></tr>`;
        else body.innerHTML = items.map((item) => { const m = item.metrics || {}; return `<tr class="row-clickable" data-event-id="${esc(item.event_id)}"><td>${esc(item.event_id)}</td><td>${esc(cycleText(item))}</td><td>${statusTag(item.status)}</td><td>${esc(item.horizon)}</td><td>${esc(dt(item.created_at))}</td><td>${esc(price(item.price_t0))}</td><td>${resultTag(m.hit)}</td><td>${esc(price(m.abs_error))}</td><td>${esc(pctRaw(m.rel_error))}</td><td>${esc(pctRaw(m.width_pct))}</td></tr>`; }).join("");
        const completed = items.filter((it) => it.status === "completed"), withResult = completed.filter((it) => it.metrics && typeof it.metrics.hit === "boolean"), hits = withResult.filter((it) => it.metrics.hit).length;
        const hitRate = withResult.length ? hits / withResult.length : null;
        const avgW = withResult.length ? withResult.reduce((s, it) => s + Number(it.metrics.width_pct || 0), 0) / withResult.length : null;
        const avgA = withResult.length ? withResult.reduce((s, it) => s + Number(it.metrics.abs_error || 0), 0) / withResult.length : null;
        el("deepStats").innerHTML = [`<span>Загружено: ${num(items.length, 0)} / ${num(state.analysis.deepTotal, 0)}</span>`, `<span>Completed: ${num(completed.length, 0)}</span>`, `<span>Hit rate: ${hitRate === null ? "-" : pct(hitRate)}</span>`, `<span>Avg width %: ${avgW === null ? "-" : pctRaw(avgW)}</span>`, `<span>Avg abs error: ${avgA === null ? "-" : price(avgA)}</span>`].join("");
        triggerScanSweep(body.closest(".panel"));
      }
      async function loadDeep() {
        const payload = await api("/api/events", { query: collectDeepFilters() });
        state.analysis.deepEvents = payload.items || [];
        state.analysis.deepTotal = Number(payload.total || 0);
        renderDeep();
      }
      async function loadAnalysis() { await Promise.all([loadStatus(), loadCompareSummary(), loadCompareScoreboard(), loadDeep()]); await loadCompareChart(); }
      function setLabTab(tab) {
        const next = tab === "create" ? "create" : "compare";
        state.activeLabTab = next;
        document.querySelectorAll(".subtab-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.labTab === next));
        el("labCreatePane").classList.toggle("active", next === "create");
        el("labComparePane").classList.toggle("active", next === "compare");
        if (state.activeTab !== "lab") return;
        if (next === "compare") {
          loadAnalysis().then(() => showBanner("")).catch(onError);
          return;
        }
        loadLabCreateData().then(() => showBanner("")).catch(onError);
      }
      function setTab(tab) {
        if (tab !== "events" && state.selectedEvent) closeEvent();
        state.activeTab = tab;
        document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tab));
        const map = { events: "paneEvents", markets: "paneMarkets", lab: "paneLab", settings: "paneSettings" };
        Object.entries(map).forEach(([name, id]) => el(id).classList.toggle("active", name === tab));
        if (tab === "events") Promise.all([loadStatus(), loadQuote(), loadEvents()]).then(() => showBanner("")).catch(onError);
        if (tab === "markets") Promise.all([loadStatus(), loadMarkets()]).then(() => showBanner("")).catch(onError);
        if (tab === "lab") setLabTab(state.activeLabTab);
      }
      function download(name, content, type) {
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = name;
        a.click();
        URL.revokeObjectURL(url);
      }
      function exportEventsJson() { download("events_export.json", JSON.stringify(state.events.items || [], null, 2), "application/json"); }
      function exportEventsCsv() {
        const header = ["event_id", "cycle_id", "cycle_seq", "cycle_total", "status", "asset", "horizon", "created_at", "expires_at", "price_t0", "pred_low", "pred_mid", "pred_high", "actual_price", "current_price", "hit", "abs_error", "rel_error", "width_pct", "model_id", "note"];
        const row = (arr) => arr.map((v) => { const s = String(v ?? ""); return /[,"\n]/.test(s) ? `"${s.replaceAll("\"", "\"\"")}"` : s; }).join(",");
        const rows = (state.events.items || []).map((it) => { const m = it.metrics || {}; return [it.event_id, it.cycle_id, it.cycle_seq, it.cycle_total, it.status, it.asset, it.horizon, it.created_at, it.expires_at, it.price_t0, it.pred_low, it.pred_mid, it.pred_high, it.actual_price, it.current_price, m.hit, m.abs_error, m.rel_error, m.width_pct, it.model_id, it.note || ""]; });
        download("events_export.csv", [row(header), ...rows.map((r) => row(r))].join("\n"), "text/csv;charset=utf-8");
      }
      function exportChartPng() { const a = document.createElement("a"); a.href = el("analysisChart").toDataURL("image/png"); a.download = "analysis_chart.png"; a.click(); }
      function exportPdf() {
        if (!state.analysis.summary) { showToast("Сначала загрузите анализ."); return; }
        const s = state.analysis.summary;
        const rows = (state.analysis.deepEvents || []).slice(0, 120).map((it) => { const m = it.metrics || {}; return `<tr><td>${esc(it.event_id)}</td><td>${esc(it.status)}</td><td>${esc(it.horizon)}</td><td>${esc(dt(it.created_at))}</td><td>${esc(price(it.price_t0))}</td><td>${esc(typeof m.hit === "boolean" ? (m.hit ? "hit" : "miss") : "-")}</td><td>${esc(price(m.abs_error))}</td><td>${esc(pctRaw(m.width_pct))}</td></tr>`; }).join("");
        const html = `<!doctype html><html><head><meta charset="utf-8"><title>UlyanAI Report</title><style>body{font-family:Arial,sans-serif;padding:16px;color:#111}table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}th,td{border:1px solid #ccc;padding:6px;text-align:left}th{background:#f2f2f2}.meta{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:6px;margin-top:8px;font-size:13px}</style></head><body><h1>UlyanAI Analysis Report</h1><div>Generated at ${new Date().toISOString()}</div><div class="meta"><div>Coverage: ${pct(s.actual_coverage)} (target ${pct(s.target_coverage)})</div><div>Best horizon: ${esc(s.best_horizon || "-")}</div><div>Completed events: ${num(s.completed_events, 0)}</div><div>Active events: ${num(s.active_events, 0)}</div><div>Avg width %: ${pctRaw(s.avg_width_pct)}</div><div>Avg abs error: ${price(s.avg_abs_error)}</div></div><h2>Deep Events (first 120)</h2><table><thead><tr><th>event_id</th><th>status</th><th>horizon</th><th>created_at</th><th>btc_open</th><th>result</th><th>abs_error</th><th>width%</th></tr></thead><tbody>${rows || "<tr><td colspan='8'>No data</td></tr>"}</tbody></table></body></html>`;
        const w = window.open("", "_blank");
        if (!w) { showToast("Разрешите всплывающие окна для Export PDF."); return; }
        w.document.open(); w.document.write(html); w.document.close(); w.focus(); w.print();
      }
      function exportComparePdf() {
        if (!state.analysis.summary) { showToast("Load compare first."); return; }
        const s = state.analysis.summary;
        const items = (s.items || []).slice(0, 100).map((item) => `<tr><td>${esc(item.model_id)}</td><td>${esc(num(item.completed_events, 0))}</td><td>${esc(item.coverage === null || item.coverage === undefined ? "-" : pct(item.coverage))}</td><td>${esc(item.avg_width_pct === null || item.avg_width_pct === undefined ? "-" : pctRaw(item.avg_width_pct))}</td><td>${esc(item.avg_abs_error === null || item.avg_abs_error === undefined ? "-" : price(item.avg_abs_error))}</td><td>${esc(num(item.group_wins, 0))}</td></tr>`).join("");
        const groups = (s.groups || []).slice(0, 100).map((item) => `<tr><td>${esc(item.key)}</td><td>${esc(dt(item.created_at))}</td><td>${esc(item.best_model_id || "-")}</td><td>${esc(item.best_abs_error === null || item.best_abs_error === undefined ? "-" : price(item.best_abs_error))}</td></tr>`).join("");
        const html = `<!doctype html><html><head><meta charset="utf-8"><title>UlyanAI Compare Report</title><style>body{font-family:Arial,sans-serif;padding:16px;color:#111}table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}th,td{border:1px solid #ccc;padding:6px;text-align:left}th{background:#f2f2f2}.meta{display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:6px;margin-top:8px;font-size:13px}</style></head><body><h1>UlyanAI Compare Desk Report</h1><div>Generated at ${new Date().toISOString()}</div><div class="meta"><div>Mode: ${esc(String(s.mode || "simple").toUpperCase())}</div><div>Cycle ID: ${esc(s.cycle_id || "-")}</div><div>Models: ${esc((s.compared_model_ids || []).join(", ") || "auto")}</div><div>Completed events: ${esc(num(s.total_completed, 0))}</div><div>Groups: ${esc(num(s.total_groups, 0))}</div><div>Best model: ${esc(s.items?.[0]?.model_id || "-")}</div></div><h2>Model Breakdown</h2><table><thead><tr><th>Model</th><th>Events</th><th>Coverage</th><th>Avg Width %</th><th>Avg Abs Error</th><th>Group Wins</th></tr></thead><tbody>${items || "<tr><td colspan='6'>No data</td></tr>"}</tbody></table><h2>Matched Groups</h2><table><thead><tr><th>Key</th><th>Created</th><th>Winner</th><th>Best Abs Error</th></tr></thead><tbody>${groups || "<tr><td colspan='4'>No data</td></tr>"}</tbody></table></body></html>`;
        const w = window.open("", "_blank");
        if (!w) { showToast("Allow popups to export PDF."); return; }
        w.document.open(); w.document.write(html); w.document.close(); w.focus(); w.print();
      }
      function exportCompareScoreboardJson() {
        if (!state.analysis.scoreboard) { showToast("Load compare scoreboard first."); return; }
        download("compare_scoreboard.json", JSON.stringify(state.analysis.scoreboard, null, 2), "application/json");
      }
      function exportCompareScoreboardCsv() {
        const payload = state.analysis.scoreboard;
        if (!payload) { showToast("Load compare scoreboard first."); return; }
        const columns = ["model_id", ...(payload.columns || [])];
        const baseline = String(payload.baseline_model_id || "").trim();
        if (baseline) {
          if (columns.includes("coverage") && !columns.includes("delta_coverage")) columns.push("delta_coverage");
          if (columns.includes("avg_width_pct") && !columns.includes("delta_avg_width_pct")) columns.push("delta_avg_width_pct");
          if (columns.includes("avg_abs_error") && !columns.includes("delta_avg_abs_error")) columns.push("delta_avg_abs_error");
          if (columns.includes("avg_rel_error") && !columns.includes("delta_avg_rel_error")) columns.push("delta_avg_rel_error");
          if (columns.includes("avg_interval_score") && !columns.includes("delta_avg_interval_score")) columns.push("delta_avg_interval_score");
          if (columns.includes("avg_wis") && !columns.includes("delta_avg_wis")) columns.push("delta_avg_wis");
        }
        const quoteCsv = (value) => {
          const text = String(value ?? "");
          return /[,"\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
        };
        const rows = (payload.items || []).map((item) => columns.map((column) => item?.[column]));
        download("compare_scoreboard.csv", [columns.map(quoteCsv).join(","), ...rows.map((row) => row.map(quoteCsv).join(","))].join("\n"), "text/csv;charset=utf-8");
      }
      function restartPolling() {
        if (state.pollTimer) clearInterval(state.pollTimer);
        state.pollTimer = null;
        if (state.settings.autoRefresh === "off") return;
        const ms = Math.max(3, Number(state.settings.refreshSec || 5)) * 1000;
        state.pollTimer = setInterval(() => refreshView().catch(onError), ms);
      }
      function ensureTicker() { if (!state.tickerTimer) state.tickerTimer = setInterval(updateTimers, 1000); }
      function restartQuoteTicker() {
        if (state.quoteLiveTimer) clearInterval(state.quoteLiveTimer);
        state.quoteLiveTimer = setInterval(async () => {
          if (state.activeTab !== "events") return;
          if (state.selectedEvent) return;
          if (state.quoteFetchInFlight) return;
          state.quoteFetchInFlight = true;
          try {
            await loadQuote({ allowFallback: false });
          } catch {
            } finally {
            state.quoteFetchInFlight = false;
          }
        }, 1000);
      }
      async function refreshView() {
        if (state.refreshInFlight) return;
        state.refreshInFlight = true;
        try {
          await loadStatus();
          if (state.activeTab === "events") {
            if (state.selectedEvent) {
              showBanner("");
              return;
            }
            await loadQuote();
            await loadEvents();
          }
          if (state.activeTab === "markets") await loadMarkets();
          if (state.activeTab === "lab" && state.activeLabTab === "compare") { await loadCompareSummary(); await loadCompareScoreboard(); await loadDeep(); await loadCompareChart(); }
          if (state.activeTab === "lab" && state.activeLabTab === "create") { await Promise.all([loadLabJobs(), loadLabModels(), loadModels()]); }
          showBanner("");
        } finally {
          state.refreshInFlight = false;
        }
      }
      function onError(error) { setConnection(false, "Disconnected"); showBanner(`Ошибка: ${error instanceof Error ? error.message : String(error)}`); }
      function openModal() { renderCreateModelOptions(); syncCreateModeUi(); el("createEventModal").classList.add("open"); }
      function closeModal() { el("createEventModal").classList.remove("open"); }
      function applyCycleFilters(cycleId) {
        if (!cycleId) return;
        el("eventsCycleFilter").value = cycleId;
        el("analysisCycleFilter").value = cycleId;
        el("deepCycleFilter").value = cycleId;
      }
      async function createEvent() {
        const mode = el("createModeSelect").value || "single";
        const runs = Math.max(1, Number(el("createRunsInput").value || "1"));
        const payload = { asset: "BTC", horizon: el("createHorizonSelect").value, price_source: el("createSourceSelect").value, note: el("createNoteInput").value.trim() || null };
        if (mode === "tournament") {
          const modelIds = selectedCreateTournamentModelIds();
          if (modelIds.length < 2) throw new Error("Tournament requires at least 2 models.");
          const createdCycle = await api("/api/cycles", { method: "POST", body: { ...payload, mode: "tournament", model_ids: modelIds, runs } });
          showToast(`Tournament cycle created: ${createdCycle.cycle_id} (models: ${modelIds.length})`);
          closeModal();
          el("createNoteInput").value = "";
          el("createRunsInput").value = "1";
          Array.from(el("createModelMultiSelect").options || []).forEach((option) => { option.selected = false; });
          const cycleId = String(createdCycle.cycle_id || "");
          applyCycleFilters(cycleId);
          setTab("events");
          await loadEvents(1);
          if (createdCycle.last_event_id) await openEvent(createdCycle.last_event_id, true);
          return;
        }
        const modelPayload = { ...payload, model_id: el("createModelSelect").value || null };
        if (mode === "cycle") {
          const createdCycle = await api("/api/cycles", { method: "POST", body: { ...modelPayload, mode: "single", runs } });
          showToast(`Cycle created: ${createdCycle.cycle_id}`);
          closeModal();
          el("createNoteInput").value = "";
          el("createRunsInput").value = "1";
          const cycleId = String(createdCycle.cycle_id || "");
          applyCycleFilters(cycleId);
          setTab("events");
          await loadEvents(1);
          if (createdCycle.last_event_id) await openEvent(createdCycle.last_event_id, true);
          return;
        }
        const created = await api("/api/events", { method: "POST", body: modelPayload });
        showToast(`Event created: ${created.event_id}`);
        closeModal();
        el("createNoteInput").value = "";
        el("createRunsInput").value = "1";
        setTab("events");
        await loadEvents(1);
        await openEvent(created.event_id, true);
      }
      async function cancelEvent() {
        if (!state.selectedEvent || state.selectedEvent.status !== "active") { showToast("Отменить можно только active событие."); return; }
        if (!window.confirm(`Отменить событие ${state.selectedEvent.event_id}?`)) return;
        await api(`/api/events/${encodeURIComponent(state.selectedEvent.event_id)}/cancel`, { method: "POST" });
        showToast("Событие отменено.");
        await loadEvents();
        await openEvent(state.selectedEvent.event_id, false);
      }
      function bind() {
        document.querySelectorAll(".tab-btn").forEach((btn) => btn.addEventListener("click", () => setTab(btn.dataset.tab)));
        document.querySelectorAll(".subtab-btn").forEach((btn) => btn.addEventListener("click", () => setLabTab(btn.dataset.labTab)));
        const builderForm = document.querySelector("#labCreatePane .lab-builder-grid > section.panel");
        const onBuilderFieldChange = () => {
          syncBuilderModeStates();
          renderSweepAxisBuilder();
          scheduleBuilderValidate();
        };
        builderForm?.querySelectorAll("input, select, textarea").forEach((node) => {
          if (!node.id || node.id === "builderQSearch" || node.closest("#builderQuantileGrid")) return;
          if (node.tagName === "TEXTAREA" || node.type === "text" || node.type === "number" || node.type === "range") {
            node.addEventListener("input", onBuilderFieldChange);
          }
          node.addEventListener("change", onBuilderFieldChange);
        });
        el("builderBudgetPreset").addEventListener("change", () => {
          const preset = String(el("builderBudgetPreset")?.value || "custom").trim();
          if (preset.toLowerCase() === "custom") {
            renderSweepAxisBuilder();
            scheduleBuilderValidate();
            return;
          }
          if (applyTrainingBudgetPreset(preset, { scheduleValidate: true })) {
            showToast(`Applied training preset ${preset.toUpperCase()}.`);
          }
        });
        el("builderAddSweepAxisBtn").addEventListener("click", () => addSweepAxis("target_coverage_percent"));
        el("builderAddCoverageAxisBtn").addEventListener("click", () => addSweepAxis("target_coverage_percent"));
        el("builderAddWindowAxisBtn").addEventListener("click", () => addSweepAxis("train_window_days"));
        el("builderAddHyperparamAxisBtn").addEventListener("click", () => addSweepAxis("hyperparams.learning_rate"));
        el("builderClearSweepAxesBtn").addEventListener("click", () => {
          state.lab.sweepAxes = [];
          renderSweepAxisBuilder();
          scheduleBuilderValidate();
        });
        el("builderSweepAxesList").addEventListener("click", (event) => {
          const button = event.target.closest("button[data-action]");
          if (!button) return;
          const axisId = button.dataset.axisId;
          const index = Number(button.dataset.index || "-1");
          const nestedIndex = Number(button.dataset.nestedIndex || "-1");
          if (button.dataset.action === "remove-axis" && axisId) {
            removeSweepAxis(axisId);
            return;
          }
          if (button.dataset.action === "add-axis-value" && axisId) {
            appendSweepAxisValue(axisId);
            return;
          }
          if (button.dataset.action === "remove-axis-value" && axisId) {
            updateSweepAxis(axisId, (axis) => ({ values: (axis.values || []).filter((_, idx) => idx !== index) }));
            return;
          }
          if (button.dataset.action === "add-axis-nested-item" && axisId) {
            updateSweepAxis(axisId, (axis) => {
              const values = [...(axis.values || [])];
              const candidate = Array.isArray(values[index]) ? [...values[index]] : [];
              candidate.push(0);
              values[index] = candidate;
              return { values };
            });
            return;
          }
          if (button.dataset.action === "remove-axis-nested-item" && axisId) {
            updateSweepAxis(axisId, (axis) => {
              const values = [...(axis.values || [])];
              const candidate = Array.isArray(values[index]) ? [...values[index]] : [];
              values[index] = candidate.filter((_, idx) => idx !== nestedIndex);
              return { values };
            });
          }
        });
        el("builderSweepAxesList").addEventListener("input", (event) => {
          const target = event.target;
          const axisId = target.getAttribute("data-axis-id");
          if (!axisId) return;
          if (target.getAttribute("data-role") === "axis-range-start") {
            updateSweepAxis(axisId, () => ({ start: target.value }));
            return;
          }
          if (target.getAttribute("data-role") === "axis-range-end") {
            updateSweepAxis(axisId, () => ({ end: target.value }));
            return;
          }
          if (target.getAttribute("data-role") === "axis-range-step") {
            updateSweepAxis(axisId, () => ({ step: target.value }));
            return;
          }
          if (target.getAttribute("data-role") === "axis-value") {
            const valueIndex = Number(target.getAttribute("data-index") || "0");
            updateSweepAxis(axisId, (axis) => {
              const spec = sweepAxisSpec(axis.path);
              const values = [...(axis.values || [])];
              let nextValue = target.value;
              if (spec?.kind === "int") nextValue = Number(target.value || "0");
              if (spec?.kind === "float") nextValue = Number(target.value || "0");
              if (spec?.kind === "bool") nextValue = String(target.value) === "true";
              values[valueIndex] = nextValue;
              return { values };
            });
            return;
          }
          if (target.getAttribute("data-role") === "axis-nested-item") {
            const valueIndex = Number(target.getAttribute("data-index") || "0");
            const nested = Number(target.getAttribute("data-nested-index") || "0");
            updateSweepAxis(axisId, (axis) => {
              const spec = sweepAxisSpec(axis.path);
              const values = [...(axis.values || [])];
              const candidate = Array.isArray(values[valueIndex]) ? [...values[valueIndex]] : [];
              candidate[nested] = (spec?.kind === "int_list") ? Number(target.value || "0") : Number(target.value || "0");
              values[valueIndex] = candidate;
              return { values };
            });
          }
        });
        el("builderSweepAxesList").addEventListener("change", (event) => {
          const target = event.target;
          const axisId = target.getAttribute("data-axis-id");
          if (!axisId) return;
          if (target.getAttribute("data-role") === "axis-path") {
            setAxisPath(axisId, target.value);
            return;
          }
          if (target.getAttribute("data-role") === "axis-mode") {
            setAxisMode(axisId, target.value);
            return;
          }
          if (target.getAttribute("data-role") === "axis-enum-list-item") {
            const valueIndex = Number(target.getAttribute("data-index") || "0");
            updateSweepAxis(axisId, (axis) => {
              const values = [...(axis.values || [])];
              const candidate = new Set(Array.isArray(values[valueIndex]) ? values[valueIndex] : []);
              if (target.checked) candidate.add(target.value); else candidate.delete(target.value);
              values[valueIndex] = [...candidate];
              return { values };
            });
            return;
          }
          if (target.getAttribute("data-role") === "axis-multi-select") {
            const valueIndex = Number(target.getAttribute("data-index") || "0");
            updateSweepAxis(axisId, (axis) => {
              const values = [...(axis.values || [])];
              values[valueIndex] = [...target.selectedOptions].map((option) => option.value);
              return { values };
            });
          }
        });
        el("builderQSearch").addEventListener("input", refreshBuilderQuantileVisuals);
        el("builderValidateBtn").addEventListener("click", () => validateBuilder({ showBanner: true }).catch(onError));
        el("builderStartTrainBtn").addEventListener("click", () => startBuilderJob("train").catch(onError));
        el("builderStartSweepBtn").addEventListener("click", () => startBuilderJob("sweep").catch(onError));
        el("builderStartWfEvalBtn").addEventListener("click", () => startBuilderJob("wf_eval").catch(onError));
        el("builderReloadJobsBtn").addEventListener("click", () => loadLabJobs().catch(onError));
        el("builderAddJobModelsToCompareBtn").addEventListener("click", () => addSelectedJobModelsToCompare().catch(onError));
        el("builderReloadModelsBtn").addEventListener("click", () => loadLabModels().catch(onError));
        el("builderUseModelForEventsBtn").addEventListener("click", useSelectedModelForEvents);
        el("builderUseModelForCompareBtn").addEventListener("click", useSelectedModelForCompare);
        el("builderSaveModelMetaBtn").addEventListener("click", () => saveSelectedModelMeta().catch(onError));
        el("builderPromoteModelBtn").addEventListener("click", () => runSelectedModelAction("promote").catch(onError));
        el("builderArchiveModelBtn").addEventListener("click", () => runSelectedModelAction("archive").catch(onError));
        el("builderUnarchiveModelBtn").addEventListener("click", () => runSelectedModelAction("unarchive").catch(onError));
        el("builderSoftDeleteModelBtn").addEventListener("click", () => runSelectedModelAction("delete").catch(onError));
        el("builderPurgeModelBtn").addEventListener("click", () => runSelectedModelAction("purge").catch(onError));
        el("builderDownloadReportBtn").addEventListener("click", () => downloadExperimentReport("pdf").catch(onError));
        el("builderDownloadReportTxtBtn").addEventListener("click", () => downloadExperimentReport("txt").catch(onError));
        el("builderAddQ50Btn").addEventListener("click", () => {
          const next = new Set(builderSelectedQuantiles());
          next.add("q50");
          setBuilderQuantileSelection([...next]);
          scheduleBuilderValidate();
        });
        el("builderSelectPrimaryBtn").addEventListener("click", () => {
          setBuilderQuantileSelection(builderPrimaryQuantiles());
          scheduleBuilderValidate();
        });
        el("builderSelectDecilesBtn").addEventListener("click", () => {
          const keys = [];
          for (let idx = 10; idx <= 90; idx += 10) keys.push(`q${String(idx).padStart(2, "0")}`);
          setBuilderQuantileSelection(keys);
          scheduleBuilderValidate();
        });
        el("builderSelectRangeBtn").addEventListener("click", () => {
          selectBuilderQuantileRange();
          scheduleBuilderValidate();
        });
        el("builderClearQBtn").addEventListener("click", () => {
          setBuilderQuantileSelection([]);
          scheduleBuilderValidate();
        });
        el("builderJobStatusFilter").addEventListener("input", () => loadLabJobs().catch(onError));
        el("builderShowArchived").addEventListener("change", () => loadLabModels().catch(onError));
        el("builderShowDeleted").addEventListener("change", () => loadLabModels().catch(onError));
        el("builderJobsBody").addEventListener("click", (event) => {
          const button = event.target.closest("button[data-action]");
          if (button) {
            const action = button.dataset.action;
            const jobId = button.dataset.jobId;
            if (!jobId) return;
            if (action === "view") {
              loadLabJobLogs(jobId).catch(onError);
              return;
            }
            if (action === "cancel") {
              api(`/api/lab/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" })
                .then(() => {
                  showToast("Cancel requested.");
                  return loadLabJobs();
                })
                .catch(onError);
              return;
            }
            if (action === "retry") {
              api(`/api/lab/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" })
                .then((job) => {
                  state.lab.selectedJobId = job?.job_id || state.lab.selectedJobId;
                  showToast("Retry job created.");
                  return loadLabJobs();
                })
                .catch(onError);
              return;
            }
            return;
          }
          const row = event.target.closest("tr[data-job-id]");
          if (!row) return;
          loadLabJobLogs(row.getAttribute("data-job-id")).catch(onError);
        });
        el("builderModelsBody").addEventListener("click", (event) => {
          const row = event.target.closest("tr[data-model-id]");
          if (!row) return;
          selectLabModel(row.getAttribute("data-model-id"));
        });
        el("reloadEventsBtn").addEventListener("click", () => refreshView().catch(onError));
        el("reloadMarketsBtn").addEventListener("click", () => loadMarkets().catch(onError));
        ["eventsStatusFilter", "eventsHorizonFilter", "eventsResultFilter"].forEach((id) => el(id).addEventListener("change", () => loadEvents(1).catch(onError)));
        const applyEventsPageSize = () => {
          state.events.page_size = normalizeEventsPageSize(el("eventsPageSizeInput").value || "10");
          renderEventPageSize();
          loadEvents(1).catch(onError);
        };
        el("eventsPageSizeInput").addEventListener("change", applyEventsPageSize);
        el("eventsPageSizeInput").addEventListener("keydown", (event) => {
          if (event.key !== "Enter") return;
          event.preventDefault();
          applyEventsPageSize();
        });
        el("eventsCycleFilter").addEventListener("input", () => { if (state.searchTimer) clearTimeout(state.searchTimer); state.searchTimer = setTimeout(() => loadEvents(1).catch(onError), 280); });
        el("eventsSearchInput").addEventListener("input", () => { if (state.searchTimer) clearTimeout(state.searchTimer); state.searchTimer = setTimeout(() => loadEvents(1).catch(onError), 280); });
        el("eventsPrevBtn").addEventListener("click", () => loadEvents(Math.max(1, Number(state.events.page || 1) - 1)).catch(onError));
        el("eventsNextBtn").addEventListener("click", () => loadEvents(Number(state.events.page || 1) + 1).catch(onError));
        el("eventsTableBody").addEventListener("click", (event) => { const row = event.target.closest("tr[data-event-id]"); if (!row) return; openEvent(row.getAttribute("data-event-id"), true).catch(onError); });
        el("closeDetailsBtn").addEventListener("click", closeEvent);
        el("eventDetailsPanel").addEventListener("click", (event) => { if (event.target.id === "eventDetailsPanel") closeEvent(); });
        el("cancelEventBtn").addEventListener("click", () => cancelEvent().catch(onError));
        el("reloadAnalysisBtn").addEventListener("click", () => loadAnalysis().catch(onError));
        el("applyAnalysisFiltersBtn").addEventListener("click", () => loadAnalysis().catch(onError));
        el("analysisHorizonFilter").addEventListener("change", () => {
          renderCompareModelOptions();
        });
        el("compareModeSelect").addEventListener("change", () => loadAnalysis().catch(onError));
        el("compareModelsSelect").addEventListener("change", () => {
          if (state.activeTab === "lab" && state.activeLabTab === "compare") {
            Promise.all([loadCompareSummary(), loadCompareScoreboard(), loadDeep()]).then(() => loadCompareChart()).catch(onError);
          }
        });
        el("compareModelSearch").addEventListener("input", () => renderCompareModelOptions());
        el("compareBaselineSelect").addEventListener("change", () => loadCompareScoreboard().then(() => loadCompareChart()).catch(onError));
        el("compareExcludeStale").addEventListener("change", () => loadAnalysis().catch(onError));
        el("tournamentComparePresetBtn").addEventListener("click", applyTournamentComparePreset);
        el("showTournamentSummaryBtn").addEventListener("click", () => loadTournamentCycleSummary().catch(onError));
        el("compareApplyChartBtn").addEventListener("click", () => loadCompareChart().catch(onError));
        el("compareColumnPicker").addEventListener("change", () => loadCompareScoreboard().then(() => loadCompareChart()).catch(onError));
        el("applyDeepAnalysisBtn").addEventListener("click", () => loadDeep().catch(onError));
        el("horizonBreakdownBody").addEventListener("click", (event) => {
          const button = event.target.closest("button[data-open-model]");
          const row = event.target.closest("tr[data-model-id]");
          const modelId = button?.getAttribute("data-open-model") || row?.getAttribute("data-model-id");
          if (!modelId) return;
          el("deepModelFilter").value = modelId;
          loadDeep().catch(onError);
        });
        el("deepEventsBody").addEventListener("click", (event) => { const row = event.target.closest("tr[data-event-id]"); if (!row) return; setTab("events"); openEvent(row.getAttribute("data-event-id"), true).catch(onError); });
        el("compareExportJsonBtn").addEventListener("click", exportCompareScoreboardJson);
        el("compareExportCsvBtn").addEventListener("click", exportCompareScoreboardCsv);
        el("exportEventsJsonBtn").addEventListener("click", exportEventsJson);
        el("exportEventsCsvBtn").addEventListener("click", exportEventsCsv);
        el("exportChartPngBtn").addEventListener("click", exportChartPng);
        el("exportReportPdfBtn").addEventListener("click", exportComparePdf);
        el("saveSettingsBtn").addEventListener("click", async () => { try { readSettings(); saveSettings(); renderSettings(); restartPolling(); await refreshView(); showToast("Настройки сохранены."); } catch (error) { onError(error); } });
        el("resetSettingsBtn").addEventListener("click", () => { state.settings = { ...defaultSettings }; saveSettings(); renderSettings(); restartPolling(); showToast("Настройки сброшены."); });
        el("openCreateEventBtn").addEventListener("click", openModal);
        el("closeCreateModalBtn").addEventListener("click", closeModal);
        el("submitCreateEventBtn").addEventListener("click", () => createEvent().catch(onError));
        el("createModeSelect").addEventListener("change", syncCreateModeUi);
        el("createRunsInput").addEventListener("input", syncCreateModeUi);
        el("createHorizonSelect").addEventListener("change", renderCreateModelOptions);
        el("createModelMultiSelect").addEventListener("change", syncCreateModeUi);
        el("createEventModal").addEventListener("click", (event) => { if (event.target.id === "createEventModal") closeModal(); });
        document.addEventListener("keydown", (event) => {
          if (event.key === "Escape" && state.selectedEvent) closeEvent();
        });
      }
      async function boot() {
        bind();
        renderBuilderQuantileGrid();
        syncBuilderModeStates();
        syncCreateModeUi();
        loadSettings();
        renderSettings();
        ensureTicker();
        restartQuoteTicker();
        try {
          await Promise.all([loadStatus(), loadQuote(), loadModels(), loadEvents(1)]);
          setConnection(true, "Connected");
          showBanner("");
        } catch (error) { onError(error); }
        restartPolling();
      }
      boot();
    })();

