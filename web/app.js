(() => {
      "use strict";
      const HORIZONS = ["5m", "15m", "1h", "4h", "1d", "1w"];
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
        statusInfo: null,
        quote: null,
        events: { items: [], total: 0, page: 1, page_size: 10 },
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
        analysis: { summary: null, deepEvents: [], deepTotal: 0 },
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
        if (state.selectedEvent) return;
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
        const prodSet = new Set(state.productionModels.map((m) => m.model_id));
        const ordered = [...state.productionModels, ...state.models.filter((m) => !prodSet.has(m.model_id))];
        const select = el("createModelSelect");
        if (!ordered.length) { select.innerHTML = `<option value="">auto (production)</option>`; return; }
        select.innerHTML = [`<option value="">auto (production)</option>`, ...ordered.map((m) => `<option value="${esc(m.model_id)}">${esc(m.model_id)}${prodSet.has(m.model_id) ? " [prod]" : ""}</option>`)].join("");
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
        const payload = await api("/api/events", { query: collectDeep() });
        state.analysis.deepEvents = payload.items || [];
        state.analysis.deepTotal = Number(payload.total || 0);
        renderDeep();
      }
      async function loadAnalysis() { await Promise.all([loadStatus(), loadSummary(), loadDeep()]); }
      function setTab(tab) {
        if (tab !== "events" && state.selectedEvent) closeEvent();
        state.activeTab = tab;
        document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tab));
        const map = { events: "paneEvents", markets: "paneMarkets", analysis: "paneAnalysis", settings: "paneSettings" };
        Object.entries(map).forEach(([name, id]) => el(id).classList.toggle("active", name === tab));
        if (tab === "events") Promise.all([loadStatus(), loadQuote(), loadEvents()]).then(() => showBanner("")).catch(onError);
        if (tab === "analysis") loadAnalysis().catch(onError);
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
          if (state.activeTab === "analysis") { await loadSummary(); await loadDeep(); }
          showBanner("");
        } finally {
          state.refreshInFlight = false;
        }
      }
      function onError(error) { setConnection(false, "Disconnected"); showBanner(`Ошибка: ${error instanceof Error ? error.message : String(error)}`); }
      function openModal() { el("createEventModal").classList.add("open"); }
      function closeModal() { el("createEventModal").classList.remove("open"); }
      async function createEvent() {
        const runs = Math.max(1, Number(el("createRunsInput").value || "1"));
        const payload = { asset: "BTC", horizon: el("createHorizonSelect").value, model_id: el("createModelSelect").value || null, price_source: el("createSourceSelect").value, note: el("createNoteInput").value.trim() || null };
        if (runs > 1) {
          const createdCycle = await api("/api/cycles", { method: "POST", body: { ...payload, runs } });
          showToast(`Cycle created: ${createdCycle.cycle_id}`);
          closeModal();
          el("createNoteInput").value = "";
          el("createRunsInput").value = "1";
          const cycleId = String(createdCycle.cycle_id || "");
          if (cycleId) {
            el("eventsCycleFilter").value = cycleId;
            el("analysisCycleFilter").value = cycleId;
            el("deepCycleFilter").value = cycleId;
          }
          setTab("events");
          await loadEvents(1);
          if (createdCycle.last_event_id) await openEvent(createdCycle.last_event_id, true);
          return;
        }
        const created = await api("/api/events", { method: "POST", body: payload });
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
        el("reloadEventsBtn").addEventListener("click", () => refreshView().catch(onError));
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
        el("applyAnalysisFiltersBtn").addEventListener("click", () => loadSummary().catch(onError));
        el("applyDeepAnalysisBtn").addEventListener("click", () => loadDeep().catch(onError));
        el("deepEventsBody").addEventListener("click", (event) => { const row = event.target.closest("tr[data-event-id]"); if (!row) return; setTab("events"); openEvent(row.getAttribute("data-event-id"), true).catch(onError); });
        el("exportEventsJsonBtn").addEventListener("click", exportEventsJson);
        el("exportEventsCsvBtn").addEventListener("click", exportEventsCsv);
        el("exportChartPngBtn").addEventListener("click", exportChartPng);
        el("exportReportPdfBtn").addEventListener("click", exportPdf);
        el("saveSettingsBtn").addEventListener("click", async () => { try { readSettings(); saveSettings(); renderSettings(); restartPolling(); await refreshView(); showToast("Настройки сохранены."); } catch (error) { onError(error); } });
        el("resetSettingsBtn").addEventListener("click", () => { state.settings = { ...defaultSettings }; saveSettings(); renderSettings(); restartPolling(); showToast("Настройки сброшены."); });
        el("openCreateEventBtn").addEventListener("click", openModal);
        el("closeCreateModalBtn").addEventListener("click", closeModal);
        el("submitCreateEventBtn").addEventListener("click", () => createEvent().catch(onError));
        el("createEventModal").addEventListener("click", (event) => { if (event.target.id === "createEventModal") closeModal(); });
        document.addEventListener("keydown", (event) => {
          if (event.key === "Escape" && state.selectedEvent) closeEvent();
        });
      }
      async function boot() {
        bind();
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

