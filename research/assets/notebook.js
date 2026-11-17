/* Notebook widgets: progressive enhancement over static tables and figures.
   No build step, no dependencies. Each <figure data-widget="..." data-src="x.json"> is
   upgraded in place; if the data cannot be loaded the static content stays visible.
   Statistics (bootstrap, seeds, formatting) mirror research/report_api.py exactly. */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const BOOTSTRAP_RESAMPLES = 1000;

  // ---------- DOM helpers ----------

  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    node.append(...children);
    return node;
  }

  function s(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const fillStyle = (key) => `fill: var(--method-${key}, #555)`;
  const strokeStyle = (key) => `stroke: var(--method-${key}, #555)`;
  const swatch = (key, dashed) =>
    h("span", { class: "swatch" + (dashed ? " dashed" : ""), style: key ? `color: var(--method-${key}, #555)` : "color: var(--reference, #8c8c8c)" });

  function labeledSelect(label, options, current, onChange) {
    const sel = h("select");
    for (const o of options) sel.append(h("option", { value: o.key, text: o.label }));
    sel.value = current;
    sel.addEventListener("change", () => onChange(sel.value));
    return h("label", { class: "control" }, label + " ", sel);
  }

  function notice(fig, text) {
    const body = fig.querySelector(".widget-body");
    for (const stale of body.querySelectorAll(":scope > p.missing")) stale.remove();
    body.append(h("p", { class: "missing", text }));
  }

  // ---------- numbers and statistics (identical to report_api.py) ----------

  // Python "%.{p}g" formatting, so tables rebuilt here match the static ones.
  function fmt(x, p = 4) {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    if (x === 0) return "0";
    const [mant, expStr] = x.toExponential(p - 1).split("e");
    const exp = Number(expStr);
    const trim = (str) => (str.includes(".") ? str.replace(/0+$/, "").replace(/\.$/, "") : str);
    if (exp < -4 || exp >= p) return `${trim(mant)}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
    return trim(x.toFixed(p - 1 - exp));
  }

  function fnv1a(str) {
    let hash = 0x811c9dc5;
    for (const ch of str) {
      hash ^= ch.codePointAt(0);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash;
  }

  function mulberry32(seed) {
    let a = seed | 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function mean(values) {
    let total = 0;
    for (const v of values) total += v;
    return total / values.length;
  }

  function quantile(sorted, q) {
    const pos = q * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  // 95% bootstrap CI of the mean: B resamples with replacement, seeded by a string key.
  function bootstrapCI(values, seedKey) {
    const n = values.length;
    if (n === 0) return [NaN, NaN];
    if (n === 1) return [values[0], values[0]];
    const rand = mulberry32(fnv1a(seedKey));
    const means = new Array(BOOTSTRAP_RESAMPLES);
    for (let b = 0; b < BOOTSTRAP_RESAMPLES; b++) {
      let total = 0;
      for (let i = 0; i < n; i++) total += values[Math.floor(rand() * n)];
      means[b] = total / n;
    }
    means.sort((x, y) => x - y);
    return [quantile(means, 0.025), quantile(means, 0.975)];
  }

  function ticks(lo, hi, count) {
    const raw = (hi - lo) / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toFixed(12)));
    return out;
  }

  function extent(arrays) {
    let lo = Infinity, hi = -Infinity;
    for (const arr of arrays) for (const v of arr) if (v !== null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!Number.isFinite(lo)) return [0, 1];
    if (lo === hi) { const pad = Math.abs(lo) * 0.1 || 1; return [lo - pad, hi + pad]; }
    const pad = (hi - lo) * 0.05;
    return [lo - pad, hi + pad];
  }

  // ---------- compare widget ----------

  function compareStats(data, regime, metric) {
    const values = data.values[regime][metric];
    const ref = data.reference_method;
    const refValues = values[ref] || [];
    return data.methods.map((m) => {
      const raw = values[m.key] || [];
      const present = raw.filter((x) => x !== null);
      const pairs = [];
      raw.forEach((x, i) => { if (x !== null && refValues[i] !== null && refValues[i] !== undefined) pairs.push(x - refValues[i]); });
      const stat = {
        key: m.key, label: m.label, values: raw, n: present.length, total: raw.length,
        mean: present.length ? mean(present) : NaN, ci: bootstrapCI(present, `${regime}/${metric}/${m.key}`),
        isRef: m.key === ref, pairs: pairs.length, diff: NaN, diffCi: [NaN, NaN],
      };
      if (!stat.isRef && pairs.length) {
        stat.diff = mean(pairs);
        stat.diffCi = bootstrapCI(pairs, `${regime}/${metric}/${m.key}-vs-${ref}`);
      }
      return stat;
    });
  }

  const ciText = (ci) => (Number.isNaN(ci[0]) ? "—" : `[${fmt(ci[0])}, ${fmt(ci[1])}]`);

  function compareTable(stats) {
    const table = h("table", { class: "compare-table" });
    const head = h("tr");
    for (const [i, text] of ["Method", "Mean", "95% CI", "n", "Δ vs reference", "95% CI (paired)", "Pairs"].entries())
      head.append(h("th", i ? { class: "num", text } : { text }));
    table.append(h("thead", {}, head));
    const body = h("tbody");
    for (const st of stats) {
      const nText = st.n === st.total ? String(st.n) : `${st.n} of ${st.total}`;
      const cells = [st.label, fmt(st.mean), ciText(st.ci), nText,
        st.isRef ? "reference" : fmt(st.diff), st.isRef ? "—" : ciText(st.diffCi), st.isRef ? "—" : String(st.pairs)];
      body.append(h("tr", {}, ...cells.map((text, i) => h("td", i ? { class: "num", text } : { text }))));
    }
    table.append(body);
    return table;
  }

  function selectionText(data, regime, metric, stats) {
    const unit = metric.unit ? ` (${metric.unit})` : "";
    const ref = data.methods.find((m) => m.key === data.reference_method);
    const cases = stats.length ? stats[0].total : 0;
    return `Selection: ${regime.label}, ${metric.label}${unit}; n = ${cases} cases. Mean with 95% bootstrap CI ` +
      `(${BOOTSTRAP_RESAMPLES} resamples); difference vs ${ref ? ref.label : data.reference_method} is paired by case, failed cases dropped from the pair.`;
  }

  function stripPlot(data, regime, metric, stats) {
    const W = 760, left = 190, right = 70, top = 10, rowH = 26, bottom = 42;
    const H = top + stats.length * rowH + bottom;
    const axisY = top + stats.length * rowH;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${metric.label} per case for each method, with mean and 95% CI` });
    const [lo, hi] = extent(stats.flatMap((st) => [st.values, st.n ? st.ci : []]));
    const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
    for (const t of ticks(lo, hi, 5)) {
      svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: axisY }));
      svg.append(s("text", { class: "tick", x: x(t), y: axisY + 14, "text-anchor": "middle" }, fmt(t)));
    }
    svg.append(s("line", { class: "axis", x1: left, x2: W - right, y1: axisY, y2: axisY }));
    const direction = metric.lower_is_better === true ? " · lower is better" : metric.lower_is_better === false ? " · higher is better" : "";
    svg.append(s("text", { class: "axis-label", x: left + (W - left - right) / 2, y: axisY + 32, "text-anchor": "middle" },
      `${metric.label}${metric.unit ? ` (${metric.unit})` : ""}${direction}`));
    const caseIds = (data.case_ids && data.case_ids[regime.key]) || [];
    stats.forEach((st, i) => {
      const cy = top + i * rowH + rowH / 2;
      svg.append(s("text", { class: "row-label", x: left - 10, y: cy + 4, "text-anchor": "end" }, st.label));
      const jitter = mulberry32(fnv1a("jitter/" + st.key));
      st.values.forEach((v, j) => {
        if (v === null) return;
        const dot = s("circle", { cx: x(v), cy: cy + (jitter() - 0.5) * 9, r: 3.2, "fill-opacity": 0.7, style: fillStyle(st.key) });
        dot.append(s("title", {}, `${caseIds[j] || "case " + (j + 1)}: ${fmt(v)}`));
        svg.append(dot);
      });
      if (st.n) {
        svg.append(s("line", { class: "stat", x1: x(st.ci[0]), x2: x(st.ci[1]), y1: cy, y2: cy }));
        svg.append(s("line", { class: "stat", x1: x(st.mean), x2: x(st.mean), y1: cy - 7, y2: cy + 7, "stroke-width": 2 }));
      }
      if (st.total - st.n) svg.append(s("text", { class: "tick", x: W - right + 8, y: cy + 4 }, `${st.total - st.n} failed`));
    });
    return svg;
  }

  function initCompare(fig, data) {
    const body = fig.querySelector(".widget-body");
    const state = { regime: fig.dataset.regime || data.regimes[0].key, metric: fig.dataset.metric || data.metrics[0].key };
    const controls = h("div", { class: "controls" });
    const plot = h("div", { class: "chart" });
    const selection = h("p", { class: "selection" });
    const tableWrap = h("div");
    const draw = () => {
      const regime = data.regimes.find((r) => r.key === state.regime);
      const metric = data.metrics.find((m) => m.key === state.metric);
      const stats = compareStats(data, regime.key, metric.key);
      plot.replaceChildren(stripPlot(data, regime, metric, stats));
      selection.textContent = selectionText(data, regime, metric, stats);
      tableWrap.replaceChildren(compareTable(stats));
    };
    controls.append(
      labeledSelect("Regime", data.regimes, state.regime, (v) => { state.regime = v; draw(); }),
      labeledSelect("Metric", data.metrics, state.metric, (v) => { state.metric = v; draw(); }));
    body.replaceChildren(controls, plot, selection, tableWrap);
    draw();
  }

  // ---------- time-series widget ----------

  function pathFor(t, values, x, y) {
    let d = "", pen = false;
    for (let i = 0; i < t.length; i++) {
      const v = values[i];
      if (v === null || v === undefined || !Number.isFinite(v)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${x(t[i]).toFixed(2)},${y(v).toFixed(2)}`;
      pen = true;
    }
    return d;
  }

  function initTimeseries(fig, data) {
    const body = fig.querySelector(".widget-body");
    const t = data.t, n = t.length;
    // data-video names a <video>: the cursor follows playback and cursor moves seek the video.
    // data-video-speed is the clip's playback rate relative to simulation time (0.25 = slow motion).
    const video = fig.dataset.video ? document.getElementById(fig.dataset.video) : null;
    const videoSpeed = Number(fig.dataset.videoSpeed) > 0 ? Number(fig.dataset.videoSpeed) : 1;
    const reduceMotion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)").matches : false;
    const state = { series: data.series[0].key, visible: new Set(data.methods.map((m) => m.key)), cursor: video ? 0 : Math.floor((n - 1) / 2) };
    const W = 760, H = 300, L = 64, R = 16, T = 18, B = 42;
    const x = (v) => L + ((v - t[0]) / (t[n - 1] - t[0] || 1)) * (W - L - R);
    let y = (v) => v, lines = [], refValues = null, cursorGroup = null;

    const legend = h("fieldset", { class: "legend" }, h("legend", { text: "Methods shown" }));
    for (const m of data.methods) {
      const box = h("input", { type: "checkbox", value: m.key });
      box.checked = true;
      box.addEventListener("change", () => { box.checked ? state.visible.add(m.key) : state.visible.delete(m.key); draw(); });
      legend.append(h("label", { class: "legend-item" }, box, swatch(m.key), m.label));
    }
    const referenceItem = h("span", { class: "legend-item" }, swatch(null, true), "commanded");
    legend.append(referenceItem);
    const controls = h("div", { class: "controls" }, labeledSelect("Series", data.series, state.series, (v) => { state.series = v; draw(); }));
    const chart = h("div", { class: "chart" });
    const readout = h("p", { class: "readout", "aria-live": "polite" });
    body.replaceChildren(controls, legend, chart, readout);

    function nearestIndex(tv) {
      let lo = 0, hi = n - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (t[mid] < tv) lo = mid + 1; else hi = mid; }
      if (lo > 0 && Math.abs(t[lo - 1] - tv) < Math.abs(t[lo] - tv)) lo--;
      return lo;
    }

    function updateCursor() {
      if (!cursorGroup) return;
      const i = state.cursor;
      cursorGroup.replaceChildren(s("line", { class: "cursor", x1: x(t[i]), x2: x(t[i]), y1: T, y2: H - B }));
      const parts = [`t = ${fmt(t[i])} s`];
      const unit = data.series.find((sr) => sr.key === state.series).unit || "";
      for (const ln of lines) {
        const v = ln.values[i];
        if (v !== null && v !== undefined && Number.isFinite(v)) cursorGroup.append(s("circle", { class: "cursor-dot", cx: x(t[i]), cy: y(v), r: 4, style: fillStyle(ln.key) }));
        parts.push(`${ln.label}: ${fmt(v)} ${unit}`.trim());
      }
      if (refValues) parts.push(`commanded: ${fmt(refValues[i])} ${unit}`.trim());
      readout.textContent = parts.join(" · ");
    }

    // ---- video link ----
    let raf = 0;
    const videoTime = () => t[0] + video.currentTime * videoSpeed;
    function followVideo() {
      state.cursor = nearestIndex(videoTime());
      updateCursor();
    }
    function seekVideo() {
      const vt = (t[state.cursor] - t[0]) / videoSpeed;
      if (Number.isFinite(vt)) video.currentTime = video.duration ? Math.min(vt, video.duration) : vt;
    }
    function tick() {
      if (video.paused || video.ended) { raf = 0; return; }
      followVideo();
      raf = requestAnimationFrame(tick);
    }
    if (video) {
      video.addEventListener("timeupdate", followVideo);
      video.addEventListener("seeked", followVideo);
      // Smooth cursor motion between timeupdate events while playing, unless the viewer prefers reduced motion.
      video.addEventListener("play", () => { if (!reduceMotion && !raf) raf = requestAnimationFrame(tick); });
      for (const ev of ["pause", "ended"]) video.addEventListener(ev, () => { if (raf) cancelAnimationFrame(raf); raf = 0; followVideo(); });
    }

    function indexAtPointer(svg, e) {
      const r = svg.getBoundingClientRect();
      const sx = ((e.clientX - r.left) / r.width) * W;
      return nearestIndex(t[0] + ((sx - L) / (W - L - R)) * (t[n - 1] - t[0]));
    }

    function draw() {
      const series = data.series.find((sr) => sr.key === state.series);
      const values = data.values[series.key] || {};
      lines = data.methods.filter((m) => state.visible.has(m.key) && values[m.key]).map((m) => ({ key: m.key, label: m.label, values: values[m.key] }));
      refValues = (data.reference && data.reference[series.key]) || null;
      referenceItem.hidden = !refValues;
      const [ylo, yhi] = extent(lines.map((ln) => ln.values).concat(refValues ? [refValues] : []));
      y = (v) => T + ((yhi - v) / (yhi - ylo)) * (H - T - B);
      const unit = series.unit ? ` (${series.unit})` : "";
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, tabindex: "0", role: "img", "aria-label": `${series.label}${unit} over time; use the left and right arrow keys to move the cursor${video ? ", which seeks the video" : ""}` });
      for (const tv of ticks(ylo, yhi, 5)) {
        svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(tv), y2: y(tv) }));
        svg.append(s("text", { class: "tick", x: L - 8, y: y(tv) + 4, "text-anchor": "end" }, fmt(tv)));
      }
      for (const tv of ticks(t[0], t[n - 1], 6)) {
        svg.append(s("line", { class: "axis", x1: x(tv), x2: x(tv), y1: H - B, y2: H - B + 4 }));
        svg.append(s("text", { class: "tick", x: x(tv), y: H - B + 16, "text-anchor": "middle" }, fmt(tv)));
      }
      svg.append(s("line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B }));
      svg.append(s("text", { class: "axis-label", x: L + (W - L - R) / 2, y: H - 8, "text-anchor": "middle" }, "Time (s)"));
      svg.append(s("text", { class: "axis-label", transform: `translate(14 ${T + (H - T - B) / 2}) rotate(-90)`, "text-anchor": "middle" }, `${series.label}${unit}`));
      for (const ev of data.events || []) {
        svg.append(s("line", { class: "event", x1: x(ev.t), x2: x(ev.t), y1: T, y2: H - B }));
        svg.append(s("text", { class: "event-label", x: x(ev.t) + 4, y: T + 10 }, ev.label));
      }
      if (refValues) svg.append(s("path", { class: "reference", d: pathFor(t, refValues, x, y) }));
      for (const ln of lines) svg.append(s("path", { class: "series", d: pathFor(t, ln.values, x, y), style: strokeStyle(ln.key) }));
      cursorGroup = s("g");
      svg.append(cursorGroup);
      svg.addEventListener("mousemove", (e) => {
        if (video && !video.paused) return; // the cursor follows playback
        state.cursor = indexAtPointer(svg, e);
        updateCursor();
      });
      if (video) {
        svg.addEventListener("mouseleave", () => { if (video.paused) followVideo(); });
        svg.addEventListener("click", (e) => { state.cursor = indexAtPointer(svg, e); updateCursor(); seekVideo(); });
      }
      svg.addEventListener("keydown", (e) => {
        const step = e.shiftKey ? 10 : 1;
        if (e.key === "ArrowLeft") state.cursor = Math.max(0, state.cursor - step);
        else if (e.key === "ArrowRight") state.cursor = Math.min(n - 1, state.cursor + step);
        else if (e.key === "Home") state.cursor = 0;
        else if (e.key === "End") state.cursor = n - 1;
        else return;
        e.preventDefault();
        updateCursor();
        if (video) seekVideo();
      });
      chart.replaceChildren(svg);
      if (video) followVideo(); else updateCursor();
    }
    draw();
  }

  // ---------- bootstrapping ----------

  const WIDGETS = { compare: initCompare, timeseries: initTimeseries };

  async function upgrade(fig) {
    const init = WIDGETS[fig.dataset.widget];
    const src = fig.dataset.src;
    if (!init || !src) return;
    try {
      const response = await fetch(src);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      init(fig, await response.json());
    } catch (err) {
      notice(fig, `Interactive view unavailable: could not load ${src} (${err.message}). ` +
        "Static content shown; for interactivity serve the site over HTTP (for example: python -m http.server in research/site).");
    }
  }

  for (const fig of document.querySelectorAll("[data-widget]")) upgrade(fig);
})();
