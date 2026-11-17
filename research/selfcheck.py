#!/usr/bin/env python3
"""Self-check for the notebook build.

    uv run python research/selfcheck.py [--keep-site DIR]

Writes a scratch report research/reports/zz-selfcheck/ with synthetic compare and time-series
data, builds the site with a temporary catalog into a scratch directory, asserts that the static
fallbacks, copied artifacts, heading ids, links and HTML structure are correct, then removes the
scratch report. The scratch site is removed too unless --keep-site DIR is given.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH))
import plotstyle  # noqa: E402
import report_api as ra  # noqa: E402

REPORT_ID = "zz-selfcheck"
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


def synthetic_compare(rng: np.random.Generator) -> dict:
    methods = [{"key": "pd_shared", "label": "PD, shared gains"}, {"key": "pd_tuned", "label": "PD, tuned per condition"},
               {"key": "learned_gains", "label": "Learned gains"}, {"key": "neural_torque", "label": "Neural torque"}]
    regimes = [{"key": "hold", "label": "Hold"}, {"key": "track", "label": "Free-space tracking"}, {"key": "contact", "label": "Sustained contact"}]
    metrics = [{"key": "pos_rmse", "label": "Position RMSE", "unit": "rad", "lower_is_better": True},
               {"key": "peak_force", "label": "Peak normal force", "unit": "N", "lower_is_better": True},
               {"key": "torque_rms", "label": "Torque RMS", "unit": "N·m", "lower_is_better": True}]
    scale = {"pos_rmse": 0.05, "peak_force": 20.0, "torque_rms": 3.0}
    values, case_ids = {}, {}
    for r, regime in enumerate(regimes):
        n = 10
        case_ids[regime["key"]] = [f"{regime['key']}-{i:02d}" for i in range(n)]
        values[regime["key"]] = {}
        for metric in metrics:
            block = {}
            for m, method in enumerate(methods):
                samples = scale[metric["key"]] * (1 + 0.3 * r) * (1 - 0.12 * m) * rng.lognormal(0, 0.2, n)
                block[method["key"]] = [float(x) for x in samples]
            if regime["key"] == "contact":
                block["neural_torque"][2] = None  # one failed case
                block["neural_torque"][7] = None
            values[regime["key"]][metric["key"]] = block
    return {"methods": methods, "regimes": regimes, "metrics": metrics, "reference_method": "pd_shared",
            "values": values, "case_ids": case_ids}


def synthetic_timeseries(rng: np.random.Generator) -> dict:
    t = np.round(np.arange(0, 3.0 + 1e-9, 0.005), 4)
    contact = (t >= 1.0) & (t < 2.2)
    commanded_q = 0.4 * np.sin(2 * np.pi * 0.5 * t)
    data = {"t": t.tolist(),
            "series": [{"key": "f_normal", "label": "Normal contact force", "unit": "N"}, {"key": "q1", "label": "Joint 1 position", "unit": "rad"}],
            "methods": [{"key": "pd_shared", "label": "PD, shared gains"}, {"key": "learned_gains", "label": "Learned gains"}],
            "values": {"f_normal": {}, "q1": {}}, "reference": {"q1": commanded_q.tolist()},
            "events": [{"t": 1.0, "label": "contact onset"}, {"t": 2.2, "label": "release"}]}
    for i, m in enumerate(data["methods"]):
        chatter = (1 - 0.6 * i) * 4.0 * np.sin(2 * np.pi * 40 * t) * np.exp(-3 * (t - 1.0))
        data["values"]["f_normal"][m["key"]] = (contact * (15.0 + chatter) + rng.normal(0, 0.2, t.size)).tolist()
        lag = 0.08 * (1 - 0.7 * i)
        data["values"]["q1"][m["key"]] = (0.4 * np.sin(2 * np.pi * 0.5 * (t - lag)) + rng.normal(0, 0.003, t.size)).tolist()
    return data


def static_timeseries_svg(ts: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotstyle.apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for m in ts["methods"]:
        ax.plot(ts["t"], ts["values"]["f_normal"][m["key"]], color=plotstyle.method_color(m["key"]), label=m["label"])
    for ev in ts["events"]:
        ax.axvline(ev["t"], color=plotstyle.reference_color(), lw=0.6, ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normal contact force (N)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


class WellFormed(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ("href", "src") and v:
                self.links.append(v)
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> at line {self.getpos()[0]} does not close {self.stack[-1:] or 'anything'}")
        else:
            self.stack.pop()


def check_html(path: Path) -> None:
    parser = WellFormed()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    if parser.stack:
        parser.errors.append(f"unclosed tags {parser.stack}")
    assert not parser.errors, f"{path}: {parser.errors}"
    for link in parser.links:
        if link.startswith(("#", "http:", "https:", "mailto:")):
            continue
        target = (path.parent / link.split("#")[0]).resolve()
        assert target.exists(), f"{path}: link target missing: {link} -> {target}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep-site", type=Path, default=None, help="build the scratch site here and keep it")
    args = parser.parse_args()

    report_dir = RESEARCH / "reports" / REPORT_ID
    site_dir = args.keep_site.resolve() if args.keep_site else report_dir / "site"
    rng = np.random.default_rng(0)
    try:
        if report_dir.exists():
            shutil.rmtree(report_dir)
        (report_dir / "data").mkdir(parents=True)
        (report_dir / "figures").mkdir()
        cmp_data, ts_data = synthetic_compare(rng), synthetic_timeseries(rng)
        (report_dir / "data" / "compare.json").write_text(json.dumps(cmp_data), encoding="utf-8")
        (report_dir / "data" / "timeseries.json").write_text(json.dumps(ts_data), encoding="utf-8")
        static_timeseries_svg(ts_data, report_dir / "figures" / "timeseries.svg")
        fragment = "\n".join([
            "<h2>Comparison</h2>",
            ra.compare_figure("data/compare.json", cmp_data,
                              "Synthetic per-case metrics; the reference is PD with shared gains. Lower is better for every metric.",
                              "fig-compare", default_regime="contact", default_metric="peak_force"),
            "<h2>Time series</h2>",
            ra.timeseries_figure("data/timeseries.json",
                                 "Synthetic contact episode at 200 Hz; the dashed gray curve is the commanded joint position.",
                                 "fig-timeseries", fallback_svg_relpath="figures/timeseries.svg"),
            "<h2>Static figure and table</h2>",
            ra.figure("figures/timeseries.svg", "The static force plot, rendered with research/plotstyle.py.", "fig-static", wide=True),
            ra.table(["Case", "Peak force (N)", "Settling time (s)"], [["contact-00", 21.37, 0.42], ["contact-01", 19.9, None]],
                     caption="Example table with right-aligned numeric columns.", numeric_cols=(1, 2)),
        ])
        (report_dir / "report.html").write_text(fragment + "\n", encoding="utf-8")

        catalog = json.loads((RESEARCH / "catalog.json").read_text(encoding="utf-8"))
        assert all(r["id"] != REPORT_ID for r in catalog["reports"]), f"{REPORT_ID} must not be in research/catalog.json"
        catalog["reports"].append({"id": REPORT_ID, "title": "Self-check: widget fallbacks", "date": "2000-01-01", "milestone": 0,
                                   "status": "scratch", "finding": "Synthetic data only.", "protocol": "docs/protocols/does-not-exist.md"})
        temp_catalog = report_dir / "catalog.json"
        temp_catalog.write_text(json.dumps(catalog), encoding="utf-8")
        subprocess.run([sys.executable, str(RESEARCH / "build.py"), "--catalog", str(temp_catalog), "--out", str(site_dir)],
                       check=True, stdout=subprocess.DEVNULL)

        # Assertions on the built output.
        index = (site_dir / "index.html").read_text(encoding="utf-8")
        assert f'href="{REPORT_ID}/index.html"' in index and "Synthetic data only." in index
        page_path = site_dir / REPORT_ID / "index.html"
        page = page_path.read_text(encoding="utf-8")
        assert 'data-widget="compare"' in page and 'data-src="data/compare.json"' in page
        assert '<table class="compare-table">' in page and "Selection: Sustained contact, Peak normal force (N); n = 10 cases." in page
        assert 'data-widget="timeseries"' in page and '<img src="figures/timeseries.svg"' in page
        for rel in ("data/compare.json", "data/timeseries.json", "figures/timeseries.svg"):
            assert (site_dir / REPORT_ID / rel).is_file(), f"artifact not copied: {rel}"
        assert json.loads((site_dir / REPORT_ID / "data" / "compare.json").read_text()) == cmp_data
        # Static table equals a recomputation from the data file that ships with the page (source equality).
        shipped = json.loads((site_dir / REPORT_ID / "data" / "compare.json").read_text(encoding="utf-8"))
        for st in ra.compare_stats(shipped, "contact", "peak_force"):
            assert f'<td class="num">{ra.fmt(st["mean"])}</td>' in page, f"mean for {st['key']} missing from static table"
            if st["key"] == "neural_torque":
                assert '<td class="num">8 of 10</td>' in page
        assert re.findall(r'<h2 id="([^"]+)">', page) == ["comparison", "time-series", "static-figure-and-table"]
        assert page.count('<nav class="contents"') == 1 and page.count('<li><a href="#') == 3
        assert 'missing: docs/protocols/does-not-exist.md' in page
        assert '<figure id="fig-static" class="wide">' in page
        assert (site_dir / "assets" / "notebook.js").is_file() and (site_dir / "assets" / "style.css").is_file()
        for html_file in sorted(site_dir.rglob("*.html")):
            check_html(html_file)
        print(f"selfcheck passed ({len(list(site_dir.rglob('*.html')))} pages checked)")
        if args.keep_site:
            print(f"scratch site kept at {site_dir}")
    finally:
        shutil.rmtree(report_dir, ignore_errors=True)
        shutil.rmtree(RESEARCH / "site" / REPORT_ID, ignore_errors=True)
        if not args.keep_site:
            shutil.rmtree(site_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
