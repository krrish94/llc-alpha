#!/usr/bin/env python3
"""Build the static research notebook.

    uv run python research/build.py [--catalog research/catalog.json] [--out research/site]

research/catalog.json owns navigation and report metadata, research/design.json the palette,
research/templates/ the layout and research/assets/ the stylesheet and widget script.
Reports are HTML fragments at research/reports/<id>/report.html (body content from h2 down);
pages are Markdown files from docs/. All links are relative so the site works from file://.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from pathlib import Path
from string import Template

import markdown

RESEARCH = Path(__file__).resolve().parent
ROOT = RESEARCH.parent
MD_EXTENSIONS = ["tables", "fenced_code"]


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def read_template(name: str) -> Template:
    return Template((RESEARCH / "templates" / name).read_text(encoding="utf-8"))


def relative_href(target: Path, page_dir: Path) -> str:
    rel = Path(os.path.relpath(target, page_dir)).as_posix()
    return "" if rel == "." else rel + ("/" if target.is_dir() else "")


def missing_notice(relpath: str) -> str:
    return f'<p class="missing">missing: {esc(relpath)}</p>'


# ---------- content transforms ----------


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=MD_EXTENSIONS)


def split_title(text: str) -> tuple[str | None, str]:
    """Return (first-line H1 title, remaining text) for a Markdown document."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            if line.startswith("# "):
                return line[2:].strip(), "\n".join(lines[i + 1:])
            break
    return None, text


def demote_headings(body: str) -> str:
    return re.sub(r"<(/?)h([1-5])\b", lambda m: f"<{m.group(1)}h{int(m.group(2)) + 1}", body)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def add_heading_ids(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Give each <h2> an id when it has none; return the body and the (id, text) list."""
    seen: set[str] = set()
    entries: list[tuple[str, str]] = []

    def replace(m: re.Match) -> str:
        attrs, inner = m.group(1) or "", m.group(2)
        text = re.sub(r"<[^>]+>", "", inner).strip()
        existing = re.search(r'\bid="([^"]*)"', attrs)
        if existing:
            hid = existing.group(1)
        else:
            base = slugify(text) or "section"
            hid, k = base, 2
            while hid in seen:
                hid, k = f"{base}-{k}", k + 1
            attrs = f' id="{hid}"' + attrs
        seen.add(hid)
        entries.append((hid, text))
        return f"<h2{attrs}>{inner}</h2>"

    return re.sub(r"<h2(\s[^>]*)?>(.*?)</h2>", replace, body, flags=re.S), entries


def contents_nav(entries: list[tuple[str, str]]) -> str:
    if len(entries) < 2:
        return ""
    items = "\n".join(f'<li><a href="#{esc(hid)}">{text}</a></li>' for hid, text in entries)
    return f'<nav class="contents" aria-labelledby="contents-label">\n<p class="nav-label" id="contents-label">Contents</p>\n<ol>\n{items}\n</ol>\n</nav>'


def palette_css(design: dict) -> str:
    decls = [f"--{k.replace('_', '-')}: {v}" for k, v in design.items() if isinstance(v, str)]
    decls += [f"--method-{k}: {v}" for k, v in design["methods"].items()]
    return ":root{" + "; ".join(decls) + "}"


# ---------- site builder ----------


class Site:
    def __init__(self, catalog: dict, design: dict, out: Path):
        self.catalog, self.design, self.out = catalog, design, out
        self.base = read_template("base.html")
        self.templates = {name: read_template(f"{name}.html") for name in ("index", "report", "page")}
        self.rendered_protocols: dict[str, Path] = {}

    def nav_html(self, page_dir: Path, current_id: str | None) -> str:
        def items(entries: list[dict]) -> str:
            out = []
            for e in entries:
                href = relative_href(self.out / e["id"] / "index.html", page_dir)
                current = ' aria-current="page"' if e["id"] == current_id else ""
                out.append(f'<li><a href="{esc(href)}"{current}>{esc(e["title"])}</a></li>')
            return "\n".join(out)
        nav = ("<p class=\"nav-label\" id=\"nav-experiments\">Experiments</p>\n"
               f"<ul aria-labelledby=\"nav-experiments\">\n{items(self.catalog['reports'])}\n</ul>")
        if self.catalog["pages"]:  # a public export has no project pages
            nav += ("\n<p class=\"nav-label\" id=\"nav-project\">Project</p>\n"
                    f"<ul aria-labelledby=\"nav-project\">\n{items(self.catalog['pages'])}\n</ul>")
        return nav

    def rewrite_links(self, body: str, source_file: Path, page_dir: Path) -> str:
        """Point relative links in rendered Markdown at the matching site page, else at the repo file."""
        targets = {(ROOT / e["source"]).resolve(): self.out / e["id"] / "index.html" for e in self.catalog["pages"] if "source" in e}
        targets.update({(ROOT / rel).resolve(): path for rel, path in self.rendered_protocols.items()})

        def replace(m: re.Match) -> str:
            attr, quote, href = m.group(1), m.group(2), m.group(3)
            if not href or href.startswith(("#", "/")) or re.match(r"^[a-z][a-z0-9+.-]*:", href, re.I):
                return m.group(0)
            path_part, _, fragment = href.partition("#")
            target = (source_file.parent / path_part).resolve()
            new_href = relative_href(targets.get(target, target), page_dir) + (f"#{fragment}" if fragment else "")
            return f"{attr}={quote}{new_href}{quote}"

        return re.sub(r"""\b(href|src)=(["'])([^"']*)\2""", replace, body)

    def write_page(self, path: Path, title: str, main: str, current_id: str | None = None) -> None:
        page_dir = path.parent
        page_dir.mkdir(parents=True, exist_ok=True)
        project = self.catalog["project"]["title"]
        document = self.base.substitute(
            title=esc(f"{title} — {project}") if title != project else esc(project),
            root=relative_href(self.out, page_dir),
            palette_css=palette_css(self.design),
            project_title=esc(project),
            nav=self.nav_html(page_dir, current_id),
            main=main,
        )
        path.write_text(document, encoding="utf-8")

    def build(self) -> Path:
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        shutil.copytree(RESEARCH / "assets", self.out / "assets")
        self.build_index()
        for entry in self.catalog["reports"]:
            if entry.get("protocol"):
                self.build_protocol(entry["protocol"])
        for entry in self.catalog["pages"]:
            self.build_doc_page(entry)
        for entry in self.catalog["reports"]:
            self.build_report(entry)
        self.strip_dead_links()
        return self.out / "index.html"

    def strip_dead_links(self) -> None:
        """Links to site pages that were not built (a tree without the project documents) become their text."""
        for page in self.out.rglob("*.html"):
            html_text = page.read_text(encoding="utf-8")

            def replace(m: re.Match) -> str:
                href = m.group(1)
                if href.startswith("#") or re.match(r"^[a-z][a-z0-9+.-]*:", href, re.I):
                    return m.group(0)
                return m.group(0) if (page.parent / href.partition("#")[0]).resolve().exists() else m.group(2)

            new = re.sub(r'<a href="([^"]+)"[^>]*>(.*?)</a>', replace, html_text, flags=re.S)
            if new != html_text:
                page.write_text(new, encoding="utf-8")

    def build_index(self) -> None:
        rows = []
        for e in self.catalog["reports"]:
            rows.append(
                f'<tr><td><a href="{esc(e["id"])}/index.html">{esc(e["title"])}</a></td>'
                f'<td class="date"><time datetime="{esc(e["date"])}">{esc(e["date"])}</time></td>'
                f'<td>{esc(e.get("finding", ""))}</td><td>{esc(e.get("status", ""))}</td></tr>')
        main = self.templates["index"].substitute(rows="\n".join(rows))
        self.write_page(self.out / "index.html", self.catalog["project"]["title"], main)

    def build_doc_page(self, entry: dict) -> None:
        page_dir = self.out / entry["id"]
        subtitle = None
        if "source" in entry:
            source = entry["source"]
            path = ROOT / source
            if path.is_file():
                # The catalog title names the page; a leading H1 in the file is kept as metadata, not a second h1.
                subtitle, rest = split_title(path.read_text(encoding="utf-8"))
                body = self.rewrite_links(md_to_html(rest), path, page_dir)
            else:
                body = missing_notice(source)
        else:
            source = entry["source_dir"].rstrip("/") + "/"
            directory = ROOT / entry["source_dir"]
            if not directory.is_dir():
                body = missing_notice(source)
            else:
                parts = []
                for md_file in sorted(directory.glob("*.md")):
                    title, rest = split_title(md_file.read_text(encoding="utf-8"))
                    rendered = self.rewrite_links(demote_headings(md_to_html(rest)), md_file, page_dir)
                    parts.append(f"<h2>{esc(title or md_file.stem)}</h2>\n{rendered}")
                body = "\n".join(parts) or f'<p class="muted">No files in {esc(source)}.</p>'
        body, headings = add_heading_ids(body)
        meta = f'Source: <a href="{esc(relative_href(ROOT / source, page_dir))}">{esc(source)}</a>'
        if subtitle and subtitle != entry["title"]:
            meta = f"{esc(subtitle)} \u00b7 {meta}"
        main = self.templates["page"].substitute(title=esc(entry["title"]), meta=meta, contents=contents_nav(headings), body=body)
        self.write_page(page_dir / "index.html", entry["title"], main, entry["id"])

    def build_protocol(self, relpath: str) -> Path | None:
        """Render a Markdown protocol once as a site page; return its output path or None if missing."""
        if relpath in self.rendered_protocols:
            return self.rendered_protocols[relpath]
        source = ROOT / relpath
        if not source.is_file():
            return None
        page_dir = self.out / "protocols" / source.stem
        title, rest = split_title(source.read_text(encoding="utf-8"))
        title = title or source.stem
        body, headings = add_heading_ids(self.rewrite_links(md_to_html(rest), source, page_dir))
        meta = f'Protocol · Source: <a href="{esc(relative_href(source, page_dir))}">{esc(relpath)}</a>'
        main = self.templates["page"].substitute(title=esc(title), meta=meta, contents=contents_nav(headings), body=body)
        self.write_page(page_dir / "index.html", title, main)
        self.rendered_protocols[relpath] = page_dir / "index.html"
        return self.rendered_protocols[relpath]

    def build_report(self, entry: dict) -> None:
        rid = entry["id"]
        source_dir = RESEARCH / "reports" / rid
        page_dir = self.out / rid
        page_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("figures", "data", "media"):
            if (source_dir / sub).is_dir():
                shutil.copytree(source_dir / sub, page_dir / sub, dirs_exist_ok=True)
        fragment = source_dir / "report.html"
        body = fragment.read_text(encoding="utf-8") if fragment.is_file() else missing_notice(f"research/reports/{rid}/report.html")
        body, headings = add_heading_ids(body)

        meta = [f'<time datetime="{esc(entry["date"])}">{esc(entry["date"])}</time>']
        if entry.get("milestone") is not None:
            meta.append(f"Milestone {esc(entry['milestone'])}")
        if entry.get("status"):
            meta.append(esc(entry["status"]))
        if entry.get("protocol"):
            rendered = self.build_protocol(entry["protocol"])
            if rendered:
                meta.append(f'Protocol: <a href="{esc(relative_href(rendered, page_dir))}">{esc(Path(entry["protocol"]).stem)}</a>')
            else:
                meta.append(f'Protocol: <span class="missing">missing: {esc(entry["protocol"])}</span>')
        source_rel = f"research/reports/{rid}/"
        meta.append(f'Source: <a href="{esc(relative_href(source_dir, page_dir))}">{esc(source_rel)}</a>')

        main = self.templates["report"].substitute(
            title=esc(entry["title"]), meta=" · ".join(meta), contents=contents_nav(headings), body=body)
        self.write_page(page_dir / "index.html", entry["title"], main, rid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the research notebook into a static site.")
    parser.add_argument("--catalog", type=Path, default=RESEARCH / "catalog.json")
    parser.add_argument("--out", type=Path, default=RESEARCH / "site")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    design = json.loads((RESEARCH / "design.json").read_text(encoding="utf-8"))
    index = Site(catalog, design, args.out.resolve()).build()
    print(index)


if __name__ == "__main__":
    main()
