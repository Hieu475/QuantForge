#!/usr/bin/env python3
"""Static site generator for the QuantForge blog (no dependencies except `markdown`).

    python blog/build.py            # content/*.md + charts -> blog/site/ (open blog/site/index.html)

Posts are Markdown files in `content/` with a small front matter block (title, date, summary, order).
`{{chart:<name>}}` on its own line is replaced by an inline SVG chart (see CHARTS below; all numbers are
taken from the measurements documented in the repository README / python/quantforge/PIPELINE.md).
Everything is plain HTML + CSS: the output folder can be copied to any static host (GitHub Pages).
"""
import html
import json
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "site"
SITE = json.loads((ROOT / "site.json").read_text())


# --------------------------------------------------------------------------- charts
def _fmt(v, decimals):
    return f"{v:.{decimals}f}"


def hbar_chart(title, rows, xmax, ref=None, unit="", decimals=2, label_w=190, note=""):
    """Horizontal bars (one series). rows: [(label, value)]. Optional reference line at `ref`."""
    bar_h, gap, top, left, right = 20, 10, 18, label_w, 70
    H = top + len(rows) * (bar_h + gap) + 24
    W = 760
    pw = W - left - right
    x = lambda v: left + pw * v / xmax  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for t in range(0, int(xmax * 100) + 1, 25 if xmax <= 1.6 else 50):
        v = t / 100
        out.append(f'<line class="grid" x1="{x(v):.1f}" y1="{top - 6}" x2="{x(v):.1f}" y2="{H - 22}"/>')
        out.append(f'<text class="tick" x="{x(v):.1f}" y="{H - 6}" text-anchor="middle">{v:g}{unit}</text>')
    for i, (label, v) in enumerate(rows):
        y = top + i * (bar_h + gap)
        out.append(f'<text class="lab" x="{left - 10}" y="{y + bar_h * 0.72:.1f}" text-anchor="end">{html.escape(label)}</text>')
        out.append(f'<path class="s1" d="M{left} {y} H{x(v) - 4:.1f} a4 4 0 0 1 4 4 V{y + bar_h - 4} a4 4 0 0 1 -4 4 H{left} Z">'
                   f'<title>{html.escape(label)}: {_fmt(v, decimals)}{unit}</title></path>')
        out.append(f'<text class="val" x="{x(v) + 8:.1f}" y="{y + bar_h * 0.72:.1f}">{_fmt(v, decimals)}{unit}</text>')
    if ref is not None:
        out.append(f'<line class="ref" x1="{x(ref):.1f}" y1="{top - 8}" x2="{x(ref):.1f}" y2="{H - 22}"/>')
        out.append(f'<text class="reflab" x="{x(ref) + 5:.1f}" y="{top - 8}">Marlin = 1.0</text>')
    out.append("</svg>")
    return _figure(title, "".join(out), note=note)


def grouped_chart(title, groups, series, ymax, unit="", decimals=0, markers=None, marker_label="", note="",
                  table_cols=None):
    """Vertical grouped bars. groups: [label]; series: [(name, [values])] (<= 2 series -> slots 1, 2)."""
    W, H, left, right, top, bottom = 760, 330, 52, 10, 20, 62
    pw, ph = W - left - right, H - top - bottom
    gw = pw / len(groups)
    bw = min(34, gw * 0.7 / len(series))
    y = lambda v: top + ph * (1 - v / ymax)  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    step = ymax / 5
    for k in range(6):
        v = step * k
        out.append(f'<line class="grid" x1="{left}" y1="{y(v):.1f}" x2="{W - right}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    for gi, g in enumerate(groups):
        cx = left + gw * (gi + 0.5)
        x0 = cx - bw * len(series) / 2 - (len(series) - 1) * 1
        for si, (name, vals) in enumerate(series):
            v = vals[gi]
            bx = x0 + si * (bw + 2)
            by = y(v)
            out.append(f'<path class="s{si + 1}" d="M{bx:.1f} {H - bottom} V{by + 4:.1f} a4 4 0 0 1 4 -4 H{bx + bw - 4:.1f} '
                       f'a4 4 0 0 1 4 4 V{H - bottom} Z"><title>{html.escape(g)} - {html.escape(name)}: {_fmt(v, decimals)}{unit}</title></path>')
            out.append(f'<text class="val" x="{bx + bw / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle">{_fmt(v, decimals)}</text>')
        if markers:
            mv = markers[gi]
            out.append(f'<line class="marker" x1="{cx - bw * len(series) / 2 - 6:.1f}" y1="{y(mv):.1f}" '
                       f'x2="{cx + bw * len(series) / 2 + 6:.1f}" y2="{y(mv):.1f}"><title>{html.escape(g)} - {html.escape(marker_label)}: {mv:g}</title></line>')
        lines = g.split("\n")
        for li, line in enumerate(lines):
            out.append(f'<text class="lab" x="{cx:.1f}" y="{H - bottom + 18 + 14 * li}" text-anchor="middle">{html.escape(line)}</text>')
    out.append("</svg>")
    legend = "".join(f'<span class="lg"><i class="sw s{i + 1}"></i>{html.escape(n)}</span>' for i, (n, _) in enumerate(series))
    if markers:
        legend += f'<span class="lg"><i class="sw mk"></i>{html.escape(marker_label)}</span>'
    # accessible table view
    head = "".join(f"<th>{html.escape(n)}</th>" for n, _ in series) + (f"<th>{html.escape(marker_label)}</th>" if markers else "")
    rows = ""
    for gi, g in enumerate(groups):
        cells = "".join(f"<td>{_fmt(vals[gi], decimals)}</td>" for _, vals in series) + (f"<td>{markers[gi]:g}</td>" if markers else "")
        rows += f"<tr><th>{html.escape(g.replace(chr(10), ' '))}</th>{cells}</tr>"
    table = f"<details class='tv'><summary>Table view</summary><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></details>"
    return _figure(title, "".join(out), legend, note, table)


def _figure(title, svg, legend="", note="", table=""):
    leg = f'<div class="legend">{legend}</div>' if legend else ""
    nt = f'<figcaption>{note}</figcaption>' if note else ""
    return f'<figure class="viz"><div class="viz-title">{html.escape(title)}</div>{leg}{svg}{nt}{table}</figure>'


CHARTS = {
    "marlin_ratio": lambda: hbar_chart(
        "QuantForge speed relative to Marlin (higher is better)",
        [("1 x 4096 x 4096", 1.30), ("4 x 4096 x 4096", 1.30), ("16 x 4096 x 4096", 1.00),
         ("1 x 11008 x 4096", 0.94), ("1 x 4096 x 11008", 0.91), ("32 x 4096 x 4096", 0.88),
         ("16 x 4096 x 11008", 0.86), ("2048 x 4096 x 4096", 0.86), ("512 x 4096 x 4096", 0.85),
         ("256 x 4096 x 4096", 0.76), ("128 x 4096 x 4096", 0.69), ("64 x 4096 x 4096", 0.67)],
        1.5, ref=1.0, unit="x", decimals=2, label_w=150,
        note="M x K x N, group size 128, weights read from DRAM (L2 flushed), device kernel time, RTX 4050 Laptop. "
             "The line is Marlin (vLLM 0.25.1) on the same INT4 weights."),
    "kernel_us": lambda: grouped_chart(
        "Kernel time at small and medium batch (microseconds, lower is better)",
        ["M=1", "M=16", "M=32", "M=64", "M=128", "M=256"],
        [("Marlin", [49.4, 50.0, 50.8, 58.6, 104.1, 198.7]), ("QuantForge", [38.0, 50.0, 57.8, 86.9, 150.8, 261.1])],
        300, unit=" us", decimals=0,
        note="K = N = 4096, group size 128. Same method as above."),
    "e2e_speed": lambda: grouped_chart(
        "Batch-1 decode speed with CUDA graphs (tokens per second, higher is better)",
        ["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "Llama-3.2\n1B", "OLMo-2\n1B"],
        [("fp16 (Hugging Face)", [129, 49, 63, 60]), ("QuantForge INT4 + INT8 head", [209, 97, 142, 138])],
        240, decimals=0,
        note="RTX 4050 Laptop, 6 GB. Round-to-nearest INT4, group size 128. Qwen2.5-3B and Phi-3.5 do not fit in fp16 on this card."),
    "gptq_ppl": lambda: grouped_chart(
        "Wikitext-2 perplexity of the quantised models (lower is better)",
        ["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "Llama-3.2\n1B", "OLMo-2\n1B"],
        [("RTN INT4", [24.08, 15.31, 24.42, 21.31]), ("GPTQ INT4", [21.38, 15.00, 23.39, 20.62])],
        30, decimals=2, markers=[19.93, 13.48, 20.15, 19.32], marker_label="fp16 reference",
        note="8 x 512 test tokens; GPTQ calibrated on 128 windows of 512 tokens from the train split; INT8 head in both variants."),
}


# --------------------------------------------------------------------------- pages
def parse(path):
    text = path.read_text()
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    return meta, m.group(2)


def render(body):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
                           extension_configs={"toc": {"permalink": "#", "toc_depth": "2-3"}})
    out = md.convert(body)
    out = re.sub(r"<p>\{\{chart:(\w+)\}\}</p>", lambda mo: CHARTS[mo.group(1)](), out)
    return out, md.toc


LAYOUT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="stylesheet" href="{root}assets/style.css">
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href="{root}index.html">{site_title}</a>
  <nav><a href="{root}index.html">Posts</a><a href="{root}about.html">About</a>{gh}</nav>
</div></header>
<main class="wrap">
{content}
</main>
<footer><div class="wrap">{footer}</div></footer>
</body>
</html>
"""


def page(title, description, content, root=""):
    gh = f'<a href="{SITE["github_url"]}">GitHub</a>' if SITE.get("github_url") else ""
    return LAYOUT.format(title=html.escape(title), description=html.escape(description), content=content, root=root,
                         site_title=html.escape(SITE["title"]), gh=gh, footer=SITE["footer"])


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "posts").mkdir(parents=True)
    shutil.copytree(ROOT / "assets", OUT / "assets")
    posts = []
    for path in sorted((ROOT / "content").glob("*.md")):
        if path.stem == "about":
            continue
        meta, body = parse(path)
        content, toc = render(body)
        posts.append((int(meta.get("order", 99)), path.stem, meta))
        article = (f'<article class="post"><p class="meta">{html.escape(meta["date"])} &middot; {html.escape(SITE["author"])}</p>'
                   f'<h1>{html.escape(meta["title"])}</h1><p class="lede">{html.escape(meta["summary"])}</p>'
                   f'<details class="toc"><summary>Contents</summary>{toc}</details>{content}</article>')
        (OUT / "posts" / f"{path.stem}.html").write_text(page(meta["title"] + " - " + SITE["title"], meta["summary"], article, "../"))
    posts.sort()
    items = "".join(
        f'<li><a href="posts/{slug}.html"><span class="pt">{html.escape(m["title"])}</span></a>'
        f'<span class="pd">{html.escape(m["date"])}</span><p>{html.escape(m["summary"])}</p></li>' for _, slug, m in posts)
    intro, _ = render((ROOT / "content" / "index_intro.txt").read_text()) if (ROOT / "content" / "index_intro.txt").exists() else ("", "")
    index = f'<section class="intro">{intro}</section><h2 class="sec">Posts</h2><ol class="posts">{items}</ol>'
    (OUT / "index.html").write_text(page(SITE["title"], SITE["tagline"], index))
    meta, body = parse(ROOT / "content" / "about.md")
    content, _ = render(body)
    (OUT / "about.html").write_text(page("About - " + SITE["title"], SITE["tagline"], f'<article class="post"><h1>{html.escape(meta["title"])}</h1>{content}</article>'))
    (OUT / ".nojekyll").write_text("")
    print(f"built {len(posts)} posts -> {OUT}")


if __name__ == "__main__":
    main()
