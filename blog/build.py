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


UI = {
    "en": {"posts": "Posts", "about": "About", "contents": "Contents", "table": "Table view", "marlin": "Marlin = 1.0",
           "other": "Tiếng Việt", "lang": "en"},
    "vi": {"posts": "Bài viết", "about": "Giới thiệu", "contents": "Mục lục", "table": "Xem dạng bảng", "marlin": "Marlin = 1.0",
           "other": "English", "lang": "vi"},
}
LANG = "en"


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
        out.append(f'<text class="reflab" x="{x(ref) + 5:.1f}" y="{top - 8}">{UI[LANG]["marlin"]}</text>')
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
    table = f"<details class='tv'><summary>{UI[LANG]["table"]}</summary><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></details>"
    return _figure(title, "".join(out), legend, note, table)


def _figure(title, svg, legend="", note="", table=""):
    leg = f'<div class="legend">{legend}</div>' if legend else ""
    nt = f'<figcaption>{note}</figcaption>' if note else ""
    return f'<figure class="viz"><div class="viz-title">{html.escape(title)}</div>{leg}{svg}{nt}{table}</figure>'


T = {
    "marlin_ratio": {
        "en": ("QuantForge speed relative to Marlin (higher is better)",
               "M x K x N, group size 128, weights read from DRAM (L2 flushed), device kernel time, RTX 4050 Laptop. "
               "The line is Marlin (vLLM 0.25.1) on the same INT4 weights."),
        "vi": ("Tốc độ QuantForge so với Marlin (càng cao càng tốt)",
               "M x K x N, group size 128, weight đọc từ DRAM (đã flush L2), thời gian kernel trên device, RTX 4050 Laptop. "
               "Đường nét đứt là Marlin (vLLM 0.25.1) trên cùng weight INT4."),
    },
    "kernel_us": {
        "en": ("Kernel time at small and medium batch (microseconds, lower is better)", "K = N = 4096, group size 128. Same method as above."),
        "vi": ("Thời gian kernel ở batch nhỏ và vừa (micro giây, càng thấp càng tốt)", "K = N = 4096, group size 128. Cùng phương pháp đo như trên."),
    },
    "e2e_speed": {
        "en": ("Batch-1 decode speed with CUDA graphs (tokens per second, higher is better)",
               "RTX 4050 Laptop, 6 GB. Round-to-nearest INT4, group size 128. Qwen2.5-3B and Phi-3.5 do not fit in fp16 on this card."),
        "vi": ("Tốc độ decode batch 1 với CUDA graph (token mỗi giây, càng cao càng tốt)",
               "RTX 4050 Laptop, 6 GB. INT4 round-to-nearest, group size 128. Qwen2.5-3B và Phi-3.5 không vừa fp16 trên card này."),
    },
    "gptq_ppl": {
        "en": ("Wikitext-2 perplexity of the quantised models (lower is better)",
               "Whole Wikitext-2 test split (564-584 windows of 512 tokens); GPTQ calibrated on 128 windows of 512 tokens from the train split; INT8 head in both variants."),
        "vi": ("Perplexity Wikitext-2 của các model đã lượng tử hóa (càng thấp càng tốt)",
               "Toàn bộ tập test Wikitext-2 (564-584 cửa sổ 512 token); GPTQ hiệu chỉnh trên 128 cửa sổ 512 token từ tập train; head INT8 ở cả hai biến thể."),
    },
}
LEG = {"en": {"fp16": "fp16 (Hugging Face)", "qf": "QuantForge INT4 + INT8 head", "ref": "fp16 reference", "rtn": "RTN INT4", "gptq": "GPTQ INT4"},
       "vi": {"fp16": "fp16 (Hugging Face)", "qf": "QuantForge INT4 + head INT8", "ref": "mốc fp16", "rtn": "RTN INT4", "gptq": "GPTQ INT4"}}


def chart(name):
    ttl, note = T[name][LANG]
    lg = LEG[LANG]
    if name == "marlin_ratio":
        return hbar_chart(ttl, [("16 x 4096 x 4096", 1.01), ("1 x 4096 x 4096", 0.99), ("4 x 4096 x 4096", 0.99), ("1 x 11008 x 4096", 0.97), ("16 x 4096 x 11008", 0.94), ("1 x 4096 x 11008", 0.93), ("32 x 4096 x 4096", 0.87), ("2048 x 4096 x 4096", 0.86), ("512 x 4096 x 4096", 0.85), ("256 x 4096 x 4096", 0.76), ("128 x 4096 x 4096", 0.69), ("64 x 4096 x 4096", 0.67)],
                          1.25, ref=1.0, unit="x", decimals=2, label_w=150, note=note)
    if name == "kernel_us":
        return grouped_chart(ttl, ["M=1", "M=16", "M=32", "M=64", "M=128", "M=256"],
                             [("Marlin", [49.6, 50.0, 50.9, 58.3, 105.1, 201.0]), ("QuantForge", [50.0, 49.8, 58.8, 87.4, 152.2, 263.0])],
                             300, unit=" us", decimals=0, note=note)
    if name == "e2e_speed":
        return grouped_chart(ttl, ["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "Llama-3.2\n1B", "OLMo-2\n1B"],
                             [(lg["fp16"], [129, 49, 63, 60]), (lg["qf"], [209, 97, 142, 138])], 240, decimals=0, note=note)
    if name == "gptq_ppl":
        return grouped_chart(ttl, ["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "Llama-3.2\n1B", "OLMo-2\n1B"],
                             [(lg["rtn"], [22.91, 14.30, 22.28, 19.02]), (lg["gptq"], [20.20, 13.38, 20.12, 18.35])],
                             25, decimals=2, markers=[18.79, 12.66, 17.91, 17.24], marker_label=lg["ref"], note=note)
    raise KeyError(name)


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
    out = re.sub(r"<p>\{\{chart:(\w+)\}\}</p>", lambda mo: chart(mo.group(1)), out)
    return out, md.toc


LAYOUT = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="stylesheet" href="{assets}assets/style.css">
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href="{root}index.html">{site_title}</a>
  <nav><a href="{home}">{t_posts}</a><a href="{about}">{t_about}</a>{gh}<a class="langsw" href="{other_href}" hreflang="{other_lang}">{other}</a></nav>
</div></header>
<main class="wrap">
{content}
</main>
<footer><div class="wrap">{footer}</div></footer>
</body>
</html>
"""


def page(title, description, content, root="", rel="index.html"):
    """root: relative path from this page to the site root; rel: this page's path inside its language folder."""
    ui = UI[LANG]
    prefix = "" if LANG == "en" else "vi/"
    other_prefix = "vi/" if LANG == "en" else ""
    # page folder depth inside the language folder = rel.count("/"); root already includes the language folder level
    to_site_root = root if LANG == "en" else root + "../"
    gh = f'<a href="{SITE["github_url"]}">GitHub</a>' if SITE.get("github_url") else ""
    S = SITE[LANG]
    return LAYOUT.format(title=html.escape(title), description=html.escape(description), content=content, root=root, lang=ui["lang"],
                         site_title=html.escape(S["title"]), assets=to_site_root, gh=gh, footer=S["footer"], home=f"{root}index.html", about=f"{root}about.html",
                         t_posts=ui["posts"], t_about=ui["about"], other=ui["other"], other_lang=("vi" if LANG == "en" else "en"),
                         other_href=f"{to_site_root}{other_prefix}{rel}")


def build_language(lang, out_dir, content_dir, root_to_site):
    global LANG
    LANG = lang
    S = SITE[lang]
    (out_dir / "posts").mkdir(parents=True, exist_ok=True)
    posts = []
    for path in sorted(content_dir.glob("*.md")):
        if path.stem == "about":
            continue
        meta, body = parse(path)
        content, toc = render(body)
        posts.append((int(meta.get("order", 99)), path.stem, meta))
        article = (f'<article class="post"><p class="meta">{html.escape(meta["date"])} &middot; {html.escape(SITE["author"])}</p>'
                   f'<h1>{html.escape(meta["title"])}</h1><p class="lede">{html.escape(meta["summary"])}</p>'
                   f'<details class="toc"><summary>{UI[lang]["contents"]}</summary>{toc}</details>{content}</article>')
        (out_dir / "posts" / f"{path.stem}.html").write_text(
            page(meta["title"] + " - " + S["title"], meta["summary"], article, "../", f"posts/{path.stem}.html"))
    posts.sort()
    items = "".join(
        f'<li><a href="posts/{slug}.html"><span class="pt">{html.escape(m["title"])}</span></a>'
        f'<span class="pd">{html.escape(m["date"])}</span><p>{html.escape(m["summary"])}</p></li>' for _, slug, m in posts)
    intro_path = content_dir / "index_intro.txt"
    intro, _ = render(intro_path.read_text()) if intro_path.exists() else ("", "")
    index = f'<section class="intro">{intro}</section><h2 class="sec">{UI[lang]["posts"]}</h2><ol class="posts">{items}</ol>'
    (out_dir / "index.html").write_text(page(S["title"], S["tagline"], index, "", "index.html"))
    meta, body = parse(content_dir / "about.md")
    content, _ = render(body)
    (out_dir / "about.html").write_text(page(meta["title"] + " - " + S["title"], S["tagline"],
                                            f'<article class="post"><h1>{html.escape(meta["title"])}</h1>{content}</article>', "", "about.html"))
    return len(posts)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "assets", OUT / "assets")
    n_en = build_language("en", OUT, ROOT / "content", "")
    n_vi = build_language("vi", OUT / "vi", ROOT / "content" / "vi", "../")
    (OUT / ".nojekyll").write_text("")
    print(f"built {n_en} English + {n_vi} Vietnamese posts -> {OUT}")


if __name__ == "__main__":
    main()
