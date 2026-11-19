# QuantForge Notes (the blog)

A dependency-light static blog: Markdown posts in `content/`, a ~200-line generator `build.py` (Python +
the `markdown` package), plain CSS with light / dark themes, and inline-SVG charts generated from the measured numbers.

```bash
pip install markdown
python blog/build.py          # writes blog/site/ ; open blog/site/index.html in a browser
```

* Add a post: create `content/NN-name.md` with a front matter block (`title`, `date`, `order`, `summary`) and run the build.
  A line `{{chart:name}}` is replaced by a chart defined in `build.py` (`CHARTS`).
* Site title, author and the GitHub link live in `site.json` (leave `github_url` empty to hide the link).
* `site/` is the complete output (it also contains `.nojekyll`, so GitHub Pages serves it as is).

## Publishing on GitHub Pages

Option A, a personal site at `https://<user>.github.io/`:

1. Create a public repository named exactly `<user>.github.io`.
2. Copy the contents of `blog/site/` to the root of that repository, commit, push.
3. Settings -> Pages -> Deploy from branch `main`, folder `/ (root)`. The site is live within a minute or two.

Option B, a project site at `https://<user>.github.io/<repo>/`: copy `blog/site/` to the `docs/` folder of the
QuantForge repository and enable Pages for `main` / `/docs`. (The generated pages use relative links, so both work.)
