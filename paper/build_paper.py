"""
Build paper/wsse.html and paper/wsse.pdf from paper/wsse.md without pandoc or LaTeX: the page renders
the Markdown in the browser (marked) and typesets the maths (MathJax), and headless Chrome or Edge
prints it to PDF. Needs an internet connection for the two scripts (cdn.jsdelivr.net).

  python paper/build_paper.py                 HTML + PDF (finds Chrome or Edge)
  python paper/build_paper.py --html-only
  python paper/build_paper.py --browser "C:/path/to/chrome.exe"
"""
import argparse, html, json, os, re, shutil, subprocess, sys, tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MARKED = "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"
MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]

CSS = """
:root { --ink: #1b1b1b; --muted: #555; --rule: #c9c9c9; --head: #f2f2f2; --link: #1a4f8a; }
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 9pt Georgia, serif; color: #666; } }
html { background: #fff; }
body { max-width: 178mm; margin: 0 auto; padding: 12mm 0; color: var(--ink); background: #fff;
       font: 10.5pt/1.45 Charter, "Bitstream Charter", Cambria, Georgia, serif; }
header.title { text-align: center; margin-bottom: 8mm; }
header.title h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 4mm; border: none; }
header.title .authors { font-size: 11.5pt; }
header.title .date { color: var(--muted); font-size: 10pt; margin-top: 1.5mm; }
header.title .links { font-size: 10.5pt; margin-top: 2.5mm; }
h1 { font-size: 14pt; margin: 9mm 0 3mm; padding-bottom: 1mm; border-bottom: 1px solid var(--rule); break-after: avoid; }
h2 { font-size: 12.5pt; margin: 7mm 0 2mm; break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm; break-after: avoid; }
p { margin: 1.6mm 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 1mm 0 1.6mm; padding-left: 6mm; }
li { margin: 0.6mm 0; }
li > p { margin: 0.6mm 0; }
code { font: 8.8pt/1.3 Consolas, "DejaVu Sans Mono", monospace; background: #f4f4f4; padding: 0 0.6mm; border-radius: 1px; }
pre { background: #f6f6f6; padding: 2mm 3mm; overflow-x: auto; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 2.5mm auto; font-size: 8.4pt; line-height: 1.3; break-inside: avoid; max-width: 100%; }
th, td { border: 1px solid var(--rule); padding: 0.8mm 1.6mm; vertical-align: top; }
th { background: var(--head); text-align: left; }
img { display: block; max-width: 100%; margin: 3mm auto 1mm; break-inside: avoid; }
p:has(> img) + p > em:only-child { display: block; text-align: center; font-size: 9pt; color: var(--muted); }
a { color: var(--link); text-decoration: none; }
mjx-container[display="true"] { margin: 2mm 0 !important; }
mjx-container { font-size: 96% !important; }
#refs + ol, .refs p { font-size: 9.2pt; }
@media print { body { padding: 0; } main > :last-child { margin-bottom: 0; } header.title .links { display: none; } }
@media (max-width: 700px) { body { padding: 6mm 4mm; font-size: 11pt; } table { font-size: 8pt; display: block; overflow-x: auto; } }
"""

def front_matter(md):
    m = re.match(r"---\n(.*?)\n---\n", md, re.S)
    if not m: return {}, md
    meta, key = {}, None
    for line in m.group(1).splitlines():
        if re.match(r"^\s+- ", line) and key: meta.setdefault(key, []).append(line.strip()[2:].strip().strip('"'))
        elif ":" in line:
            key, val = line.split(":", 1); key = key.strip(); val = val.strip().strip('"')
            if val: meta[key] = val
    return meta, md[m.end():]

def head_meta(meta, authors, abstract, pdf_name):
    """Sharing previews (Open Graph) and the citation_* tags Google Scholar reads, when the front
    matter names the published site (`site:`); `doi:` and `image:` are optional."""
    site, doi, title = meta.get("site", ""), meta.get("doi", ""), meta.get("title", "")
    tags = [("name", "description", abstract)]
    if site:
        site = site.rstrip("/") + "/"
        tags += [("name", "citation_title", title)] + [("name", "citation_author", a) for a in authors]
        try: tags.append(("name", "citation_publication_date", datetime.strptime(meta.get("date", "").split(" (")[0], "%B %d, %Y").strftime("%Y/%m/%d")))
        except ValueError: pass
        tags += [("name", "citation_pdf_url", site + pdf_name), ("property", "og:type", "article"), ("property", "og:title", title),
                 ("property", "og:description", abstract), ("property", "og:url", site)]
        if meta.get("image"): tags += [("property", "og:image", site + meta["image"]), ("name", "twitter:card", "summary_large_image")]
    if doi: tags.append(("name", "citation_doi", doi))
    return "\n".join(f'<meta {k}="{v}" content="{html.escape(c, quote=True)}">' for k, v, c in tags)

def build_html(md_path, html_path):
    meta, body = front_matter(open(md_path, encoding="utf-8").read())
    body = re.sub(r"<!--.*?-->\n?", "", body, flags=re.S)
    authors = meta.get("author", [])
    authors = authors if isinstance(authors, list) else [authors]
    pdf_name = os.path.splitext(os.path.basename(md_path))[0] + ".pdf"
    m = re.search(r"## Abstract\s+(.+?)\n\n", body, re.S)
    abstract = re.sub(r"\s+", " ", re.sub(r"[*`_]|\s*\[\d+(?:, *\d+)*\]", "", m.group(1))).strip() if m else ""
    links = [("PDF", pdf_name)] + ([("Code", meta["code"])] if meta.get("code") else []) \
            + ([(f"DOI {meta['doi']}", "https://doi.org/" + meta["doi"])] if meta.get("doi") else [])
    head = (f'<header class="title"><h1>{html.escape(meta.get("title", ""))}</h1>'
            f'<div class="authors">{" &nbsp;·&nbsp; ".join(html.escape(a) for a in authors)}</div>'
            f'<div class="date">{html.escape(meta.get("date", ""))}</div>'
            f'<nav class="links">{" &nbsp;·&nbsp; ".join(f"<a href={chr(34)}{html.escape(u)}{chr(34)}>{html.escape(t)}</a>" for t, u in links)}</nav></header>')
    payload = json.dumps(body).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(meta.get("title", "paper"))}</title>
{head_meta(meta, authors, abstract, pdf_name)}
<style>{CSS}</style>
<script>window.MathJax = {{ tex: {{ inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] }},
  svg: {{ fontCache: "global" }}, startup: {{ typeset: false }} }};</script>
<script src="{MARKED}"></script><script src="{MATHJAX}"></script>
</head><body>{head}<main id="paper"></main>
<script id="md" type="application/json">{payload}</script>
<script>
(function () {{
  const md = JSON.parse(document.getElementById("md").textContent), math = [];
  const keep = s => "@@M" + (math.push(s) - 1) + "@@";
  let src = md.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, (m, t) => keep("\\\\[" + t + "\\\\]"));
  src = src.replace(/(^|[^\\\\$])\\$([^\\s$](?:[^$\\n]*?[^\\s$])?)\\$/g, (m, pre, t) => pre + keep("\\\\(" + t + "\\\\)"));
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  document.getElementById("paper").innerHTML = marked.parse(src, {{ gfm: true }}).replace(/@@M(\\d+)@@/g, (m, i) => esc(math[+i]));
  MathJax.startup.promise.then(() => MathJax.typesetPromise()).then(() => document.body.setAttribute("data-ready", "1"));
}})();
</script></body></html>
"""
    open(html_path, "w", encoding="utf-8", newline="\n").write(page)

def find_browser(explicit):
    for b in ([explicit] if explicit else BROWSERS):
        if b and (os.path.exists(b) or shutil.which(b)): return shutil.which(b) or b
    raise SystemExit("no Chrome/Edge found; pass --browser")

def print_copy(html_path, tmp, max_width=1800, quality=90):
    """The PDF uses JPEG copies of the PNG figures (the repository keeps the lossless PNGs)."""
    from PIL import Image
    page = open(html_path, encoding="utf-8").read(); figdir = os.path.join(os.path.dirname(html_path), "figures")
    for f in sorted(set(re.findall(r"figures/([\w.-]+\.png)", page))):
        im = Image.open(os.path.join(figdir, f)).convert("RGB")
        if im.width > max_width: im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        jpg = os.path.join(tmp, f[:-4] + ".jpg"); im.save(jpg, quality=quality, optimize=True)
        page = page.replace(f"figures/{f}", "file:///" + jpg.replace("\\", "/"))
    out = os.path.join(tmp, "print.html"); open(out, "w", encoding="utf-8").write(page)
    return out

def build_pdf(html_path, pdf_path, browser):
    tmp = tempfile.mkdtemp(prefix="wsse-pdf-")
    try:
        src = print_copy(html_path, tmp)
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={os.path.join(tmp, 'profile')}", "--no-pdf-header-footer", "--print-to-pdf-no-header",
               "--allow-file-access-from-files", "--virtual-time-budget=60000", f"--print-to-pdf={pdf_path}",
               "file:///" + src.replace("\\", "/")]
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--browser"); ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--md", default=os.path.join(HERE, "wsse.md"))
    a = ap.parse_args()
    out_html = os.path.splitext(a.md)[0] + ".html"; build_html(a.md, out_html); print("wrote", out_html)
    if not a.html_only:
        out_pdf = os.path.splitext(a.md)[0] + ".pdf"; build_pdf(out_html, out_pdf, find_browser(a.browser))
        print("wrote", out_pdf, f"({os.path.getsize(out_pdf) / 1e6:.1f} MB)")
