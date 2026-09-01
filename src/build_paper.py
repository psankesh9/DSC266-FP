"""Render paper.md to a print-ready, two-column paper.html.

This is the submission document: ACL-style two columns, 4-6 pages. It is a
separate pipeline from build_report.py, which renders the longer single-column
report.md that this paper condenses.

Layout: the title block and abstract span the full page width, the body runs in
two columns, and every table and figure spans both columns. Wide tables in a
half-width column are unreadable, and ACL papers span them by convention.
Figures from outputs/plots are inlined as data URIs so the file can be moved or
emailed without breaking.
"""
import base64
import re
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "outputs" / "plots"

# (figure file, caption) keyed by the h3 heading they follow. The long report
# carries a fourth figure (the gazetteer ablation); at this length that result
# is one sentence in 3.2 and does not earn a panel.
FIGURES = {
    "3.2 Task formulation and the candidate ceiling": [
        ("fig2_candidate_window_sweep.png",
         "Figure 1: Candidate recall (the ceiling) and mean candidates per "
         "quotation as the enumeration window widens, dev. Explicit recall is "
         "flat; the whole gain is on anaphoric and implicit quotations."),
    ],
    "4.1 Main results": [
        ("fig1_headroom_by_quote_type.png",
         "Figure 2: Accuracy by quote type against the candidate ceiling, dev. "
         "The bar a system does not fill is its avoidable error; the gap above "
         "the ceiling is unreachable."),
    ],
    "4.3 Error analysis": [
        ("fig3_per_novel_variation.png",
         "Figure 3: Per-novel accuracy, dev. Between-novel spread exceeds the "
         "gaps between systems, which is why every interval in this paper is "
         "bootstrapped over novels."),
    ],
}

CSS = """
  body {
    font-family: "Times New Roman", Times, serif;
    font-size: 10pt;
    line-height: 1.26;
    color: #000000;
    background: #ffffff;
    max-width: 7.0in;
    margin: 0 auto;
    padding: 0;
    text-align: justify;
  }
  .titleblock { text-align: center; margin-bottom: 14pt; }
  .titleblock h1 {
    font-size: 15pt; font-weight: bold; margin: 0 0 8pt 0; line-height: 1.25;
    text-align: center;
  }
  .byline { font-size: 10.5pt; margin: 0 0 12pt 0; text-align: center; }
  .abstract { margin: 0 auto 6pt auto; width: 85%; }
  .abstract h2 {
    font-size: 11pt; text-align: center; margin: 0 0 5pt 0; border: none;
  }
  .abstract p { font-size: 9.5pt; margin: 0; }
  .cols { column-count: 2; column-gap: 0.28in; }
  h2 {
    font-size: 11.5pt; font-weight: bold; margin: 11pt 0 4pt 0;
    break-after: avoid;
  }
  h3 {
    font-size: 10.5pt; font-weight: bold; font-style: italic;
    margin: 9pt 0 3pt 0; break-after: avoid;
  }
  p { margin: 0 0 5pt 0; }
  /* Wide tables span both columns: a 6-column table with bracketed intervals
     set in a half-width column is illegible, and spanning is the ACL
     convention. Narrow ones stay in-column, since every span forces a column
     break and costs vertical space. build_paper.py picks per table. */
  table.wide { column-span: all; }
  table { break-inside: avoid; }
  /* Figures stay column-width: spanning them costs roughly a page of vertical
     space each, and these three read fine at half width. */
  figure { break-inside: avoid; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 5pt 0 2pt 0;
    font-size: 7.8pt;
    text-align: left;
  }
  th, td { border: none; padding: 2.5pt 4pt; vertical-align: top; }
  thead th { border-bottom: 0.75pt solid #000000; font-weight: bold; }
  thead tr:first-child th { border-top: 0.75pt solid #000000; }
  tbody tr:last-child td { border-bottom: 0.75pt solid #000000; }
  td:not(:first-child), th:not(:first-child) { text-align: right; }
  /* A paragraph of the form "Table N: ..." directly after a table is its
     caption; mistune gives us no caption element to target. */
  p.tablecaption { font-size: 8.2pt; margin: 0 0 7pt 0; text-align: left; }
  p.tablecaption.wide { column-span: all; }
  ol, ul { margin: 0 0 6pt 0; padding-left: 14pt; }
  li { margin-bottom: 3pt; font-size: 9pt; }
  figure { margin: 5pt 0 7pt 0; text-align: center; }
  figure img {
    width: 100%; height: auto; border: 0.5pt solid #bbbbbb;
  }
  figcaption {
    font-size: 8.5pt; margin-top: 4pt; line-height: 1.3; text-align: left;
  }
  @page { margin: 0.75in 0.7in; size: letter; }
"""


def data_uri(name: str) -> str:
    blob = (PLOTS / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")


def figure_html(name: str, caption: str) -> str:
    return (f'<figure>\n<img src="{data_uri(name)}" alt="{caption[:60]}">\n'
            f"<figcaption>{caption}</figcaption>\n</figure>\n")


def main() -> None:
    md = (ROOT / "paper.md").read_text(encoding="utf-8")
    body = mistune.create_markdown(plugins=["table", "strikethrough"])(md)

    # Attach figures after their section heading. Matching on the rendered <h3>
    # keeps a reworded heading a loud failure rather than a silently dropped
    # figure.
    missing = []
    for heading, figs in FIGURES.items():
        pattern = re.compile(r"(<h3>" + re.escape(heading) + r"</h3>\s*)",
                             re.IGNORECASE)
        if not pattern.search(body):
            missing.append(heading)
            continue
        block = "".join(figure_html(n, c) for n, c in figs)
        body = pattern.sub(lambda m: m.group(1) + block, body, count=1)
    if missing:
        raise SystemExit("figure anchors not found in paper.md: "
                         + "; ".join(missing))

    # A table spans both columns only if it needs the width. Everything else
    # stays in-column, because each span forces a column break. A table earns a
    # span by having many columns or a long cell; the caption inherits whatever
    # the table got, or it detaches from it in the layout.
    stats = {"wide": 0, "narrow": 0, "caps": 0}

    def classify(m):
        table_html, caption = m.group(1), m.group(2)
        cells = re.findall(r"<t[hd]>(.*?)</t[hd]>", table_html, re.DOTALL)
        first_row = re.search(r"<tr>(.*?)</tr>", table_html, re.DOTALL)
        n_cols = len(re.findall(r"<t[hd]>", first_row.group(1))) if first_row else 0
        longest = max((len(re.sub(r"<[^>]+>", "", c)) for c in cells), default=0)
        wide = n_cols > 5 or longest > 20
        stats["wide" if wide else "narrow"] += 1
        stats["caps"] += 1
        cls = " wide" if wide else ""
        table_html = table_html.replace("<table>", f'<table class="t{cls}">', 1)
        return (table_html + f'<p class="tablecaption{cls}">' + caption + "</p>")

    body = re.sub(r"(<table>.*?</table>)\s*<p>(Table \d+:.*?)</p>",
                  classify, body, flags=re.DOTALL)
    n_caps = stats["caps"]

    # Split the title block and abstract out of the two-column flow.
    m = re.search(r"<h2>1\. Introduction</h2>", body)
    if not m:
        raise SystemExit("could not find '1. Introduction' to split the "
                         "title block from the two-column body")
    head_html, cols_html = body[:m.start()], body[m.start():]

    head_html = head_html.replace("<h2>Abstract</h2>",
                                  '</div><div class="abstract"><h2>Abstract</h2>')

    title = "Who Said That? Speaker Attribution for Untagged Dialogue in Novels"
    html = (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f'<div class="titleblock">{head_html}</div>\n'
        f'<div class="cols">\n{cols_html}\n</div>\n</body>\n</html>\n'
    )
    out = ROOT / "paper.html"
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(md.split()):,} words, "
          f"{stats['wide']} wide + {stats['narrow']} narrow tables, {sum(len(v) for v in FIGURES.values())} figures)")


if __name__ == "__main__":
    main()
