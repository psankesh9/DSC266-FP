"""Render report.md to a print-ready report.html.

Same visual conventions as proposal.html so the two documents read as one
submission: Arial, 11pt, 7.5in column, ruled tables. Figures from
outputs/plots are inlined as data URIs so the file can be moved or emailed
without breaking, and so printing to PDF never races a missing image.
"""
import base64
import re
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "outputs" / "plots"

# (figure file, caption) in the order they are inserted, keyed by the section
# heading they follow.
FIGURES = {
    "4. Task formulation and the candidate ceiling": [
        ("fig2_candidate_window_sweep.png",
         "Figure 1. Candidate recall (the ceiling) and mean candidates per "
         "quotation as the enumeration window widens, dev. Explicit recall is "
         "flat; the whole gain is on anaphoric and implicit quotations."),
        ("fig4_gazetteer_ablation.png",
         "Figure 2. PDNC aliases alone vs aliases plus names derived from the "
         "text, dev. Derived names buy 7.8 points of ceiling for 0.8 more "
         "candidates per quotation."),
    ],
    "7. Results": [
        ("fig1_headroom_by_quote_type.png",
         "Figure 3. Accuracy by quote type against the candidate ceiling, dev. "
         "The bar a system does not fill is its avoidable error; the gap above "
         "the ceiling is unreachable."),
    ],
    "9. Error analysis": [
        ("fig3_per_novel_variation.png",
         "Figure 4. Per-novel accuracy, dev. Between-novel spread exceeds the "
         "gaps between systems, which is why every interval in this report is "
         "bootstrapped over novels."),
    ],
}

CSS = """
  body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #000000;
    background: #ffffff;
    max-width: 7.5in;
    margin: 40px auto;
    padding: 0 24px;
  }
  h1 { font-size: 16pt; font-weight: bold; margin: 0 0 4pt 0; line-height: 1.3; }
  h2 { font-size: 13pt; font-weight: bold; margin: 20pt 0 6pt 0; }
  h3 { font-size: 11.5pt; font-weight: bold; margin: 14pt 0 5pt 0; }
  p  { margin: 0 0 10pt 0; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0 14pt 0;
    font-size: 9.5pt;
  }
  th, td {
    border: 1px solid #000000;
    padding: 4pt 6pt;
    text-align: left;
    vertical-align: top;
  }
  th { background: #efefef; font-weight: bold; }
  td:not(:first-child), th:not(:first-child) { text-align: right; }
  hr { border: none; border-top: 1px solid #000000; margin: 18pt 0; }
  ol, ul { margin: 0 0 10pt 0; padding-left: 22pt; }
  li { margin-bottom: 6pt; }
  code {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9.5pt;
    background: #f4f4f4;
    padding: 0 2px;
  }
  figure { margin: 12pt 0 16pt 0; page-break-inside: avoid; }
  figure img { width: 100%; height: auto; border: 1px solid #cccccc; }
  figcaption { font-size: 9.5pt; margin-top: 5pt; line-height: 1.4; }
  h2, h3 { page-break-after: avoid; }
  @media print {
    body { margin: 0 auto; }
    @page { margin: 0.75in; }
  }
"""


def data_uri(name: str) -> str:
    blob = (PLOTS / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")


def figure_html(name: str, caption: str) -> str:
    return (f'<figure>\n<img src="{data_uri(name)}" alt="{caption[:60]}">\n'
            f"<figcaption>{caption}</figcaption>\n</figure>\n")


def main() -> None:
    md = (ROOT / "report.md").read_text(encoding="utf-8")
    body = mistune.create_markdown(plugins=["table", "strikethrough"])(md)

    # Insert each figure block after the closing tag of the heading it belongs
    # to. Matching on the rendered <h2> keeps this robust to the heading text
    # being reworded in report.md without the anchor silently going stale.
    missing = []
    for heading, figs in FIGURES.items():
        pattern = re.compile(
            r"(<h2>" + re.escape(heading) + r"</h2>\s*)", re.IGNORECASE)
        if not pattern.search(body):
            missing.append(heading)
            continue
        block = "".join(figure_html(n, c) for n, c in figs)
        body = pattern.sub(lambda m: m.group(1) + block, body, count=1)
    if missing:
        raise SystemExit(
            "figure anchors not found in report.md: " + "; ".join(missing))

    title = "Who Said That? Speaker Attribution for Untagged Dialogue in Novels"
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n\n"
        f"{body}\n</body>\n</html>\n"
    )
    out = ROOT / "report.html"
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(md.split()):,} words of source)")


if __name__ == "__main__":
    main()
