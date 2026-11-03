"""The paper's overview figure (Fig. 1): colour-coded cards with a rule under each header.

One layout, in points at final size (506 pt = the 178 mm text width), written as overview.svg;
Inkscape renders it to figures/overview.pdf. Type: Nimbus Roman, the Times face of the paper's
body text, 8 pt. All text #000000. Card colours: categorical slots 1-4 of the palette used by
src/figures.py (blue, orange, aqua, yellow), as light tints with the slot darkened 15% for rules
and borders; the embedding card is neutral. The input crop is a PBC test neutrophil (CC BY 4.0).
"""
import base64
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
FONT = "Nimbus Roman, 'Times New Roman', Times, serif"
INK = "#000000"
W, H, BAND, PAD = 506, 46, 12, 5          # figure width, card height, header band, text inset (pt)
BASE = (21.4, 30.8, 40.2)                 # body baselines (pt)

# (header, band tint, body tint, rule/border, x, width, body lines). A body line is a list of runs,
# or a (left runs, right runs) pair for a right-aligned label. Run: (text, style), style in
# "", "i" (italic), "b" (bold), "sup" (superscript).
CARDS = [
    ("DinoBloom-B", "#e8e6e1", "#f6f5f2", "#8a877f", 58, 60,
     [[("ViT-B/14", "")], [("CLS token ", ""), ("z", "b")], [("768-d", "")]]),
    ("3.1 Linear models", "#bbd4f2", "#eaf2fb", "#2466b6", 130, 80,
     [[("ŷ", "i"), (" = ", ""), ("w", "b"), ("⊤", "sup"), ("z", "b"), (" + ", ""), ("b", "i")],
      ([("R", "i"), ("² 0.45–0.86", "")], [("Matek", "")]),
      ([("R", "i"), ("² 0.70–0.95", "")], [("PBC", "")])]),
    ("3.2 Embedding positions", "#f9cfbe", "#fdf0eb", "#c8582c", 222, 96,
     [[("top-", ""), ("K", "i"), (" refits", "")], [("best position", "")],
      [("at most 37% of ", ""), ("R", "i"), ("²", "")]]),
    ("3.3 Decomposition", "#b6e5d4", "#e8f7f2", "#179568", 330, 79,
     [[("ŷ", "i"), (" = Σ writes + MLP", "")], [("patch writes", "")],
      [("66–98% of Var(", ""), ("ŷ", "i"), (")", "")]]),
    ("3.4 Setup transfer", "#f9e1ad", "#fdf6e6", "#c98900", 421, 85,
     [[("fit on Raabin train", "")],
      ([("ρ", "i"), (" 0.67–0.94", "")], [("validation", "")]),
      ([("ρ", "i"), (" 0.19–0.63", "")], [("set B", "")])]),
]
IMAGE = (0, 46)                           # the input crop: x, side (pt)
ARROWS = [(46, 58), (118, 130), (210, 222), (318, 330), (409, 421)]   # from right edge to left edge
RADIUS = 2.5


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- SVG, the paper's master
def svg_runs(runs):
    out, back = [], ""
    for text, style in runs:
        if style == "sup":   # raised 2.9 pt and kerned in; the next run drops back
            out.append(f'<tspan font-size="5.6" dx="-0.35" dy="-2.9">{esc(text)}</tspan>')
            back = ' dx="-0.2" dy="2.9"'
            continue
        attr = {"i": ' font-style="italic"', "b": ' font-weight="bold"'}.get(style, "") + back
        out.append(f"<tspan{attr}>{esc(text)}</tspan>")
        back = ""
    return "".join(out)


def svg(image_href):
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
         f'width="{W + 2}pt" height="{H + 2}pt" viewBox="-1 -1 {W + 2} {H + 2}" '
         f'font-family="{FONT}" font-size="8" fill="{INK}" xml:space="preserve">', "<defs>"]
    x0, side = IMAGE
    p.append(f'<clipPath id="clip-img"><rect x="{x0}" y="0" width="{side}" height="{side}" rx="{RADIUS}"/></clipPath>')
    for k, (_, _, _, _, x, w, _) in enumerate(CARDS):
        p.append(f'<clipPath id="clip-{k}"><rect x="{x}" y="0" width="{w}" height="{H}" rx="{RADIUS}"/></clipPath>')
    p.append("</defs>")
    p.append(f'<image x="{x0}" y="0" width="{side}" height="{side}" preserveAspectRatio="xMidYMid slice" '
             f'clip-path="url(#clip-img)" xlink:href="{image_href}"/>')
    p.append(f'<rect x="{x0}" y="0" width="{side}" height="{side}" rx="{RADIUS}" fill="none" stroke="#8a877f" stroke-width="0.6"/>')
    for a, b in ARROWS:   # 0.75 pt shaft, filled head 3.6 pt long, tip 0.6 pt short of the card
        tip, y = b - 0.6, H / 2
        p.append(f'<line x1="{a + 0.6}" y1="{y}" x2="{tip - 3.2}" y2="{y}" stroke="{INK}" stroke-width="0.75"/>')
        p.append(f'<polygon points="{tip - 3.6},{y - 1.7} {tip},{y} {tip - 3.6},{y + 1.7}" fill="{INK}"/>')
    for k, (head, band, body, rule, x, w, lines) in enumerate(CARDS):
        p.append(f'<g clip-path="url(#clip-{k})"><rect x="{x}" y="0" width="{w}" height="{BAND}" fill="{band}"/>'
                 f'<rect x="{x}" y="{BAND}" width="{w}" height="{H - BAND}" fill="{body}"/></g>')
        p.append(f'<line x1="{x}" y1="{BAND}" x2="{x + w}" y2="{BAND}" stroke="{rule}" stroke-width="0.8"/>')
        p.append(f'<rect x="{x}" y="0" width="{w}" height="{H}" rx="{RADIUS}" fill="none" stroke="{rule}" stroke-width="0.6"/>')
        p.append(f'<text x="{x + PAD}" y="8.7" font-weight="bold">{esc(head)}</text>')
        for base, line in zip(BASE, lines):
            if isinstance(line, tuple):
                left, right = line
                p.append(f'<text x="{x + PAD}" y="{base}">{svg_runs(left)}</text>')
                p.append(f'<text x="{x + w - PAD}" y="{base}" text-anchor="end">{svg_runs(right)}</text>')
            else:
                p.append(f'<text x="{x + PAD}" y="{base}">{svg_runs(line)}</text>')
    p.append("</svg>\n")
    return "\n".join(p)


if __name__ == "__main__":
    png = base64.b64encode((HERE / "cell_neutrophil.png").read_bytes()).decode()
    (HERE / "overview.svg").write_text(svg(f"data:image/png;base64,{png}"), encoding="utf8")
    print("wrote", HERE / "overview.svg", "-- render: inkscape overview.svg --export-type=pdf "
          "--export-filename=../overview.pdf")
