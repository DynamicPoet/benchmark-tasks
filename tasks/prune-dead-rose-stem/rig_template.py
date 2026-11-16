"""The real-world rig, from the same numbers as the simulation.

  python rig_template.py            writes rig_template.svg (A4, print at 100 %) and prints the cane table for the README

The template is a disc that lies on the pot's soil with its arrow pointing away from the robot. Each stem's base point is
marked, with an arrow in the direction it leans. Below it, an angle guide for the lean angles.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import scene_assets as sa

HERE = Path(__file__).resolve().parent
PAGE = (210.0, 297.0)  # A4, mm
DISC_C = (105.0, 112.0)  # the disc's centre on the page, mm
DISC_R = 95.0  # mm: inside the soil's 201 mm, so it lies flat inside the rim
INSERT = 0.06  # each stem goes this far into the pot, along the stem, below the soil surface


def dead_cane() -> dict:
    base = sa.CANE_IN_POT
    return dict(name="dead", base=(float(base[0]), float(base[1])), lean=math.degrees(sa.LEAN), toward=90.0, length=sa.TOP_S,
                top=sa.on_cane(sa.TOP_S), radius=sa.R0)


def rows() -> list[dict]:
    out = []
    for c in sa.LIVE_CANES:
        out.append(dict(name=c.name, base=c.base, lean=c.lean, toward=c.toward, length=c.length, top=c.top, radius=c.radius,
                        leaves_from=min(s for s, *_ in c.leaves), head=c.head))
    d = dead_cane()
    out.insert(1, d)  # in the table, the dead cane sits between its two neighbours
    return out


def table() -> str:
    head = {"bloom": "open bloom", "bud": "flower in tight bud", "cut": "flower head cut off"}
    lines = ["| Cane | Base in the pot (x, y) | Base on the station (x, y) | Lean from upright | Leans toward | Above the soil | Stem to cut | "
             "Tip on the station (x, y, height) | Carries |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows():
        bx, by = r["base"]
        sx, sy = bx + sa.POT_XY[0], by + sa.POT_XY[1]
        tx, ty, tz = r["top"][0] + sa.POT_XY[0], r["top"][1] + sa.POT_XY[1], r["top"][2]
        toward = f"{r['toward']:.0f} deg"
        if r["name"] == "dead":
            carries = (f"bud {sa.BUD_S * 100:.1f} cm along, facing the robot; brown from {sa.BROWN_S * 100:.1f} cm along; leafless; "
                       "square-cut tip")
        else:
            carries = f"leaves from {r['leaves_from'] * 100:.1f} cm along; {head[r['head']]}"
        lines.append(f"| {r['name']} | {bx * 100:+.1f}, {by * 100:+.1f} cm | {sx:.3f}, {sy:.3f} m | {r['lean']:.0f} deg | {toward} | "
                     f"{r['length'] * 100:.0f} cm | {(r['length'] + INSERT) * 100:.0f} cm | {tx:.3f}, {ty:.3f}, {tz:.3f} m | {carries} |")
    return "\n".join(lines)


def _page(x: float, y: float) -> tuple[float, float]:
    """Pot frame (m) to page (mm): away from the robot is up the page, the robot's left is left."""
    return DISC_C[0] - y * 1000.0, DISC_C[1] - x * 1000.0


def svg() -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE[0]:.0f}mm" height="{PAGE[1]:.0f}mm" viewBox="0 0 {PAGE[0]:.0f} {PAGE[1]:.0f}" '
           'font-family="Helvetica, Arial, sans-serif">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="105" y="9" font-size="5" text-anchor="middle">Rose bush template: print at 100 % on A4; the bar must measure 100 mm</text>',
           '<g stroke="black" stroke-width="0.4"><line x1="55" y1="14" x2="155" y2="14"/>'
           + "".join(f'<line x1="{55 + 10 * k}" y1="12.5" x2="{55 + 10 * k}" y2="15.5"/>' for k in range(11)) + "</g>",
           f'<circle cx="{DISC_C[0]}" cy="{DISC_C[1]}" r="{DISC_R}" fill="none" stroke="black" stroke-width="0.5" stroke-dasharray="3 1.5"/>',
           f'<text x="{DISC_C[0]}" y="{DISC_C[1] + DISC_R + 6}" font-size="4" text-anchor="middle">robot side (the pot\'s centre is '
           f'at x {sa.POT_XY[0]:.2f} m, y {sa.POT_XY[1]:+.2f} m on the station)</text>',
           f'<text x="{DISC_C[0] + 6}" y="{DISC_C[1] + 6}" font-size="3.2">pot centre</text>',
           f'<g stroke="black" stroke-width="0.3"><line x1="{DISC_C[0] - 4}" y1="{DISC_C[1]}" x2="{DISC_C[0] + 4}" y2="{DISC_C[1]}"/>'
           f'<line x1="{DISC_C[0]}" y1="{DISC_C[1] - 4}" x2="{DISC_C[0]}" y2="{DISC_C[1] + 4}"/></g>',
           f'<line x1="{DISC_C[0]}" y1="{DISC_C[1] - 70}" x2="{DISC_C[0]}" y2="{DISC_C[1] - 88}" stroke="black" stroke-width="0.6"/>',
           f'<path d="M{DISC_C[0] - 3},{DISC_C[1] - 84} L{DISC_C[0]},{DISC_C[1] - 90} L{DISC_C[0] + 3},{DISC_C[1] - 84} Z"/>',
           f'<text x="{DISC_C[0] + 5}" y="{DISC_C[1] - 84}" font-size="4">away from the robot</text>']
    for r in rows():
        px, py = _page(*r["base"])
        a = math.radians(r["toward"])
        ex, ey = _page(r["base"][0] + 0.022 * math.cos(a), r["base"][1] + 0.022 * math.sin(a))
        color = "#7a3b10" if r["name"] == "dead" else "#1b6e20"
        hx, hy = ex - px, ey - py
        n = math.hypot(hx, hy)
        ux, uy = hx / n, hy / n
        head = (f'M{ex:.2f},{ey:.2f} L{ex - 3.5 * ux + 1.8 * uy:.2f},{ey - 3.5 * uy - 1.8 * ux:.2f} '
                f'L{ex - 3.5 * ux - 1.8 * uy:.2f},{ey - 3.5 * uy + 1.8 * ux:.2f} Z')
        out += [f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.6" fill="none" stroke="{color}" stroke-width="0.5"/>',
                f'<g stroke="{color}" stroke-width="0.25"><line x1="{px - 3:.2f}" y1="{py:.2f}" x2="{px + 3:.2f}" y2="{py:.2f}"/>'
                f'<line x1="{px:.2f}" y1="{py - 3:.2f}" x2="{px:.2f}" y2="{py + 3:.2f}"/></g>',
                f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" stroke="{color}" stroke-width="0.6"/>',
                f'<path d="{head}" fill="{color}"/>',
                f'<text x="{px - 3 * ux + 2:.2f}" y="{py - 3 * uy + 5:.2f}" font-size="3.6" fill="{color}">{r["name"]} {r["lean"]:.0f}&#176;</text>']
    # The angle guide: stand it upright beside a stem, bottom edge on the soil, corner at the stem's base.
    steps = ["1 Cut out the disc and lay it on the soil, centred, its arrow pointing away from the robot.",
             "2 At each ring, push a skewer 6 cm in along its arrow, at its lean angle. Lift the disc off.",
             "3 Push each stem 6 cm into its hole (measured along the stem from the soil), leaning along its arrow."]
    out += [f'<text x="12" y="{221 + 5 * i}" font-size="3.6">{t}</text>' for i, t in enumerate(steps)]
    out.append('<text x="12" y="240" font-size="3.6">Angle guides: cut out; stand one upright beside the stem, bottom edge on the soil, '
               'corner at the stem\'s base.</text>')
    leans = sorted({round(r["lean"]) for r in rows()})
    size = 40.0
    for i, lean in enumerate(leans):
        ox, oy = 14.0 + i * (size + 8.0), 290.0
        a = math.radians(lean)
        names = ", ".join(r["name"] for r in rows() if round(r["lean"]) == lean)
        # the stem's line, from the corner to the card's edge
        ex, ey = (ox + size * math.tan(a), oy - size) if lean <= 45 else (ox + size, oy - size / math.tan(a))
        out += [f'<path d="M{ox},{oy} L{ox + size},{oy} L{ox + size},{oy - size} L{ox},{oy - size} Z" fill="none" stroke="black" '
                'stroke-width="0.3" stroke-dasharray="2 1"/>',
                f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - size}" stroke="black" stroke-width="0.6"/>',
                f'<line x1="{ox}" y1="{oy}" x2="{ox + size}" y2="{oy}" stroke="black" stroke-width="0.6"/>',
                f'<line x1="{ox}" y1="{oy}" x2="{ex:.2f}" y2="{ey:.2f}" stroke="#1b6e20" stroke-width="0.8"/>',
                f'<text x="{ox + 2}" y="{oy - size + 5}" font-size="3.4">{lean}&#176;: {names}</text>',
                f'<text x="{ox + 2}" y="{oy - 2}" font-size="2.6">soil</text>']
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (HERE / "rig_template.svg").write_text(svg())
    print(table())
