"""Write layout_template.svg: a 1:1 A4 template of the cereal rings' start spots, read from task.py.

Print it at 100 % and check the 100 mm scale bar. Run: python layout_template.py
"""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 210.0, 297.0  # A4 portrait, mm
LEFT_Y = -0.055  # world y (m) of the sheet's left edge: 5.5 cm right of the centre line
X_TOP = 0.50  # world x (m) of the sheet's top edge


def constants() -> dict:
    """SPILL_XY and RING_XY as written in task.py, without importing it (it needs the simulator)."""
    tree = ast.parse((HERE / "task.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"SPILL_XY", "RING_XY"}:
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def sheet(x: float, y: float) -> tuple[float, float]:
    """World point (m; x forward from the arms, y to their left) to sheet mm (right, down)."""
    return (LEFT_Y - y) * 1000.0, (X_TOP - x) * 1000.0


def text(x: float, y: float, s: str, size: float = 3.5) -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="sans-serif">{s}</text>'


def main() -> None:
    c = constants()
    sx, sy = c["SPILL_XY"]
    x0, y0 = sheet(sx + 0.10, sy + 0.10)
    near = (sx - 0.10) * 100
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="200" height="200" fill="none" stroke="#777" stroke-width="0.3" '
        f'stroke-dasharray="3 2"/>',
        text(8, 12, "Cereal ring start spots, 1:1. Up the sheet is forward, away from the arms."),
        text(8, 18, f"Left edge of the sheet: parallel to the centre line between the arms, {-LEFT_Y * 100:.1f} cm to its right."),
        text(8, 24, f"Dashed square: the 20 x 20 cm spill patch; its near (bottom) edge is {near:.0f} cm in front of"),
        text(8, 30, "the arms' base line (the line through both arm mounts)."),
        text(8, 36, "Prick each ring centre through onto the table with a pencil point, then remove the sheet."),
    ]
    for i, (dx, dy) in enumerate(c["RING_XY"]):
        u, v = sheet(sx + dx, sy + dy)
        parts.append(f'<circle cx="{u:.2f}" cy="{v:.2f}" r="6" fill="none" stroke="black" stroke-width="0.3"/>')
        parts.append(f'<circle cx="{u:.2f}" cy="{v:.2f}" r="0.6" fill="black"/>')
        parts.append(text(u + 7, v + 1.2, str(i + 1), 3))
    parts += [
        f'<line x1="8" y1="{H - 20}" x2="108" y2="{H - 20}" stroke="black" stroke-width="0.8"/>',
        f'<line x1="8" y1="{H - 23}" x2="8" y2="{H - 17}" stroke="black" stroke-width="0.8"/>',
        f'<line x1="108" y1="{H - 23}" x2="108" y2="{H - 17}" stroke="black" stroke-width="0.8"/>',
        text(8, H - 11, "100 mm (print at 100 %, check this bar)"),
        "</svg>",
    ]
    (HERE / "layout_template.svg").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
