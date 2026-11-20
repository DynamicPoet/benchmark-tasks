"""Draws the textures in textures/: the two vial labels and the two bin cards. Run once with any Python that has Pillow;
the PNGs are committed, so the task itself needs no Pillow.

The vial labels follow the layout of the training vials' labels (a white wrap label with an orange edge stripe, the name,
the bold second line, the strength, the per-mL strength, the fill), drawn here as plain text: the two differ only in
their second line and strengths, as the real pair do.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
PX_MM = 20  # pixels per millimetre
FONTS = Path("/usr/share/fonts/truetype")
BOLD = FONTS / "liberation" / "LiberationSansNarrow-Bold.ttf"
REG = FONTS / "liberation" / "LiberationSansNarrow-Regular.ttf"
if not BOLD.exists():
    BOLD = FONTS / "dejavu" / "DejaVuSansCondensed-Bold.ttf"
    REG = FONTS / "dejavu" / "DejaVuSansCondensed.ttf"
ORANGE = (238, 118, 30)
INK = (25, 25, 25)

# the label: 300 degrees of a 12.35 mm radius (64.7 mm) by 35 mm tall; its printed panel centred
LABEL_W_MM, LABEL_H_MM = 64.67, 35.0


def font(path: Path, mm: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), int(round(mm * PX_MM * 1.38)))  # cap height ~0.72 of the em


def label(lines: list[tuple[str, Path, float]], path: Path) -> None:
    w, h = int(LABEL_W_MM * PX_MM), int(LABEL_H_MM * PX_MM)
    img = Image.new("RGB", (w, h), (246, 246, 242))
    d = ImageDraw.Draw(img)
    cx = w / 2
    left = cx - 12.0 * PX_MM
    # orange edge stripe down the left of the printed panel, and the bottom band's outline
    d.rectangle([left - 2.4 * PX_MM, 1.0 * PX_MM, left - 1.0 * PX_MM, h - 1.0 * PX_MM], fill=ORANGE)
    y = 2.2 * PX_MM
    for text, f, mm in lines:
        fnt = font(f, mm)
        d.text((left, y), text, font=fnt, fill=INK)
        y += mm * PX_MM * 1.42
    band_top = h - 5.0 * PX_MM
    d.rectangle([left, band_top, left + 24.0 * PX_MM, h - 1.6 * PX_MM], outline=ORANGE, width=int(0.35 * PX_MM))
    d.text((left + 1.0 * PX_MM, band_top + 0.7 * PX_MM), "FOR INSTRUCTIONAL USE ONLY", font=font(BOLD, 1.5), fill=ORANGE)
    img.save(path)


def card(lines: list[tuple[str, float]], path: Path, w_mm: float, h_mm: float, bg, fg) -> None:
    w, h = int(w_mm * PX_MM), int(h_mm * PX_MM)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    total = sum(mm * PX_MM * 1.35 for _, mm in lines)
    y = (h - total) / 2
    for text, mm in lines:
        fnt = font(BOLD, mm)
        tw = d.textlength(text, font=fnt)
        d.text(((w - tw) / 2, y), text, font=fnt, fill=fg)
        y += mm * PX_MM * 1.35
    img.save(path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    common_top = [("Multi-Dose Vial", REG, 1.7), ("DEMO-HEPARIN", BOLD, 3.2)]
    label(common_top + [("LOCK FLUSH", BOLD, 3.2), ("1,000 units per 10 mL", BOLD, 2.3), ("(100 units per mL)", REG, 2.1),
                        ("10 mL", BOLD, 2.8)], OUT / "label_flush.png")
    label(common_top + [("Injection", BOLD, 3.0), ("50,000 units per 10 mL", BOLD, 2.3), ("(5,000 units per mL)", REG, 2.1),
                        ("10 mL", BOLD, 2.8)], OUT / "label_injection.png")
    card([("HEPARIN LOCK FLUSH", 5.0), ("100 units/mL  10 mL vial", 4.0)], OUT / "card_flush.png", 90.0, 24.0,
         (250, 250, 248), (20, 20, 20))
    card([("RETURN TO PHARMACY", 5.0)], OUT / "card_returns.png", 90.0, 24.0, (250, 250, 248), (150, 20, 20))


if __name__ == "__main__":
    main()
