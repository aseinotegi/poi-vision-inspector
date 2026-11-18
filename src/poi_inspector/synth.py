"""Synthetic inspection photos with a known answer.

* ``gauge_photo(value)``: a pressure gauge dial reading exactly ``value``, with
  the usual trouble of a robot photo (tilt, blur, glare, noise, dim light).
* ``extinguisher_photo(state)``: an extinguisher on its wall bracket, either fine,
  missing, or with its pressure needle out of the green band.

They are not a substitute for real photos. They give a labelled test set that
anyone can regenerate, and they feed the Unified Namespace demo with images
whose true value is known.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 640
EXTINGUISHER_STATES = ("ok", "ausente", "presion_baja")


@dataclass(frozen=True)
class Trouble:
    tilt_deg: float = 0.0
    blur: float = 0.0
    glare: float = 0.0  # 0..1
    darkness: float = 0.0  # 0..1
    noise: float = 0.0  # pixel std

    @classmethod
    def random(cls, rng: random.Random, hard: bool = False) -> Trouble:
        k = 2.0 if hard else 1.0
        return cls(
            tilt_deg=rng.uniform(-12, 12) * k,
            blur=rng.choice([0, 0, 0.6, 1.2]) * k,
            glare=rng.choice([0, 0, 0, 0.35, 0.6]) * (k if hard else 1),
            darkness=rng.uniform(0, 0.35) * k,
            noise=rng.uniform(0, 6) * k,
        )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _background(rng: random.Random) -> Image.Image:
    base = rng.choice([(118, 124, 130), (92, 104, 96), (140, 132, 118), (70, 78, 90)])
    img = Image.new("RGB", (SIZE, SIZE), base)
    draw = ImageDraw.Draw(img)
    for _ in range(14):  # pipes, beams and shadows of a plant behind the instrument
        y = rng.randint(0, SIZE)
        shade = tuple(max(0, min(255, c + rng.randint(-40, 40))) for c in base)
        draw.rectangle([0, y, SIZE, y + rng.randint(6, 40)], fill=shade)
    return img


def _dial(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    r: int,
    lo: float,
    hi: float,
    unit: str,
    value: float,
    green: tuple[float, float] | None = None,
) -> None:
    """A dial with a 270° scale, starting at the bottom left, clockwise."""
    start, sweep = 135.0, 270.0
    draw.ellipse([cx - r - 14, cy - r - 14, cx + r + 14, cy + r + 14], fill=(55, 58, 62))  # bezel
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(246, 245, 240))
    if green:
        a0 = start + sweep * (green[0] - lo) / (hi - lo)
        a1 = start + sweep * (green[1] - lo) / (hi - lo)
        draw.arc([cx - r + 10, cy - r + 10, cx + r - 10, cy + r - 10], a0, a1, fill=(40, 160, 70), width=14)
    major = 10
    label = _font(int(r * 0.15))
    for i in range(major * 5 + 1):
        frac = i / (major * 5)
        ang = math.radians(start + sweep * frac)
        inner = r * (0.78 if i % 5 == 0 else 0.86)
        draw.line(
            [
                (cx + inner * math.cos(ang), cy + inner * math.sin(ang)),
                (cx + r * 0.94 * math.cos(ang), cy + r * 0.94 * math.sin(ang)),
            ],
            fill=(20, 20, 20),
            width=4 if i % 5 == 0 else 2,
        )
        if i % 5 == 0:
            text = f"{lo + (hi - lo) * frac:g}"
            tx, ty = cx + r * 0.62 * math.cos(ang), cy + r * 0.62 * math.sin(ang)
            draw.text((tx, ty), text, fill=(20, 20, 20), font=label, anchor="mm")
    draw.text((cx, cy + r * 0.42), unit, fill=(30, 30, 30), font=_font(int(r * 0.16)), anchor="mm")
    frac = (min(max(value, lo), hi) - lo) / (hi - lo)
    ang = math.radians(start + sweep * frac)
    tip = (cx + r * 0.84 * math.cos(ang), cy + r * 0.84 * math.sin(ang))
    tail = (cx - r * 0.16 * math.cos(ang), cy - r * 0.16 * math.sin(ang))
    draw.line([tail, tip], fill=(190, 30, 30), width=max(4, r // 28))
    draw.ellipse([cx - r * 0.07, cy - r * 0.07, cx + r * 0.07, cy + r * 0.07], fill=(30, 30, 30))


def _finish(img: Image.Image, trouble: Trouble, rng: random.Random) -> bytes:
    if trouble.tilt_deg:
        img = img.rotate(trouble.tilt_deg, resample=Image.Resampling.BICUBIC, fillcolor=(80, 84, 90))
    if trouble.glare:
        overlay = Image.new("L", img.size, 0)
        gx, gy = rng.randint(180, 460), rng.randint(140, 400)
        ImageDraw.Draw(overlay).ellipse([gx - 90, gy - 50, gx + 90, gy + 50], fill=int(255 * trouble.glare))
        overlay = overlay.filter(ImageFilter.GaussianBlur(30))
        img = Image.composite(Image.new("RGB", img.size, (255, 255, 255)), img, overlay)
    if trouble.blur:
        img = img.filter(ImageFilter.GaussianBlur(trouble.blur))
    if trouble.darkness:
        img = img.point(lambda p: int(p * (1 - trouble.darkness)))
    if trouble.noise:
        noise = Image.effect_noise(img.size, trouble.noise).convert("RGB")
        img = Image.blend(img, noise, 0.08)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88)
    return out.getvalue()


def gauge_photo(
    value: float, lo: float = 0, hi: float = 10, unit: str = "bar", trouble: Trouble | None = None, seed: int = 0
) -> bytes:
    rng = random.Random(seed)
    img = _background(rng)
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2 + rng.randint(-40, 40), SIZE // 2 + rng.randint(-30, 30)
    r = rng.randint(190, 230)
    draw.rectangle([cx - 14, cy + r, cx + 14, SIZE], fill=(150, 150, 150))  # stem to the pipe
    _dial(draw, cx, cy, r, lo, hi, unit, value)
    return _finish(img, trouble or Trouble(), rng)


def extinguisher_photo(state: str, trouble: Trouble | None = None, seed: int = 0) -> bytes:
    if state not in EXTINGUISHER_STATES:
        raise ValueError(f"state must be one of {EXTINGUISHER_STATES}")
    rng = random.Random(seed)
    img = _background(rng)
    draw = ImageDraw.Draw(img)
    cx = SIZE // 2 + rng.randint(-50, 50)
    # wall sign and bracket are always there: an empty bracket is the finding
    draw.rectangle([cx - 70, 30, cx + 70, 120], fill=(200, 30, 30))
    draw.text((cx, 75), "EXTINTOR", fill=(255, 255, 255), font=_font(26), anchor="mm")
    draw.rectangle([cx - 60, 190, cx + 60, 206], fill=(60, 60, 60))
    if state != "ausente":
        draw.rounded_rectangle([cx - 65, 200, cx + 65, 600], radius=55, fill=(205, 25, 30))
        draw.rectangle([cx - 20, 160, cx + 20, 205], fill=(40, 40, 40))  # valve
        draw.line([(cx + 20, 175), (cx + 95, 240), (cx + 90, 420)], fill=(25, 25, 25), width=12)  # hose
        draw.rectangle([cx - 50, 330, cx + 50, 450], fill=(240, 240, 235))  # label
        draw.text((cx, 390), "ABC", fill=(30, 30, 30), font=_font(34), anchor="mm")
        needle = rng.uniform(12, 16) if state == "ok" else rng.uniform(1, 5)
        _dial(draw, cx - 75, 165, 38, 0, 20, "bar", needle, green=(11, 17))
    return _finish(img, trouble or Trouble(), rng)
