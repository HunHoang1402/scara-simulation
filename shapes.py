"""Point sets for drawing shapes and text (BTL 3.3: multi-point paths).

Every function returns a list of polylines; each polyline is a list of (x, y) points
in metres in the base frame {0}. A polyline is drawn with the pen down; moving
between polylines is done with the pen up (see scara_trajectory.strokes).
"""
import math

GLYPH_W, GLYPH_ADVANCE = 0.6, 0.8   # in units of the character height

# Single-stroke font in a 0.6 x 1.0 box (origin bottom-left). Kept deliberately
# simple: straight segments only, so every stroke is a plain line path.
_O = [(0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.15), (0.15, 0)]
_P = [(0, 0), (0, 1), (0.45, 1), (0.6, 0.85), (0.6, 0.65), (0.45, 0.5), (0, 0.5)]
FONT = {
    "A": [[(0, 0), (0.3, 1), (0.6, 0)], [(0.12, 0.4), (0.48, 0.4)]],
    "B": [[(0, 0), (0, 1), (0.4, 1), (0.55, 0.9), (0.55, 0.6), (0.4, 0.5), (0, 0.5)],
          [(0.4, 0.5), (0.6, 0.4), (0.6, 0.1), (0.45, 0), (0, 0)]],
    "C": [[(0.6, 0.9), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.15), (0.15, 0), (0.45, 0), (0.6, 0.1)]],
    "D": [[(0, 0), (0, 1), (0.35, 1), (0.6, 0.75), (0.6, 0.25), (0.35, 0), (0, 0)]],
    "E": [[(0.6, 1), (0, 1), (0, 0), (0.6, 0)], [(0, 0.5), (0.45, 0.5)]],
    "F": [[(0.6, 1), (0, 1), (0, 0)], [(0, 0.5), (0.45, 0.5)]],
    "G": [[(0.6, 0.9), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.15), (0.15, 0), (0.45, 0),
           (0.6, 0.15), (0.6, 0.45), (0.35, 0.45)]],
    "H": [[(0, 0), (0, 1)], [(0.6, 0), (0.6, 1)], [(0, 0.5), (0.6, 0.5)]],
    "I": [[(0.1, 1), (0.5, 1)], [(0.3, 1), (0.3, 0)], [(0.1, 0), (0.5, 0)]],
    "J": [[(0.6, 1), (0.6, 0.2), (0.45, 0), (0.15, 0), (0, 0.2)]],
    "K": [[(0, 0), (0, 1)], [(0.6, 1), (0, 0.4)], [(0.2, 0.6), (0.6, 0)]],
    "L": [[(0, 1), (0, 0), (0.6, 0)]],
    "M": [[(0, 0), (0, 1), (0.3, 0.5), (0.6, 1), (0.6, 0)]],
    "N": [[(0, 0), (0, 1), (0.6, 0), (0.6, 1)]],
    "O": [_O],
    "P": [_P],
    "Q": [_O, [(0.35, 0.25), (0.6, 0)]],
    "R": [_P, [(0.3, 0.5), (0.6, 0)]],
    "S": [[(0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.65), (0.15, 0.5), (0.45, 0.5),
           (0.6, 0.35), (0.6, 0.15), (0.45, 0), (0.15, 0), (0, 0.15)]],
    "T": [[(0, 1), (0.6, 1)], [(0.3, 1), (0.3, 0)]],
    "U": [[(0, 1), (0, 0.15), (0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 1)]],
    "V": [[(0, 1), (0.3, 0), (0.6, 1)]],
    "W": [[(0, 1), (0.15, 0), (0.3, 0.6), (0.45, 0), (0.6, 1)]],
    "X": [[(0, 0), (0.6, 1)], [(0, 1), (0.6, 0)]],
    "Y": [[(0, 1), (0.3, 0.5), (0.6, 1)], [(0.3, 0.5), (0.3, 0)]],
    "Z": [[(0, 1), (0.6, 1), (0, 0), (0.6, 0)]],
    "0": [_O, [(0.6, 0.85), (0, 0.15)]],
    "1": [[(0.15, 0.8), (0.3, 1), (0.3, 0)], [(0.1, 0), (0.5, 0)]],
    "2": [[(0, 0.85), (0.15, 1), (0.45, 1), (0.6, 0.85), (0.6, 0.6), (0, 0), (0.6, 0)]],
    "3": [[(0, 0.85), (0.15, 1), (0.45, 1), (0.6, 0.85), (0.6, 0.65), (0.45, 0.5), (0.2, 0.5)],
          [(0.45, 0.5), (0.6, 0.35), (0.6, 0.15), (0.45, 0), (0.15, 0), (0, 0.15)]],
    "4": [[(0.45, 0), (0.45, 1), (0, 0.3), (0.6, 0.3)]],
    "5": [[(0.6, 1), (0, 1), (0, 0.55), (0.45, 0.55), (0.6, 0.4), (0.6, 0.15), (0.45, 0), (0.15, 0), (0, 0.15)]],
    "6": [[(0.55, 0.9), (0.4, 1), (0.15, 1), (0, 0.8), (0, 0.15), (0.15, 0), (0.45, 0), (0.6, 0.15),
           (0.6, 0.4), (0.45, 0.55), (0.15, 0.55), (0, 0.4)]],
    "7": [[(0, 1), (0.6, 1), (0.2, 0)]],
    "8": [[(0.15, 0.5), (0, 0.65), (0, 0.85), (0.15, 1), (0.45, 1), (0.6, 0.85), (0.6, 0.65), (0.45, 0.5),
           (0.15, 0.5), (0, 0.35), (0, 0.15), (0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 0.35), (0.45, 0.5)]],
    "9": [[(0.6, 0.6), (0.45, 0.45), (0.15, 0.45), (0, 0.6), (0, 0.85), (0.15, 1), (0.45, 1), (0.6, 0.85),
           (0.6, 0.15), (0.45, 0), (0.2, 0)]],
    " ": [],
}


def text_size(text, height):
    """(width, height) in metres of text written with text_to_strokes."""
    n = len(text)
    return (max(0, n - 1) * GLYPH_ADVANCE + GLYPH_W) * height if n else 0.0, height


def text_to_strokes(text, x0, y0, height):
    """Polylines for text whose bottom-left corner is (x0, y0); height in metres.
    Unsupported characters raise ValueError."""
    strokes = []
    for i, ch in enumerate(text.upper()):
        if ch not in FONT:
            raise ValueError(f"character {ch!r} is not in the font (use A-Z, 0-9, space)")
        ox = x0 + i * GLYPH_ADVANCE * height
        strokes += [[(ox + x * height, y0 + y * height) for x, y in pl] for pl in FONT[ch]]
    return strokes


def centered_text(text, cx, cy, height):
    """text_to_strokes with the text block centred on (cx, cy)."""
    w, h = text_size(text, height)
    return text_to_strokes(text, cx - w / 2, cy - h / 2, height)


def regular_polygon(cx, cy, radius, n, rotation=math.pi / 2):
    pts = [(cx + radius * math.cos(rotation + 2 * math.pi * k / n),
            cy + radius * math.sin(rotation + 2 * math.pi * k / n)) for k in range(n)]
    return [pts + [pts[0]]]


def square(cx, cy, size):
    h = size / 2
    return [[(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]]


def triangle(cx, cy, size):
    """Equilateral triangle with side `size` centred on (cx, cy)."""
    return regular_polygon(cx, cy, size / math.sqrt(3), 3)


def star(cx, cy, radius, points=5, inner_ratio=0.382):
    pts = []
    for k in range(2 * points):
        r = radius if k % 2 == 0 else radius * inner_ratio
        a = math.pi / 2 + math.pi * k / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return [pts + [pts[0]]]


SHAPES = {"Square": square, "Triangle": triangle, "Star": star}
