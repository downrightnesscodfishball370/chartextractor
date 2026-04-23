"""Stage 3a: OCR of numeric tick labels.

Tesseract is the only learned component in this pipeline. We constrain
it aggressively with a numeric whitelist and parse the output with a
strict regex so that failed reads surface as `None` rather than
corrupting downstream calibration.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover
    pytesseract = None
    Output = None


def _autodetect_tesseract_binary() -> Optional[str]:
    # If already on PATH, nothing to do.
    found = shutil.which("tesseract")
    if found:
        return found
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


if pytesseract is not None:
    _bin = _autodetect_tesseract_binary()
    if _bin:
        pytesseract.pytesseract.tesseract_cmd = _bin


# Matches a numeric substring: 123, -1.5, 1e3, 1.2e-4, 2.5E+10. Anchored so
# that re.search finds the full numeric run, not a digit fragment.
_NUMERIC_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

# Unicode mathematical minus + en/em dashes that OCR commonly emits.
_MINUS_CHARS = str.maketrans({"−": "-", "–": "-", "—": "-"})


def parse_numeric(text: str) -> Optional[float]:
    s = text.strip().translate(_MINUS_CHARS).replace(",", "")
    if not s:
        return None
    # Labels often arrive with stray characters from the tick mark bleeding
    # into the OCR crop ("100-", "-50-"). Take the longest plausible numeric
    # substring instead of demanding a full-string match.
    candidates = _NUMERIC_RE.findall(s)
    if not candidates:
        return None
    best = max(candidates, key=len)
    # A bare sign ("-") or solitary "." from a slipped tick is not a number.
    if best in {"", "-", "+", ".", "-.", "+."}:
        return None
    try:
        return float(best)
    except ValueError:
        return None


def _ocr_region(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    psm: int = 7,
    binarise: bool = True,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    if pytesseract is None:
        return []
    x, y, w, h = bbox
    x = max(0, x); y = max(0, y)
    w = min(gray.shape[1] - x, w); h = min(gray.shape[0] - y, h)
    if w <= 2 or h <= 2:
        return []
    crop = gray[y : y + h, x : x + w]
    # Upscale so digit height is ~60-80 px. Tesseract handles bold sans-serif
    # much better at larger sizes; at native 20 px it misreads bold "2" as "9".
    scale = max(2, int(np.ceil(60.0 / max(1, h))))
    crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    if binarise:
        # Otsu helps Tesseract on antialiased/bold glyphs; hurts nothing on thin ones.
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789.+-eE"
    data = pytesseract.image_to_data(crop, config=cfg, output_type=Output.DICT)
    out = []
    for i, txt in enumerate(data["text"]):
        if not txt or not txt.strip():
            continue
        conf = float(data["conf"][i]) if data["conf"][i] not in ("", "-1") else 0.0
        bx = x + data["left"][i] // scale
        by = y + data["top"][i] // scale
        bw = max(1, data["width"][i] // scale)
        bh = max(1, data["height"][i] // scale)
        out.append((txt, conf / 100.0, (bx, by, bw, bh)))
    return out


def read_x_axis_labels(
    gray: np.ndarray,
    x_axis_row: float,
    y_axis_col: float,
    strip_height: int = 40,
) -> list[tuple[float, float, Optional[float], float]]:
    """Return (label_centre_x_px, label_centre_y_px, parsed_value, confidence)."""
    y0 = int(round(x_axis_row)) + 2
    region = (int(y_axis_col) - 5, y0, gray.shape[1] - int(y_axis_col) + 5, strip_height)
    found = _ocr_region(gray, region, psm=6)
    results = []
    for txt, conf, (bx, by, bw, bh) in found:
        val = parse_numeric(txt)
        results.append((bx + bw / 2.0, by + bh / 2.0, val, conf))
    return results


def read_x_axis_labels_per_tick(
    gray: np.ndarray,
    x_axis_row: float,
    tick_pixels: list[float],
    strip_height: int = 42,
) -> list[tuple[float, float, Optional[float], float]]:
    """OCR each tick individually (single-character PSM).

    Full-strip OCR drops bold sans-serif digits at a surprising rate
    (e.g. "2" reads as "9", "5" as "A"). Cropping tight to each tick and
    running single-char PSM lets Tesseract focus on one glyph at a time,
    which recovers more labels on charts with chunky fonts.
    """
    if pytesseract is None or not tick_pixels:
        return []
    y0 = int(round(x_axis_row)) + 2
    y1 = min(gray.shape[0], y0 + strip_height)
    diffs = np.diff(np.asarray(tick_pixels, dtype=np.float64))
    step = float(np.median(diffs)) if len(diffs) > 0 else 40.0
    half_w = max(8, int(step * 0.45))
    results: list[tuple[float, float, Optional[float], float]] = []
    for tp in tick_pixels:
        cx = int(round(tp))
        x0 = max(0, cx - half_w)
        x1 = min(gray.shape[1], cx + half_w)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        scale = max(3, int(np.ceil(90.0 / max(1, crop.shape[0]))))
        big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # psm=10 = single character. Whitelist excludes e/E because a stray
        # tick mark misread as "E" would otherwise sneak through.
        cfg = "--psm 10 -c tessedit_char_whitelist=0123456789.-"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=Output.DICT)
        best_txt, best_conf = "", 0.0
        for i, txt in enumerate(data["text"]):
            if not txt or not txt.strip():
                continue
            raw_conf = data["conf"][i]
            conf = float(raw_conf) / 100.0 if raw_conf not in ("", "-1") else 0.0
            if conf > best_conf:
                best_txt, best_conf = txt.strip(), conf
        if not best_txt:
            continue
        val = parse_numeric(best_txt)
        if val is None:
            continue
        results.append((float(tp), float(y0 + strip_height / 2.0), val, best_conf))
    return results


def read_y_axis_labels_per_tick(
    gray: np.ndarray,
    y_axis_col: float,
    tick_pixels: list[float],
    strip_width: int = 120,
) -> list[tuple[float, float, Optional[float], float]]:
    """Y-axis analogue of read_x_axis_labels_per_tick."""
    if pytesseract is None or not tick_pixels:
        return []
    avail = max(10, int(round(y_axis_col)) - 2)
    strip_width = min(strip_width, avail)
    x0 = max(0, int(round(y_axis_col)) - strip_width - 2)
    x1 = int(round(y_axis_col)) - 2
    diffs = np.diff(np.asarray(tick_pixels, dtype=np.float64))
    step = float(np.median(np.abs(diffs))) if len(diffs) > 0 else 30.0
    half_h = max(8, int(step * 0.45))
    results: list[tuple[float, float, Optional[float], float]] = []
    for tp in tick_pixels:
        cy = int(round(tp))
        y0 = max(0, cy - half_h)
        y1 = min(gray.shape[0], cy + half_h)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        scale = max(3, int(np.ceil(90.0 / max(1, crop.shape[0]))))
        big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # psm=7 = treat as one text line (may hold "450", "0.5").
        cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.-"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=Output.DICT)
        best_txt, best_conf = "", 0.0
        for i, txt in enumerate(data["text"]):
            if not txt or not txt.strip():
                continue
            raw_conf = data["conf"][i]
            conf = float(raw_conf) / 100.0 if raw_conf not in ("", "-1") else 0.0
            if conf > best_conf:
                best_txt, best_conf = txt.strip(), conf
        if not best_txt:
            continue
        val = parse_numeric(best_txt)
        if val is None:
            continue
        results.append((float(x0 + (x1 - x0) / 2.0), float(tp), val, best_conf))
    return results


def read_y_axis_labels(
    gray: np.ndarray,
    x_axis_row: float,
    y_axis_col: float,
    strip_width: int = 120,
) -> list[tuple[float, float, Optional[float], float]]:
    """Return (label_centre_x_px, label_centre_y_px, parsed_value, confidence)."""
    # Strip must fit 3-digit labels like "450"; 60 px used to clip the leading digit.
    avail = max(10, int(round(y_axis_col)) - 2)
    strip_width = min(strip_width, avail)
    x0 = max(0, int(round(y_axis_col)) - strip_width - 2)
    region = (x0, 0, strip_width, int(round(x_axis_row)) + 5)
    found = _ocr_region(gray, region, psm=6)
    results = []
    for txt, conf, (bx, by, bw, bh) in found:
        val = parse_numeric(txt)
        results.append((bx + bw / 2.0, by + bh / 2.0, val, conf))
    return results
