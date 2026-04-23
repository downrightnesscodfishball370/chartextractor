"""Stage 2: Axis and tick localisation.

We combine three signals: Probabilistic Hough (orientation-aware line
detection), a sweep-line projection (robust to breaks caused by ticks/
labels), and subpixel refinement via parabolic fit on the projection.
The sweep-line result is authoritative; Hough is used to sanity-check
the assumption that axes are (nearly) image-aligned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import AxisInfo, TickLabel


@dataclass
class AxisFrame:
    x_axis: AxisInfo
    y_axis: AxisInfo
    origin_uv: tuple[float, float]
    rotation_deg: float  # estimated frame rotation; should be ~0 for axis-aligned plots


# --- Hough-based rotation check ------------------------------------------------

def estimate_frame_rotation(edges: np.ndarray) -> float:
    # Probabilistic Hough on edges; the dominant near-horizontal/vertical
    # angle cluster tells us if the image is tilted. Used only as a sanity
    # check and to trigger perspective rectification if needed.
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 720, threshold=80,
        minLineLength=max(edges.shape) // 4, maxLineGap=5,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Fold into [-45, 45] around horizontal/vertical
        a = ((a + 45) % 90) - 45
        angles.append(a)
    return float(np.median(angles))


# --- Sweep-line axis localisation ---------------------------------------------

def _pick_axis_candidate(profile: np.ndarray, region_mask: np.ndarray, prefer_high: bool) -> int:
    """Pick the axis index from `profile` restricted to `region_mask`.

    `prefer_high=True` returns the largest-indexed near-max (bottom for rows,
    right for columns); `prefer_high=False` returns the smallest.
    """
    masked = np.where(region_mask, profile, -np.inf)
    peak = masked.max()
    if not np.isfinite(peak) or peak <= 0:
        return int(np.argmax(profile))
    candidates = np.where(masked >= 0.97 * peak)[0]
    return int(candidates.max() if prefer_high else candidates.min())


def _subpixel_peak(profile: np.ndarray, idx: int) -> float:
    # Parabolic interpolation of three samples around the detected maximum.
    if idx <= 0 or idx >= len(profile) - 1:
        return float(idx)
    y0, y1, y2 = profile[idx - 1 : idx + 2].astype(np.float64)
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(idx)
    delta = 0.5 * (y0 - y2) / denom
    return float(idx) + float(np.clip(delta, -0.5, 0.5))


def locate_axes(binary: np.ndarray) -> AxisFrame:
    """Return subpixel positions of the x- and y-axis lines.

    Assumption: the plot area occupies the majority of the image and the
    two axes are the longest continuous horizontal and vertical runs of
    foreground. This holds for all standard scientific charts.
    """
    h, w = binary.shape
    fg = (binary > 0).astype(np.uint8)

    # Long horizontal/vertical structuring elements isolate axis candidates
    # while erasing the curve and most gridlines (which are usually thinner).
    horiz = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 20), 1)))
    vert = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 20))))

    row_sum = horiz.sum(axis=1).astype(np.float32)
    col_sum = vert.sum(axis=0).astype(np.float32)

    # Charts without a bold axis line — only gridlines — produce many rows
    # with identical `row_sum`. Plain `argmax` then picks the top-most tied
    # gridline rather than the actual x-axis. Among near-maximal candidates
    # (>= 97% of the peak), the x-axis is the bottom-most and the y-axis is
    # the left-most, since those sit at the frame boundary.
    lower_mask = np.arange(h) > h // 2
    left_mask = np.arange(w) < w // 2
    x_row = _pick_axis_candidate(row_sum, lower_mask, prefer_high=True)
    y_col = _pick_axis_candidate(col_sum, left_mask, prefer_high=False)

    x_row_sub = _subpixel_peak(row_sum, x_row)
    y_col_sub = _subpixel_peak(col_sum, y_col)

    # Coverage = fraction of the axis line that is continuous foreground.
    # Low coverage -> axis detection is unreliable.
    x_coverage = float(horiz[x_row].mean())
    y_coverage = float(vert[:, y_col].mean())

    x_axis = AxisInfo(orientation="x", pixel_position=x_row_sub, confidence=x_coverage)
    y_axis = AxisInfo(orientation="y", pixel_position=y_col_sub, confidence=y_coverage)

    return AxisFrame(
        x_axis=x_axis,
        y_axis=y_axis,
        origin_uv=(y_col_sub, x_row_sub),
        rotation_deg=0.0,
    )


# --- Tick detection ------------------------------------------------------------

def detect_ticks(
    binary: np.ndarray,
    frame: AxisFrame,
    tick_length: int = 8,
) -> tuple[list[float], list[float]]:
    """Return subpixel pixel positions of ticks along x and y axes.

    Charts vary: some draw ticks outward (outside the plot area), others
    inward. We scan both strips and return whichever yields the clearer
    tick signal, falling back to the union when both are plausible.
    """
    h, w = binary.shape
    fg = (binary > 0).astype(np.uint8)

    # X-axis: below (outward) and above (inward) the axis row.
    x_row = int(round(frame.x_axis.pixel_position))
    below = fg[x_row : min(h, x_row + tick_length + 2), :].sum(axis=0).astype(np.float32)
    above = fg[max(0, x_row - tick_length - 1) : x_row, :].sum(axis=0).astype(np.float32)
    x_ticks = _select_tick_peaks(below, above, min_spacing=max(5, w // 100))

    # Y-axis: left (outward) and right (inward) of the axis column.
    y_col = int(round(frame.y_axis.pixel_position))
    left = fg[:, max(0, y_col - tick_length - 2) : y_col].sum(axis=1).astype(np.float32)
    right = fg[:, y_col + 1 : min(w, y_col + tick_length + 2)].sum(axis=1).astype(np.float32)
    y_ticks = _select_tick_peaks(left, right, min_spacing=max(5, h // 100))

    return x_ticks, y_ticks


def _select_tick_peaks(outward: np.ndarray, inward: np.ndarray, min_spacing: int) -> list[float]:
    """Pick whichever side is clearly better; if both look useful, merge."""
    out_peaks = _find_tick_peaks(outward, min_spacing=min_spacing)
    in_peaks = _find_tick_peaks(inward, min_spacing=min_spacing)
    # Heuristic: the "real" tick strip produces many, well-separated peaks.
    # A nearly-empty outward strip gives only 1–2 noise peaks.
    if len(in_peaks) >= max(3, 2 * len(out_peaks)):
        return in_peaks
    if len(out_peaks) >= max(3, 2 * len(in_peaks)):
        return out_peaks
    # Both sides have comparable signal — union and dedupe by min_spacing.
    merged = sorted(out_peaks + in_peaks)
    deduped: list[float] = []
    for p in merged:
        if not deduped or p - deduped[-1] >= min_spacing:
            deduped.append(p)
        else:
            deduped[-1] = 0.5 * (deduped[-1] + p)
    return deduped


def _find_tick_peaks(profile: np.ndarray, min_spacing: int) -> list[float]:
    from scipy.signal import find_peaks

    if profile.max() <= 0:
        return []
    # Height threshold derived from the profile itself; prominence guards
    # against residual label ink bleeding into the strip.
    peaks, _ = find_peaks(
        profile,
        height=max(1.0, 0.3 * profile.max()),
        distance=min_spacing,
        prominence=max(1.0, 0.2 * profile.max()),
    )
    return [_subpixel_peak(profile, int(p)) for p in peaks]


def attach_tick_labels(
    axis: AxisInfo,
    tick_pixels: list[float],
    labels: list[tuple[float, float, Optional[float], float]],
    tolerance: int,
    min_confidence: float = 0.5,
) -> AxisInfo:
    """Associate OCR'd numeric labels to detected ticks.

    `labels` entries: (label_centre_pixel_along_axis, _unused, parsed_value, confidence).

    Strategy: iterate over labels (not ticks) so each label binds to the
    tick that is physically closest to it. The earlier tick-first greedy
    order could claim a label from a farther tick if that tick happened
    to be processed before the real (closer) tick. Low-confidence OCR
    reads are excluded outright — they usually come from label fragments
    that were partially split or letters touching the axis line.
    """
    # Tesseract's word-level confidence is a mixed signal. `conf == 0` can
    # mean either "very uncertain" or "no word-confidence reported" (the
    # latter happens routinely when the character-level whitelist trimmed
    # output or when stray tick ink got glued to a clean digit string).
    # Low-but-nonzero confidences (e.g. 0.2-0.4) are the ones that tend to
    # carry genuine misreads like "570" in place of "50". Reject only that
    # band; keep conf == 0 and conf >= threshold.
    usable: list[tuple[int, float, float, float]] = []
    for i, (lp, _, val, conf) in enumerate(labels):
        if val is None:
            continue
        if 0.0 < conf < min_confidence:
            continue
        usable.append((i, lp, val, conf))

    used_ticks: set[int] = set()
    mapping: dict[int, tuple[float, float]] = {}  # tick_idx -> (value, conf)
    # Process labels in order of descending confidence so the cleanest reads
    # bind their closest tick first.
    for _, lp, val, conf in sorted(usable, key=lambda r: -r[3]):
        best_tick, best_d = -1, float("inf")
        for ti, tp in enumerate(tick_pixels):
            if ti in used_ticks:
                continue
            d = abs(lp - tp)
            if d < best_d and d <= tolerance:
                best_tick, best_d = ti, d
        if best_tick >= 0:
            used_ticks.add(best_tick)
            mapping[best_tick] = (val, conf)

    attached: list[TickLabel] = []
    for ti, tp in enumerate(tick_pixels):
        if ti in mapping:
            val, conf = mapping[ti]
            attached.append(TickLabel(pixel=tp, value=val, ocr_confidence=conf))
        else:
            attached.append(TickLabel(pixel=tp, value=None, ocr_confidence=0.0))
    axis.ticks = attached
    return axis
