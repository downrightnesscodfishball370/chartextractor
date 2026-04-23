"""Stage 4: Curve isolation, skeletonisation, and subpixel extraction.

The critical precision step. We:
  1. Remove axes/gridlines via directional morphological opening.
  2. Optionally isolate a colour-coded curve in HSV.
  3. Thin to a 1-px skeleton (Zhang-Suen via scikit-image).
  4. Walk columns and fit a parabola to the original grayscale profile
     to locate each curve sample to subpixel precision.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _axis_stripe_bounds(
    fg: np.ndarray, axis_pos: float, orientation: str, max_check: int = 40
) -> tuple[int, int]:
    """Walk outward from the detected axis centreline to find the full width of
    the axis stripe in the binary. Adaptive thresholding often thickens axis
    lines and produces edge-ghost stripes parallel to the centre; using a
    fixed band underestimates this and leaves residue inside the plot.

    Returns inclusive (low, high) index bounds of the stripe.
    """
    h, w = fg.shape
    pos = int(round(axis_pos))
    if orientation == "y":
        density = fg.mean(axis=0)
        length = w
    else:
        density = fg.mean(axis=1)
        length = h
    if density.max() <= 0:
        return pos, pos
    peak = density[max(0, pos - 2) : min(length, pos + 3)].max()
    threshold = max(0.25, 0.5 * peak)
    low = pos
    while low > 0 and density[low - 1] >= threshold and pos - low < max_check:
        low -= 1
    high = pos
    while high < length - 1 and density[high + 1] >= threshold and high - pos < max_check:
        high += 1
    return low, high


def _drop_small_components(fg: np.ndarray, min_bbox_dim: int, min_area: int) -> np.ndarray:
    """Drop connected components whose max bbox dimension < `min_bbox_dim`
    or whose area < `min_area`. The curve spans most of the plot, so its
    component is massive; tick-marks, text fragments, dotted-grid dots, and
    edge-ghost debris are all small. Classical CV hygiene step.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 1:
        return fg
    out = np.zeros_like(fg)
    for i in range(1, num):
        _, _, w_cc, h_cc, area = stats[i]
        if area >= min_area and max(w_cc, h_cc) >= min_bbox_dim:
            out[labels == i] = 1
    return out


def remove_axes_and_grid(
    binary: np.ndarray,
    x_axis_row: float,
    y_axis_col: float,
    grid_min_length_frac: float = 0.25,
    grid_thickness: int = 5,
    min_component_bbox: int = 25,
    min_component_area: int = 40,
) -> np.ndarray:
    """Return a binary with axes and long horizontal/vertical grid lines removed.

    - Axis band widths are derived from the actual binary (not a fixed 3 px),
      so thick axes and anti-aliased edge-ghosts are erased cleanly.
    - After axis + grid removal we drop small connected components (tick
      marks, label fragments, dotted-grid dots, isolated debris) via
      bbox-size filtering. The curve spans the full plot width/height so its
      component is always far larger than the threshold.
    """
    h, w = binary.shape
    fg = (binary > 0).astype(np.uint8)

    # Erase the axis lines + their edge-ghost stripes + tick-mark attachments.
    # Stripe width is measured directly; we then extend by a small margin
    # on the plot-side to catch inward-pointing tick marks.
    tick_margin = 4
    y_lo, y_hi = _axis_stripe_bounds(fg, y_axis_col, "y")
    x_lo, x_hi = _axis_stripe_bounds(fg, x_axis_row, "x")
    fg[max(0, x_lo - 1) : min(h, x_hi + 2), :] = 0
    fg[:, max(0, y_lo - 1) : min(w, y_hi + 2)] = 0

    # Gridlines: long axis-aligned strokes. Kernel length defined by what we
    # are willing to call "a grid line" vs "a locally-horizontal curve".
    kx = max(15, int(grid_min_length_frac * w))
    ky = max(15, int(grid_min_length_frac * h))
    grid_h = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    grid_v = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
    if grid_thickness > 1:
        grid_h = cv2.dilate(grid_h, cv2.getStructuringElement(cv2.MORPH_RECT, (1, grid_thickness)))
        grid_v = cv2.dilate(grid_v, cv2.getStructuringElement(cv2.MORPH_RECT, (grid_thickness, 1)))
    no_grid = cv2.subtract(fg, cv2.bitwise_or(grid_h, grid_v))

    # Close tiny gaps where the curve intersected a grid line we just removed.
    no_grid = cv2.morphologyEx(no_grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Drop small debris: tick marks, label fragments, dotted-grid dots.
    no_grid = _drop_small_components(no_grid, min_component_bbox, min_component_area)

    # Axis-gutter cleanup. Even after widening band-erase, a "ladder" of the
    # axis's edge-ghost + attached tick-marks can survive as one tall thin
    # component adjacent to the y-axis (or one long thin component along the
    # x-axis). Kill only components that (a) sit mostly (>85%) inside the
    # gutter AND (b) actually span far along that axis — axis residue runs
    # the length of the plot, whereas a curve segment dwelling near an axis
    # (e.g. a low-y minimum or x=0 tail) is locally short.
    gutter = 25
    axis_span_frac = 0.40  # component must reach at least this fraction of plot extent along the axis to be treated as residue
    y_gutter = np.zeros_like(no_grid)
    y_gutter[:, : min(w, y_hi + gutter + 1)] = 1
    x_gutter = np.zeros_like(no_grid)
    x_gutter[max(0, x_lo - gutter) :, :] = 1
    num, labels, stats, _ = cv2.connectedComponentsWithStats(no_grid, connectivity=8)
    for i in range(1, num):
        comp = (labels == i)
        total = int(comp.sum())
        if total == 0:
            continue
        _, _, cw, ch, _ = stats[i]
        overlap_y = int((comp & y_gutter.astype(bool)).sum())
        overlap_x = int((comp & x_gutter.astype(bool)).sum())
        # y-gutter residue: elongated vertically along the y-axis.
        if overlap_y / total > 0.85 and ch >= axis_span_frac * h:
            no_grid[comp] = 0
            continue
        # x-gutter residue: elongated horizontally along the x-axis.
        if overlap_x / total > 0.85 and cw >= axis_span_frac * w:
            no_grid[comp] = 0

    return (no_grid > 0).astype(np.uint8) * 255


def auto_detect_curve_hue(
    bgr: np.ndarray,
    plot_bounds: tuple[int, int, int, int],
    sat_min: int = 50,
    val_min: int = 40,
    val_max: int = 230,
) -> Optional[int]:
    """Return the dominant saturated hue inside the plot region, or None.

    Rationale: if a plotted curve is colour-coded (blue line, red line, etc.),
    pixels belonging to the curve have far higher saturation than the grey
    gridlines and black axes/labels. Histogramming hue over pixels that pass
    a saturation+value filter surfaces the curve's colour automatically.
    """
    u_min, v_min, u_max, v_max = plot_bounds
    roi = bgr[v_min : v_max + 1, u_min : u_max + 1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    mask = (s_chan >= sat_min) & (v_chan >= val_min) & (v_chan <= val_max)
    if int(mask.sum()) < 50:
        return None
    # Coloured-pixel count must be > ~0.2% of plot area to avoid triggering
    # on stray JPEG artifacts.
    if mask.sum() / mask.size < 0.002:
        return None
    hues = h_chan[mask]
    hist = cv2.calcHist([hues], [0], None, [180], [0, 180]).flatten()
    # Smooth circularly before peak-picking (hue wraps around 180).
    kernel = np.ones(5) / 5.0
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smoothed = np.convolve(padded, kernel, mode="valid")

    # A plotted curve spans most of the plot height; a title banner or
    # coloured legend patch sits in a narrow v-range. Weight each hue by
    # the vertical spread of its pixels so the curve wins even when a
    # banner has more pixels.
    plot_h = mask.shape[0]
    v_idx = np.repeat(np.arange(plot_h)[:, None], mask.shape[1], axis=1)[mask]
    # Bucket rows-per-hue into the same 180-bin grid for O(n) computation.
    rows_top = np.full(180, plot_h, dtype=np.int32)
    rows_bot = np.full(180, -1, dtype=np.int32)
    for h_val, v_val in zip(hues.astype(np.int32), v_idx.astype(np.int32)):
        if v_val < rows_top[h_val]: rows_top[h_val] = v_val
        if v_val > rows_bot[h_val]: rows_bot[h_val] = v_val
    span = np.where(rows_bot >= 0, rows_bot - rows_top + 1, 0).astype(np.float32)
    span_frac = np.clip(span / max(1, plot_h), 0.0, 1.0)
    # Kill hues that don't span a meaningful portion of the plot (banners,
    # legend swatches, title text). A real curve always spans >= 30%.
    # Also penalise hues whose pixels sit near the top or bottom edge only,
    # which catches coloured title bands that otherwise slip through.
    plot_h_i = int(plot_h)
    top_in_edge = rows_top < 0.15 * plot_h_i
    bot_in_interior = rows_bot < 0.5 * plot_h_i
    banner_like = top_in_edge & bot_in_interior & (span_frac < 0.3)
    gate = np.where(span_frac >= 0.3, 1.0, 0.0) * np.where(banner_like, 0.0, 1.0)
    weighted = smoothed * gate
    if float(weighted.max()) <= 0:
        return None
    peak = int(np.argmax(weighted))
    return peak


def isolate_color_curve(bgr: np.ndarray, hue: int, hue_tol: int = 12) -> np.ndarray:
    """HSV mask for a single coloured curve. `hue` in OpenCV range [0, 179]."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([max(0, hue - hue_tol), 60, 40], dtype=np.uint8)
    hi = np.array([min(179, hue + hue_tol), 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    if hue - hue_tol < 0:  # red wraps around
        lo2 = np.array([180 + (hue - hue_tol), 60, 40], dtype=np.uint8)
        hi2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    return mask


def skeletonise(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning. Fast; adequate for most single-curve charts."""
    return (skeletonize(binary > 0)).astype(np.uint8) * 255


def prune_short_branches(skeleton: np.ndarray, min_branch_len: int = 6) -> np.ndarray:
    """Remove skeleton branches shorter than `min_branch_len` pixels.

    These short spurs usually come from data-point markers (circles, squares)
    placed on the curve. Identifying junction pixels and walking outward to
    each endpoint is the deterministic way to do this; we drop any walk
    shorter than the threshold.
    """
    skel = (skeleton > 0).astype(np.uint8)
    if skel.sum() == 0:
        return skeleton

    # Count 8-connected neighbours for every skeleton pixel.
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbour_count = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere((skel == 1) & (neighbour_count == 1))

    to_remove: set[tuple[int, int]] = set()
    for ey, ex in endpoints:
        path = [(int(ey), int(ex))]
        cur = (int(ey), int(ex))
        prev = None
        for _ in range(min_branch_len + 1):
            y, x = cur
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if (ny, nx_) == prev:
                        continue
                    if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1] and skel[ny, nx_]:
                        # Hit a junction (>2 neighbours) -> stop here.
                        if neighbour_count[ny, nx_] > 2:
                            nxt = (ny, nx_)
                            break
                        if nxt is None:
                            nxt = (ny, nx_)
                if nxt is not None and neighbour_count[nxt[0], nxt[1]] > 2:
                    break
            if nxt is None:
                break
            if neighbour_count[nxt[0], nxt[1]] > 2:
                # Reached a junction: remove everything walked so far (short spur).
                if len(path) < min_branch_len:
                    to_remove.update(path)
                break
            prev = cur
            cur = nxt
            path.append(cur)

    if to_remove:
        out = skel.copy()
        for y, x in to_remove:
            out[y, x] = 0
        return out.astype(np.uint8) * 255
    return skeleton


# --- Subpixel extraction -------------------------------------------------------

def extract_points_by_column(
    skeleton: np.ndarray,
    gray: np.ndarray,
    plot_bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertical-profile scan with parabolic subpixel y-refinement.

    plot_bounds: (u_min, v_min, u_max, v_max) in pixel coords, defining the
    region of interest (typically axes-bounded).

    Returns (u, v_sub, sigma_v) where u is integer column and v_sub is a
    subpixel row. `sigma_v` is a naive uncertainty estimate (pixels).
    """
    u_min, v_min, u_max, v_max = plot_bounds
    cols, rows_sub, sigmas = [], [], []
    prev_v: Optional[float] = None

    # Invert grayscale so "dark curve" -> high intensity for fitting.
    inv = 255.0 - gray.astype(np.float32)

    for u in range(u_min, u_max + 1):
        rows = np.where(skeleton[v_min : v_max + 1, u] > 0)[0]
        if rows.size == 0:
            continue
        rows = rows + v_min
        # If multiple skeleton pixels in this column, keep the one closest
        # to the previous sample (continuity heuristic). This is what gives
        # robustness against residual grid speckle.
        if prev_v is not None:
            v = int(rows[np.argmin(np.abs(rows - prev_v))])
        else:
            v = int(rows[np.argmin(np.abs(rows - (v_min + v_max) / 2.0))])

        # Parabolic subpixel refinement on the *grayscale* profile
        # (the skeleton itself is binary and carries no subpixel info).
        if 0 < v < gray.shape[0] - 1:
            a, b, c = inv[v - 1, u], inv[v, u], inv[v + 1, u]
            denom = a - 2 * b + c
            if abs(denom) > 1e-6:
                delta = 0.5 * (a - c) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                v_sub = v + delta
            else:
                v_sub = float(v)
        else:
            v_sub = float(v)

        # Uncertainty proxy: inverse of local profile curvature (normalised).
        # Sharp, high-contrast peaks -> low sigma. This is a rough estimator.
        if 0 < v < gray.shape[0] - 1:
            curvature = max(1e-3, float(inv[v - 1, u] - 2 * inv[v, u] + inv[v + 1, u]))
            sigma = min(1.0, 1.0 / np.sqrt(abs(curvature) + 1e-3))
        else:
            sigma = 1.0

        cols.append(u)
        rows_sub.append(v_sub)
        sigmas.append(sigma)
        prev_v = v_sub

    return np.asarray(cols, dtype=np.float64), np.asarray(rows_sub, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)


def order_contour(skeleton: np.ndarray) -> np.ndarray:
    """Walk the skeleton as a connected polyline (for non-functional curves).

    Returns an Nx2 array of (u, v) pixel coordinates ordered along the path.
    This is used when the curve is not a single-valued function of x
    (e.g. closed loops, vertical segments). Uses a simple endpoint-start
    DFS which is deterministic for a 1-px skeleton with at most one branch.
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return np.empty((0, 2))
    pts = set(zip(xs.tolist(), ys.tolist()))

    def neighbours(p):
        x, y = p
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in pts:
                    yield q

    # Endpoints have exactly one 8-neighbour; start from one if present.
    start = None
    for p in pts:
        if sum(1 for _ in neighbours(p)) == 1:
            start = p
            break
    if start is None:
        start = min(pts)  # deterministic fallback

    ordered = [start]
    visited = {start}
    cur = start
    while True:
        nxt = None
        for q in neighbours(cur):
            if q not in visited:
                nxt = q
                break
        if nxt is None:
            break
        ordered.append(nxt)
        visited.add(nxt)
        cur = nxt
    return np.asarray(ordered, dtype=np.float64)
