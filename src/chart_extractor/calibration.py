"""Stage 3b/5: Calibration (pixel <-> data) and scale detection.

Two sources of truth are available after axis/OCR: a set of (pixel, value)
pairs per axis. With n >= 2 pairs we over-determine the 2-parameter
affine fit and can (a) reject OCR outliers via RANSAC and (b) decide
between linear and log scales by comparing residuals.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .types import AxisInfo, Calibration, ScaleKind, TickLabel


def _linear_fit(pixels: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    # Closed-form least squares for y = a*x + b; returns (a, b, residual_std).
    A = np.vstack([pixels, np.ones_like(pixels)]).T
    sol, *_ = np.linalg.lstsq(A, values, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    res = values - (a * pixels + b)
    return a, b, float(np.std(res)) if len(res) > 1 else 0.0


def _ransac_fit(
    pixels: np.ndarray,
    values: np.ndarray,
    tol: float,
    iters: int = 200,
    seed: int = 0,
) -> tuple[float, float, np.ndarray, float]:
    """Return (a, b, inlier_mask, residual_std). Falls back to LS if n < 3."""
    n = len(pixels)
    if n < 3:
        a, b, s = _linear_fit(pixels, values)
        return a, b, np.ones(n, dtype=bool), s
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        if pixels[i] == pixels[j]:
            continue
        a = (values[j] - values[i]) / (pixels[j] - pixels[i])
        b = values[i] - a * pixels[i]
        res = np.abs(values - (a * pixels + b))
        inliers = res <= tol
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers.sum() < 2:
        a, b, s = _linear_fit(pixels, values)
        return a, b, np.ones(n, dtype=bool), s
    a, b, s = _linear_fit(pixels[best_inliers], values[best_inliers])
    return a, b, best_inliers, s


def _fit_and_score(
    pixels: np.ndarray,
    values: np.ndarray,
    scale: ScaleKind,
) -> tuple[float, float, np.ndarray, float]:
    # Transform values into the space where the mapping is linear.
    if scale == "linear":
        v = values
    elif scale == "log10":
        if np.any(values <= 0):
            return math.nan, math.nan, np.zeros_like(values, dtype=bool), math.inf
        v = np.log10(values)
    elif scale == "ln":
        if np.any(values <= 0):
            return math.nan, math.nan, np.zeros_like(values, dtype=bool), math.inf
        v = np.log(values)
    else:
        raise ValueError(scale)
    span = max(1.0, float(v.max() - v.min()))
    # RANSAC tolerance scaled to data range; 2% keeps 1-digit OCR errors out.
    tol = 0.02 * span
    return _ransac_fit(pixels, v, tol=tol)


def calibrate_axis(
    axis: AxisInfo,
    candidate_scales: tuple[ScaleKind, ...] = ("linear", "log10"),
) -> tuple[float, float, ScaleKind, AxisInfo]:
    """Pick the scale with the smallest relative residual and fit it.

    Returns (a, b, scale, updated_axis_with_inliers).
    """
    labelled: list[TickLabel] = [t for t in axis.ticks if t.value is not None]
    if len(labelled) < 2:
        raise ValueError(
            f"Need at least 2 recognised tick labels on {axis.orientation}-axis; "
            f"got {len(labelled)}."
        )
    pixels = np.array([t.pixel for t in labelled], dtype=np.float64)
    values = np.array([t.value for t in labelled], dtype=np.float64)

    best: Optional[tuple[float, float, ScaleKind, np.ndarray, float, int]] = None
    for scale in candidate_scales:
        a, b, inliers, resid = _fit_and_score(pixels, values, scale)
        if math.isnan(a):
            continue
        # Normalise residual by transformed range so we can compare scales.
        v = values if scale == "linear" else (np.log10 if scale == "log10" else np.log)(values)
        rel = resid / max(1e-9, float(np.ptp(v[inliers])))
        n_in = int(inliers.sum())
        # Prefer more inliers; a fit that drops half the labels to claim a
        # zero residual is worse than a fit with all labels and small error.
        # Residual is the tiebreaker among fits with equal inlier count.
        if best is None or (n_in, -rel) > (best[5], -best[4]):
            best = (a, b, scale, inliers, rel, n_in)

    assert best is not None
    a, b, scale, inliers, rel, _ = best

    # Mark outliers in the axis so the pipeline can report them.
    kept: list[TickLabel] = []
    for i, t in enumerate(labelled):
        if inliers[i]:
            kept.append(t)
    axis.ticks = kept
    axis.fit_residual = rel
    # Confidence: combines coverage (set upstream) with fit tightness.
    axis.confidence = float(axis.confidence * max(0.0, 1.0 - min(1.0, rel * 10.0)))
    return a, b, scale, axis


def build_calibration(x_axis: AxisInfo, y_axis: AxisInfo) -> Calibration:
    ax, bx, sx, _ = calibrate_axis(x_axis)
    ay, by, sy, _ = calibrate_axis(y_axis)
    return Calibration(ax=ax, bx=bx, ay=ay, by=by, x_scale=sx, y_scale=sy)


def homography_from_corners(
    image_corners: np.ndarray,
    data_corners: np.ndarray,
) -> np.ndarray:
    """DLT homography for perspective-distorted plot areas.

    image_corners: 4x2 pixel coordinates (TL, TR, BR, BL).
    data_corners:  4x2 corresponding rectified plot-area coordinates.
    Use this only when the plot frame is not axis-aligned.
    """
    import cv2
    H, _ = cv2.findHomography(image_corners, data_corners, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return H
