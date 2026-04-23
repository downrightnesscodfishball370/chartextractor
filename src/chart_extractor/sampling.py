"""Stage 6/7: Resampling, smoothing, and uncertainty propagation."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def savitzky_golay_smooth(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Preserves peaks and inflections far better than a moving average.

    Assumes `x` is monotonic (enforce by sorting upstream).
    """
    n = len(y)
    if n < max(5, window):
        return y.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    w = max(polyorder + 2 + (polyorder % 2), w)
    if w > n:
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=polyorder, mode="interp")


def uniform_arclength_resample(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline so output points are evenly spaced in arc length."""
    if len(x) < 2:
        return x.copy(), y.copy()
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total <= 0:
        return x.copy(), y.copy()
    targets = np.linspace(0, total, n_out)
    return np.interp(targets, s, x), np.interp(targets, s, y)


def curvature_adaptive_resample(
    x: np.ndarray,
    y: np.ndarray,
    n_out: int,
    curvature_weight: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Place more samples where local curvature is high.

    Builds a CDF over (alpha + (1-alpha) * |kappa|) parameterised by arc
    length, then inverts it to draw `n_out` equi-density samples.
    `curvature_weight` = 1 gives pure-curvature weighting; 0 gives
    uniform arclength (a sanity check).
    """
    if len(x) < 4:
        return uniform_arclength_resample(x, y, n_out)

    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(seg)))

    # Second derivatives via cubic spline (preferred over finite differences
    # because the curve has already been smoothed and spline gives us a
    # continuous curvature we can integrate).
    cs_x = CubicSpline(s, x)
    cs_y = CubicSpline(s, y)
    x1, x2 = cs_x(s, 1), cs_x(s, 2)
    y1, y2 = cs_y(s, 1), cs_y(s, 2)
    num = np.abs(x1 * y2 - y1 * x2)
    den = np.power(x1 * x1 + y1 * y1, 1.5) + 1e-9
    kappa = num / den

    # Density ~ (1 - w) + w * normalised_curvature; strictly positive.
    k_norm = kappa / (kappa.max() + 1e-9)
    density = (1.0 - curvature_weight) + curvature_weight * k_norm
    # CDF of density along arc length.
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(s))))
    cdf /= cdf[-1]
    u = np.linspace(0, 1, n_out)
    s_new = np.interp(u, cdf, s)
    return cs_x(s_new), cs_y(s_new)
