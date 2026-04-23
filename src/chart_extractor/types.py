from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

ScaleKind = Literal["linear", "log10", "ln"]


@dataclass
class TickLabel:
    # Subpixel pixel coordinate of the tick centre along its axis.
    pixel: float
    # Data value parsed from the OCR label (None if reading failed).
    value: Optional[float]
    # OCR confidence in [0, 1].
    ocr_confidence: float = 0.0


@dataclass
class AxisInfo:
    # "x" is the horizontal data axis (a horizontal line in the image),
    # "y" is the vertical data axis (a vertical line in the image).
    orientation: Literal["x", "y"]
    # Subpixel position of the axis line in the perpendicular direction
    # (row index for x-axis, column index for y-axis).
    pixel_position: float
    # Inlier ticks with recognised labels, sorted by pixel coordinate.
    ticks: list[TickLabel] = field(default_factory=list)
    # Residual std (in data units) of the tick linear/log fit.
    fit_residual: float = 0.0
    # Detection confidence in [0, 1].
    confidence: float = 0.0


@dataclass
class Calibration:
    # Affine pixel -> data mapping per axis.
    # For linear:  data = a * pixel + b
    # For log10:   log10(data) = a * pixel + b
    ax: float
    bx: float
    ay: float
    by: float
    x_scale: ScaleKind = "linear"
    y_scale: ScaleKind = "linear"
    # Optional full 3x3 homography for perspective-rectified images.
    homography: Optional[np.ndarray] = None

    def pixel_to_data(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.homography is not None:
            pts = np.stack([u, v, np.ones_like(u)], axis=0)
            w = self.homography @ pts
            u = w[0] / w[2]
            v = w[1] / w[2]
        x = self.ax * u + self.bx
        y = self.ay * v + self.by
        if self.x_scale == "log10":
            x = np.power(10.0, x)
        elif self.x_scale == "ln":
            x = np.exp(x)
        if self.y_scale == "log10":
            y = np.power(10.0, y)
        elif self.y_scale == "ln":
            y = np.exp(y)
        return x, y


@dataclass
class PointSeries:
    x: np.ndarray
    y: np.ndarray
    # Per-point localisation uncertainty in data units (std estimate).
    sigma_x: Optional[np.ndarray] = None
    sigma_y: Optional[np.ndarray] = None
