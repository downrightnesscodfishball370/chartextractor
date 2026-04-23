"""End-to-end orchestration.

Each stage is a pure function so individual components can be unit-tested
with synthetic inputs. `extract_curve` wires them together and returns a
structured result with per-stage confidence metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import axes, calibration, curve, ocr, preprocess, sampling
from .types import Calibration, PointSeries


@dataclass
class ExtractionResult:
    points: PointSeries
    calibration: Calibration
    x_axis_confidence: float
    y_axis_confidence: float
    warnings: list[str] = field(default_factory=list)
    # Back-projection of extracted points into pixel space, for visual QA.
    reprojected_pixels: Optional[np.ndarray] = None


def extract_curve(
    image: np.ndarray,
    curve_color_hue: Optional[int] = None,
    n_output_points: int = 500,
    adaptive_sampling: bool = True,
    smooth_window: int = 11,
) -> ExtractionResult:
    warnings: list[str] = []

    # --- 1. Preprocess -------------------------------------------------------
    stages = preprocess.preprocess(image)
    gray = stages["gray"]
    binary = stages["binary"]
    edges = stages["edges"]

    # --- 2. Axis + ticks -----------------------------------------------------
    rotation = axes.estimate_frame_rotation(edges)
    if abs(rotation) > 1.5:
        warnings.append(
            f"Detected frame rotation of {rotation:.2f}deg; consider perspective rectification."
        )

    frame = axes.locate_axes(binary)
    if frame.x_axis.confidence < 0.4:
        warnings.append(f"Low x-axis coverage ({frame.x_axis.confidence:.2f}); detection may be unreliable.")
    if frame.y_axis.confidence < 0.4:
        warnings.append(f"Low y-axis coverage ({frame.y_axis.confidence:.2f}); detection may be unreliable.")

    x_ticks, y_ticks = axes.detect_ticks(binary, frame)

    # --- 3. OCR labels -------------------------------------------------------
    x_label_raw = ocr.read_x_axis_labels(gray, frame.x_axis.pixel_position, frame.y_axis.pixel_position)
    y_label_raw = ocr.read_y_axis_labels(gray, frame.x_axis.pixel_position, frame.y_axis.pixel_position)

    # Per-tick single-char OCR supplements the full-strip reader. Bold
    # sans-serif digits that full-strip Tesseract mangles ("2"->"9",
    # "5"->"A") tend to read correctly when cropped tight to one glyph.
    x_label_per = ocr.read_x_axis_labels_per_tick(gray, frame.x_axis.pixel_position, x_ticks)
    y_label_per = ocr.read_y_axis_labels_per_tick(gray, frame.y_axis.pixel_position, y_ticks)

    x_labels = [(lx, ly, v, c) for (lx, ly, v, c) in x_label_raw] + \
               [(lx, ly, v, c) for (lx, ly, v, c) in x_label_per]
    y_labels_axis = [(ly, lx, v, c) for (lx, ly, v, c) in y_label_raw] + \
                    [(ly, lx, v, c) for (lx, ly, v, c) in y_label_per]

    x_axis = axes.attach_tick_labels(frame.x_axis, x_ticks, x_labels, tolerance=25)
    y_axis = axes.attach_tick_labels(frame.y_axis, y_ticks, y_labels_axis, tolerance=25)

    labelled_x = sum(1 for t in x_axis.ticks if t.value is not None)
    labelled_y = sum(1 for t in y_axis.ticks if t.value is not None)
    if labelled_x < 2 or labelled_y < 2:
        raise RuntimeError(
            f"Insufficient OCR labels (x={labelled_x}, y={labelled_y}). "
            "Provide manual calibration points or improve image quality."
        )

    # --- 3b. Calibration ----------------------------------------------------
    calib = calibration.build_calibration(x_axis, y_axis)

    # --- 4. Curve extraction ------------------------------------------------
    # Plot interior — everything right of the y-axis and above the x-axis.
    u_min = int(round(frame.y_axis.pixel_position)) + 1
    u_max = gray.shape[1] - 1
    v_min = 0
    v_max = int(round(frame.x_axis.pixel_position)) - 1

    # Auto-detect a coloured curve unless the caller specified a hue.
    auto_hue: Optional[int] = None
    if curve_color_hue is None and image.ndim == 3:
        auto_hue = curve.auto_detect_curve_hue(image, (u_min, v_min, u_max, v_max))
        if auto_hue is not None:
            warnings.append(f"Auto-detected curve hue ~{auto_hue} (OpenCV scale). Using HSV mask for extraction.")

    hue = curve_color_hue if curve_color_hue is not None else auto_hue

    if hue is not None and image.ndim == 3:
        colour_mask = curve.isolate_color_curve(image, hue)
        cleaned = curve.remove_axes_and_grid(colour_mask, frame.x_axis.pixel_position, frame.y_axis.pixel_position)
    else:
        cleaned = curve.remove_axes_and_grid(binary, frame.x_axis.pixel_position, frame.y_axis.pixel_position)

    skeleton = curve.skeletonise(cleaned)
    skeleton = curve.prune_short_branches(skeleton, min_branch_len=6)

    u_px, v_px, sigma_v = curve.extract_points_by_column(
        skeleton, gray, plot_bounds=(u_min, v_min, u_max, v_max)
    )
    if u_px.size < 2:
        raise RuntimeError("No curve points were recovered; check binarisation or colour mask.")

    # --- 5. Pixel -> data ---------------------------------------------------
    x_data, y_data = calib.pixel_to_data(u_px, v_px)

    # Propagate pixel sigma to data units (linear approximation).
    # d(data)/d(pixel) = |a| in linear, = ln(base) * base^(a*u+b) * a in log.
    if calib.y_scale == "linear":
        sigma_y = np.abs(calib.ay) * sigma_v
    elif calib.y_scale == "log10":
        sigma_y = np.log(10.0) * y_data * np.abs(calib.ay) * sigma_v
    else:  # ln
        sigma_y = y_data * np.abs(calib.ay) * sigma_v
    sigma_x = np.full_like(x_data, np.abs(calib.ax) * 0.5)  # half-pixel in x

    # --- 6. Smoothing + resampling -----------------------------------------
    order = np.argsort(x_data)
    x_sorted, y_sorted = x_data[order], y_data[order]
    y_smoothed = sampling.savitzky_golay_smooth(x_sorted, y_sorted, window=smooth_window)

    if adaptive_sampling:
        x_out, y_out = sampling.curvature_adaptive_resample(x_sorted, y_smoothed, n_out=n_output_points)
    else:
        x_out, y_out = sampling.uniform_arclength_resample(x_sorted, y_smoothed, n_out=n_output_points)

    # --- 7. Back-projection for visual QA -----------------------------------
    # Invert calibration to map (x_out, y_out) back to pixel coordinates,
    # so callers can overlay extracted points on the source image.
    u_back = (np.log10(x_out) - calib.bx) / calib.ax if calib.x_scale == "log10" else (x_out - calib.bx) / calib.ax
    v_back = (np.log10(y_out) - calib.by) / calib.ay if calib.y_scale == "log10" else (y_out - calib.by) / calib.ay
    reproj = np.stack([u_back, v_back], axis=1)

    return ExtractionResult(
        points=PointSeries(
            x=x_out,
            y=y_out,
            sigma_x=np.interp(x_out, x_sorted, sigma_x[order]),
            sigma_y=np.interp(x_out, x_sorted, sigma_y[order]),
        ),
        calibration=calib,
        x_axis_confidence=x_axis.confidence,
        y_axis_confidence=y_axis.confidence,
        warnings=warnings,
        reprojected_pixels=reproj,
    )
