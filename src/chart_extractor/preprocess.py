"""Stage 1: Preprocessing.

Goal: normalise illumination, suppress noise *without* blurring curve/axis
edges, and produce (a) a clean grayscale image for subpixel work and
(b) a binary foreground mask for morphological work.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def edge_preserving_denoise(gray: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    # Guided filter with self-guidance: O(N) and avoids the staircase/
    # gradient-reversal artefacts of the bilateral filter, which matters
    # because those artefacts bias the subpixel parabolic fit later.
    g = gray.astype(np.float32) / 255.0
    try:
        # Available in opencv-contrib. Falls back to bilateral otherwise.
        out = cv2.ximgproc.guidedFilter(guide=g, src=g, radius=radius, eps=eps)
    except AttributeError:
        out = cv2.bilateralFilter(gray, d=2 * radius + 1, sigmaColor=25, sigmaSpace=radius)
        return out
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def normalise_illumination(gray: np.ndarray, block: int = 51) -> np.ndarray:
    # Subtract a local median estimate of background. Neutralises scanner
    # shading without attenuating thin strokes (unlike morphological
    # top-hat with a round kernel, which can eat narrow curves).
    if block % 2 == 0:
        block += 1
    background = cv2.medianBlur(gray, block)
    flat = cv2.subtract(background, gray)  # dark ink -> bright foreground
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def adaptive_binarise(gray: np.ndarray, block: int = 25, C: int = 7) -> np.ndarray:
    # Input here is the output of `normalise_illumination`, which has already
    # removed illumination gradient (background ~ 0, ink ~ 255). A global
    # threshold is therefore both simpler and more reliable than an adaptive
    # one: Otsu finds the optimal split between the two modes and produces a
    # clean bipartite result with ink = 255 (foreground), background = 0.
    #
    # History note: the previous version ran adaptive_threshold with
    # THRESH_BINARY_INV, which double-inverted relative to normalise_illumination
    # and silently swapped the "foreground" of the binary to be the chart's
    # background — breaking every downstream morphology step.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def detect_edges(gray: np.ndarray) -> np.ndarray:
    # Auto hysteresis thresholds from the image's own gradient distribution;
    # robust across exposures without per-image tuning.
    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    return cv2.Canny(gray, lo, hi, L2gradient=True)


def preprocess(image: np.ndarray) -> dict:
    gray = to_grayscale(image)
    denoised = edge_preserving_denoise(gray)
    flat = normalise_illumination(denoised)
    binary = adaptive_binarise(flat)
    edges = detect_edges(denoised)
    return {
        "gray": gray,
        "denoised": denoised,
        "flat": flat,
        "binary": binary,
        "edges": edges,
    }
