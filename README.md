# ChartExtractor

A production-grade Python application that automatically digitizes 2D graphs and scientific charts. Given a chart image (PNG, JPG, etc.), it produces precise `(x, y)` coordinate datasets — including per-point uncertainty estimates — using a deterministic computer vision pipeline. No machine learning is used for pixel-level work; only Tesseract OCR is used for reading axis labels.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Installation & Setup](#4-installation--setup)
5. [Running the Application](#5-running-the-application)
6. [Using the Python API](#6-using-the-python-api)
7. [The 7-Stage Pipeline — Deep Dive](#7-the-7-stage-pipeline--deep-dive)
   - [Stage 1: Preprocessing](#stage-1-preprocessing)
   - [Stage 2: Axis & Tick Localization](#stage-2-axis--tick-localization)
   - [Stage 3a: OCR — Reading Labels](#stage-3a-ocr--reading-labels)
   - [Stage 3b: Calibration & Scale Detection](#stage-3b-calibration--scale-detection)
   - [Stage 4: Curve Isolation & Skeletonization](#stage-4-curve-isolation--skeletonization)
   - [Stage 5: Pixel → Data Coordinate Conversion](#stage-5-pixel--data-coordinate-conversion)
   - [Stage 6: Smoothing](#stage-6-smoothing)
   - [Stage 7: Resampling](#stage-7-resampling)
8. [Data Types Reference](#8-data-types-reference)
9. [Key Algorithms Explained](#9-key-algorithms-explained)
10. [Confidence & Quality Metrics](#10-confidence--quality-metrics)
11. [Error Handling & Warnings](#11-error-handling--warnings)
12. [Contributing Guide](#12-contributing-guide)

---

## 1. What It Does

When scientists or engineers publish graphs in papers or reports, the underlying data is rarely available. ChartExtractor solves this by treating the chart image as input and recovering the original numerical data.

**Input:** A raster image of a 2D chart (line graph, scatter plot, etc.)

**Output:**
- `(x, y)` coordinate pairs in the same units as the chart's axis labels
- Per-point uncertainty `σ_x` and `σ_y` (how precisely we could locate each point)
- The detected axis scales (`linear`, `log10`, or `ln`)
- Confidence scores for both axes
- Non-fatal warnings (e.g., low-coverage axis, chart rotation)

**Capabilities:**
- Handles linear and logarithmic (base-10 and natural) axes — auto-detected
- Optional color-based curve isolation for multi-curve charts
- Subpixel precision (typically ±0.1 pixel) using parabolic profile fitting
- Automatic RANSAC-based outlier rejection for noisy OCR labels
- Curvature-adaptive resampling to place more output points at sharp bends

---

## 2. Architecture Overview

The entire extraction is orchestrated by a single function `extract_curve()` in `pipeline.py`. It calls each processing stage in sequence, passing data forward as plain Python dataclasses.

```
Image (BGR numpy array)
        │
        ▼
┌─────────────────────┐
│  Stage 1            │  preprocess.py
│  Preprocessing      │  Denoise, flatten illumination, binarize, detect edges
└────────┬────────────┘
         │ gray, binary, edges
         ▼
┌─────────────────────┐
│  Stage 2            │  axes.py
│  Axis & Tick        │  Find x/y axes; detect tick positions at subpixel accuracy
│  Localization       │
└────────┬────────────┘
         │ AxisFrame (axis pixel positions + tick lists)
         ▼
┌─────────────────────┐
│  Stage 3a           │  ocr.py
│  OCR Label Reading  │  Run Tesseract on axis label strips; parse numeric values
└────────┬────────────┘
         │ (pixel, value, confidence) tuples for each tick
         ▼
┌─────────────────────┐
│  Stage 3b           │  calibration.py
│  Calibration &      │  Fit pixel→data mapping; auto-detect linear/log scale;
│  Scale Detection    │  RANSAC outlier rejection
└────────┬────────────┘
         │ Calibration object (coefficients + scale kind)
         ▼
┌─────────────────────┐
│  Stage 4            │  curve.py
│  Curve Isolation &  │  Remove axes/gridlines; skeletonize to 1-pixel width;
│  Skeletonization    │  extract ordered (u, v) pixel points per column
└────────┬────────────┘
         │ (u_pixels, v_subpixel, σ_v uncertainty)
         ▼
┌─────────────────────┐
│  Stage 5            │  pipeline.py + calibration.py
│  Pixel → Data       │  Apply calibration mapping with uncertainty propagation
└────────┬────────────┘
         │ (x_data, y_data, σ_x, σ_y)
         ▼
┌─────────────────────┐
│  Stage 6            │  sampling.py
│  Smoothing          │  Savitzky-Golay filter to suppress digitization noise
└────────┬────────────┘
         │ smoothed (x, y)
         ▼
┌─────────────────────┐
│  Stage 7            │  sampling.py
│  Resampling         │  Uniform or curvature-adaptive resampling to n points
└────────┬────────────┘
         │
         ▼
    ExtractionResult
    (PointSeries, Calibration, confidences, warnings, reprojected pixels)
```

Every stage is a set of **pure functions** — no global state, no side effects. This makes each stage independently testable and replaceable.

---

## 3. Project Structure

```
D:\Dev\ChartExtractor\
│
├── app.py                          # Streamlit web UI — entry point
├── requirements.txt                # Python package dependencies
├── Research.txt                    # Academic references and algorithm notes
│
└── src/
    └── chart_extractor/            # Core library (importable package)
        ├── __init__.py             # Public API: exports extract_curve() and types
        ├── types.py                # All dataclass definitions
        ├── pipeline.py             # extract_curve() — orchestrates stages 1–7
        ├── preprocess.py           # Stage 1: image normalization
        ├── axes.py                 # Stage 2: axis/tick detection
        ├── ocr.py                  # Stage 3a: Tesseract OCR wrapper
        ├── calibration.py          # Stage 3b+5: pixel↔data mapping
        ├── curve.py                # Stage 4: curve isolation and extraction
        └── sampling.py             # Stages 6+7: smoothing and resampling
```

**Rule of thumb:** Every file corresponds to exactly one pipeline stage. When debugging, start by identifying which stage produces wrong output, then open only that file.

---

## 4. Installation & Setup

### Prerequisites

**Python 3.10+** is required.

**Tesseract OCR** must be installed as a system binary — it is not a Python package:

- **Windows:** Download from [UB Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki). Install to the default path (`C:\Program Files\Tesseract-OCR\`). The code auto-detects this path.
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

### Python Dependencies

```bash
# Create and activate a virtual environment (strongly recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**What each package does:**

| Package | Version | Role |
|---|---|---|
| `numpy` | ≥1.24 | All array/matrix math; backbone of every stage |
| `opencv-python` | ≥4.8 | Core CV: morphology, Hough, Canny, HSV, Gaussian filter |
| `scikit-image` | ≥0.22 | Zhang-Suen skeletonization (`skimage.morphology.skeletonize`) |
| `scipy` | ≥1.11 | Savitzky-Golay filter, spline interpolation, peak finding |
| `pytesseract` | ≥0.3.10 | Python wrapper around the Tesseract binary |
| `Pillow` | ≥10.0 | Image I/O (used as fallback by some loaders) |
| `streamlit` | ≥1.32 | Web UI framework |
| `pandas` | ≥2.1 | DataFrame for CSV/JSON export |

> **Note on opencv-contrib:** The guided filter (`cv2.ximgproc.guidedFilter`) lives in `opencv-contrib-python`, not `opencv-python`. If you install `opencv-contrib-python`, you get both. If only `opencv-python` is installed, the code automatically falls back to bilateral filtering — correct but slightly slower on large images.

---

## 5. Running the Application

### Web UI (Recommended)

```bash
streamlit run app.py
```

This starts a local web server. Open the URL printed in your terminal (typically `http://localhost:8501`).

**UI walkthrough:**

1. **Upload** a chart image using the file uploader (PNG, JPG, BMP, TIF, or WEBP).
2. **Configure** extraction parameters in the left sidebar:
   - **Output points** (`n_points`): How many `(x, y)` pairs to produce (10–10000). More points = smoother output curve. Default: 500.
   - **Adaptive sampling**: If enabled, more points are placed at high-curvature regions (sharp bends). If disabled, points are evenly spaced in arc length.
   - **Smooth window** (`smooth_window`): Width of the Savitzky-Golay smoothing window (must be odd, 3–51). Larger = smoother but reduces detail. Default: 11.
   - **Use color isolation**: Enable if your chart has multiple curves — lets you isolate one curve by color.
   - **Hue** (0–179): OpenCV HSV hue of the curve color. Appears only when color isolation is on.
   - **Overlay dot size**: Visual size of the orange reprojected-point dots drawn on the output image.
3. **Click "Extract coordinates"** — the pipeline runs (typically 1–5 seconds).
4. **Inspect results:**
   - Output image with orange dots showing where each extracted point lands
   - Quality metrics: number of points, x/y axis confidence, detected scale types
   - Warnings if anything looks suspicious
   - Data table with `x`, `y`, `sigma_x`, `sigma_y` columns
   - Line chart preview
   - Download buttons for CSV and JSON

### Python API

See [Section 6](#6-using-the-python-api) below.

---

## 6. Using the Python API

```python
import cv2
from src.chart_extractor import extract_curve

# Load image with OpenCV (BGR format — the pipeline expects BGR, not RGB)
image = cv2.imread("my_chart.png")

# Run extraction with default settings
result = extract_curve(image)

# Or with all parameters specified:
result = extract_curve(
    image,
    curve_color_hue=120,   # Isolate a blue curve (HSV hue 120 ≈ green-blue)
    n_output_points=500,   # How many (x, y) pairs to output
    adaptive_sampling=True, # More points at sharp bends
    smooth_window=11,       # Savitzky-Golay smoothing window (must be odd)
)

# Access results
print(f"Extracted {len(result.points.x)} points")
print(f"X-axis scale: {result.calibration.x_scale}")   # "linear", "log10", or "ln"
print(f"Y-axis scale: {result.calibration.y_scale}")
print(f"X confidence: {result.x_axis_confidence:.2f}")
print(f"Y confidence: {result.y_axis_confidence:.2f}")

# The actual data
x = result.points.x       # numpy array of x data values
y = result.points.y       # numpy array of y data values
sx = result.points.sigma_x  # uncertainty in x (same units as x)
sy = result.points.sigma_y  # uncertainty in y (same units as y)

# Any non-fatal warnings
for w in result.warnings:
    print(f"Warning: {w}")

# Pixel coordinates of extracted points (for overlay drawing)
# result.reprojected_pixels is an Nx2 array of (u, v) pixel coords
```

**Key import:** The public API is `src.chart_extractor`. Import only from there, not from internal submodules — internal APIs may change.

---

## 7. The 7-Stage Pipeline — Deep Dive

### Stage 1: Preprocessing

**File:** `src/chart_extractor/preprocess.py`

**Goal:** Produce a clean binary (black/white) image and a grayscale image from the raw input, while preserving sharp edges and suppressing noise and uneven illumination.

**Functions and what they do:**

#### `to_grayscale(image)` — lines 13–19
Converts BGR (or BGRA) to 8-bit grayscale. Handles 4-channel images by dropping the alpha channel first.

```
BGR image → Gray 8-bit image
```

#### `edge_preserving_denoise(gray)` — lines 21–33
Removes sensor noise while keeping edge sharpness. Uses a **guided filter** (O(N) time), which smooths flat regions while leaving transitions intact. Falls back to bilateral filtering if `opencv-contrib` is not installed.

Why not Gaussian blur? Gaussian blurs edges, which hurts tick detection. Why not median filter? Median is slow and doesn't preserve sub-pixel sharpness as well.

#### `normalise_illumination(gray)` — lines 35–44
Fixes uneven illumination (common in scanned images or photos). Subtracts a large-radius local median from the image. Anything brighter than its neighborhood becomes bright; anything darker becomes dark. This flattens background gradients.

```
flat_gray = gray - cv2.medianBlur(gray, radius=large) + 128
```

#### `adaptive_binarise(flat)` — lines 46–58
Converts grayscale to binary (0 or 255). Uses **adaptive thresholding** — each pixel's threshold is determined from its local neighborhood (Gaussian-weighted mean), so the same image can have different thresholds in bright and dark regions. This handles uneven ink density or faded prints. Result is inverted so foreground (ink) = 255, background = 0.

#### `detect_edges(gray)` — lines 60–67
Runs Canny edge detection. Thresholds are set automatically from the image histogram:
```
lo = 0.66 × median_gray_value
hi = 1.33 × median_gray_value
```
This avoids hard-coded thresholds that fail on high- or low-contrast images.

#### `preprocess(image)` — lines 69–81
The entry point called by `pipeline.py`. Chains all the above and returns a dict:
```python
{
    "gray":     gray image,        # for OCR and subpixel fitting
    "denoised": denoised gray,     # for guided filter output
    "flat":     illumination-flat, # for binarization
    "binary":   binary foreground, # for morphology, axis detection, skeletonization
    "edges":    Canny edges,       # for rotation estimation
}
```

---

### Stage 2: Axis & Tick Localization

**File:** `src/chart_extractor/axes.py`

**Goal:** Find the x-axis (horizontal line) and y-axis (vertical line) in pixel coordinates, detect tick positions, and return everything at subpixel precision.

#### Subpixel peak fitting — `_subpixel_peak(profile, idx)` — lines 51–60

This is the most important utility in the file. Given a 1D intensity profile and a coarse peak index, it fits a parabola through 3 neighboring samples and finds the vertex analytically:

```
Profile values at three adjacent positions: f₋₁, f₀, f₊₁

Parabolic vertex offset: δ = 0.5 × (f₋₁ − f₊₁) / (f₋₁ − 2·f₀ + f₊₁)

Subpixel position: idx + clip(δ, −0.5, +0.5)
```

This achieves ~0.1 pixel localization accuracy. The clipping prevents wild extrapolation from noisy profiles.

#### `estimate_frame_rotation(edges)` — lines 30–46

Uses probabilistic Hough line detection on the edge map. Clusters detected line angles around 0° (horizontal) and 90° (vertical). Returns the dominant residual rotation. If `|rotation| > 1.5°`, a warning is emitted. At large rotations, the columnar curve extraction becomes inaccurate.

#### `locate_axes(binary)` — lines 63–104

Finds the pixel positions of the x and y axis lines:

1. Apply **morphological opening** with elongated horizontal kernel `(w//20, 1)` → isolates long horizontal lines (x-axis candidates). Repeat with `(1, h//20)` for vertical (y-axis).
2. Sum along columns (for y-axis detection) and along rows (for x-axis). This gives projection profiles where peaks correspond to continuous horizontal/vertical structures.
3. Find the **strongest peak** in the lower half of the horizontal projection (x-axis) and in the left half of the vertical projection (y-axis) — standard chart layout assumption.
4. Apply `_subpixel_peak()` to refine to subpixel accuracy.
5. Compute **coverage**: what fraction of the axis line is continuously foreground? Lower coverage → lower confidence.

Returns an `AxisFrame` with `x_row` (float pixel row) and `y_col` (float pixel column).

#### `detect_ticks(binary, frame)` — lines 109–130

For the x-axis: extract a thin horizontal strip of pixels directly below the x-axis line (~8px tall). Project this strip vertically (sum each column). Peaks in this 1D profile are tick positions. Use `scipy.signal.find_peaks` with:
- `height` threshold (must exceed background noise)
- `distance` ≈ `width // 100` (minimum spacing between ticks)
- `prominence` (tick must stand above its surroundings)

Apply `_subpixel_peak()` to each detected tick. Repeat symmetrically for y-axis (strip left of y-axis, project horizontally).

#### `attach_tick_labels(axis_info, tick_pixels, ocr_labels)` — lines 149–176

Associates OCR-read labels with their nearest tick. For each tick position, finds the closest OCR label centroid within a tolerance (25 pixels). Unmatched ticks get `value=None` and are marked as uncalibrated.

---

### Stage 3a: OCR — Reading Labels

**File:** `src/chart_extractor/ocr.py`

**Goal:** Read the numeric values printed next to tick marks on both axes.

#### Auto-detection of Tesseract — lines 26–43

The code searches standard installation paths for the Tesseract binary across Windows, macOS, and Linux. If found, it sets `pytesseract.pytesseract.tesseract_cmd`. If not found, OCR calls will raise a clear error.

#### `parse_numeric(text)` — lines 59–69

Converts an OCR string to a float:
- Replaces Unicode minus variants (`−`, `–`, `—`) with ASCII `-`
- Strips whitespace, removes commas
- Validates with regex: `^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`
- Returns `None` if string does not match a valid number

This handles: `123`, `-1.5`, `1.2e3`, `2.5E-4`, `0.001`.

#### `_ocr_region(gray, bbox, psm)` — lines 72–101

Runs Tesseract on a region of the grayscale image:
1. Crops to `bbox = (x, y, w, h)`
2. If the region is smaller than ~30px tall, upscales by 2–4× (Tesseract needs minimum font size)
3. Runs `pytesseract.image_to_data()` with:
   - `psm` (page segmentation mode): 7 = single line, 6 = block of text
   - `tessedit_char_whitelist=0123456789.+-eE` — discard any OCR output that isn't a digit or valid numeric character
4. Returns list of `(text, confidence, bbox)` tuples

#### `read_x_axis_labels(gray, frame)` — lines 104–118

Defines a strip below the x-axis (full width of the plot, 40px tall) and calls `_ocr_region()` with PSM=6. Returns centroids as `(label_centre_x, label_centre_y, numeric_value, confidence)`.

#### `read_y_axis_labels(gray, frame)` — lines 121–135

Same but for a strip left of the y-axis (60px wide, full height of plot). OCR text in this region is typically rotated, so the strip is wider to ensure labels are captured.

---

### Stage 3b: Calibration & Scale Detection

**File:** `src/chart_extractor/calibration.py`

**Goal:** Turn `(tick_pixel, tick_value)` pairs into a precise mathematical mapping from any pixel coordinate to its data-space value, and automatically determine whether the scale is linear, log₁₀, or ln.

#### Why is scale detection necessary?

On a logarithmic axis, the pixel positions of 1, 10, 100 are evenly spaced, but the values are multiplicative. A linear fit on raw values would fail; the correct fit is on `log₁₀(value)` vs. pixel.

#### `_linear_fit(pixels, values)` — lines 18–24

Closed-form least squares: solve `A @ [a, b]ᵀ = values` where `A` is the 2-column matrix `[pixels, 1s]`. Returns `(a, b, residual_std)`.

```
data_value = a × pixel + b
```

#### `_ransac_fit(pixels, values, n_iter, tol)` — lines 27–55

RANSAC (Random Sample Consensus) robustly fits a line even when some OCR labels are wrong:

1. Repeat `n_iter` (200) times:
   - Randomly pick 2 points
   - Fit a line through those 2 points
   - Count how many remaining points are within `tol` of that line (inliers)
2. Keep the fit with the most inliers
3. Refit using all inliers only
4. Tolerance = 2% of data range

If fewer than 2 inliers are ever found, falls back to using all points.

#### `calibrate_axis(axis_info)` — lines 82–122

1. Extract all `(pixel, value)` pairs where `value is not None`
2. For each scale candidate (`linear`, `log10`, `ln`):
   - Transform values: linear → `v`, log10 → `log₁₀(v)`, ln → `ln(v)`
   - Run RANSAC on `(pixel, transformed_value)`
   - Compute relative residual: `residual_std / data_range` (normalized so large-scale and small-scale axes compare fairly)
3. Pick the scale with the smallest relative residual
4. Refit final coefficients on RANSAC inliers for that scale
5. Compute confidence: `confidence *= max(0, 1 − min(1, relative_residual × 10))`

#### `build_calibration(x_axis, y_axis)` — lines 125–128

Calls `calibrate_axis()` on both axes independently and assembles a `Calibration` object.

#### `homography_from_corners(src_corners, dst_corners)` — lines 131–143

For perspective-distorted images (photos of printed charts). Computes a 3×3 homography matrix using `cv2.findHomography` with RANSAC. The `Calibration` object can hold this matrix. Not used in the default pipeline but available for extension.

---

### Stage 4: Curve Isolation & Skeletonization

**File:** `src/chart_extractor/curve.py`

**Goal:** Isolate the curve line from everything else (axes, tick marks, gridlines, labels) and thin it down to a 1-pixel-wide skeleton for precise coordinate extraction.

#### `remove_axes_and_grid(binary, frame)` — lines 19–46

1. **Erase axis bands:** Zero out a ±2px band around each axis line.
2. **Isolate gridlines morphologically:**
   - Horizontal gridlines: opening with kernel `(max(15, w//4), 1)` — only structures longer than w//4 survive
   - Vertical gridlines: opening with kernel `(1, max(15, h//4))`
3. **Subtract extracted gridlines** from the binary image → only short structures (curve segments, tick marks) remain.
4. **Closing operation** to repair small gaps where the curve crossed a gridline and was accidentally erased.

#### `isolate_color_curve(image_bgr, hue)` — lines 49–59

Used only when `curve_color_hue` is provided:
1. Convert BGR → HSV
2. Create a mask for pixels within ±12 hue units of the target
3. Add constraint: saturation ≥ 60, value ≥ 40 (excludes white/grey/black regions)
4. Handle red hue wraparound (hues near 0° and near 179° are both "red")

Returns a binary mask where 255 = curve color detected.

#### `skeletonise(binary)` — lines 62–64

Uses scikit-image's Zhang-Suen thinning algorithm to reduce any curve thickness down to exactly 1 pixel. The result is a binary image where connected pixels represent the curve centerline.

Why skeletonization rather than edge detection? Edges produce two parallel lines (both sides of a thick curve). Skeletonization gives one centerline, which is what we want to sample from.

#### `extract_points_by_column(skeleton, gray, bounds)` — lines 69–129

The core curve extraction loop. For each column `u` from `u_min` to `u_max`:

1. Find all skeleton pixels in column `u` within the plot's vertical bounds
2. If multiple candidates (e.g., noise speckles), pick the one closest to the previous column's y position — **continuity heuristic** to follow the curve rather than jump to noise
3. Apply **parabolic subpixel refinement** on the inverted grayscale profile at pixel `v`:
   ```
   Invert: I = 255 - gray  (so dark curve → bright peak)
   δ = 0.5 × (I[v-1] - I[v+1]) / (I[v-1] - 2·I[v] + I[v+1])
   v_sub = v + clip(δ, -0.5, +0.5)
   ```
4. Estimate per-point uncertainty from profile sharpness:
   ```
   curvature = I[v-1] - 2·I[v] + I[v+1]
   σ_v = 1 / sqrt(|curvature|)   (sharper peak → lower uncertainty)
   ```

Returns arrays `(u_pixels, v_subpixel, sigma_v)`.

#### `order_contour(skeleton)` — lines 132–178

For non-functional curves (closed loops, curves with multiple y-values at a single x). Walks the skeleton as a connected polyline using depth-first search from an endpoint. Returns an ordered Nx2 array of `(u, v)` pixel coordinates.

---

### Stage 5: Pixel → Data Coordinate Conversion

**Done in:** `pipeline.py` lines 105–115, using `Calibration.pixel_to_data()` from `types.py` lines 51–67.

For each extracted pixel `(u, v)`, apply the calibration mapping:

| Scale | Formula |
|---|---|
| `linear` | `x = ax × u + bx` |
| `log10` | `x = 10^(ax × u + bx)` |
| `ln` | `x = e^(ax × u + bx)` |

**Uncertainty propagation:**

The pixel uncertainty `σ_v` (from profile curvature) propagates to data-space uncertainty `σ_y`:

| Scale | Uncertainty |
|---|---|
| `linear` | `σ_y = |ay| × σ_v` |
| `log10` | `σ_y = ln(10) × y_data × |ay| × σ_v` |
| `ln` | `σ_y = y_data × |ay| × σ_v` |

X-axis uncertainty `σ_x` is fixed at half a pixel in data units: `σ_x = 0.5 × |ax|`.

---

### Stage 6: Smoothing

**File:** `src/chart_extractor/sampling.py`, lines 11–30

#### `savitzky_golay_smooth(x, y, window, polyorder=3)`

Applies a **Savitzky-Golay filter** to the y values. This filter fits a low-degree polynomial to each local window of points and evaluates it at the center — mathematically equivalent to a weighted moving average but preserving peaks and inflection points (unlike simple moving average which flattens them).

Parameters:
- `window`: Must be odd. Larger values = stronger smoothing but less detail.
- `polyorder`: Degree of the local polynomial. Default 3 (cubic).
- Auto-reduces window if fewer than `window` data points exist.

---

### Stage 7: Resampling

**File:** `src/chart_extractor/sampling.py`, lines 33–88

After smoothing, the points are on a per-column grid (one per image column). This does not give uniform arc-length spacing or any particular density distribution. Resampling fixes this.

#### `uniform_arclength_resample(x, y, n)`

Computes cumulative arc length along the curve, then places `n` points uniformly in arc length. The result looks evenly spaced when plotted, regardless of the curve's actual direction.

#### `curvature_adaptive_resample(x, y, n, curvature_weight=0.7)`

Used when `adaptive_sampling=True`. Places more points at high-curvature regions:

1. Compute arc-length parameterization
2. Fit cubic splines `x(s)` and `y(s)` (where `s` = arc length)
3. Compute curvature `κ` from spline derivatives:
   ```
   κ = |x'·y'' - y'·x''| / (x'² + y'²)^(3/2)
   ```
4. Define a **density function** combining uniform and curvature terms:
   ```
   d(s) = (1 - w) + w × κ_normalized(s)     (strictly positive)
   ```
5. Integrate to get CDF, invert to draw samples at equal density intervals

The result: flat stretches of the curve get fewer points; sharp bends get more.

---

## 8. Data Types Reference

**File:** `src/chart_extractor/types.py`

### `TickLabel`
```python
@dataclass
class TickLabel:
    pixel: float           # Subpixel position of tick centre (row for x-axis, col for y-axis)
    value: Optional[float] # Numeric value from OCR (None if OCR failed or no label found)
    ocr_confidence: float  # OCR confidence normalized to [0.0, 1.0]
```

### `AxisInfo`
```python
@dataclass
class AxisInfo:
    orientation: Literal["x", "y"]  # Which axis
    pixel_position: float            # Subpixel position of the axis line itself
    ticks: list[TickLabel]           # All ticks detected on this axis
    fit_residual: float              # Std dev of the calibration line fit
    confidence: float                # Overall detection confidence [0.0, 1.0]
```

### `Calibration`
```python
@dataclass
class Calibration:
    ax: float               # x = ax × u + bx  (or similar for log scales)
    bx: float
    ay: float               # y = ay × v + by
    by: float
    x_scale: ScaleKind      # "linear" | "log10" | "ln"
    y_scale: ScaleKind
    homography: Optional[np.ndarray]  # 3×3 perspective rectification (usually None)
```

Key method:
```python
def pixel_to_data(self, u: float, v: float) -> tuple[float, float]:
    # Converts pixel (u=col, v=row) to (x_data, y_data)
```

### `PointSeries`
```python
@dataclass
class PointSeries:
    x: np.ndarray                    # Output x coordinates (data units)
    y: np.ndarray                    # Output y coordinates (data units)
    sigma_x: Optional[np.ndarray]    # Per-point uncertainty in x
    sigma_y: Optional[np.ndarray]    # Per-point uncertainty in y
```

### `ScaleKind`
```python
ScaleKind = Literal["linear", "log10", "ln"]
```

### `ExtractionResult` (returned by `extract_curve()`)
```python
@dataclass
class ExtractionResult:
    points: PointSeries
    calibration: Calibration
    x_axis_confidence: float         # [0.0, 1.0]
    y_axis_confidence: float
    warnings: list[str]              # Non-fatal issues
    reprojected_pixels: np.ndarray   # Nx2 array of (u, v) pixel coords for overlay
```

---

## 9. Key Algorithms Explained

### Parabolic Subpixel Refinement

Used in axis localization and curve extraction to achieve ~0.1 pixel accuracy without iterative optimization.

**Idea:** Given a discrete intensity profile with a peak at index `i`, assume the true peak lies between `i-1` and `i+1`. Fit a parabola through the three samples and find its vertex analytically:

```
f(i-1) = a(i-1)² + b(i-1) + c
f(i)   = ai² + bi + c
f(i+1) = a(i+1)² + b(i+1) + c

Vertex at: i* = i - (f(i+1) - f(i-1)) / (2 × (f(i+1) - 2f(i) + f(i-1)))
```

This is clamped to ±0.5 pixels to handle cases where the profile is flat or noisy.

### RANSAC for Calibration

When OCR reads axis labels, some readings will be wrong (misread digits, partial text). RANSAC handles this:

```
Repeat 200 times:
    1. Randomly pick 2 (pixel, value) pairs
    2. Fit a line through them
    3. Count how many OTHER pairs agree with this line (within 2% tolerance)
    4. Remember the line with the most agreeing pairs (inliers)

Final fit: least-squares over all inliers of the best line
```

If even 1 in 3 OCR readings is wrong, RANSAC still recovers the correct calibration.

### Morphological Axis/Grid Isolation

OpenCV's morphological **opening** with an elongated structuring element is a powerful way to selectively keep or remove image structures by length:

- Kernel `(1, h//4)` = vertical bar, height `h//4`
- Opening with this kernel: only vertical structures longer than `h//4` survive
- Opening with `(w//4, 1)`: only horizontal structures longer than `w//4` survive

This is how axes and long gridlines are separated from the curve (which has no single long straight segment when viewed as horizontal or vertical lines).

### Savitzky-Golay vs. Moving Average

A simple moving average blurs everything, including peaks. Savitzky-Golay fits a polynomial locally and evaluates it — this preserves the shape of peaks and inflections while still suppressing high-frequency noise. For digitized curves where the peak position and height matter, Savitzky-Golay is the correct choice.

---

## 10. Confidence & Quality Metrics

### Axis Confidence (per axis)

Computed in `axes.py` and updated in `calibration.py`:

```
coverage = fraction of axis line that is continuous foreground pixels
           (gaps indicate the axis is partially invisible or merged with background)

relative_residual = calibration_std / data_range
                    (how well OCR labels align with a line fit)

confidence = coverage × max(0, 1 − min(1, relative_residual × 10))
```

A confidence of 1.0 means: full axis coverage + perfect calibration fit. A confidence below 0.4 triggers a warning.

### When to Trust the Output

| Confidence | Interpretation |
|---|---|
| 0.8–1.0 | High quality; all labels read correctly, full axis coverage |
| 0.5–0.8 | Acceptable; some OCR noise but RANSAC corrected it |
| 0.3–0.5 | Low; inspect visually, check for chart rotation or faint axes |
| < 0.3 | Very low; results may be unreliable |

---

## 11. Error Handling & Warnings

### Fatal Errors (raise `RuntimeError`)

| Condition | Where | Message |
|---|---|---|
| Fewer than 2 ticks with valid OCR labels on either axis | `pipeline.py:73–76` | "Insufficient calibration points" |
| No curve pixels found after isolation | `pipeline.py:101–102` | "No curve points recovered" |

### Non-Fatal Warnings (added to `result.warnings`)

| Condition | Threshold |
|---|---|
| Chart rotation detected | `\|rotation\| > 1.5°` |
| Axis coverage too low | `coverage < 0.4` |
| Too few calibration ticks | `< 2 ticks with value` |
| Low OCR confidence | Average confidence < 0.5 |

Warnings do not stop extraction. The result may still be usable — inspect the overlay image and confidence scores.

---

## 12. Contributing Guide

### Getting Oriented

1. **Read `types.py` first.** All data flows through the dataclasses defined there. Understanding `AxisInfo`, `Calibration`, and `PointSeries` is the prerequisite for understanding any other file.

2. **Read `pipeline.py` second.** It is short (~150 lines) and shows exactly how the stages connect. Each function call in `extract_curve()` is one stage.

3. **Then read the stage file you want to modify.** Each file is self-contained and ~80–180 lines.

### Design Principles to Follow

- **Pure functions:** Every processing function takes explicit inputs and returns explicit outputs. Do not read from or write to global state. Do not mutate inputs.
- **No ML for pixel work:** The project uses classical CV (morphology, Hough, thresholding, etc.) deliberately. Tesseract is the only ML/statistical model allowed, and only for text recognition.
- **Subpixel accuracy:** Whenever you detect a spatial location (tick, axis line, curve point), apply parabolic subpixel refinement. Never return integer pixel coordinates as a final answer.
- **Robustness first:** Prefer algorithms that degrade gracefully (RANSAC, adaptive thresholds) over algorithms that require tuning per image.

### Adding a New Feature

**Example: adding support for axis grid detection and suppression of dotted grids**

1. Identify which stage it belongs to. Dotted grid removal belongs in Stage 4 (curve.py), specifically in `remove_axes_and_grid()`.
2. Write the logic as a new function in that file. Test it in isolation with a few images before wiring it in.
3. Add it to the `remove_axes_and_grid()` call chain.
4. If the feature needs a new parameter, add it to `extract_curve()` in `pipeline.py` and thread it through. Update the Streamlit UI in `app.py` to expose it.
5. Add the new parameter to the public API docstring in `__init__.py` if applicable.

### Common Debugging Workflow

If extraction produces wrong results:

1. **Check warnings first.** `result.warnings` tells you if confidence is low.
2. **Look at the overlay image.** The orange dots should track the curve. If they don't, the issue is in Stage 4 (curve isolation) or Stage 2 (axis localization).
3. **Check axis confidence.** Low x-axis confidence → tick detection or OCR failed → check Stage 2 or Stage 3a.
4. **Add intermediate output.** In `pipeline.py`, temporarily save intermediate images with `cv2.imwrite()`:
   ```python
   cv2.imwrite("debug_binary.png", prep["binary"])
   cv2.imwrite("debug_skeleton.png", skeleton * 255)
   ```
5. **Isolate the stage.** Call individual functions directly from a test script rather than running the full pipeline.

### Code Style

- Python 3.10+ type hints throughout
- Dataclasses for all structured data (no dicts as function return values)
- No global variables
- Line length ≤ 100 characters
- Comments only when the **why** is non-obvious — well-named variables document the **what**
