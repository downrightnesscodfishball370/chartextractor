"""Streamlit frontend for ChartExtractor.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Make the src-layout package importable without pip install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from chart_extractor import extract_curve


st.set_page_config(page_title="Chart Extractor", layout="wide")
st.title("Chart Extractor")
st.caption("Upload a 2D graph image and extract (x, y) coordinates from the plotted curve.")


# --- Sidebar controls ---------------------------------------------------------
with st.sidebar:
    st.header("Extraction settings")

    n_points = st.number_input(
        "Number of output points",
        min_value=10, max_value=10000, value=500, step=10,
        help="How many (x, y) pairs to resample along the extracted curve.",
    )
    adaptive = st.checkbox(
        "Curvature-adaptive sampling",
        value=True,
        help="Place more samples where the curve bends; fewer on flat sections.",
    )
    smooth_window = st.slider(
        "Smoothing window (Savitzky-Golay)",
        min_value=3, max_value=51, value=11, step=2,
        help="Larger = smoother. Odd values only. Set low for noisy-but-sharp data.",
    )

    st.divider()
    use_color = st.checkbox(
        "Curve is a specific colour",
        value=False,
        help="Isolate the curve by hue (useful when axes/grid share intensity with the curve).",
    )
    hue = None
    if use_color:
        hue = st.slider(
            "Curve hue (OpenCV 0-179)",
            min_value=0, max_value=179, value=120,
            help="0=red, 30=yellow, 60=green, 90=cyan, 120=blue, 150=magenta.",
        )

    st.divider()
    overlay_size = st.slider("Overlay dot size (px)", 1, 6, 2)


# --- Main panel ---------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload chart image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
)

if uploaded is None:
    st.info("Upload a chart image to begin. Works best with axis-aligned plots that have visible numeric tick labels on both axes.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if image is None:
    st.error("Could not decode the uploaded image.")
    st.stop()

input_col, output_col = st.columns(2)
with input_col:
    st.subheader("Input")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

run = st.button("Extract coordinates", type="primary", use_container_width=True)

if not run:
    st.stop()

# --- Run the pipeline ---------------------------------------------------------
with st.spinner("Running extraction pipeline..."):
    try:
        result = extract_curve(
            image,
            curve_color_hue=hue,
            n_output_points=int(n_points),
            adaptive_sampling=adaptive,
            smooth_window=int(smooth_window),
        )
    except RuntimeError as e:
        st.error(f"Extraction failed: {e}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

# --- Overlay ------------------------------------------------------------------
overlay = image.copy()
if result.reprojected_pixels is not None:
    for u, v in result.reprojected_pixels:
        u_i, v_i = int(round(u)), int(round(v))
        if 0 <= u_i < overlay.shape[1] and 0 <= v_i < overlay.shape[0]:
            cv2.circle(overlay, (u_i, v_i), overlay_size, (0, 165, 255), -1)

with output_col:
    st.subheader("Extracted points (overlay)")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

# --- Metrics + warnings -------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Points extracted", len(result.points.x))
m2.metric("X-axis confidence", f"{result.x_axis_confidence:.2f}")
m3.metric("Y-axis confidence", f"{result.y_axis_confidence:.2f}")
m4.metric(
    "Scale",
    f"{result.calibration.x_scale} / {result.calibration.y_scale}",
    help="Detected x-axis / y-axis scale (linear or log10).",
)

for w in result.warnings:
    st.warning(w)

# --- Table + plot + download --------------------------------------------------
df = pd.DataFrame({
    "x": result.points.x,
    "y": result.points.y,
})

tab_table, tab_plot = st.tabs(["Coordinates", "Plot"])
with tab_table:
    st.dataframe(df, use_container_width=True, height=420)
with tab_plot:
    st.line_chart(df.set_index("x")["y"], height=420)

txt_bytes = "\n".join(f"{x}\t{y}" for x, y in zip(result.points.x, result.points.y)).encode("utf-8")
json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")

d1, d2 = st.columns(2)
d1.download_button(
    "Download TXT",
    data=txt_bytes,
    file_name=f"{Path(uploaded.name).stem}_extracted.txt",
    mime="text/plain",
    use_container_width=True,
)
d2.download_button(
    "Download JSON",
    data=json_bytes,
    file_name=f"{Path(uploaded.name).stem}_extracted.json",
    mime="application/json",
    use_container_width=True,
)
