"""Diagnostic: run full pipeline end-to-end and summarise."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from chart_extractor import extract_curve

IMG = r"C:\Users\User\Pictures\Relationship-between-the-valve-opening-and-the-flow-area-of-the-butterfly-valve.webp"
OUT = Path(r"D:\Dev\ChartExtractor\debug_out")
OUT.mkdir(exist_ok=True)

image = cv2.imread(IMG, cv2.IMREAD_COLOR)
result = extract_curve(image, n_output_points=50)

print(f"Points: {len(result.points.x)}")
print(f"x range: [{result.points.x.min():.3f}, {result.points.x.max():.3f}]")
print(f"y range: [{result.points.y.min():.3f}, {result.points.y.max():.3f}]")
print(f"Axis confidence: x={result.x_axis_confidence:.2f}, y={result.y_axis_confidence:.2f}")
for w in result.warnings:
    print(f"  WARN: {w}")

print("\nSample points (first 10):")
for i in range(min(10, len(result.points.x))):
    print(f"  x={result.points.x[i]:7.3f}  y={result.points.y[i]:7.4f}")
print("...")
print("Last 5:")
for i in range(max(0, len(result.points.x) - 5), len(result.points.x)):
    print(f"  x={result.points.x[i]:7.3f}  y={result.points.y[i]:7.4f}")

# Draw overlay
overlay = image.copy()
for u, v in result.reprojected_pixels.astype(int):
    if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
        cv2.circle(overlay, (int(u), int(v)), 3, (0, 0, 255), -1)
cv2.imwrite(str(OUT / "99_result_overlay.png"), overlay)
print(f"\nOverlay saved: {OUT / '99_result_overlay.png'}")
