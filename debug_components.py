"""Dump connected-component stats from the cleaned binary to see why the ladder isn't being removed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from chart_extractor import axes, curve, preprocess

IMG = r"C:\Users\User\Pictures\Relationship-between-the-valve-opening-and-the-flow-area-of-the-butterfly-valve.webp"
image = cv2.imread(IMG, cv2.IMREAD_COLOR)
stages = preprocess.preprocess(image)
binary = stages["binary"]
frame = axes.locate_axes(binary)

# Reproduce the innards of remove_axes_and_grid to see component stats at each step.
h, w = binary.shape
fg = (binary > 0).astype(np.uint8)

y_lo, y_hi = curve._axis_stripe_bounds(fg, frame.y_axis.pixel_position, "y")
x_lo, x_hi = curve._axis_stripe_bounds(fg, frame.x_axis.pixel_position, "x")
print(f"Y-axis stripe cols: {y_lo}-{y_hi}   X-axis stripe rows: {x_lo}-{x_hi}")

fg[max(0, x_lo - 1) : min(h, x_hi + 2), :] = 0
fg[:, max(0, y_lo - 1) : min(w, y_hi + 2)] = 0

# Same grid removal
kx = max(15, int(0.25 * w))
ky = max(15, int(0.25 * h))
grid_h = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
grid_v = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
grid_h = cv2.dilate(grid_h, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
grid_v = cv2.dilate(grid_v, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
no_grid = cv2.subtract(fg, cv2.bitwise_or(grid_h, grid_v))
no_grid = cv2.morphologyEx(no_grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

num, labels, stats, _ = cv2.connectedComponentsWithStats(no_grid, connectivity=8)
print(f"\nAll components (after grid removal, before any filter):  n={num-1}")
ranked = sorted(range(1, num), key=lambda i: -stats[i, 4])
for rank, i in enumerate(ranked[:25]):
    x, y, cw, ch, area = stats[i]
    print(f"  #{rank:2d}  bbox=(x={x:4d},y={y:4d},w={cw:4d},h={ch:4d})  area={area:6d}")

# Now apply small-component drop.
no_grid2 = curve._drop_small_components(no_grid, 25, 40)
num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(no_grid2, connectivity=8)
print(f"\nAfter small-component drop: n={num2-1}")
ranked2 = sorted(range(1, num2), key=lambda i: -stats2[i, 4])
for rank, i in enumerate(ranked2[:15]):
    x, y, cw, ch, area = stats2[i]
    print(f"  #{rank:2d}  bbox=(x={x:4d},y={y:4d},w={cw:4d},h={ch:4d})  area={area:6d}")

# Show gutter overlap analysis for largest components.
gutter = 25
y_gutter = np.zeros_like(no_grid2, dtype=bool)
y_gutter[:, : min(w, y_hi + gutter + 1)] = True
x_gutter = np.zeros_like(no_grid2, dtype=bool)
x_gutter[max(0, x_lo - gutter) :, :] = True

print(f"\nGutter overlap analysis (y_gutter cols 0-{y_hi+gutter}, x_gutter rows {x_lo-gutter}+):")
for rank, i in enumerate(ranked2[:10]):
    comp = (labels2 == i)
    total = int(comp.sum())
    oy = int((comp & y_gutter).sum())
    ox = int((comp & x_gutter).sum())
    x, y, cw, ch, _ = stats2[i]
    print(f"  #{rank:2d}  bbox=(x={x},y={y},w={cw},h={ch})  total={total}  y_overlap={oy/total:.2f}  x_overlap={ox/total:.2f}")
