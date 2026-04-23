"""Dump every intermediate stage so we can see exactly where the left-edge noise comes from."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from chart_extractor import axes, curve, preprocess

IMG = r"C:\Users\User\Pictures\Relationship-between-the-valve-opening-and-the-flow-area-of-the-butterfly-valve.webp"
OUT = Path(r"D:\Dev\ChartExtractor\debug_out")
OUT.mkdir(exist_ok=True)

image = cv2.imread(IMG, cv2.IMREAD_COLOR)
print(f"Image shape: {image.shape}")

stages = preprocess.preprocess(image)
gray = stages["gray"]
binary = stages["binary"]

frame = axes.locate_axes(binary)
print(f"Frame: x_axis row={frame.x_axis.pixel_position:.2f} conf={frame.x_axis.confidence:.2f}")
print(f"       y_axis col={frame.y_axis.pixel_position:.2f} conf={frame.y_axis.confidence:.2f}")

u_min = int(round(frame.y_axis.pixel_position)) + 1
u_max = gray.shape[1] - 1
v_min = 0
v_max = int(round(frame.x_axis.pixel_position)) - 1
print(f"Plot bounds: u=[{u_min},{u_max}]  v=[{v_min},{v_max}]")

# Hue check
auto_hue = curve.auto_detect_curve_hue(image, (u_min, v_min, u_max, v_max))
print(f"Auto-detected hue: {auto_hue}")

cleaned = curve.remove_axes_and_grid(binary, frame.x_axis.pixel_position, frame.y_axis.pixel_position)
skeleton = curve.skeletonise(cleaned)
skeleton_pruned = curve.prune_short_branches(skeleton, min_branch_len=6)

cv2.imwrite(str(OUT / "01_binary.png"), binary)
cv2.imwrite(str(OUT / "02_cleaned.png"), cleaned)
cv2.imwrite(str(OUT / "03_skeleton.png"), skeleton)
cv2.imwrite(str(OUT / "04_skeleton_pruned.png"), skeleton_pruned)

# Look at the first 30 columns past the y-axis. Count skeleton pixels per column.
print(f"\nSkeleton pixels in first 30 columns past y-axis (u_min={u_min}):")
for u in range(u_min, min(u_min + 30, u_max + 1)):
    rows = np.where(skeleton_pruned[v_min:v_max + 1, u] > 0)[0]
    if rows.size:
        rows = rows + v_min
        print(f"  u={u}  count={rows.size}  v_rows={rows.tolist()[:10]}")

# Crop the bottom-left corner at the start of the curve for close inspection.
crop = binary[max(0, v_max - 150) : v_max + 10, u_min - 5 : u_min + 80]
cv2.imwrite(str(OUT / "05_start_binary.png"), crop)
crop_sk = skeleton_pruned[max(0, v_max - 150) : v_max + 10, u_min - 5 : u_min + 80]
cv2.imwrite(str(OUT / "06_start_skeleton.png"), crop_sk * 255 if crop_sk.max() == 1 else crop_sk)
