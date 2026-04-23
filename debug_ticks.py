import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import cv2
import numpy as np
from chart_extractor import axes, preprocess

IMG = r"C:\Users\User\Pictures\Relationship-between-the-valve-opening-and-the-flow-area-of-the-butterfly-valve.webp"
image = cv2.imread(IMG, cv2.IMREAD_COLOR)
stages = preprocess.preprocess(image)
binary = stages["binary"]
frame = axes.locate_axes(binary)

y_col = int(round(frame.y_axis.pixel_position))
tick_length = 8

# Left strip (outward ticks)
left_strip = binary[:, max(0, y_col - tick_length - 2):y_col + 1]
left_profile = (left_strip > 0).sum(axis=1)
print(f"Y-axis at col {y_col}. Left strip cols {max(0, y_col-tick_length-2)}-{y_col}. Width {left_strip.shape[1]}")
print(f"Left profile stats: min={left_profile.min()}, max={left_profile.max()}, mean={left_profile.mean():.2f}")
print(f"Left profile peaks-above-mean rows: {np.where(left_profile > left_profile.mean() + 2)[0].tolist()[:30]}")

# Right strip (inward ticks) - the plot-side
right_strip = binary[:, y_col:min(binary.shape[1], y_col + tick_length + 2)]
right_profile = (right_strip > 0).sum(axis=1)
print(f"\nRight strip cols {y_col}-{y_col+tick_length+2}. Width {right_strip.shape[1]}")
print(f"Right profile stats: min={right_profile.min()}, max={right_profile.max()}, mean={right_profile.mean():.2f}")
print(f"Right profile peaks-above-mean rows: {np.where(right_profile > right_profile.mean() + 2)[0].tolist()[:30]}")

# Now the x-axis
x_row = int(round(frame.x_axis.pixel_position))
# Strip below (outward)
below = binary[x_row:min(binary.shape[0], x_row + tick_length + 2), :]
below_profile = (below > 0).sum(axis=0)
print(f"\nX-axis at row {x_row}. Below strip rows {x_row}-{x_row+tick_length+2}.")
print(f"Below profile max={below_profile.max()}, top-10 cols: {np.argsort(-below_profile)[:10].tolist()}")

# Strip above (inward)
above = binary[max(0, x_row - tick_length - 2):x_row + 1, :]
above_profile = (above > 0).sum(axis=0)
print(f"Above profile max={above_profile.max()}, top-10 cols: {np.argsort(-above_profile)[:10].tolist()}")
