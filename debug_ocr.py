import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import cv2
from chart_extractor import axes, ocr, preprocess

IMG = r"C:\Users\User\Pictures\Relationship-between-the-valve-opening-and-the-flow-area-of-the-butterfly-valve.webp"
image = cv2.imread(IMG, cv2.IMREAD_COLOR)
stages = preprocess.preprocess(image)
gray = stages["gray"]
binary = stages["binary"]
frame = axes.locate_axes(binary)
print(f"x_axis row={frame.x_axis.pixel_position:.2f}   y_axis col={frame.y_axis.pixel_position:.2f}")

x_raw = ocr.read_x_axis_labels(gray, frame.x_axis.pixel_position, frame.y_axis.pixel_position)
y_raw = ocr.read_y_axis_labels(gray, frame.x_axis.pixel_position, frame.y_axis.pixel_position)
print(f"\nX-axis labels (center_x, center_y, parsed_value, conf):")
for r in x_raw:
    print(f"  {r}")
print(f"\nY-axis labels (center_x, center_y, parsed_value, conf):")
for r in y_raw:
    print(f"  {r}")

x_ticks, y_ticks = axes.detect_ticks(binary, frame)
print(f"\nX ticks (pixel cols): {[f'{t:.1f}' for t in x_ticks]}")
print(f"Y ticks (pixel rows): {[f'{t:.1f}' for t in y_ticks]}")
