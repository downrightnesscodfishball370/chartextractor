import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import cv2
import numpy as np
from chart_extractor import preprocess

IMG = r"C:\Users\User\Pictures\Relationship-between-the-valve-opening-and-the-flow-area-of-the-butterfly-valve.webp"
image = cv2.imread(IMG, cv2.IMREAD_COLOR)
g = preprocess.to_grayscale(image)
d = preprocess.edge_preserving_denoise(g)
f = preprocess.normalise_illumination(d)
b = preprocess.adaptive_binarise(f)

# Sample pixel values at known locations: on the curve (black ink) vs background (white paper).
# The valve curve passes roughly through (x=45°→ col~413, y=20%→ row~491).
# Origin-ish: col 100, row 600 (should be black ink, the curve near bottom-left).
print("Grayscale (original): high=bright/white bg, low=dark/ink")
print(f"  gray at (col=100, row=610) (curve-near-bottom):  g={g[610,100]}   d={d[610,100]}  flat={f[610,100]}  bin={b[610,100]}")
print(f"  gray at (col=400, row=300) (empty plot area):    g={g[300,400]}   d={d[300,400]}  flat={f[300,400]}  bin={b[300,400]}")
print(f"  gray at y-axis (col=68, row=300):                g={g[300,68]}    d={d[300,68]}   flat={f[300,68]}   bin={b[300,68]}")

# Where is the curve at a middle column?
col = 400
column = g[:, col]
dark_rows = np.where(column < 100)[0]
print(f"\nAt col={col}, very dark (ink) rows (g<100): {dark_rows.tolist()[:20]} ...")
print(f"At col={col}, binary foreground (bin>0) rows: {np.where(b[:,col]>0)[0].tolist()[:20]} ...")
