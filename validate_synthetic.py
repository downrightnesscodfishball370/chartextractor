"""Generate synthetic single-line graphs with known ground-truth curves,
run the pipeline, and report per-point error statistics.

Tests three shapes:
  - Quadratic:    y = 0.01 * x^2         over x in [0, 100]
  - Exponential:  y = 10 * (1 - e^(-x/20)) over x in [0, 100]
  - Sine:         y = 50 + 40 * sin(x/15) over x in [0, 100]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chart_extractor.pipeline import extract_curve


def make_chart(x_gt: np.ndarray, y_gt: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=140)
    ax.plot(x_gt, y_gt, "k-", linewidth=2.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle=":", color="0.7")
    ax.set_xlim(x_gt.min(), x_gt.max())
    ax.set_ylim(min(0, y_gt.min()), max(y_gt.max() * 1.05, y_gt.max() + 1))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def evaluate(name: str, fn) -> None:
    out_dir = Path("debug_out") / "validate"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = out_dir / f"{name}.png"
    overlay_path = out_dir / f"{name}_overlay.png"

    x_gt = np.linspace(0, 100, 400)
    y_gt = fn(x_gt)
    make_chart(x_gt, y_gt, name, chart_path)

    img = cv2.imread(str(chart_path), cv2.IMREAD_COLOR)
    result = extract_curve(img, n_output_points=200)
    x_ext = np.asarray(result.points.x)
    y_ext = np.asarray(result.points.y)
    if x_ext.size == 0:
        print(f"[{name}] extraction returned zero points")
        return
    y_true = fn(np.clip(x_ext, x_gt.min(), x_gt.max()))

    # Mask to the ground-truth input domain.
    mask = (x_ext >= x_gt.min()) & (x_ext <= x_gt.max())
    x_ext, y_ext, y_true = x_ext[mask], y_ext[mask], y_true[mask]

    # y error normalised by the chart's y range.
    y_range = float(y_gt.max() - y_gt.min())
    err = np.abs(y_ext - y_true) / y_range
    print(
        f"[{name}] n={len(y_ext):3d} "
        f"x=[{x_ext.min():.2f},{x_ext.max():.2f}]  "
        f"y-err: mean={err.mean()*100:.3f}%, median={np.median(err)*100:.3f}%, "
        f"p95={np.percentile(err, 95)*100:.3f}%, max={err.max()*100:.3f}% "
        f"(of y-range {y_range:.2f})"
    )

    # Save overlay for visual check.
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=140)
    ax.plot(x_gt, y_gt, "k-", linewidth=2.5, label="ground truth")
    ax.plot(x_ext, y_ext, "r--", linewidth=1.2, label="extracted")
    ax.legend()
    ax.grid(True, linestyle=":", color="0.7")
    fig.savefig(overlay_path)
    plt.close(fig)


if __name__ == "__main__":
    evaluate("quadratic", lambda x: 0.01 * x**2)
    evaluate("exponential", lambda x: 10 * (1 - np.exp(-x / 20)))
    evaluate("sine", lambda x: 50 + 40 * np.sin(x / 15))
