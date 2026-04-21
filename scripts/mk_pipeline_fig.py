from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=1.5, facecolor="#f7f7fb")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def arr(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#333")
    ax.add_patch(a)


def main():
    out = Path("docs/img/pipeline.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 3.2))
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    box(ax, 0.3, 2.1, 2.2, 0.8, "Extracted OASIS discs\n(disc1/disc2/disc12)")
    box(ax, 3.0, 2.1, 1.9, 0.8, "Index\n`index.csv`")
    box(ax, 5.2, 2.1, 2.1, 0.8, "Manifest\n`manifest.csv`")
    box(ax, 7.7, 2.1, 2.0, 0.8, "Subject splits\n(train/val/test)")

    box(ax, 2.0, 0.4, 2.6, 0.9, "Tabular baseline\n(log-reg)")
    box(ax, 5.0, 0.4, 2.6, 0.9, "2D CNN baseline\n(processed MRI)")
    box(ax, 8.0, 0.4, 1.7, 0.9, "Reports\n(ROC/CM/errors)")

    arr(ax, 2.5, 2.5, 3.0, 2.5)
    arr(ax, 4.9, 2.5, 5.2, 2.5)
    arr(ax, 7.3, 2.5, 7.7, 2.5)

    arr(ax, 4.1, 2.1, 3.2, 1.3)
    arr(ax, 6.2, 2.1, 6.2, 1.3)
    arr(ax, 8.7, 2.1, 8.9, 1.3)

    arr(ax, 4.6, 0.85, 8.0, 0.85)
    arr(ax, 7.6, 0.85, 8.0, 0.85)

    plt.tight_layout()
    plt.savefig(out, dpi=200)


if __name__ == "__main__":
    main()

