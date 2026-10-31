"""Reproduce the representative face grid: python -m src.figures."""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .data import load_orl, source_manifest
from .partition import DEFAULT_ROOT

def image_grid(images, labels, paths, output):
    subjects = np.unique(labels)
    # Evenly spaced identities and samples, selected without randomness.
    selected = subjects[np.linspace(0, len(subjects) - 1, min(8, len(subjects)), dtype=int)]
    cols = min(5, min(np.count_nonzero(labels == s) for s in selected))
    fig, axes = plt.subplots(len(selected), cols, figsize=(cols * 1.45, len(selected) * 1.65), squeeze=False)
    for row, subject in enumerate(selected):
        indices = np.flatnonzero(labels == subject)
        indices = indices[np.linspace(0, len(indices) - 1, cols, dtype=int)]
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"s{subject}/{paths[idx].name}", fontsize=9)
            ax.axis("off")
    fig.suptitle("ORL: variation within and between identities\nFixed display scale: 0–255", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "representative_faces.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1]/'figures')
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    if root == output or root in output.parents:
        raise ValueError('Output must be outside the original dataset')
    before = source_manifest(root)
    images, labels, paths = load_orl(root)
    output.mkdir(parents=True, exist_ok=True)
    image_grid(images, labels, paths, output)
    if before != source_manifest(root):
        raise RuntimeError('Original dataset changed during figure generation')


if __name__ == '__main__':
    main()
