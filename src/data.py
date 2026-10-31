"""Read subject folders in numeric order without modifying source files."""

import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image


def load_orl(root):
    """Return uint8 images (N,H,W), integer subject IDs (N,), and source paths.

    Labels preserve folder IDs (s1 -> 1). Samples are ordered by subject and
    numeric filename, so 2.pgm precedes 10.pgm. Unexpected files inside subject
    folders fail explicitly; metadata files at the root are not image samples.
    No expected class count or resolution is assumed during discovery.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    folders = []
    for path in root.iterdir():
        if path.is_dir():
            if not re.fullmatch(r"s[1-9]\d*", path.name):
                raise ValueError(f"Unexpected dataset subdirectory: {path}")
            folders.append(path)
    folders.sort(key=lambda p: int(p.name[1:]))
    images, labels, paths = [], [], []
    expected_shape = None
    for folder in folders:
        files = list(folder.iterdir())
        if not files:
            raise ValueError(f"Empty subject folder: {folder}")
        for path in files:
            if not path.is_file() or not re.fullmatch(r"[1-9]\d*\.pgm", path.name):
                raise ValueError(f"Unexpected subject entry: {path}")
        for path in sorted(files, key=lambda p: int(p.stem)):
            with Image.open(path) as image:
                # Pillow identifies PGM as part of its PPM format family.
                if image.format != "PPM" or image.mode != "L":
                    raise ValueError(f"Expected 8-bit grayscale PGM: {path}")
                pixels = np.array(image)
            if pixels.ndim != 2 or pixels.dtype != np.uint8:
                raise ValueError(f"Unexpected pixel representation: {path}")
            if expected_shape is None:
                expected_shape = pixels.shape
            if pixels.shape != expected_shape:
                raise ValueError(f"Inconsistent image dimensions: {path}")
            images.append(pixels)
            labels.append(int(folder.name[1:]))
            paths.append(path)
    if not images:
        raise ValueError(f"No subject images found in {root}")
    return np.stack(images), np.asarray(labels, dtype=np.int64), tuple(paths)


def source_manifest(root):
    """Hash every source file, including metadata, to verify read-only execution."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }

