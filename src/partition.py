"""Fixed identity-stratified partitions for all subsequent experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_orl
from .preprocessing import preprocess_images

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "dataset-faces"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results"
COUNTS = {"train": 6, "validation": 2, "test": 2}


def validate_indices(y, partitions):
    """Reject missing/repeated/out-of-bounds indices or incorrect class counts."""
    y = np.asarray(y)
    if y.ndim != 1 or not np.array_equal(np.unique(y), np.arange(1, 41)):
        raise ValueError("Expected identity labels 1–40")
    if not all(np.count_nonzero(y == label) == 10 for label in range(1, 41)):
        raise ValueError("Expected ten samples per identity")
    if set(partitions) != set(COUNTS):
        raise ValueError("Expected train, validation and test partitions")
    all_indices = []
    for name, count in COUNTS.items():
        indices = np.asarray(partitions[name])
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name}: indices must be a one-dimensional integer array")
        if len(indices) != 40 * count or np.any(indices < 0) or np.any(indices >= len(y)):
            raise ValueError(f"{name}: invalid size or index bounds")
        if not all(np.count_nonzero(y[indices] == label) == count for label in range(1, 41)):
            raise ValueError(f"{name}: incorrect per-identity counts")
        all_indices.extend(indices.tolist())
    if not np.array_equal(np.sort(all_indices), np.arange(len(y))):
        raise ValueError("Indices must be unique, disjoint and collectively exhaustive")


def make_split_indices(y, seed=42):
    """Shuffle each identity using one PCG64 stream, returning 6/2/2 indices.

    Iterate identities 1–40 in numeric order. For each identity, permute its
    indices in their input order; take the first six, next two, and last two.
    Return each partition grouped by identity, retaining shuffled sample order.
    Does not modify y or the global NumPy random state.
    """
    y = np.asarray(y)
    # Validate the input before allocating the RNG or partitioning.
    if y.ndim != 1 or not np.array_equal(np.unique(y), np.arange(1, 41)):
        raise ValueError("Expected identity labels 1–40")
    if any(np.count_nonzero(y == label) != 10 for label in range(1, 41)):
        raise ValueError("Expected ten samples per identity")
    rng = np.random.Generator(np.random.PCG64(seed))
    partitions = {name: [] for name in COUNTS}
    for label in range(1, 41):
        shuffled = rng.permutation(np.flatnonzero(y == label))
        offset = 0
        for name, count in COUNTS.items():
            partitions[name].extend(shuffled[offset:offset + count].tolist())
            offset += count
    partitions = {name: np.asarray(indices, dtype=np.int64) for name, indices in partitions.items()}
    validate_indices(y, partitions)
    return partitions


def _load_features(root):
    root = Path(root).expanduser().resolve()
    images, y, paths = load_orl(root)
    X = preprocess_images(images)
    relative_paths = np.asarray([p.relative_to(root).as_posix() for p in paths])
    return X, y, relative_paths


def _fingerprint(X, y, paths):
    """Bind saved indices to exact feature values, labels and sample ordering."""
    digest = hashlib.sha256()
    digest.update(np.asarray(X, dtype="<f4").tobytes())
    digest.update(np.asarray(y, dtype="<i8").tobytes())
    digest.update(json.dumps(paths.tolist(), separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def load_partitions(data_root=DEFAULT_ROOT, split_file=DEFAULT_OUTPUT / "split_indices.json", names=None):
    """Load and verify a saved split, returning dicts of X, y, paths, indices.

    All future models should reuse this saved definition. No fitting, centering
    or normalization beyond the fixed intensity scaling occurs here. names can
    restrict returned arrays, e.g. ('train', 'validation') for model selection.
    Full source fingerprint/split integrity checks still cover all samples.
    """
    saved = json.loads(Path(split_file).read_text(encoding="utf-8"))
    selected = tuple(COUNTS) if names is None else tuple(names)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(COUNTS):
        raise ValueError("names must select distinct train/validation/test partitions")
    if saved["schema_version"] != 1 or saved["rng"] != "numpy.random.Generator(PCG64)":
        raise ValueError("Unsupported split format or RNG")
    X, y, paths = _load_features(data_root)
    if saved["dataset_fingerprint_sha256"] != _fingerprint(X, y, paths):
        raise ValueError("Dataset features, labels or ordering differ from the saved split")
    partitions = {name: np.asarray(part["indices"]) for name, part in saved["partitions"].items()}
    validate_indices(y, partitions)
    expected = make_split_indices(y, saved["seed"])
    result = {}
    for name, indices in partitions.items():
        if not np.array_equal(indices, expected[name]):
            raise ValueError(f"{name}: indices disagree with saved seed")
        if saved["partitions"][name]["paths"] != paths[indices].tolist():
            raise ValueError(f"{name}: paths disagree with indices")
        if name in selected:
            result[name] = dict(X=X[indices], y=y[indices], paths=paths[indices], indices=indices)
    return result


def run(data_root=DEFAULT_ROOT, output_dir=DEFAULT_OUTPUT, seed=42):
    from .data import source_manifest

    root, output = Path(data_root).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("Output directory must be outside the original dataset")
    before = source_manifest(root)
    X, y, paths = _load_features(root)
    partitions = make_split_indices(y, seed)
    repeated = make_split_indices(y, seed)
    if any(not np.array_equal(indices, repeated[name]) for name, indices in partitions.items()):
        raise RuntimeError("Seed reproducibility failed")
    saved = dict(schema_version=1, seed=seed, rng="numpy.random.Generator(PCG64)",
                 numpy_version=np.__version__, dataset_fingerprint_sha256=_fingerprint(X, y, paths),
                 sample_order="numeric subject, then numeric image filename; zero-based global indices",
                 partitions={name: dict(indices=indices.tolist(), paths=paths[indices].tolist())
                             for name, indices in partitions.items()})
    output.mkdir(parents=True, exist_ok=True)
    split_file = output / "split_indices.json"
    split_file.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    loaded = load_partitions(root, split_file)
    for name, indices in partitions.items():
        if not (np.array_equal(loaded[name]["X"], X[indices])
                and np.array_equal(loaded[name]["y"], y[indices])
                and np.array_equal(loaded[name]["paths"], paths[indices])):
            raise RuntimeError(f"{name}: saved split alignment failed")
    if before != source_manifest(root):
        raise RuntimeError("Source dataset changed during partitioning")
    print(f"Seed {seed}: " + ", ".join(f"{name}={len(idx)}" for name, idx in partitions.items()))
    print(f"Verified disjoint, exhaustive 6/2/2 partitions; saved to {split_file}")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_root, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
