"""Prepare and validate image data using repository-relative default paths."""
from pathlib import Path
import hashlib
import json
import random
import shutil

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _sources(raw_dir):
    """Read identity labels and resolve the image directory."""
    raw = Path(raw_dir)
    labels = pd.read_csv(raw / "identity_CelebA.txt", sep=r"\s+", names=["image", "identity"])
    images = raw / "img_align_celeba"
    if (images / "img_align_celeba").is_dir():
        images /= "img_align_celeba"
    return labels, images


def _identity_files(labels, images, identity):
    """Return the 23-25 distinct, existing image paths for one identity."""
    names = labels.loc[labels.identity == int(identity), "image"].tolist()
    if not 23 <= len(names) <= 25 or len(set(names)) != len(names):
        raise ValueError(f"ID {identity}: expected 23-25 distinct filenames, found {len(names)}")
    paths = [images / name for name in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    return paths


def _digest(path):
    """Compute an RGB pixel hash for exact duplicate-image checks."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def plot_candidates(raw_dir=ROOT / "data/raw", grid_output=ROOT / "visual_diversity_candidates.png"):
    """Render a seeded sample of identities with 23-25 images."""
    import matplotlib.pyplot as plt
    labels, images = _sources(raw_dir)
    counts = labels.identity.value_counts()
    eligible = counts[counts.between(23, 25)].index.tolist()
    candidates = random.Random(42).sample(eligible, min(20, len(eligible)))
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for axis in axes.flat:
        axis.axis("off")
    for axis, identity in zip(axes.flat, candidates):
        paths = _identity_files(labels, images, identity)
        with Image.open(paths[0]) as img:
            axis.imshow(img.copy())
        axis.set_title(f"ID {identity} (N={counts[identity]})")
    fig.tight_layout()
    fig.savefig(grid_output)
    plt.close(fig)
    return Path(grid_output)


def extract_claimed_id(claimed_id, raw_dir=ROOT / "data/raw"):
    """Export all verified images for one identity to TA_Share_Pool."""
    labels, images = _sources(raw_dir)
    paths = _identity_files(labels, images, claimed_id)
    destination = ROOT / "TA_Share_Pool" / str(int(claimed_id))
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)
    return destination


def audit_subset(data_dir=ROOT / "data/selected_subset"):
    """Validate class consistency, readable images and cross-split exact duplicates."""
    root = Path(data_dir)
    rows, expected, filenames, hashes = [], None, {}, {}
    for split in SPLITS:
        folder = root / split
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if not 4 <= len(classes) <= 6 or (expected is not None and classes != expected):
            raise ValueError(f"Inconsistent or invalid class directories in {folder}")
        expected = classes
        for index, identity in enumerate(classes):
            paths = sorted(p for p in (folder / identity).iterdir()
                           if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f"Empty class: {split}/{identity}")
            for path in paths:
                digest = _digest(path)
                for value, seen in ((path.name, filenames), (digest, hashes)):
                    if value in seen and seen[value] != (split, identity):
                        raise ValueError(f"Duplicate across splits or identities: {path}")
                    seen[value] = (split, identity)
                rows.append(dict(split=split, identity=identity, class_index=index,
                                 filename=path.name, relative_path=path.relative_to(root).as_posix(),
                                 pixel_sha256=digest))
    frame = pd.DataFrame(rows)
    counts = frame.groupby("identity").size()
    if not counts.between(23, 25).all():
        raise ValueError(f"Expected 23-25 images per identity: {counts.to_dict()}")
    return frame, expected


def prepare_pipeline_subset(raw_dir=ROOT / "data/raw", output_dir=ROOT / "data/selected_subset",
                            selected_ids=None, seed=42):
    """Create a new dataset split and preserve existing output directories.

    Filename order and the seed determine the reproducible two-stage split. With 25 images the realized counts are 17/4/4.
    Identity selection is documented in the data preparation notebook.
    """
    ids = [str(int(i)) for i in (selected_ids or [])]
    if not 4 <= len(ids) <= 6 or len(set(ids)) != len(ids):
        raise ValueError("Choose 4-6 distinct identities")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing dataset: {output}. Audit it or choose a new output_dir.")
    labels, images = _sources(raw_dir)
    plan, hashes = [], {}
    for identity in ids:
        paths = _identity_files(labels, images, identity)
        train, remainder = train_test_split(paths, test_size=0.3, random_state=seed)
        val, test = train_test_split(remainder, test_size=0.5, random_state=seed)
        for split, files in zip(SPLITS, (train, val, test)):
            for path in files:
                digest = _digest(path)
                if digest in hashes and hashes[digest] != (split, identity):
                    raise ValueError(f"Duplicate pixels across splits/identities: {path}")
                hashes[digest] = (split, identity)
                plan.append((path, split, identity))
    # All inputs are checked before the first dataset write.
    output.mkdir(parents=True)
    for source, split, identity in plan:
        destination = output / split / identity
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)
    frame, classes = audit_subset(output)
    frame.to_csv(output / "split_manifest.csv", index=False)
    (output / "split_config.json").write_text(json.dumps(dict(
        seed=seed, selected_ids=ids, class_names=classes,
        target_split_ratios=[0.7, 0.15, 0.15]), indent=2), encoding="utf-8")
    return frame
