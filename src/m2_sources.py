"""Inventory shared CelebA portraits and assign source images to detection splits."""

from collections import Counter
import csv
import hashlib
from pathlib import Path
from PIL import Image


IDENTITY_IDS = (3, 7, 797, 1212, 2336, 2619, 2970, 4422, 4428, 5695, 7007, 8335, 10002)
SPLITS = ("train", "val", "test")


def _pixel_hash(path):
    """Hash decoded RGB pixels to find exact duplicates across identities."""
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(), rgb.size


def _m1_assignments(path):
    """Preserve the existing M1 split for its six identities."""
    assignments = {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = (row["identity"], row["filename"])
            if key in assignments or row["split"] not in SPLITS:
                raise ValueError("Invalid M1 split manifest")
            assignments[key] = row["split"]
    return assignments


def _new_split(paths, identity, seed):
    """Assign 70/15/15 source proportions in a repeatable order."""
    ordered = sorted(paths, key=lambda p: hashlib.sha256(
        f"{seed}:{identity}:{p.name}".encode()).hexdigest())
    n = len(ordered)
    train_end = int(n * 0.70)
    val_end = train_end + round(n * 0.15)
    return {p.name: ("train" if i < train_end else "val" if i < val_end else "test")
            for i, p in enumerate(ordered)}


def inventory_shared_portraits(source_dir, m1_manifest, seed=42):
    """Return audited source rows and one fixed identity-to-class mapping.

    The source directory may contain one extra nested directory per identity.
    Jupyter checkpoint folders and temporary sync files are ignored; only decoded
    JPG images enter the inventory.
    """
    source_dir = Path(source_dir).resolve()
    ids = [str(value) for value in IDENTITY_IDS]
    # Jupyter may create .ipynb_checkpoints beside the identity folders.
    actual = sorted(p.name for p in source_dir.iterdir()
                    if p.is_dir() and not p.name.startswith("."))
    if set(actual) != set(ids):
        raise ValueError(f"Expected shared identities {ids}; found {actual}")
    m1 = _m1_assignments(m1_manifest)
    rows, hashes, seen_names = [], {}, set()
    for class_index, identity in enumerate(ids):
        folder = source_dir / identity
        paths = sorted(p for p in folder.rglob("*.jpg") if p.is_file())
        if not 23 <= len(paths) <= 25:
            raise ValueError(f"ID {identity}: expected 23-25 JPG images, found {len(paths)}")
        new_assignments = _new_split(paths, identity, seed)
        for path in paths:
            key = (identity, path.name)
            if key in seen_names:
                raise ValueError(f"Duplicate filename within identity: {key}")
            seen_names.add(key)
            pixel_hash, size = _pixel_hash(path)
            if pixel_hash in hashes:
                raise ValueError(f"Duplicate decoded image: {path} and {hashes[pixel_hash]}")
            hashes[pixel_hash] = path
            split = m1.get(key, new_assignments[path.name])
            rows.append(dict(identity=identity, class_index=class_index,
                             filename=path.name, relative_path=path.relative_to(source_dir).as_posix(),
                             split=split, width=size[0], height=size[1], pixel_sha256=pixel_hash))
        inherited = {key[1] for key in m1 if key[0] == identity}
        if inherited and inherited != {path.name for path in paths}:
            raise ValueError(f"M1 source filenames changed for identity {identity}")
        counts = Counter(row["split"] for row in rows if row["identity"] == identity)
        if set(counts) != set(SPLITS):
            raise ValueError(f"Missing split for identity {identity}")
    return rows, ids


def write_inventory(rows, destination):
    """Write the audited source split in a stable CSV column order."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = ("identity", "class_index", "filename", "relative_path", "split",
               "width", "height", "pixel_sha256")
    with destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination
