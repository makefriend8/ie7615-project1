"""Build a reproducible multi-identity detection dataset with YOLO labels."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .m2_sources import SPLITS, inventory_shared_portraits, write_inventory


SIZE = 640
FRAME_COUNTS = {"train": 156, "val": 39, "test": 39}


def _background(path, size, rng):
    """Use a blurred shared-source photograph as one coherent backdrop."""
    with Image.open(path) as source:
        image = ImageOps.fit(source.convert("RGB"), (size, size))
    image = image.filter(ImageFilter.GaussianBlur(radius=48))
    tint = Image.new("RGB", (size, size),
                     tuple(rng.randint(170, 215) for _ in range(3)))
    return Image.blend(image, tint, .45)


def _source_crop(path, rng):
    """Crop the aligned portrait around the head and apply recorded appearance changes."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    # CelebA aligned portraits center the head; the crop keeps hair and the face.
    image = image.crop((round(width * .08), round(height * .03),
                        round(width * .92), round(height * .96)))
    flip = rng.random() < .5
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    brightness = rng.uniform(.85, 1.15)
    contrast = rng.uniform(.90, 1.10)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image, dict(horizontal_flip=flip, brightness=round(brightness, 4),
                       contrast=round(contrast, 4))


def _layout(count, rng, size, grid=False):
    """Choose non-overlapping tile layouts, including 3x3 for nine portraits."""
    if grid:
        columns, rows = 3, 3
    else:
        columns, rows = rng.choice({
            2: [(2, 1)],
            3: [(3, 1), (2, 2)],
            4: [(2, 2)],
            5: [(3, 2), (2, 3)],
        }[count])
    # Unequal row and column widths vary the scale and spacing of the portraits.
    column_weights = [rng.uniform(.82, 1.18) for _ in range(columns)]
    row_weights = [rng.uniform(.82, 1.18) for _ in range(rows)]
    column_edges = [0.0]
    row_edges = [0.0]
    for weight in column_weights:
        column_edges.append(column_edges[-1] + size * weight / sum(column_weights))
    for weight in row_weights:
        row_edges.append(row_edges[-1] + size * weight / sum(row_weights))
    cells = [(col, row) for row in range(rows) for col in range(columns)]
    rng.shuffle(cells)
    return [(column_edges[col], row_edges[row],
             column_edges[col + 1] - column_edges[col],
             row_edges[row + 1] - row_edges[row])
            for col, row in cells[:count]]


def _render(rows, source_dir, rng, size=SIZE, grid=False):
    """Compose portraits on a shared backdrop and return their pixel boxes."""
    canvas = _background(source_dir / rows[0]["relative_path"], size, rng)
    layout = _layout(len(rows), rng, size, grid=grid)
    objects = []
    for row, (cx, cy, cell_w, cell_h) in zip(rows, layout):
        portrait, augmentation = _source_crop(source_dir / row["relative_path"], rng)
        scale = rng.uniform(.78, .98)
        factor = min(cell_w * scale / portrait.width, cell_h * scale / portrait.height)
        width, height = max(1, round(portrait.width * factor)), max(1, round(portrait.height * factor))
        portrait = portrait.resize((width, height), Image.Resampling.LANCZOS)
        margin_x = max(0, round((cell_w - width) / 2))
        margin_y = max(0, round((cell_h - height) / 2))
        x = round(cx + margin_x + rng.uniform(-.70, .70) * margin_x)
        y = round(cy + margin_y + rng.uniform(-.70, .70) * margin_y)
        if not (0 <= x < x + width <= size and 0 <= y < y + height <= size):
            raise ValueError("Portrait placement exceeds the frame")
        canvas.paste(portrait, (x, y))
        objects.append(dict(source=row, box=(x, y, x + width, y + height),
                            augmentation=augmentation))
    return canvas, objects


def _yolo_line(class_index, box, size=SIZE):
    """Convert a placed portrait's pixel bounds into a normalized YOLO row."""
    x1, y1, x2, y2 = box
    values = ((x1 + x2) / (2 * size), (y1 + y2) / (2 * size),
              (x2 - x1) / size, (y2 - y1) / size)
    if not (0 <= x1 < x2 <= size and 0 <= y1 < y2 <= size and
            all(0 < value <= 1 for value in values)):
        raise ValueError(f"Invalid bounding box: {box}")
    return str(class_index) + " " + " ".join(f"{value:.6f}" for value in values)


def _annotated(image, objects):
    """Draw generated portrait boxes and identity IDs for visual inspection."""
    image = image.copy()
    pen = ImageDraw.Draw(image)
    for item in objects:
        box = item["box"]
        label = item["source"]["identity"]
        pen.rectangle(box, outline="#17c653", width=3)
        pen.text((box[0] + 3, max(0, box[1] - 13)), label, fill="#087b2f",
                 stroke_width=1, stroke_fill="white")
    return image


def _write_frame(image, objects, output, split, stem, size=SIZE):
    image_path = output / "images" / split / (stem + ".jpg")
    label_path = output / "labels" / split / (stem + ".txt")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, quality=92)
    label_path.write_text("\n".join(_yolo_line(item["source"]["class_index"], item["box"], size)
                                     for item in objects) + "\n", encoding="utf-8")
    return image_path


def build_detection_dataset(source_dir, m1_manifest, output_dir, seed=42,
                            frame_counts=None):
    """Generate split-safe YOLO data, examples and four additional 3x3 grids."""
    source_dir = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    rows, identities = inventory_shared_portraits(source_dir, m1_manifest, seed)
    counts = dict(FRAME_COUNTS if frame_counts is None else frame_counts)
    if set(counts) != set(SPLITS) or any(not isinstance(v, int) or v < 13 for v in counts.values()):
        raise ValueError("Each train/val/test frame count must be at least 13")
    by_split = {split: defaultdict(list) for split in SPLITS}
    for row in rows:
        by_split[row["split"]][row["identity"]].append(row)
    output.mkdir(parents=True, exist_ok=False)
    write_inventory(rows, output / "source_manifest.csv")
    (output / "identity_mapping.json").write_text(json.dumps(
        {str(index): identity for index, identity in enumerate(identities)}, indent=2),
        encoding="utf-8")
    object_rows, coverage = [], {}
    for split in SPLITS:
        rng = random.Random(f"{seed}:{split}")
        order = identities.copy()
        rng.shuffle(order)
        pools = {identity: by_split[split][identity][:] for identity in identities}
        for pool in pools.values():
            rng.shuffle(pool)
        used = Counter()
        cursor = 0
        for frame_index in range(counts[split]):
            # Include dense nine-person frames alongside smaller compositions.
            count = (2, 3, 4, 5, 9)[frame_index % 5]
            chosen = [order[(cursor + j) % len(order)] for j in range(count)]
            cursor += count
            chosen_rows = []
            for identity in chosen:
                pool = pools[identity]
                chosen_rows.append(pool[used[identity] % len(pool)])
                used[identity] += 1
            image, objects = _render(chosen_rows, source_dir, rng, grid=(count == 9))
            stem = f"{split}_{frame_index:04d}"
            _write_frame(image, objects, output, split, stem)
            if frame_index in (0, 1, 2, 4):
                example = output / "examples" / f"{stem}_annotated.png"
                example.parent.mkdir(exist_ok=True)
                _annotated(image, objects).save(example)
            for item in objects:
                source = item["source"]
                object_rows.append(dict(frame=f"images/{split}/{stem}.jpg", split=split,
                                        identity=source["identity"], class_index=source["class_index"],
                                        source_relative_path=source["relative_path"],
                                        pixel_sha256=source["pixel_sha256"],
                                        x1=item["box"][0], y1=item["box"][1],
                                        x2=item["box"][2], y2=item["box"][3],
                                        **item["augmentation"]))
        coverage[split] = dict(used)
    # Save four grids without boxes for the TA, separate from annotated examples.
    rng = random.Random(f"{seed}:ta_grids")
    grid_source = by_split["test"]
    grid_pools = {identity: grid_source[identity][:] for identity in identities}
    for pool in grid_pools.values():
        rng.shuffle(pool)
    grid_used = Counter()
    for index in range(4):
        chosen = [identities[(index * 9 + j) % len(identities)] for j in range(9)]
        chosen_rows = []
        for identity in chosen:
            pool = grid_pools[identity]
            if grid_used[identity] >= len(pool):
                raise ValueError(f"Not enough test portraits for TA grids: {identity}")
            chosen_rows.append(pool[grid_used[identity]])
            grid_used[identity] += 1
        image, objects = _render(chosen_rows, source_dir, rng, grid=True)
        target = output / "ta_grids" / f"grid_{index + 1:02d}.png"
        target.parent.mkdir(exist_ok=True)
        image.save(target)
        preview = output / "ta_grids_annotated" / target.name
        preview.parent.mkdir(exist_ok=True)
        _annotated(image, objects).save(preview)
        (target.with_suffix(".json")).write_text(json.dumps([
            dict(identity=item["source"]["identity"],
                 source=item["source"]["relative_path"], box=item["box"])
            for item in objects], indent=2), encoding="utf-8")
    with (output / "object_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(object_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(object_rows)
    config = dict(seed=seed, image_size=SIZE, frame_counts=counts,
                  people_per_frame=[2, 3, 4, 5, 9], identities=identities,
                  source_images=len(rows), object_counts=coverage,
                  augmentation=dict(horizontal_flip_probability=.5, brightness=[.85, 1.15],
                                    contrast=[.90, 1.10], placement_scale=[.78, .98]),
                  annotation="Placed portrait bounds in normalized YOLO xywh format")
    (output / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data_yaml = (f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\n"
                 f"test: images/test\nnames:\n" + "".join(
                     f"  {i}: '{identity}'\n" for i, identity in enumerate(identities)))
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return output, config


def audit_detection_dataset(output_dir):
    """Check label pairing, normalized boxes, identity coverage and source isolation."""
    output = Path(output_dir)
    mapping = json.loads((output / "identity_mapping.json").read_text(encoding="utf-8"))
    config = json.loads((output / "generation_config.json").read_text(encoding="utf-8"))
    source_splits = {}
    with (output / "source_manifest.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["pixel_sha256"]
            if key in source_splits and source_splits[key] != row["split"]:
                raise ValueError("Source image appears in multiple splits")
            source_splits[key] = row["split"]
    recorded = defaultdict(list)
    with (output / "object_manifest.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if source_splits.get(row["pixel_sha256"]) != row["split"]:
                raise ValueError(f"Source split mismatch: {row['frame']}")
            if mapping.get(row["class_index"]) != row["identity"]:
                raise ValueError(f"Identity mapping mismatch: {row['frame']}")
            recorded[row["frame"]].append(row)
    summary = {}
    for split in SPLITS:
        images = sorted((output / "images" / split).glob("*.jpg"))
        labels = sorted((output / "labels" / split).glob("*.txt"))
        if len(images) != config["frame_counts"][split] or {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError(f"Image/label pairing mismatch in {split}")
        classes = Counter()
        for image in images:
            key = f"images/{split}/{image.name}"
            lines = (output / "labels" / split / (image.stem + ".txt")).read_text(
                encoding="utf-8").splitlines()
            objects = recorded[key]
            if len(lines) != len(objects) or len(lines) not in (2, 3, 4, 5, 9):
                raise ValueError(f"Object count mismatch: {key}")
            with Image.open(image) as im:
                if im.size != (SIZE, SIZE):
                    raise ValueError(f"Incorrect image size: {key}")
            if len({item["identity"] for item in objects}) < 2:
                raise ValueError(f"Image lacks two distinct identities: {key}")
            for line, item in zip(lines, objects):
                fields = line.split()
                if len(fields) != 5 or fields[0] != item["class_index"]:
                    raise ValueError(f"Invalid YOLO row: {key}")
                x, y, w, h = map(float, fields[1:])
                if not (0 < w <= 1 and 0 < h <= 1 and
                        0 <= x-w/2 < x+w/2 <= 1 and 0 <= y-h/2 < y+h/2 <= 1):
                    raise ValueError(f"Out-of-frame YOLO box: {key}")
                classes[item["identity"]] += 1
        if set(classes) != set(mapping.values()):
            raise ValueError(f"Missing identity in {split}")
        summary[split] = dict(frames=len(images), objects=sum(classes.values()),
                              per_identity=dict(sorted(classes.items(), key=lambda kv: int(kv[0]))))
    grids = sorted((output / "ta_grids").glob("grid_*.png"))
    if len(grids) != 4 or len({image.read_bytes() for image in grids}) != 4:
        raise ValueError("Expected four distinct TA grids")
    previews = sorted((output / "ta_grids_annotated").glob("grid_*.png"))
    if {grid.name for grid in previews} != {grid.name for grid in grids}:
        raise ValueError("Missing annotated TA grid previews")
    for grid in grids:
        with Image.open(grid) as image:
            if image.size != (SIZE, SIZE):
                raise ValueError(f"Incorrect TA grid size: {grid}")
        if len(json.loads(grid.with_suffix(".json").read_text(encoding="utf-8"))) != 9:
            raise ValueError(f"TA grid must contain nine portraits: {grid}")
    return summary


def main():
    """Rebuild or audit the detection dataset from a command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Directory with one folder per CelebA identity")
    parser.add_argument("--m1-manifest", required=True, type=Path,
                        help="Saved split_manifest.csv from the M1 experiment")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.exists():
        build_detection_dataset(args.source, args.m1_manifest, args.output)
    print(json.dumps(audit_detection_dataset(args.output), indent=2))


if __name__ == "__main__":
    main()
