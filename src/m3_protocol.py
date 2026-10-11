"""Shared dataset and evaluation helpers for Milestone 3 detection."""

from collections import Counter
import json
import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "m2_detection"
ARTIFACTS = ROOT / "m3_artifacts"
RUNS = ROOT / "runs" / "m3"
SPLIT_COUNTS = {"train": 156, "val": 39, "test": 39}


def validate_dataset():
    """Check the committed M2 images, labels, and 13-class mapping before training."""
    mapping_path = ROOT / "m2_artifacts" / "identity_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in mapping.items()}
    if set(names) != set(range(13)) or len(set(names.values())) != 13:
        raise ValueError("Expected the 13-class M2 identity mapping")

    for split, count in SPLIT_COUNTS.items():
        images = sorted((DATASET / "images" / split).glob("*.jpg"))
        labels = sorted((DATASET / "labels" / split).glob("*.txt"))
        if len(images) != len(labels) or len(images) != count:
            raise ValueError(f"Incorrect image/label count in {split}")
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"Image/label filenames differ in {split}")
        classes = Counter()
        for image_path in images:
            with Image.open(image_path) as image:
                if image.size != (640, 640):
                    raise ValueError(f"Expected a 640 x 640 image: {image_path}")
            label_path = DATASET / "labels" / split / f"{image_path.stem}.txt"
            rows = label_path.read_text(encoding="utf-8").splitlines()
            if not rows:
                raise ValueError(f"Empty YOLO label: {label_path}")
            for row in rows:
                fields = row.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO row: {label_path}")
                class_id = int(fields[0])
                x, y, width, height = map(float, fields[1:])
                if (class_id not in names or
                        not all(math.isfinite(value) for value in (x, y, width, height)) or
                        width <= 0 or height <= 0 or
                        x - width / 2 < 0 or x + width / 2 > 1 or
                        y - height / 2 < 0 or y + height / 2 > 1):
                    raise ValueError(f"Invalid YOLO box: {label_path}")
                classes[class_id] += 1
        if set(classes) != set(names):
            raise ValueError(f"Some identities are absent from {split}")
    return names


def write_data_yaml(destination):
    """Write a portable YOLO dataset file with a path for this checkout."""
    names = validate_dataset()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = [f"path: {json.dumps(str(DATASET.resolve()))}",
               "train: images/train", "val: images/val", "test: images/test", "names:"]
    content.extend(f"  {index}: {json.dumps(names[index])}" for index in sorted(names))
    destination.write_text("\n".join(content) + "\n", encoding="utf-8")
    return destination


def read_label_boxes(path, image_size=(640, 640)):
    """Convert normalized YOLO rows to pixel xyxy boxes and identity indices."""
    width, height = image_size
    boxes = []
    for row in Path(path).read_text(encoding="utf-8").splitlines():
        class_id, x, y, w, h = map(float, row.split())
        boxes.append((int(class_id), ((x - w / 2) * width,
                                      (y - h / 2) * height,
                                      (x + w / 2) * width,
                                      (y + h / 2) * height)))
    return boxes


def box_iou(a, b):
    """Compute intersection over union of two xyxy pixel boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_detections(truth, predictions, threshold=0.5):
    """Match predictions by confidence, class, and IoU for localization review.

    Each prediction is (class_id, confidence, xyxy). A ground-truth box can
    match only once. This fixed-threshold review is separate from mAP.
    """
    unmatched = set(range(len(truth)))
    matched = []
    extra = []
    for class_id, confidence, box in sorted(predictions, key=lambda item: -item[1]):
        candidates = [(box_iou(box, truth[index][1]), index)
                      for index in unmatched if truth[index][0] == class_id]
        overlap, index = max(candidates, default=(0.0, None))
        if index is not None and overlap >= threshold:
            unmatched.remove(index)
            matched.append((class_id, overlap, confidence))
        else:
            extra.append(class_id)
    missed = [truth[index][0] for index in sorted(unmatched)]
    return {"matched": matched, "missed": missed, "extra": extra,
            "ground_truth_count": len(truth), "prediction_count": len(predictions)}
