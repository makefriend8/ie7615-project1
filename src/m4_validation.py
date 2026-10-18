"""Review the M3 detector on original validation portraits and a separate grid."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "m2_artifacts" / "source_manifest.csv"
MAPPING = ROOT / "m2_artifacts" / "identity_mapping.json"
SELECTION = ROOT / "m3_artifacts" / "selection.json"
CHECKPOINT = ROOT / "m3_artifacts" / "best.pt"


def _sha256(path):
    """Check that inference uses the validation-selected checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(source_dir, row):
    """Locate a manifest image in either the flat or nested identity folder."""
    relative = Path(row["relative_path"])
    paths = (source_dir / relative, source_dir / row["identity"] / relative)
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one source file for {relative}; found {matches}")
    return matches[0]


def _validation_images(source_dir):
    """Verify the original validation portraits against the M2 source manifest."""
    with MANIFEST.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "val"]
    if len(rows) != 50 or len({row["relative_path"] for row in rows}) != 50:
        raise ValueError("Expected 50 distinct original validation portraits")
    images = []
    for row in rows:
        path = _source_path(source_dir, row)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            size = rgb.size
        if (digest != row["pixel_sha256"] or
                size != (int(row["width"]), int(row["height"]))):
            raise ValueError(f"Source portrait differs from the manifest: {path}")
        images.append((row, path))
    return images


def _boxes(result, names):
    """Return detections in descending confidence order."""
    boxes = []
    for class_id, confidence, xyxy in zip(
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.xyxy.cpu().tolist()):
        boxes.append({
            "identity": names[int(class_id)],
            "confidence": round(float(confidence), 6),
            "xyxy": [round(float(value), 2) for value in xyxy],
        })
    return sorted(boxes, key=lambda box: -box["confidence"])


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _archive(directory):
    """Package the small review output for transfer back to the local workspace."""
    archive = directory.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                target.write(path, arcname=path.relative_to(directory.parent))
    return archive


def main():
    """Run the selected detector without changing any training or test artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "shared_pool")
    parser.add_argument("--grid", type=Path,
                        default=ROOT / "data" / "m3_external_samples" / "ta_week3_grid_01.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "m4_validation")
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or not 0 < args.nms_iou < 1:
        raise ValueError("Confidence and NMS IoU thresholds must lie between 0 and 1")

    source_dir, grid, output = args.source.resolve(), args.grid.resolve(), args.output.resolve()
    if not grid.is_file():
        raise FileNotFoundError(grid)
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError(f"Preserving existing review output: {output}")
    images = _validation_images(source_dir)
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    names = {int(index): str(identity) for index, identity in mapping.items()}
    if set(names) != set(range(13)) or len(set(names.values())) != 13:
        raise ValueError("Expected 13 distinct identities in the M2 mapping")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if _sha256(CHECKPOINT) != selection["checkpoint_sha256"]:
        raise ValueError("The checkpoint differs from the M3 selected checkpoint")

    from ultralytics import YOLO

    model = YOLO(str(CHECKPOINT))
    if {int(index): str(identity) for index, identity in model.names.items()} != names:
        raise ValueError("Checkpoint class names differ from the M2 identity mapping")
    image_size = int(selection["settings"]["imgsz"])
    predict_args = dict(imgsz=image_size, conf=args.confidence, iou=args.nms_iou,
                        device=args.device, verbose=False)

    output.mkdir(parents=True)
    annotated = output / "annotated_single"
    annotated.mkdir()
    per_image, all_predictions = [], {}
    for row, image_path in images:
        result = model.predict(source=str(image_path), **predict_args)[0]
        boxes = _boxes(result, names)
        predicted = boxes[0]["identity"] if boxes else ""
        status = ("no_detection" if not boxes else
                  "correct" if predicted == row["identity"] and len(boxes) == 1 else
                  "correct_with_extra_boxes" if predicted == row["identity"] else
                  "wrong_identity")
        key = row["relative_path"]
        per_image.append({
            "image": key, "true_identity": row["identity"],
            "top_identity": predicted,
            "top_confidence": boxes[0]["confidence"] if boxes else "",
            "detection_count": len(boxes), "status": status,
        })
        all_predictions[key] = boxes
        result.save(filename=str(annotated / f'{row["identity"]}_{row["filename"]}'))

    _write_csv(output / "single_predictions.csv", per_image,
               ["image", "true_identity", "top_identity", "top_confidence",
                "detection_count", "status"])
    (output / "single_boxes.json").write_text(
        json.dumps(all_predictions, indent=2) + "\n", encoding="utf-8")
    by_identity = defaultdict(Counter)
    for row in per_image:
        by_identity[row["true_identity"]][row["status"]] += 1
    counts = Counter(row["status"] for row in per_image)
    summary = {
        "split": "val", "source_images": len(per_image),
        "top1_identity_correct": counts["correct"] + counts["correct_with_extra_boxes"],
        "exactly_one_box_and_correct": counts["correct"],
        "no_detection": counts["no_detection"],
        "wrong_top_identity": counts["wrong_identity"],
        "correct_top_identity_with_extra_boxes": counts["correct_with_extra_boxes"],
        "per_identity": {identity: dict(by_identity[identity])
                         for identity in sorted(names.values(), key=int)},
        "confidence_threshold": args.confidence, "nms_iou": args.nms_iou,
        "image_size": image_size, "checkpoint_sha256": selection["checkpoint_sha256"],
        "note": "This is image-level identity review. Source portraits have no face-box ground truth, so detection mAP and box IoU are not reported.",
    }
    (output / "single_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    grid_result = model.predict(source=str(grid), **predict_args)[0]
    grid_boxes = _boxes(grid_result, names)
    (output / "grid_predictions.json").write_text(
        json.dumps({"image": grid.name, "detections": grid_boxes,
                    "note": "Predictions only; ground-truth identities for this grid were not supplied."},
                   indent=2) + "\n", encoding="utf-8")
    grid_result.save(filename=str(output / "grid_annotated.jpg"))
    archive = _archive(output)
    print(f"Validation portraits: {len(per_image)}")
    print(f"Top identity correct: {summary['top1_identity_correct']}/{len(per_image)}")
    print(f"Exactly one correct box: {summary['exactly_one_box_and_correct']}/{len(per_image)}")
    print(f"Week 3 grid detections: {len(grid_boxes)}")
    print(f"Review archive: {archive}")


if __name__ == "__main__":
    main()
