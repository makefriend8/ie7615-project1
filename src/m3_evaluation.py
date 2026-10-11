"""Evaluate the selected YOLOv8 detector and create a predicted-box gallery."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from PIL import Image

from .m3_protocol import (ARTIFACTS, DATASET, ROOT, RUNS, match_detections,
                          read_label_boxes, write_data_yaml)


def _selected_model():
    """Load the validation-selected weights and verify their saved digest."""
    from ultralytics import YOLO

    selection = json.loads((ARTIFACTS / "selection.json").read_text(encoding="utf-8"))
    checkpoint = ARTIFACTS / "best.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint differs from selection.json")
    return YOLO(str(checkpoint)), selection


def _predicted_boxes(result):
    """Read predicted class, confidence, and pixel box from one YOLO result."""
    return [(int(class_id), float(confidence), [float(value) for value in box])
            for class_id, confidence, box in zip(
                result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                result.boxes.xyxy.cpu().tolist())]


def evaluate(device="0", confidence=0.25, nms_iou=0.7):
    """Evaluate held-out test images once, after validation-based selection."""
    ARTIFACTS.mkdir(exist_ok=True)
    output = ARTIFACTS / "test_metrics.json"
    if output.exists():
        raise FileExistsError(f"Test metrics already exist: {output}")
    validation_dir = RUNS / "selected_test"
    if validation_dir.exists():
        raise FileExistsError(f"Test validation output already exists: {validation_dir}")
    data_yaml = write_data_yaml(RUNS / "data.yaml")
    model, selection = _selected_model()
    image_size = int(selection["settings"]["imgsz"])

    # Ultralytics calculates AP, precision, and recall over the labeled test split.
    metrics = model.val(data=str(data_yaml), split="test", imgsz=image_size,
                        device=device, plots=True, project=str(RUNS),
                        name="selected_test", exist_ok=False)
    box = metrics.box
    names = {int(key): str(value) for key, value in model.names.items()}
    per_class = {}
    for position, class_id in enumerate(box.ap_class_index):
        precision, recall, ap50, ap = box.class_result(position)
        per_class[int(class_id)] = {
            "identity": names[int(class_id)],
            "precision": float(precision), "recall": float(recall),
            "map50": float(ap50), "map50_95": float(ap),
        }
    if set(per_class) != set(names):
        raise ValueError("Test metrics are missing one or more of the 13 identities")

    # Report matched-box IoU separately from AP at the fixed prediction threshold.
    per_image = []
    per_class_ious = defaultdict(list)
    all_ious = []
    predictions_by_image = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for image_path in sorted((DATASET / "images" / "test").glob("*.jpg")):
        with Image.open(image_path) as image:
            size = image.size
        truth = read_label_boxes(
            DATASET / "labels" / "test" / f"{image_path.stem}.txt", size)
        result = model.predict(source=str(image_path), imgsz=image_size,
                               conf=confidence, iou=nms_iou, device=device,
                               verbose=False)[0]
        predictions = _predicted_boxes(result)
        match = match_detections(truth, predictions, threshold=0.5)
        ious = [item[1] for item in match["matched"]]
        for class_id, overlap, _ in match["matched"]:
            per_class_ious[class_id].append(overlap)
        all_ious.extend(ious)
        tp, fp, fn = len(match["matched"]), len(match["extra"]), len(match["missed"])
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        row = {
            "image": image_path.name, "ground_truth": len(truth),
            "predictions": len(predictions), "tp": tp, "fp": fp, "fn": fn,
            "mean_matched_iou": mean(ious) if ious else None,
            "missed_identities": [names[index] for index in match["missed"]],
            "extra_predicted_identities": [names[index] for index in match["extra"]],
        }
        per_image.append(row)
        predictions_by_image[image_path.name] = predictions

    for class_id, values in per_class.items():
        matched_ious = per_class_ious[class_id]
        values["mean_matched_iou"] = (
            mean(matched_ious) if matched_ious else None)
        values["matched_boxes"] = len(matched_ious)

    report = {
        "selected_run": selection["selected_run"],
        "split": "test", "images": len(per_image),
        "map50": float(box.map50), "map50_95": float(box.map),
        "precision": float(box.mp), "recall": float(box.mr),
        "mean_matched_iou": mean(all_ious) if all_ious else None,
        "iou_definition": "Mean IoU of class-correct, one-to-one test matches at IoU >= 0.5",
        "iou_confidence": confidence, "nms_iou": nms_iou,
        "matched_counts": totals,
        "ultralytics_version": selection["environment"]["ultralytics"],
        "per_class": {names[index]: values for index, values in sorted(per_class.items())},
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ARTIFACTS / "test_predictions.json").write_text(
        json.dumps(predictions_by_image, indent=2) + "\n", encoding="utf-8")
    (ARTIFACTS / "per_image_review.json").write_text(
        json.dumps(per_image, indent=2) + "\n", encoding="utf-8")
    with (ARTIFACTS / "per_class_metrics.csv").open(
            "w", newline="", encoding="utf-8") as target:
        fields = ("identity", "precision", "recall", "map50", "map50_95",
                  "mean_matched_iou", "matched_boxes")
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_class[index] for index in sorted(per_class))
    print(f"Saved test results: {output}")
    return model, selection, per_image


def create_gallery(model=None, selection=None, per_image=None, device="0",
                   confidence=0.25, nms_iou=0.7):
    """Save ten selected test images with the model's predicted boxes."""
    saved_metrics = json.loads((ARTIFACTS / "test_metrics.json").read_text(
        encoding="utf-8"))
    if (confidence != saved_metrics["iou_confidence"] or
            nms_iou != saved_metrics["nms_iou"]):
        raise ValueError("Gallery thresholds must match the saved test review")
    if model is None:
        model, selection = _selected_model()
    if per_image is None:
        per_image = json.loads((ARTIFACTS / "per_image_review.json").read_text(
            encoding="utf-8"))
    successful = sorted(
        (row for row in per_image if row["fp"] == row["fn"] == 0),
        key=lambda row: (-(row["mean_matched_iou"] or 0), -row["ground_truth"]))
    failures = sorted(
        (row for row in per_image if row["fp"] or row["fn"]),
        key=lambda row: (-(row["fp"] + row["fn"]), row["image"]))
    chosen = (successful[:5] + failures[:5])
    chosen_names = {row["image"] for row in chosen}
    for row in successful[5:] + failures[5:]:
        if len(chosen) >= 10:
            break
        if row["image"] not in chosen_names:
            chosen.append(row)
            chosen_names.add(row["image"])
    if len(chosen) < 10:
        raise ValueError("Need at least ten test images for the gallery")

    gallery = ROOT / "m3_gallery"
    if gallery.exists():
        raise FileExistsError(f"Preserving existing gallery: {gallery}")
    gallery.mkdir()
    for number, row in enumerate(chosen, start=1):
        image_path = DATASET / "images" / "test" / row["image"]
        result = model.predict(source=str(image_path),
                               imgsz=int(selection["settings"]["imgsz"]),
                               conf=confidence, iou=nms_iou, device=device,
                               verbose=False)[0]
        filename = f"{number:02d}_{image_path.stem}.jpg"
        result.save(filename=str(gallery / filename))
    print(f"Saved {len(chosen)} predicted examples in {gallery}")


def main():
    """Run final test evaluation once or rebuild its gallery from saved metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "gallery"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or not 0 < args.nms_iou < 1:
        raise ValueError("Confidence and NMS IoU thresholds must lie between 0 and 1")
    if args.action == "evaluate":
        model, selection, per_image = evaluate(
            device=args.device, confidence=args.confidence, nms_iou=args.nms_iou)
        create_gallery(model, selection, per_image, args.device,
                       args.confidence, args.nms_iou)
    else:
        saved_metrics = json.loads((ARTIFACTS / "test_metrics.json").read_text(
            encoding="utf-8"))
        create_gallery(device=args.device,
                       confidence=saved_metrics["iou_confidence"],
                       nms_iou=saved_metrics["nms_iou"])


if __name__ == "__main__":
    main()
