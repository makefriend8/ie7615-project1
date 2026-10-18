"""Compare YOLO identities with classifier identities on validation predicted boxes.

The detector and classifier run in separate processes so PyTorch and TensorFlow
release GPU allocations between phases. No test images or test labels are read.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/m4/cascade_validation"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.


def match_boxes(truth, predictions, threshold=0.5):
    """Match by detector confidence then IoU, without using predicted identity.

    A prediction takes the highest-IoU available truth box at IoU >= threshold.
    The same localization assignment is used for both identity methods.
    """
    available = set(range(len(truth)))
    matched, extra = [], []
    order = sorted(range(len(predictions)),
                   key=lambda i: (-predictions[i]["yolo_confidence"], i))
    for index in order:
        candidates = [(box_iou(predictions[index]["xyxy"], truth[j]["xyxy"]), j)
                      for j in sorted(available)]
        overlap, target = max(candidates, key=lambda item: (item[0], -item[1]),
                              default=(0., None))
        if target is not None and overlap >= threshold:
            available.remove(target)
            matched.append((index, target, overlap))
        else:
            extra.append(index)
    return matched, extra, sorted(available)


def crop_bounds(box, size):
    """Clip and round outward for classifier input; retain float boxes for IoU."""
    if (len(box) != 4 or not all(math.isfinite(v) for v in box) or
            box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError(f"Invalid predicted box: {box}")
    width, height = size
    bounds = (max(0, min(width, math.floor(box[0]))),
              max(0, min(height, math.floor(box[1]))),
              max(0, min(width, math.ceil(box[2]))),
              max(0, min(height, math.ceil(box[3]))))
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError(f"Empty predicted crop: {box}, {size}")
    return bounds


def audit_frames(dataset, artifacts):
    """Read only validation image/label files; verify them against M2 metadata."""
    dataset, artifacts = Path(dataset), Path(artifacts)
    mapping = read_json(artifacts / "identity_mapping.json")
    names = [str(mapping[str(i)]) for i in range(13)]
    if len(mapping) != 13 or len(set(names)) != 13:
        raise ValueError("Expected the fixed 13-class M2 mapping")
    sources = {r["relative_path"]: r for r in read_csv(artifacts / "source_manifest.csv")}
    groups = defaultdict(list)
    for row in read_csv(artifacts / "object_manifest.csv"):
        if row["split"] != "val":
            continue
        source = sources[row["source_relative_path"]]
        if (source["split"] != "val" or source["identity"] != row["identity"] or
                source["pixel_sha256"] != row["pixel_sha256"] or
                names[int(row["class_index"])] != row["identity"]):
            raise ValueError(f"Invalid validation source relation: {row}")
        if Path(row["frame"]).parts[:2] != ("images", "val"):
            raise ValueError(f"Unexpected validation frame path: {row['frame']}")
        groups[row["frame"]].append(row)
    paths = sorted((dataset / "images/val").glob("*.jpg"))
    labels = sorted((dataset / "labels/val").glob("*.txt"))
    if (len(paths) != 39 or len(groups) != 39 or
            {p.stem for p in paths} != {p.stem for p in labels} or
            sum(map(len, groups.values())) != 175):
        raise ValueError("Expected exactly 39 validation frames and 175 objects")
    frames = []
    for path in paths:
        key = path.relative_to(dataset).as_posix()
        objects = groups[key]
        with Image.open(path) as image:
            if image.size != (640, 640):
                raise ValueError(f"Expected a 640x640 frame: {path}")
            image.verify()
        label_path = dataset / "labels/val" / (path.stem + ".txt")
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(objects):
            raise ValueError(f"Label/object count mismatch: {path}")
        truth = []
        for line, row in zip(lines, objects):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid label row: {label_path}")
            class_id = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            converted = [(x-w/2)*640, (y-h/2)*640, (x+w/2)*640, (y+h/2)*640]
            box = [int(row[k]) for k in ("x1", "y1", "x2", "y2")]
            if (class_id != int(row["class_index"]) or
                    not all(math.isfinite(v) for v in converted) or
                    max(abs(a-b) for a, b in zip(box, converted)) > 0.001 or
                    not (0 <= box[0] < box[2] <= 640 and
                         0 <= box[1] < box[3] <= 640)):
                raise ValueError(f"Label differs from object manifest: {label_path}")
            truth.append({"identity": row["identity"], "class_index": class_id,
                          "xyxy": box, "source": row["source_relative_path"]})
        frames.append({"image": key, "ground_truth": truth,
                       "image_sha256": sha256(path), "label_sha256": sha256(label_path)})
    return names, frames


def validation_portraits(source, artifacts):
    from .m4_classifier import _source_path, _pixel_hash

    rows = [r for r in read_csv(Path(artifacts) / "source_manifest.csv") if r["split"] == "val"]
    if len(rows) != 50 or len({r["relative_path"] for r in rows}) != 50:
        raise ValueError("Expected 50 original validation portraits")
    for row in rows:
        path = _source_path(Path(source), row)
        with Image.open(path) as image:
            if (image.size != (int(row["width"]), int(row["height"])) or
                    _pixel_hash(image) != row["pixel_sha256"]):
                raise ValueError(f"Validation portrait differs from M2: {path}")
    return rows


def detection_worker(output):
    config = read_json(output / "run_config.json")
    from ultralytics import YOLO
    import ultralytics
    import torch

    detector = YOLO(config["detector_checkpoint"])
    if {int(i): str(n) for i, n in detector.names.items()} != dict(enumerate(config["class_names"])):
        raise ValueError("YOLO class names differ from the M2 mapping")
    frames = read_json(output / "validation_frames.json")
    start = time.perf_counter()
    for frame in frames:
        result = detector.predict(
            source=str(Path(config["dataset"]) / frame["image"]),
            imgsz=config["image_size"], conf=config["confidence"],
            iou=config["nms_iou"], device=config["device"], verbose=False)[0]
        predictions = []
        for class_id, confidence, box in zip(result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist()):
            crop_bounds(box, (640, 640))
            predictions.append({"yolo_identity": config["class_names"][int(class_id)],
                                "yolo_confidence": float(confidence),
                                "xyxy": [float(v) for v in box]})
        frame["predictions"] = sorted(predictions, key=lambda p: -p["yolo_confidence"])
        print(f"YOLO {frame['image']}: {len(predictions)} predicted boxes", flush=True)
    write_json(output / "detector_predictions.json", frames)
    write_json(output / "detector_environment.json", {
        "python": platform.python_version(), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "phase_seconds_including_io": time.perf_counter() - start})


def _pixels(image):
    return np.asarray(image.convert("RGB").resize((224, 224),
                      Image.Resampling.BILINEAR), dtype=np.float32)


def classify_worker(output):
    config = read_json(output / "run_config.json")
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(42)
    from .models import build_resnet
    from .m4_classifier import _source_path

    model = build_resnet(13, mode="finetune", weights=None)
    model.load_weights(config["classifier_checkpoint"])
    frames = read_json(output / "detector_predictions.json")
    names = config["class_names"]
    dataset = Path(config["dataset"])
    originals = validation_portraits(config["source"], ROOT / "m2_artifacts")
    images, keys, truth = [], [], []
    for row in originals:
        with Image.open(_source_path(Path(config["source"]), row)) as image:
            images.append(_pixels(image))
        keys.append(("portrait", row["relative_path"]))
        truth.append(row["identity"])
    for frame in frames:
        with Image.open(dataset / frame["image"]) as image:
            rgb = image.convert("RGB")
            for target in frame["ground_truth"]:
                images.append(_pixels(rgb.crop(target["xyxy"])))
                keys.append(("crop", frame["image"] + ":" + ",".join(map(str, target["xyxy"]))))
                truth.append(target["identity"])
    # Verify restoration and unchanged validation inputs before using predicted crops.
    probabilities = model.predict(np.stack(images), batch_size=config["batch_size"], verbose=0)
    if probabilities.shape != (225, 13) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid restored-classifier validation probabilities")
    reference = read_csv(Path(config["classifier_run"]) / "validation_predictions.csv")
    reference = {(r["domain"], r["image"]): r for r in reference}
    if len(reference) != 225 or set(reference) != set(keys):
        raise ValueError("Classifier reference predictions differ from validation inputs")
    restored = []
    for i, key in enumerate(keys):
        predicted = names[int(probabilities[i].argmax())]
        confidence = float(probabilities[i].max())
        previous = reference[key]
        restored.append({"domain": key[0], "image": key[1], "true_identity": truth[i],
            "predicted_identity": predicted, "confidence": confidence,
            "correct": predicted == truth[i],
            "reference_identity_matches": predicted == previous["predicted_identity"],
            "reference_confidence_difference": abs(confidence-float(previous["confidence"]))})
        if previous["true_identity"] != truth[i]:
            raise ValueError(f"Reference identity differs from M2: {key}")
    write_csv(output / "restored_validation_predictions.csv", restored, list(restored[0]))
    check = {"images": len(restored),
        "identity_mismatches": sum(not r["reference_identity_matches"] for r in restored),
        "maximum_confidence_difference": max(r["reference_confidence_difference"] for r in restored),
        "confidence_tolerance": 0.0001,
        "validation": {d: {"images": sum(r["domain"] == d for r in restored),
            "correct": sum(r["domain"] == d and r["correct"] for r in restored)}
            for d in ("portrait", "crop")}}
    check["passed"] = check["identity_mismatches"] == 0 and check["maximum_confidence_difference"] <= 0.0001
    write_json(output / "checkpoint_restore_check.json", check)
    if not check["passed"]:
        raise ValueError("Restored predictions differ from the training review; inspect checkpoint_restore_check.json")
    print("Selected checkpoint restored: reference predictions agree for all 225 inputs", flush=True)
    del images, probabilities
    crops, locations = [], []
    for frame_index, frame in enumerate(frames):
        with Image.open(dataset / frame["image"]) as image:
            rgb = image.convert("RGB")
            for prediction_index, prediction in enumerate(frame["predictions"]):
                bounds = crop_bounds(prediction["xyxy"], rgb.size)
                prediction["classifier_crop_xyxy"] = list(bounds)
                crops.append(_pixels(rgb.crop(bounds)))
                locations.append((frame_index, prediction_index))
    if crops:
        probabilities = model.predict(np.stack(crops), batch_size=config["batch_size"], verbose=0)
        if (probabilities.shape != (len(crops), 13) or not np.isfinite(probabilities).all() or
                not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5)):
            raise ValueError("Invalid predicted-crop classifier probabilities")
        for (frame_index, prediction_index), probability in zip(locations, probabilities):
            prediction = frames[frame_index]["predictions"][prediction_index]
            prediction.update(classifier_identity=names[int(probability.argmax())],
                classifier_confidence=float(probability.max()),
                classifier_probabilities=[float(p) for p in probability])
    write_json(output / "predictions.json", frames)
    write_json(output / "classifier_environment.json", {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "keras": tf.keras.__version__, "numpy": np.__version__,
        "gpu": [tf.config.experimental.get_device_details(d).get("device_name", d.name) for d in devices]})
    print(f"Classified {len(crops)} YOLO predicted crops", flush=True)


def score_frames(frames, names, threshold=0.5):
    """Count localization and identity errors, including missed and extra boxes.

    Identity micro TP = localized, correctly identified objects. Wrong identity
    contributes both an identity FP and FN. These fixed-threshold metrics are
    not AP and are not a replacement for the historical M3 test metrics.
    """
    totals = Counter()
    transitions = Counter()
    per_class = {method: {n: Counter() for n in names} for method in ("yolo", "classifier")}
    confusion = {method: Counter() for method in per_class}
    per_image, prediction_rows = [], []
    for frame in frames:
        truth, predictions = frame["ground_truth"], frame["predictions"]
        matched, extra, missed = match_boxes(truth, predictions, threshold)
        matches = {p: (g, overlap) for p, g, overlap in matched}
        totals.update(images=1, ground_truth=len(truth), predictions=len(predictions),
                      localized=len(matched), extra=len(extra), missed=len(missed))
        row = {"image": frame["image"], "ground_truth": len(truth),
            "predictions": len(predictions), "localized": len(matched),
            "extra": len(extra), "missed": len(missed),
            "mean_localized_iou": sum(v for _, _, v in matched)/len(matched) if matched else None,
            "disagreements": sum(p["yolo_identity"] != p["classifier_identity"] for p in predictions)}
        for method in per_class:
            correct = sum(predictions[p][method+"_identity"] == truth[g]["identity"] for p, g, _ in matched)
            row[method+"_correct"] = correct
            row[method+"_all_correct"] = len(extra) == len(missed) == 0 and correct == len(truth)
            totals[method+"_correct"] += correct
            totals[method+"_all_correct_images"] += row[method+"_all_correct"]
            for p, g, _ in matched:
                true, predicted = truth[g]["identity"], predictions[p][method+"_identity"]
                confusion[method][(true, predicted)] += 1
                if true == predicted:
                    per_class[method][true]["tp"] += 1
                else:
                    per_class[method][true]["fn"] += 1
                    per_class[method][predicted]["fp"] += 1
            for p in extra:
                predicted = predictions[p][method+"_identity"]
                per_class[method][predicted]["fp"] += 1
                confusion[method][("__background__", predicted)] += 1
            for g in missed:
                true = truth[g]["identity"]
                per_class[method][true]["fn"] += 1
                confusion[method][(true, "__missed__")] += 1
        for p, g, _ in matched:
            yolo_ok = predictions[p]["yolo_identity"] == truth[g]["identity"]
            classifier_ok = predictions[p]["classifier_identity"] == truth[g]["identity"]
            transitions[("both_correct" if yolo_ok and classifier_ok else
                "classifier_only_correct" if classifier_ok else
                "yolo_only_correct" if yolo_ok else "both_wrong")] += 1
        for i, prediction in enumerate(predictions):
            target, overlap = matches.get(i, (None, None))
            prediction["matched_gt_index"] = target
            prediction["matched_iou"] = overlap
            prediction_rows.append({"image": frame["image"], "prediction_index": i,
                **{k: prediction[k] for k in ("yolo_identity", "yolo_confidence", "classifier_identity", "classifier_confidence")},
                **dict(zip(("x1", "y1", "x2", "y2"), prediction["xyxy"])),
                "matched_gt_index": target, "matched_iou": overlap,
                "true_identity": truth[target]["identity"] if target is not None else "",
                "disagreement": prediction["yolo_identity"] != prediction["classifier_identity"]})
        per_image.append(row)
    def metrics(tp, predictions, ground_truth):
        precision = tp/predictions if predictions else 0.
        recall = tp/ground_truth if ground_truth else 0.
        return {"tp": tp, "fp": predictions-tp, "fn": ground_truth-tp,
            "precision": precision, "recall": recall,
            "f1": 2*tp/(predictions+ground_truth) if predictions+ground_truth else 0.}
    summary = {"split": "val", "test_evaluated": False, "counts": dict(totals),
        "localization": metrics(totals["localized"], totals["predictions"], totals["ground_truth"]),
        "identity": {}, "matched_identity_transitions": dict(transitions),
        "matching": {"iou_threshold": threshold, "class_agnostic": True,
            "rule": "Descending YOLO confidence; highest-IoU available truth; one-to-one"},
        "metric_note": "Fixed-threshold micro counts; not mAP. Wrong identity adds both identity FP and FN."}
    class_rows, confusion_rows = [], []
    for method in per_class:
        summary["identity"][method] = metrics(totals[method+"_correct"], totals["predictions"], totals["ground_truth"])
        summary["identity"][method]["accuracy_given_localized"] = (
            totals[method+"_correct"]/totals["localized"] if totals["localized"] else None)
        for name, counts in per_class[method].items():
            class_rows.append({"method": method, "identity": name, **metrics(
                counts["tp"], counts["tp"]+counts["fp"], counts["tp"]+counts["fn"])})
        confusion_rows.extend({"method": method, "true_identity": a,
            "predicted_identity": b, "count": count} for (a, b), count in sorted(confusion[method].items()))
    return summary, per_image, prediction_rows, class_rows, confusion_rows


def save_overlays(frames, dataset, output):
    """Save only predicted boxes; truth controls validation review colors."""
    for folder in ("annotated_yolo", "annotated_classifier", "comparison"):
        (output / folder).mkdir()
    for frame in frames:
        with Image.open(Path(dataset) / frame["image"]) as source:
            panels = []
            for method, title in (("yolo", "YOLO identities"), ("classifier", "Classifier on YOLO crops")):
                panel = Image.new("RGB", (640, 670), "white")
                panel.paste(source.convert("RGB"), (0, 30))
                draw = ImageDraw.Draw(panel)
                draw.text((8, 8), title + " | validation", fill="black")
                for prediction in frame["predictions"]:
                    target = prediction["matched_gt_index"]
                    correct = target is not None and prediction[method+"_identity"] == frame["ground_truth"][target]["identity"]
                    color = "#168a35" if correct else "#d12332"
                    x1, y1, x2, y2 = prediction["xyxy"]
                    label = f"{prediction[method+'_identity']} {prediction[method+'_confidence']:.3f}"
                    draw.rectangle((x1, y1+30, x2, y2+30), outline=color, width=3)
                    tx, ty = min(x1+2, 530), max(30, y1+30-14)
                    draw.rectangle((tx, ty, tx+108, ty+14), fill="white")
                    draw.text((tx, ty), label, fill=color)
                filename = Path(frame["image"]).name
                panel.save(output / ("annotated_"+method) / filename, quality=92)
                panels.append(panel)
            pair = Image.new("RGB", (1280, 670), "white")
            pair.paste(panels[0], (0, 0))
            pair.paste(panels[1], (640, 0))
            pair.save(output / "comparison" / filename, quality=92)


def _worker(phase, output):
    command = [sys.executable, "-u", "-m", "src.m4_cascade_validation",
               "--worker", phase, "--output", str(output)]
    with (output / (phase+".log")).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def run(source=ROOT / "data/shared_pool", output=DEFAULT_RUN, device="0", batch_size=8):
    """Run validation inference and comparison, preserving every previous run."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    artifacts, dataset = ROOT / "m2_artifacts", ROOT / "data/m2_detection"
    classifier_run = ROOT / "runs/m4/classifier13"
    names, frames = audit_frames(dataset, artifacts)
    validation_portraits(source, artifacts)
    selection = read_json(classifier_run / "selection.json")
    detector_selection = read_json(ROOT / "m3_artifacts/selection.json")
    if names != read_json(classifier_run / "class_names.json") or names != selection["class_names"]:
        raise ValueError("Classifier class names differ from M2")
    for filename, key in (("source_manifest.csv", "source_manifest_sha256"),
                          ("object_manifest.csv", "object_manifest_sha256")):
        if sha256(artifacts / filename) != read_json(classifier_run / "data_counts.json")[key]:
            raise ValueError(f"Classifier training manifest changed: {filename}")
    checkpoints = {"classifier": classifier_run / "best.weights.h5",
                   "detector": ROOT / "m3_artifacts/best.pt"}
    for kind, metadata in (("classifier", selection), ("detector", detector_selection)):
        if sha256(checkpoints[kind]) != metadata["checkpoint_sha256"]:
            raise ValueError(f"Selected {kind} checkpoint hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    config = {"split": "val", "test_evaluated": False, "source": str(source),
        "dataset": str(dataset), "classifier_run": str(classifier_run),
        "class_names": names, "device": str(device), "batch_size": batch_size,
        "confidence": 0.25, "nms_iou": 0.7, "match_iou": 0.5,
        "image_size": int(detector_selection["settings"]["imgsz"]),
        **{k+"_checkpoint": str(v) for k, v in checkpoints.items()},
        "checkpoint_sha256": {k: sha256(v) for k, v in checkpoints.items()},
        "crop_preprocessing": "RGB; clipped floor/ceil predicted box; bilinear resize 224x224; raw float32 0..255; model ResNetV2 preprocessing"}
    output.mkdir(parents=True)
    write_json(output / "run_config.json", config)
    write_json(output / "validation_frames.json", frames)
    snapshot = output / "input_snapshot"
    snapshot.mkdir()
    for name, path in {
        "classifier_selection.json": classifier_run / "selection.json",
        "classifier_data_counts.json": classifier_run / "data_counts.json",
        "classifier_reference_predictions.csv": classifier_run / "validation_predictions.csv",
        "detector_selection.json": ROOT / "m3_artifacts/selection.json",
        **{f: artifacts / f for f in ("identity_mapping.json", "source_manifest.csv", "object_manifest.csv")},
        **{f: ROOT / "src" / f for f in ("models.py", "m4_classifier.py", "m4_cascade_validation.py")},
    }.items():
        shutil.copyfile(path, snapshot / name)
    try:
        print("Phase 1/2: YOLO validation prediction in an isolated process", flush=True)
        _worker("detect", output)
        print("Phase 2/2: restore and validate classifier, then classify predicted crops", flush=True)
        _worker("classify", output)
        frames = read_json(output / "predictions.json")
        summary, images, predictions, classes, confusion = score_frames(frames, names)
        write_json(output / "summary.json", summary)
        write_json(output / "predictions.json", frames)
        write_csv(output / "per_image.csv", images, list(images[0]))
        fields = ["image", "prediction_index", "yolo_identity", "yolo_confidence",
            "classifier_identity", "classifier_confidence", "x1", "y1", "x2", "y2",
            "matched_gt_index", "matched_iou", "true_identity", "disagreement"]
        write_csv(output / "predicted_boxes.csv", predictions, fields)
        write_csv(output / "disagreements.csv", [r for r in predictions if r["disagreement"]], fields)
        write_csv(output / "per_identity.csv", classes, list(classes[0]))
        write_csv(output / "identity_confusion.csv", confusion,
                  ["method", "true_identity", "predicted_identity", "count"])
        save_overlays(frames, dataset, output)
        (output / "README.md").write_text(
            "# M4 validation cascade review\n\n"
            "Only the 39 synthetic validation frames (175 labeled portrait regions) are evaluated. "
            "A separate restoration check repeats the existing 50 portrait / 175 ground-truth-crop "
            "validation predictions. No training or test images are read.\n\n"
            "YOLO-only and classifier-on-YOLO-crops use identical float predicted boxes and "
            "class-agnostic one-to-one matching at IoU >= 0.5. Identity micro precision/recall/F1 "
            "include missed and extra predictions; they are fixed-threshold metrics, not mAP. "
            "Wrong identity counts as both an identity FP and FN. Conditional identity accuracy "
            "uses only localization matches.\n\n"
            "All 39 comparison images show predicted boxes only: green means localized and "
            "identity-correct; red means wrong identity or unmatched. Missing objects remain "
            "in per_image.csv and predictions.json. These are validation review images, not "
            "the final test gallery. YOLO confidence and classifier softmax confidence are "
            "different scores and should not be directly compared as calibrated probabilities.\n\n"
            "predictions.json retains all 13 classifier probabilities in run_config.json class order, "
            "ground truth, crop bounds, predicted boxes and matching indices. input_snapshot/ "
            "records the actual code and manifests. Worker logs and separate environment files "
            "support debugging. No identity fusion policy is selected automatically.\n",
            encoding="utf-8")
        files = {p.relative_to(output).as_posix(): sha256(p)
                 for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output / "artifact_hashes.json", files)
        write_json(output / "COMPLETE.json", {"status": "complete", "split": "val",
            "test_evaluated": False, "summary_sha256": sha256(output / "summary.json"),
            "artifact_hashes_sha256": sha256(output / "artifact_hashes.json")})
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Review complete: {output}", flush=True)
        return output
    except Exception:
        write_json(output / "FAILED.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/shared_pool")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="0", help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--worker", choices=("detect", "classify"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        {"detect": detection_worker, "classify": classify_worker}[args.worker](args.output.resolve())
    else:
        run(args.source, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
