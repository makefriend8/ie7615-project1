"""Run the frozen M4 system on new images or evaluate the held-out test split.

Examples:
  python -m src.m4_pipeline infer --single portrait.jpg --multi grid.jpg --output runs/m4/demo_01
  python -m src.m4_pipeline evaluate --source data/shared_pool
"""

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from .m4_cascade_validation import (ROOT, read_json, write_json, read_csv,
    write_csv, sha256, crop_bounds, score_frames, match_boxes, _pixels)
from .m4_policy import suppress_duplicates, choose_identity, apply_policy


POLICY_PATH = ROOT / "m4_artifacts/integration_policy.json"
FINAL_RUN = ROOT / "runs/m4/final_evaluation"


def verify_complete(output):
    output = Path(output)
    complete = read_json(output / "COMPLETE.json")
    if complete.get("status") != "complete" or (output / "FAILED.json").exists():
        raise ValueError("Review is not complete")
    if sha256(output / "artifact_hashes.json") != complete["artifact_hashes_sha256"]:
        raise ValueError("Artifact hash list was changed")
    for relative, expected in read_json(output / "artifact_hashes.json").items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != expected:
            raise ValueError(f"Saved result changed: {relative}")
    return complete


def configuration(device, batch_size):
    policy = read_json(POLICY_PATH)
    names = policy["class_names"]
    mapping = read_json(ROOT / "m2_artifacts/identity_mapping.json")
    classifier = ROOT / "runs/m4/classifier13"
    if (policy["selection_split"] != "val" or policy["test_used_for_selection"] is not False or
            names != [str(mapping[str(i)]) for i in range(13)] or
            names != read_json(classifier / "class_names.json")):
        raise ValueError("Invalid policy or inconsistent class names")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    checkpoints = {"classifier": classifier / "best.weights.h5",
                   "detector": ROOT / "m3_artifacts/best.pt"}
    for kind, path in checkpoints.items():
        if sha256(path) != policy["checkpoint_sha256"][kind]:
            raise ValueError(f"Checkpoint differs from the frozen policy: {kind}")
    data = read_json(classifier / "data_counts.json")
    for filename, key in (("source_manifest.csv", "source_manifest_sha256"),
                          ("object_manifest.csv", "object_manifest_sha256")):
        if sha256(ROOT / "m2_artifacts" / filename) != data[key]:
            raise ValueError(f"Training manifest changed: {filename}")
    return {"policy": policy, "policy_sha256": sha256(POLICY_PATH),
        "class_names": names, "device": str(device), "batch_size": batch_size,
        **{k+"_checkpoint": str(v) for k, v in checkpoints.items()}}


def portrait_tasks(source, split):
    from .m4_classifier import _source_path, _pixel_hash

    rows = [r for r in read_csv(ROOT / "m2_artifacts/source_manifest.csv") if r["split"] == split]
    if len(rows) != {"val": 50, "test": 52}[split]:
        raise ValueError(f"Unexpected original portrait count: {split}")
    tasks = []
    for row in rows:
        path = _source_path(Path(source), row)
        with Image.open(path) as image:
            if (image.size != (int(row["width"]), int(row["height"])) or
                    _pixel_hash(image) != row["pixel_sha256"]):
                raise ValueError(f"Original portrait differs from M2: {path}")
        tasks.append({"kind": "single", "split": split, "image": row["relative_path"],
                      "path": str(path), "true_identity": row["identity"]})
    return tasks


def test_frame_tasks(names):
    dataset = ROOT / "data/m2_detection"
    sources = {r["relative_path"]: r for r in read_csv(ROOT / "m2_artifacts/source_manifest.csv")}
    groups = {}
    for row in read_csv(ROOT / "m2_artifacts/object_manifest.csv"):
        if row["split"] == "test":
            groups.setdefault(row["frame"], []).append(row)
    paths = sorted((dataset / "images/test").glob("*.jpg"))
    labels = sorted((dataset / "labels/test").glob("*.txt"))
    if (len(paths) != 39 or len(groups) != 39 or sum(map(len, groups.values())) != 175 or
            {p.stem for p in paths} != {p.stem for p in labels}):
        raise ValueError("Expected 39 test frames and 175 objects")
    tasks = []
    for path in paths:
        key = path.relative_to(dataset).as_posix()
        with Image.open(path) as image:
            if image.size != (640, 640):
                raise ValueError(f"Invalid frame dimensions: {path}")
        label = dataset / "labels/test" / (path.stem+".txt")
        lines = label.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(groups[key]):
            raise ValueError(f"Test label count differs from manifest: {label}")
        targets = []
        for line, row in zip(lines, groups[key]):
            class_id, *values = line.split()
            if len(values) != 4:
                raise ValueError(f"Invalid test label: {label}")
            x, y, w, h = map(float, values)
            converted = np.asarray([(x-w/2)*640, (y-h/2)*640, (x+w/2)*640, (y+h/2)*640])
            box = [int(row[k]) for k in ("x1", "y1", "x2", "y2")]
            source = sources[row["source_relative_path"]]
            if (class_id != row["class_index"] or names[int(class_id)] != row["identity"] or
                    source["split"] != "test" or source["identity"] != row["identity"] or
                    source["pixel_sha256"] != row["pixel_sha256"] or
                    not np.isfinite(converted).all() or np.max(np.abs(converted-box)) > .001 or
                    not (0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 640)):
                raise ValueError(f"Test label or source differs from manifest: {label}")
            targets.append({"identity": row["identity"], "xyxy": box,
                            "source": row["source_relative_path"]})
        tasks.append({"kind": "multi", "split": "test", "image": key, "path": str(path),
            "ground_truth": targets, "image_sha256": sha256(path), "label_sha256": sha256(label)})
    return tasks


def validation_replay(policy, directory):
    directory = Path(directory)
    verify_complete(directory)
    if sha256(directory / "summary.json") != policy["validation_evidence"]["original_summary_sha256"]:
        raise ValueError("Validation review differs from the policy selection evidence")
    frames = apply_policy(read_json(directory / "predictions.json"), policy)
    for frame in frames:
        for prediction in frame["predictions"]:
            prediction["classifier_identity"] = prediction["final_identity"]
    summary, *_ = score_frames(frames, policy["class_names"])
    counts = summary["counts"]
    actual = {k: counts[k] for k in ("images", "ground_truth", "predictions", "localized", "extra", "missed")}
    actual.update(final_correct=counts["classifier_correct"],
                  final_all_correct_images=counts["classifier_all_correct_images"])
    if actual != policy["validation_evidence"]["expected_counts"]:
        raise ValueError("Frozen policy replay does not reproduce validation counts")
    return {"passed": True, "counts": actual, "identity": summary["identity"]["classifier"]}


def detect_worker(output):
    config = read_json(output / "run_config.json")
    tasks = read_json(output / "tasks.json")
    if not any(task["kind"] == "multi" for task in tasks):
        write_json(output / "detected_tasks.json", tasks)
        write_json(output / "detector_environment.json", {"skipped": True,
            "reason": "All inputs use single-portrait classification mode"})
        return
    from ultralytics import YOLO
    import ultralytics
    import torch

    model = YOLO(config["detector_checkpoint"])
    if {int(i): str(n) for i, n in model.names.items()} != dict(enumerate(config["class_names"])):
        raise ValueError("Detector class names differ from the policy")
    settings = config["policy"]["detector"]
    start = time.perf_counter()
    for task in tasks:
        if task["kind"] != "multi":
            continue
        result = model.predict(source=task["path"], imgsz=settings["image_size"],
            conf=settings["confidence"], iou=settings["nms_iou"],
            agnostic_nms=False, device=config["device"], verbose=False)[0]
        raw = [{"yolo_identity": config["class_names"][int(c)], "yolo_confidence": float(p),
                "xyxy": [float(v) for v in box]}
               for c, p, box in zip(result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist())]
        task["raw_detector_predictions"] = raw
        task["predictions"] = suppress_duplicates(raw, config["policy"]["post_nms_iou"])
        print(f"{task['image']}: {len(raw)} YOLO boxes -> {len(task['predictions'])} after post-NMS", flush=True)
    write_json(output / "detected_tasks.json", tasks)
    write_json(output / "detector_environment.json", {"python": platform.python_version(),
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "phase_seconds_including_io": time.perf_counter()-start})


def classify_worker(output):
    config = read_json(output / "run_config.json")
    tasks = read_json(output / "detected_tasks.json")
    import tensorflow as tf

    if config["device"] == "cpu":
        tf.config.set_visible_devices([], "GPU")
    devices = tf.config.get_visible_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(42)
    from .models import build_resnet

    start = time.perf_counter()
    model = build_resnet(13, mode="finetune", weights=None)
    model.load_weights(config["classifier_checkpoint"])
    pixels, destinations = [], []
    for task in tasks:
        with Image.open(task["path"]) as source:
            rgb = source.convert("RGB")
            if task["kind"] == "single":
                pixels.append(_pixels(rgb))
                destinations.append(task)
            else:
                for prediction in task["predictions"]:
                    bounds = crop_bounds(prediction["xyxy"], rgb.size)
                    prediction["classifier_crop_xyxy"] = list(bounds)
                    pixels.append(_pixels(rgb.crop(bounds)))
                    destinations.append(prediction)
    if pixels:
        probabilities = model.predict(np.stack(pixels), batch_size=config["batch_size"], verbose=0)
        if (probabilities.shape != (len(pixels), 13) or not np.isfinite(probabilities).all() or
                not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5)):
            raise ValueError("Invalid classifier probability vectors")
        for destination, probability in zip(destinations, probabilities):
            destination.update(classifier_identity=config["class_names"][int(probability.argmax())],
                classifier_confidence=float(probability.max()),
                classifier_probabilities=[float(v) for v in probability])
    for task in tasks:
        if task["kind"] == "single":
            task["final_identity"] = task["classifier_identity"]
            task["identity_rule"] = "single_classifier"
            task["localization"] = "Not estimated: classification of an entire single-portrait input"
        else:
            for prediction in task["predictions"]:
                identity, reason = choose_identity(prediction, config["policy"]["classifier_conflict_threshold"])
                prediction.update(final_identity=identity, identity_rule=reason)
    write_json(output / "predictions.json", tasks)
    write_json(output / "classifier_environment.json", {"python": platform.python_version(),
        "tensorflow": tf.__version__, "keras": tf.keras.__version__, "numpy": np.__version__,
        "gpu": [tf.config.experimental.get_device_details(d).get("device_name", d.name) for d in devices],
        "phase_seconds_including_io": time.perf_counter()-start})
    print(f"Classified {len(destinations)} portraits or predicted crops", flush=True)


def multi_scores(tasks, names, method):
    frames = copy.deepcopy([t for t in tasks if t["kind"] == "multi"])
    for frame in frames:
        if method == "raw_yolo":
            frame["predictions"] = frame["raw_detector_predictions"]
        for p in frame["predictions"]:
            if method in ("raw_yolo", "yolo_after_post_nms"):
                p["classifier_identity"] = p["yolo_identity"]
                p["classifier_confidence"] = p["yolo_confidence"]
            elif method == "integrated":
                p["classifier_identity"] = p["final_identity"]
    summary, images, rows, classes, confusion = score_frames(frames, names)
    metrics = summary["identity"]["classifier"]
    metrics["all_correct_images"] = summary["counts"]["classifier_all_correct_images"]
    return metrics, summary["localization"], images, rows, classes, confusion


def evaluate_predictions(output, tasks, config):
    single = {}
    single_rows = []
    reference = {(r["domain"], r["image"]): r for r in read_csv(
        ROOT / "runs/m4/classifier13/validation_predictions.csv")}
    for split in ("val", "test"):
        subset = [t for t in tasks if t["kind"] == "single" and t["split"] == split]
        correct = sum(t["final_identity"] == t["true_identity"] for t in subset)
        single[split] = {"images": len(subset), "correct": correct, "accuracy": correct/len(subset)}
        for t in subset:
            if split == "val":
                previous = reference[("portrait", t["image"])]
                if (t["final_identity"] != previous["predicted_identity"] or
                        abs(t["classifier_confidence"]-float(previous["confidence"])) > 1e-4):
                    raise ValueError("Final single mode differs from the validated classifier")
            single_rows.append({"split": split, "image": t["image"],
                "true_identity": t["true_identity"], "predicted_identity": t["final_identity"],
                "confidence": t["classifier_confidence"], "correct": t["final_identity"] == t["true_identity"]})
    write_csv(output / "single_predictions.csv", single_rows, list(single_rows[0]))
    multi, localization = {}, {}
    for method in ("raw_yolo", "yolo_after_post_nms", "classifier_after_post_nms", "integrated"):
        metrics, loc, images, rows, classes, confusion = multi_scores(tasks, config["class_names"], method)
        multi[method], localization[method] = metrics, loc
        if method == "integrated":
            images = [{k.replace("classifier_", "integrated_"): v for k, v in row.items()} for row in images]
            write_csv(output / "multi_per_image.csv", images, list(images[0]))
            rows = [{k.replace("classifier_", "integrated_"): v for k, v in row.items()} for row in rows]
            # This field is the raw classifier score, not a calibrated final score.
            for row in rows:
                row["raw_classifier_confidence"] = row.pop("integrated_confidence")
            write_csv(output / "multi_predictions.csv", rows, list(rows[0]) if rows else
                ["image", "prediction_index", "yolo_identity", "integrated_identity", "true_identity"])
            class_rows = [dict(r, method="integrated") for r in classes if r["method"] == "classifier"]
            confusion_rows = [dict(r, method="integrated") for r in confusion if r["method"] == "classifier"]
            write_csv(output / "multi_per_identity.csv", class_rows, list(classes[0]))
            write_csv(output / "multi_confusion.csv", confusion_rows,
                      ["method", "true_identity", "predicted_identity", "count"])
    summary = {"status": "complete", "test_evaluated": True,
        "policy_id": config["policy"]["policy_id"], "policy_sha256": config["policy_sha256"],
        "single": single, "multi_test": {"images": 39, "ground_truth_objects": 175,
            "identity": multi, "localization": localization},
        "note": "Fixed-threshold class-agnostic matching at IoU >= 0.5; not mAP. Validation portraits are gallery supplements, not test samples."}
    write_json(output / "summary.json", summary)
    return summary


def render_tasks(output, tasks):
    for kind in ("single", "multi"):
        (output / ("annotated_"+kind)).mkdir()
    for index, task in enumerate(tasks):
        with Image.open(task["path"]) as source:
            rgb = source.convert("RGB")
        if task["kind"] == "single" and rgb.width < 320:
            rgb = rgb.resize((rgb.width*2, rgb.height*2), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (max(320, rgb.width), rgb.height+60), "white")
        canvas.paste(rgb, (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"M4 {task['kind']} | {task.get('split', 'unseen')}", fill="black")
        if task["kind"] == "single":
            draw.text((8, 21), f"Prediction ID {task['final_identity']} | CLS {task['classifier_confidence']:.3f}", fill="black")
            if "true_identity" in task:
                ok = task["final_identity"] == task["true_identity"]
                draw.text((8, 37), f"True ID {task['true_identity']} | {'correct' if ok else 'misclassified'}",
                          fill="#168a35" if ok else "#d12332")
        else:
            predictions = task["predictions"]
            matches = {}
            if "ground_truth" in task:
                matched, extra, missed = match_boxes(task["ground_truth"], predictions)
                matches = {p: g for p, g, _ in matched}
                draw.text((8, 37), f"GT {len(task['ground_truth'])} | extra {len(extra)} | missed {len(missed)}", fill="black")
            draw.text((8, 21), f"{len(predictions)} predicted regions | frozen validation policy", fill="black")
            for i, p in enumerate(predictions):
                color = "#1368b5"
                if "ground_truth" in task:
                    ok = i in matches and p["final_identity"] == task["ground_truth"][matches[i]]["identity"]
                    color = "#168a35" if ok else "#d12332"
                x1, y1, x2, y2 = p["xyxy"]
                draw.rectangle((x1, y1+60, x2, y2+60), outline=color, width=3)
                label = f"ID {p['final_identity']}"
                tx, ty = min(x1+2, max(0, rgb.width-85)), max(60, y1+60-14)
                draw.rectangle((tx, ty, tx+85, ty+14), fill="white")
                draw.text((tx, ty), label, fill=color)
        relative = f"annotated_{task['kind']}/{index:03d}_{Path(task['image']).stem}.jpg"
        canvas.save(output / relative, quality=92)
        task["annotated_file"] = relative


def create_gallery(output, tasks, names):
    gallery = output / "gallery"
    gallery.mkdir()
    singles = sorted([t for t in tasks if t["kind"] == "single"],
                     key=lambda t: (t["split"] != "test", t["image"]))
    chosen = {"single_correct": [t for t in singles if t["final_identity"] == t["true_identity"]][:20],
              "single_errors": [t for t in singles if t["final_identity"] != t["true_identity"]][:5],
              "multi": [t for t in tasks if t["kind"] == "multi" and t["predictions"]][:10]}
    lines = ["# Final M4 results gallery", "", "Actual outputs of the frozen M4 pipeline. "
        "Test examples are selected first; validation supplements are explicitly labeled. "
        "Single portraits are classified without an estimated face box. Multi-person boxes "
        "are predictions. Scores are uncalibrated; identity IDs follow the M2 mapping.", ""]
    index = []
    for group, selected in chosen.items():
        (gallery / group).mkdir()
        lines += ["## " + group.replace("_", " ").title(), ""]
        for number, task in enumerate(selected, 1):
            name = f"{number:02d}_{task['split']}_{Path(task['image']).stem}.jpg"
            shutil.copyfile(output / task["annotated_file"], gallery / group / name)
            record = {"group": group, "split": task["split"], "image": task["image"],
                      "gallery_file": group+"/"+name}
            caption = f"{task['split']} | {task['image']}"
            if task["kind"] == "single":
                record.update(true_identity=task["true_identity"], predicted_identity=task["final_identity"],
                              confidence=task["classifier_confidence"])
                caption += f" | true {task['true_identity']} | predicted {task['final_identity']}"
                if group == "single_errors":
                    true_probability = task["classifier_probabilities"][names.index(task["true_identity"])]
                    record["true_identity_probability"] = true_probability
                    caption += (f". Error analysis: the wrong identity receives {task['classifier_confidence']:.3f} "
                                f"softmax probability; the true identity receives {true_probability:.3f}. "
                                "The ranking fails on this labeled portrait; confidence is not a correctness guarantee.")
            else:
                caption += f" | {len(task['ground_truth'])} labeled regions, {len(task['predictions'])} predicted regions"
            lines += [caption, "", f"![{group} example {number}]({group}/{name})", ""]
            index.append(record)
    counts = {g: len(v) for g, v in chosen.items()}
    satisfied = counts["single_correct"] >= 20 and counts["single_errors"] >= 5 and counts["multi"] >= 10
    write_json(gallery / "index.json", {"counts": counts, "quantity_requirements_met": satisfied, "examples": index})
    (gallery / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not satisfied:
        raise ValueError("Insufficient genuine gallery examples; inspect gallery/index.json")
    return counts


def execute(tasks, output, config, evaluation=False, replay=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing run: {output}")
    start = time.perf_counter()
    output.mkdir(parents=True)
    config.update(evaluation=evaluation, test_evaluated=evaluation)
    write_json(output / "run_config.json", config)
    write_json(output / "tasks.json", tasks)
    snapshot = output / "input_snapshot"
    snapshot.mkdir()
    for path in [POLICY_PATH, *(ROOT / "src" / f for f in
            ("m4_pipeline.py", "m4_policy.py", "m4_cascade_validation.py", "m4_classifier.py", "models.py")),
            *(ROOT / "m2_artifacts" / f for f in
            ("source_manifest.csv", "object_manifest.csv", "identity_mapping.json"))]:
        shutil.copyfile(path, snapshot / path.name)
    if replay is not None:
        write_json(output / "validation_policy_replay.json", replay)
    try:
        # Reuse the logging transport, with this module's worker entry point.
        for phase in ("detect", "classify"):
            import subprocess
            command = [sys.executable, "-u", "-m", "src.m4_pipeline", "--worker", phase, "--output", str(output)]
            with (output / (phase+".log")).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        tasks = read_json(output / "predictions.json")
        summary = evaluate_predictions(output, tasks, config) if evaluation else {
            "status": "complete", "test_evaluated": False, "input_images": len(tasks),
            "policy_id": config["policy"]["policy_id"]}
        render_tasks(output, tasks)
        write_json(output / "predictions.json", tasks)
        if evaluation:
            summary["gallery_counts"] = create_gallery(output, tasks, config["class_names"])
        summary["cold_run_seconds_including_workers_io_and_rendering"] = time.perf_counter()-start
        write_json(output / "summary.json", summary)
        write_json(output / "artifact_hashes.json", {p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*")) if p.is_file()})
        write_json(output / "COMPLETE.json", {"status": "complete", "test_evaluated": evaluation,
            "policy_sha256": config["policy_sha256"], "summary_sha256": sha256(output / "summary.json"),
            "artifact_hashes_sha256": sha256(output / "artifact_hashes.json")})
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Saved outputs: {output}", flush=True)
        return output
    except Exception:
        write_json(output / "FAILED.json", {"traceback": traceback.format_exc()})
        raise


def evaluate(source=ROOT / "data/shared_pool", output=FINAL_RUN, device="0", batch_size=8,
             validation_review=ROOT / "runs/m4/cascade_validation"):
    output = Path(output).resolve()
    if (output / "COMPLETE.json").is_file():
        complete = verify_complete(output)
        if complete.get("test_evaluated") is not True or complete["policy_sha256"] != sha256(POLICY_PATH):
            raise ValueError("Existing results do not match this final evaluation")
        print(f"Reusing verified completed evaluation; no inference repeated: {output}")
        return output
    if output.exists():
        raise FileExistsError(f"Preserving incomplete evaluation: {output}")
    config = configuration(device, batch_size)
    replay = validation_replay(config["policy"], validation_review)
    tasks = portrait_tasks(Path(source).resolve(), "val") + portrait_tasks(Path(source).resolve(), "test")
    tasks += test_frame_tasks(config["class_names"])
    return execute(tasks, output, config, evaluation=True, replay=replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("infer", "evaluate"), nargs="?", default="infer")
    parser.add_argument("--single", type=Path, nargs="*", default=[])
    parser.add_argument("--multi", type=Path, nargs="*", default=[])
    parser.add_argument("--source", type=Path, default=ROOT / "data/shared_pool")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0", help="Use cpu for CPU-only inference")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--worker", choices=("detect", "classify"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        {"detect": detect_worker, "classify": classify_worker}[args.worker](args.output.resolve())
    elif args.action == "evaluate":
        evaluate(args.source, args.output or FINAL_RUN, args.device, args.batch_size)
    else:
        if not args.single and not args.multi:
            parser.error("Provide --single and/or --multi image paths")
        tasks = []
        for kind, paths in (("single", args.single), ("multi", args.multi)):
            for path in paths:
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                tasks.append({"kind": kind, "split": "unseen", "image": path.name, "path": str(path)})
        execute(tasks, args.output or ROOT / "runs/m4/demo_01", configuration(args.device, args.batch_size))


if __name__ == "__main__":
    main()
