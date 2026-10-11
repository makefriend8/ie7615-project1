"""Train and select a YOLOv8 detector using M2 train/validation data."""

import argparse
import csv
import gc
import hashlib
import json
import platform
import shutil

from .m3_protocol import ARTIFACTS, ROOT, RUNS, write_data_yaml


# Each alternative changes one factor from the baseline.
SWEEP = {
    "lr001_img640": {"lr0": 0.001, "imgsz": 640},
    "lr0005_img640": {"lr0": 0.0005, "imgsz": 640},
    "lr001_img512": {"lr0": 0.001, "imgsz": 512},
}


def _environment(ultralytics, torch):
    """Record software and GPU details with the experiment."""
    return {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def train_run(name, settings, args, data_yaml, environment):
    """Train once and evaluate the saved best checkpoint on validation."""
    run_dir = RUNS / name
    checkpoint = run_dir / "weights" / "best.pt"
    metrics_path = run_dir / "validation_metrics.json"
    if run_dir.exists():
        if checkpoint.is_file() and metrics_path.is_file():
            print(f"Completed run preserved: {name}")
            return
        if (metrics_path.is_file() and
                all((run_dir / filename).is_file() for filename in
                    ("args.yaml", "results.csv", "results.png")) and
                _published_selection() is not None):
            print(f"Saved training record preserved: {name}")
            return
        raise FileExistsError(f"Incomplete run exists: {run_dir}; inspect and archive it before retrying")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml), project=str(RUNS), name=name, exist_ok=False,
        epochs=args.epochs, patience=args.patience, imgsz=settings["imgsz"],
        batch=args.batch, workers=args.workers, device=args.device,
        optimizer="AdamW", lr0=settings["lr0"], seed=42, deterministic=True,
        plots=True, save=True, val=True,
        # M2 already varies appearance and placement; online changes stay mild.
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
        translate=0.05, scale=0.10, fliplr=0.5,
        hsv_h=0.0, hsv_s=0.10, hsv_v=0.10,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training did not save best.pt: {checkpoint}")
    del model
    gc.collect()

    best = YOLO(str(checkpoint))
    metrics = best.val(
        data=str(data_yaml), split="val", imgsz=settings["imgsz"],
        device=args.device, plots=True, project=str(RUNS),
        name=f"{name}_validation", exist_ok=False,
    )
    values = {
        "run": name, "lr0": settings["lr0"], "imgsz": settings["imgsz"],
        "val_map50": float(metrics.box.map50),
        "val_map50_95": float(metrics.box.map),
        "val_precision": float(metrics.box.mp),
        "val_recall": float(metrics.box.mr),
        "checkpoint": str(checkpoint.relative_to(ROOT).as_posix()),
        "epochs_requested": args.epochs, "patience": args.patience,
        "batch": args.batch, "device": str(args.device), "optimizer": "AdamW",
        "pretrained_model": args.model, "environment": environment,
    }
    metrics_path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(f"Validation result saved: {metrics_path}")
    del best
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _published_selection():
    """Return the saved selection only when its checkpoint is intact."""
    selection_path = ARTIFACTS / "selection.json"
    checkpoint = ARTIFACTS / "best.pt"
    if not selection_path.is_file() or not checkpoint.is_file():
        return None
    record = json.loads(selection_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return record if digest == record["checkpoint_sha256"] else None


def select_model():
    """Select by validation mAP50-95 and copy weights for evaluation/demo."""
    results = []
    for name in SWEEP:
        path = RUNS / name / "validation_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Finish validation before selection: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result["run"] != name or result["lr0"] != SWEEP[name]["lr0"] or
                result["imgsz"] != SWEEP[name]["imgsz"]):
            raise ValueError(f"Sweep settings differ from the saved plan: {name}")
        results.append(result)
    fixed = ("batch", "epochs_requested", "patience", "pretrained_model", "optimizer")
    if any(tuple(row[key] for key in fixed) != tuple(results[0][key] for key in fixed)
           for row in results[1:]):
        raise ValueError("Sweep runs used different fixed training settings")
    ARTIFACTS.mkdir(exist_ok=True)
    columns = ("run", "lr0", "imgsz", "val_map50_95", "val_map50",
               "val_precision", "val_recall", "checkpoint")
    with (ARTIFACTS / "validation_comparison.csv").open(
            "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in results)

    selected = max(results, key=lambda row: (row["val_map50_95"], row["val_map50"]))
    source = ROOT / selected["checkpoint"]
    destination = ARTIFACTS / "best.pt"
    if not source.is_file():
        saved = _published_selection()
        if saved is None or saved["selected_run"] != selected["run"]:
            raise FileNotFoundError(f"Selected run checkpoint is unavailable: {source}")
        if (saved["validation"]["val_map50_95"] != selected["val_map50_95"] or
                saved["validation"]["val_map50"] != selected["val_map50"]):
            raise ValueError("Published selection differs from the sweep records")
        print(f"Published selection preserved: {selected['run']}")
        return saved
    shutil.copyfile(source, destination)
    record = {
        "selected_run": selected["run"],
        "selection_rule": "Highest validation mAP50-95; mAP50 breaks ties",
        "settings": {key: selected[key] for key in (
            "lr0", "imgsz", "batch", "device", "epochs_requested",
            "patience", "pretrained_model", "optimizer")},
        "validation": {key: selected[key] for key in (
            "val_map50_95", "val_map50", "val_precision", "val_recall")},
        "checkpoint_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "environment": selected["environment"],
    }
    (ARTIFACTS / "selection.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {selected['run']} by validation mAP50-95")
    return record


def main():
    """Run a selected training configuration or choose among completed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "select"))
    parser.add_argument("--run", choices=tuple(SWEEP) + ("all",), default="all")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.action == "select":
        select_model()
        return
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        raise ValueError("Epochs and batch must be positive; workers cannot be negative")
    import torch
    import ultralytics

    RUNS.mkdir(parents=True, exist_ok=True)
    data_yaml = write_data_yaml(RUNS / "data.yaml")
    environment = _environment(ultralytics, torch)
    environment_path = RUNS / "environment.json"
    if not environment_path.is_file():
        environment_path.write_text(
            json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    targets = SWEEP if args.run == "all" else {args.run: SWEEP[args.run]}
    for name, settings in targets.items():
        train_run(name, settings, args, data_yaml, environment)


if __name__ == "__main__":
    main()
