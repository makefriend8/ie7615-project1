"""Train a 13-identity classifier for original portraits and detected crops."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "m2_artifacts" / "source_manifest.csv"
OBJECT_MANIFEST = ROOT / "m2_artifacts" / "object_manifest.csv"
MAPPING = ROOT / "m2_artifacts" / "identity_mapping.json"
DETECTION_DATA = ROOT / "data" / "m2_detection"
IMAGE_SIZE = (224, 224)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _source_path(source_dir, row):
    """Resolve a manifest image in the original or nested identity folder."""
    relative = Path(row["relative_path"])
    candidates = (source_dir / relative, source_dir / row["identity"] / relative)
    paths = [path for path in candidates if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected one source image for {relative}; found {paths}")
    return paths[0]


def _pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def _load_examples(source_dir):
    """Load only train/validation examples from the fixed M2 source split."""
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    names = {int(index): str(identity) for index, identity in mapping.items()}
    if set(names) != set(range(13)) or len(set(names.values())) != 13:
        raise ValueError("Expected the fixed 13-identity M2 mapping")
    source_rows = _read_csv(SOURCE_MANIFEST)
    object_rows = _read_csv(OBJECT_MANIFEST)
    if Counter(row["split"] for row in source_rows) != {
            "train": 217, "val": 50, "test": 52}:
        raise ValueError("Unexpected source split counts")
    if Counter(row["split"] for row in object_rows) != {
            "train": 715, "val": 175, "test": 175}:
        raise ValueError("Unexpected object split counts")

    # Each synthetic crop inherits its source portrait's split and identity.
    source_lookup = {row["relative_path"]: row for row in source_rows}
    if len(source_lookup) != len(source_rows):
        raise ValueError("Duplicate source path in the manifest")
    for row in object_rows:
        source = source_lookup.get(row["source_relative_path"])
        if (source is None or source["split"] != row["split"] or
                source["identity"] != row["identity"] or
                source["pixel_sha256"] != row["pixel_sha256"] or
                int(row["class_index"]) != int(source["class_index"])):
            raise ValueError(f"Object crosses a source split or identity: {row['frame']}")

    examples = {"train": [], "val": []}
    for row in source_rows:
        if row["split"] == "test":
            continue
        if names[int(row["class_index"])] != row["identity"]:
            raise ValueError("Source class index differs from the identity mapping")
        path = _source_path(source_dir, row)
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            if (rgb.size != (int(row["width"]), int(row["height"])) or
                    _pixel_hash(rgb) != row["pixel_sha256"]):
                raise ValueError(f"Source portrait differs from the manifest: {path}")
            pixels = np.asarray(rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR),
                                dtype=np.uint8)
        examples[row["split"]].append(
            (pixels, int(row["class_index"]), "portrait", row["relative_path"]))

    for row in object_rows:
        if row["split"] == "test":
            continue
        frame = DETECTION_DATA / row["frame"]
        if not frame.is_file():
            raise FileNotFoundError(frame)
        box = tuple(int(row[key]) for key in ("x1", "y1", "x2", "y2"))
        with Image.open(frame) as image:
            if image.size != (640, 640) or not (
                    0 <= box[0] < box[2] <= 640 and
                    0 <= box[1] < box[3] <= 640):
                raise ValueError(f"Invalid M2 crop: {frame}, {box}")
            pixels = np.asarray(
                image.convert("RGB").crop(box).resize(
                    IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)
        key = f"{row['frame']}:{','.join(map(str, box))}"
        examples[row["split"]].append(
            (pixels, int(row["class_index"]), "crop", key))
    return examples, [names[index] for index in range(13)]


def _arrays(examples):
    images = np.stack([item[0] for item in examples])
    labels = np.asarray([item[1] for item in examples], dtype=np.int32)
    domains = [item[2] for item in examples]
    keys = [item[3] for item in examples]
    counts = Counter(domains)
    if set(counts) != {"portrait", "crop"}:
        raise ValueError("Expected both original portraits and synthetic crops")
    # Equal total weight prevents the more numerous crops from dominating loss.
    weights = np.asarray(
        [len(examples) / (2 * counts[domain]) for domain in domains],
        dtype=np.float32)
    return images, labels, weights, domains, keys


def _dataset(tf, images, labels, weights, batch_size, training, seed):
    data = tf.data.Dataset.from_tensor_slices(
        (images.astype(np.float32), labels, weights))
    if training:
        data = data.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _callbacks(tf, checkpoint, log, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience,
            restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(log)),
    ]


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_predictions(model, images, labels, domains, keys, names, batch_size):
    probabilities = model.predict(images.astype(np.float32),
                                  batch_size=batch_size, verbose=0)
    predicted = probabilities.argmax(axis=1)
    rows = []
    summary = {}
    for domain in ("portrait", "crop"):
        indices = [index for index, value in enumerate(domains) if value == domain]
        correct = sum(int(predicted[index] == labels[index]) for index in indices)
        summary[domain] = {
            "images": len(indices), "correct": correct,
            "accuracy": correct / len(indices),
        }
    for index, key in enumerate(keys):
        rows.append({
            "domain": domains[index], "image": key,
            "true_identity": names[int(labels[index])],
            "predicted_identity": names[int(predicted[index])],
            "confidence": float(probabilities[index, predicted[index]]),
            "correct": bool(predicted[index] == labels[index]),
        })
    return rows, summary


def train(source_dir, output, batch_size=8, warmup_epochs=15,
          finetune_epochs=30, seed=42):
    """Train only on train data and select one checkpoint using validation loss."""
    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing run: {output}")
    examples, names = _load_examples(source_dir)
    train_data = _arrays(examples["train"])
    val_data = _arrays(examples["val"])
    train_images, train_labels, train_weights, train_domains, _ = train_data
    val_images, val_labels, val_weights, val_domains, val_keys = val_data

    import tensorflow as tf
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    from .models import build_resnet
    train_ds = _dataset(tf, train_images, train_labels, train_weights,
                        batch_size, True, seed)
    val_ds = _dataset(tf, val_images, val_labels, val_weights,
                      batch_size, False, seed)
    output.mkdir(parents=True)
    _write_json(output / "class_names.json", names)
    _write_json(output / "data_counts.json", {
        "train": dict(Counter(train_domains)),
        "val": dict(Counter(val_domains)),
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "object_manifest_sha256": _sha256(OBJECT_MANIFEST),
    })
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "gpu": [device.name for device in tf.config.list_physical_devices("GPU")],
    })

    frozen = build_resnet(len(names), mode="frozen")
    warm_checkpoint = output / "warmup_best.weights.h5"
    warm_history = frozen.fit(
        train_ds, validation_data=val_ds, epochs=warmup_epochs,
        callbacks=_callbacks(tf, warm_checkpoint, output / "warmup_epochs.csv", 4))
    frozen.load_weights(str(warm_checkpoint))

    # Fine-tuning starts from the best frozen checkpoint and keeps BatchNorm
    # inference statistics, as in the M1 ResNet50V2 classifier.
    finetuned = build_resnet(len(names), mode="finetune", weights=None)
    finetuned.set_weights(frozen.get_weights())
    fine_checkpoint = output / "finetune_best.weights.h5"
    fine_history = finetuned.fit(
        train_ds, validation_data=val_ds, epochs=finetune_epochs,
        callbacks=_callbacks(tf, fine_checkpoint, output / "finetune_epochs.csv", 6))

    warm_loss = min(float(value) for value in warm_history.history["val_loss"])
    fine_loss = min(float(value) for value in fine_history.history["val_loss"])
    stage = "finetune" if fine_loss < warm_loss else "warmup"
    chosen = fine_checkpoint if stage == "finetune" else warm_checkpoint
    checkpoint = output / "best.weights.h5"
    shutil.copyfile(chosen, checkpoint)
    finetuned.load_weights(str(checkpoint))
    rows, validation = _validation_predictions(
        finetuned, val_images, val_labels, val_domains, val_keys, names, batch_size)
    with (output / "validation_predictions.csv").open(
            "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=(
            "domain", "image", "true_identity", "predicted_identity",
            "confidence", "correct"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output / "selection.json", {
        "selected_stage": stage, "selection_metric": "validation_loss",
        "validation_loss": min(warm_loss, fine_loss),
        "warmup_best_validation_loss": warm_loss,
        "finetune_best_validation_loss": fine_loss,
        "validation": validation, "class_names": names,
        "checkpoint_sha256": _sha256(checkpoint),
        "seed": seed, "batch_size": batch_size,
        "warmup_epochs_requested": warmup_epochs,
        "finetune_epochs_requested": finetune_epochs,
        "test_evaluated": False,
    })
    print(f"Selected {stage} checkpoint: {checkpoint}")
    for domain, metrics in validation.items():
        print(f"{domain} validation: {metrics['correct']}/{metrics['images']} "
              f"({metrics['accuracy']:.1%})")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "shared_pool")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "m4" / "classifier13")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.warmup_epochs, args.finetune_epochs) < 1:
        raise ValueError("Batch size and epoch limits must be positive")
    train(args.source, args.output, args.batch_size, args.warmup_epochs,
          args.finetune_epochs, args.seed)


if __name__ == "__main__":
    main()
