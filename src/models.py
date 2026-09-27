"""Classification models and auditable, per-run experiment artifacts."""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from .data_utils import ROOT, audit_subset


def _json(path, value):
    """Write experiment metadata as readable UTF-8 JSON."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_datasets(data_dir=ROOT / "data/selected_subset", batch_size=8, img_size=(224, 224), seed=42):
    """Audit before loading; return datasets, actual class names and file manifest."""
    manifest, classes = audit_subset(data_dir)
    datasets = []
    for split in ("train", "val", "test"):
        ds = tf.keras.utils.image_dataset_from_directory(
            Path(data_dir) / split, class_names=classes, image_size=img_size,
            batch_size=None, shuffle=False)
        ds = ds.cache()
        if split == "train":
            ds = ds.shuffle(len(manifest[manifest.split == split]), seed=seed,
                            reshuffle_each_iteration=True)
        datasets.append(ds.batch(batch_size).prefetch(tf.data.AUTOTUNE))
    return (*datasets, classes, manifest)


def build_custom_cnn(num_classes):
    """Two convolution blocks; Flatten yields 11,963,782 parameters for six classes."""
    model = models.Sequential([
        layers.Input((224, 224, 3)), layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),  # +/-36 degrees, only during training.
        layers.Rescaling(1./255),  # Normalize pixel values to [0, 1].
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(64, activation="relu"),
        layers.Dropout(0.5), layers.Dense(num_classes, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_resnet(num_classes, mode="frozen", weights="imagenet"):
    """Build a ResNet50V2 classifier initialized independently from ImageNet weights.

    weights=None supports offline reconstruction before loading saved weights.
    BatchNorm uses inference statistics even when convolution weights are trainable.
    """
    if mode not in {"frozen", "finetune"}:
        raise ValueError("mode must be frozen or finetune")
    base = applications.ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights=weights)
    base.trainable = mode == "finetune"
    inputs = layers.Input((224, 224, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = applications.resnet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    model = models.Model(inputs, layers.Dense(num_classes, activation="softmax")(x))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4 if mode == "finetune" else 1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(model, model_name, train_ds, val_ds, test_ds,
                       class_names, manifest, run_dir, epochs=20, seed=42):
    """Save the best validation-loss checkpoint and evaluate it with fixed BatchNorm statistics.

    Fit time measures the model.fit call, including tracing, validation and callbacks.
    Caller sets the seed before both dataset and model creation for each variant.
    """
    folder = Path(run_dir) / model_name
    folder.mkdir(exist_ok=False)
    # Record the environment for each model, including the physical GPU model.
    devices = tf.config.list_physical_devices("GPU")
    _json(folder / "environment.json", dict(
        python=platform.python_version(), tensorflow=tf.__version__,
        keras=tf.keras.__version__, numpy=np.__version__,
        gpu=[tf.config.experimental.get_device_details(d).get("device_name", d.name)
             for d in devices], deterministic_operations=True))
    manifest.to_csv(folder / "split_manifest.csv", index=False)
    _json(folder / "class_names.json", class_names)
    summary = []
    model.summary(print_fn=lambda line, **kwargs: summary.append(line))
    (folder / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    checkpoint = folder / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min",
                                           save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(folder / "tensorboard")),
        tf.keras.callbacks.CSVLogger(str(folder / "epochs.csv"))]
    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    elapsed = time.perf_counter() - start
    values = {key: list(map(float, val)) for key, val in history.history.items()}
    _json(folder / "history.json", values)
    model.load_weights(str(checkpoint))
    # Test evaluation uses the checkpoint selected by validation loss.
    evaluation = model.evaluate(test_ds, verbose=0, return_dict=True)
    probabilities = model.predict(test_ds, verbose=0)
    truth = np.concatenate([labels.numpy() for _, labels in test_ds])
    predicted = probabilities.argmax(axis=1)
    test_rows = manifest[manifest.split == "test"].sort_values(["identity", "filename"]).copy()
    if len(test_rows) != len(truth) or not np.array_equal(test_rows.class_index.to_numpy(), truth):
        raise ValueError("Test file order does not match dataset label order")
    test_rows["predicted_index"] = predicted
    test_rows["predicted_identity"] = [class_names[i] for i in predicted]
    for i in range(len(class_names)):
        test_rows[f"probability_{i}"] = probabilities[:, i]
    test_rows.to_csv(folder / "predictions.csv", index=False)
    accuracy = float(np.mean(truth == predicted))
    if not np.isclose(accuracy, evaluation["accuracy"], atol=1e-6):
        raise ValueError("evaluate() and predict() accuracy disagree")
    cm = confusion_matrix(truth, predicted, labels=range(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(folder / "confusion_matrix.csv")
    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", ax=axis, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    axis.set(xlabel="Predicted CelebA ID", ylabel="True CelebA ID", title=model_name)
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png")
    plt.close(fig)
    report = classification_report(truth, predicted, labels=range(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    _json(folder / "classification_report.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(range(1, len(values[metric])+1), values[metric], label="train")
        axis.plot(range(1, len(values[metric])+1), values["val_"+metric], label="validation")
        axis.set(xlabel="Epoch", ylabel=metric)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend()
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png")
    plt.close(fig)
    # Measure synchronized, batch-one forward latency after warm-up.
    # Time the forward pass on an already decoded input tensor.
    sample = next(iter(test_ds))[0][:1]
    for _ in range(10):
        model(sample, training=False).numpy()
    timings = []
    for _ in range(30):
        begin = time.perf_counter()
        model(sample, training=False).numpy()
        timings.append((time.perf_counter() - begin) * 1000)
    result = dict(model=model_name,
                  inference_ms_median=float(np.median(timings)),
                  inference_repeats=30, inference_batch_size=1, total_parameters=int(model.count_params()),
                  trainable_parameters=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                  training_seconds=elapsed, epochs_completed=len(history.epoch),
                  best_epoch=int(np.argmin(values["val_loss"])+1),
                  best_val_loss=float(min(values["val_loss"])),
                  test_accuracy=accuracy, test_loss=float(evaluation["loss"]),
                  correct=int((truth == predicted).sum()), test_count=len(truth), seed=seed,
                  learning_rate=float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
                  batch_size=int(next(iter(train_ds))[0].shape[0]),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    _json(folder / "metrics.json", result)
    return result
