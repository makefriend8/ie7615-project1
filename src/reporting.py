"""Validate classification artifacts and export comparison data and figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import MODEL_NAMES


def _read_json(path):
    """Read one structured experiment artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_experiment(run_dir):
    """Validate three completed models and export their numerical comparison.

    The returned selection records the minimum-validation-loss ranking.
    All generated artifacts are written inside the supplied experiment directory.
    """
    run = Path(run_dir)
    results = []
    reference_manifest = (run / "split_manifest.csv").read_text(encoding="utf-8")
    reference_classes = _read_json(run / "experiment.json")["class_names"]
    for name in MODEL_NAMES:
        folder = run / name
        # Verify that each model has all required results and figures.
        for filename in ("learning_curves.png", "confusion_matrix.png", "history.json",
                         "epochs.csv", "model_summary.txt", "classification_report.json",
                         "environment.json", "reload_verification.json"):
            if not (folder / filename).is_file():
                raise FileNotFoundError(folder / filename)
        result = _read_json(folder / "metrics.json")
        if result["model"] != name:
            raise ValueError("Model directory and metrics disagree")
        if not _read_json(folder / "reload_verification.json")["passed"]:
            raise ValueError(f"Checkpoint reload failed: {name}")
        if _read_json(folder / "class_names.json") != reference_classes:
            raise ValueError("Model class mappings differ")
        if (folder / "split_manifest.csv").read_text(encoding="utf-8") != reference_manifest:
            raise ValueError("Models were evaluated on different splits")
        predictions = pd.read_csv(folder / "predictions.csv", dtype={"identity": str})
        actual = predictions.class_index.to_numpy()
        predicted = predictions.predicted_index.to_numpy()
        correct = int((actual == predicted).sum())
        if correct != result["correct"] or len(actual) != result["test_count"]:
            raise ValueError("Prediction counts and metrics disagree")
        if not np.isclose(correct / len(actual), result["test_accuracy"]):
            raise ValueError("Prediction accuracy and metrics disagree")
        recomputed = np.zeros((len(reference_classes), len(reference_classes)), dtype=int)
        np.add.at(recomputed, (actual, predicted), 1)
        stored = pd.read_csv(folder / "confusion_matrix.csv", index_col=0).to_numpy()
        if not np.array_equal(recomputed, stored):
            raise ValueError("Predictions and confusion matrix disagree")
        results.append(result)
    table = pd.DataFrame(results)
    table.to_csv(run / "comparison.csv", index=False)
    best = min(results, key=lambda r: (r["best_val_loss"], r["trainable_parameters"]))
    selection = dict(model=best["model"], criterion="minimum validation loss; trainable parameters break ties",
                     checkpoint=f"{best['model']}/best.weights.h5", class_names=reference_classes)
    (run / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    # Export a compact Markdown table for direct viewing on GitHub.
    rows = ["| Model | Total parameters | Trainable parameters | Fit time (s) | Inference (ms) | Validation loss | Test correct | Test accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        rows.append(
            f"| {result['model']} | {result['total_parameters']:,} | "
            f"{result['trainable_parameters']:,} | {result['training_seconds']:.2f} | "
            f"{result['inference_ms_median']:.2f} | {result['best_val_loss']:.4f} | "
            f"{result['correct']}/{result['test_count']} | {result['test_accuracy']:.2%} |")
    (run / "comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Combine the training and validation histories into one figure.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    figure, axes = plt.subplots(3, 2, figsize=(10, 8))
    per_class, confusions, mistakes = [], [], []
    for row, name in enumerate(MODEL_NAMES):
        folder = run / name
        history = _read_json(folder / "history.json")
        for col, metric in enumerate(("loss", "accuracy")):
            axis = axes[row, col]
            epochs = range(1, len(history[metric]) + 1)
            axis.plot(epochs, history[metric], label="train")
            axis.plot(epochs, history["val_" + metric], label="validation")
            axis.set(title=name, xlabel="Epoch", ylabel=metric)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(fontsize=8)
        matrix = pd.read_csv(folder / "confusion_matrix.csv", index_col=0).to_numpy()
        for i, identity in enumerate(reference_classes):
            count, correct = int(matrix[i].sum()), int(matrix[i, i])
            per_class.append(dict(model=name, identity=identity, correct=correct,
                                  support=count, accuracy=correct / count if count else None))
            for j, predicted_identity in enumerate(reference_classes):
                if i != j and matrix[i,j] > 0:
                    confusions.append(dict(model=name, true_identity=identity,
                                           predicted_identity=predicted_identity,
                                           count=int(matrix[i,j])))
        predictions = pd.read_csv(folder / "predictions.csv", dtype={"identity": str})
        wrong = predictions[predictions.class_index != predictions.predicted_index].copy()
        wrong.insert(0, "model", name)
        mistakes.append(wrong)
    figure.tight_layout()
    figure.savefig(run / "comparison_curves.png")
    plt.close(figure)
    pd.DataFrame(per_class).to_csv(run / "per_class_accuracy.csv", index=False)
    pd.DataFrame(confusions, columns=["model", "true_identity", "predicted_identity", "count"]).to_csv(
        run / "confusion_pairs.csv", index=False)
    pd.concat(mistakes, ignore_index=True).to_csv(run / "misclassified_images.csv", index=False)
    return table, selection
