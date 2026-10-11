# Project 1: Celebrity Classification and Detection

This repository contains the work for Milestones 1–3. Milestone 1 compares classifiers for six CelebA identities; Milestone 2 prepares a labeled detection dataset covering 13 identities; Milestone 3 trains and evaluates a YOLOv8 detector on that dataset. The notebooks include saved experiment outputs.

## Notebooks

| File | What it shows |
|---|---|
| [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb) | Selects the six Milestone 1 identities and prepares their train/validation/test split. |
| [02_train_custom_cnn.ipynb](notebooks/02_train_custom_cnn.ipynb) | Trains and evaluates the custom CNN. |
| [03_train_resnet_frozen.ipynb](notebooks/03_train_resnet_frozen.ipynb) | Trains and evaluates the frozen ResNet50V2 feature extractor. |
| [04_train_resnet_finetuned.ipynb](notebooks/04_train_resnet_finetuned.ipynb) | Fine-tunes and evaluates ResNet50V2. |
| [05_results_analysis.ipynb](notebooks/05_results_analysis.ipynb) | Compares the three classifiers, learning curves, and class-level results. |
| [06_detection_data_preparation.ipynb](notebooks/06_detection_data_preparation.ipynb) | Builds and audits the synthetic YOLO dataset, with annotated examples. |
| [07_yolov8_detection.ipynb](notebooks/07_yolov8_detection.ipynb) | Shows three YOLOv8 runs, validation-based selection, test metrics, predicted examples, and image inference. |

## Code

| File | Role |
|---|---|
| [src/config.py](src/config.py) | Fixed Milestone 1 identities and training settings. |
| [src/data_utils.py](src/data_utils.py) | CelebA image selection, split preparation, and data checks. |
| [src/experiment.py](src/experiment.py) | Milestone 1 run creation, source snapshot, and input checks. |
| [src/models.py](src/models.py) | Classification models, training, checkpoint saving, and evaluation. |
| [src/reporting.py](src/reporting.py) | Validates classification results and writes comparison data, a Markdown table, and figures. |
| [utils.py](utils.py) | Random seeds and deterministic TensorFlow operations for classification. |
| [src/m2_sources.py](src/m2_sources.py) | Inventories shared portraits and assigns source images to splits. |
| [src/m2_detection.py](src/m2_detection.py) | Composes multi-person images, writes YOLO labels, and audits the dataset. |
| [src/m3_protocol.py](src/m3_protocol.py) | Checks the detection dataset and supports box matching and IoU calculations. |
| [src/m3_training.py](src/m3_training.py) | Trains the YOLOv8 sweep and selects a checkpoint using validation metrics. |
| [src/m3_evaluation.py](src/m3_evaluation.py) | Computes held-out test metrics and saves ten images with predicted boxes. |
| [src/m3_reproduce.py](src/m3_reproduce.py) | Runs the Milestone 3 stages or verifies the saved result. |
| [demo_m3.py](demo_m3.py) | Runs the selected detector on one image and saves an annotated prediction. |

## Datasets and results

| Location | Contents |
|---|---|
| `runs/20260926T151014Z_08ffd273/` | Milestone 1 split, model histories and metrics, comparison table and figures, and source snapshot. The six IDs are 2619, 797, 2970, 4422, 3, and 2336. |
| `data/m2_detection/images/{train,val,test}/` and `labels/{train,val,test}/` | Synthetic detection dataset: 156/39/39 images and matching YOLO label files. |
| `m2_artifacts/source_manifest.csv` and `object_manifest.csv` | Source-image split assignments and the source portrait and box used for each placed object. |
| `m2_artifacts/identity_mapping.json` and `generation_config.json` | YOLO class-to-identity mapping and dataset-generation settings. [M2_DATASET.md](M2_DATASET.md) explains the dataset and annotation convention. |
| `runs/m3/{lr001_img640,lr0005_img640,lr001_img512}/` | The three YOLOv8 runs; each contains `args.yaml`, `results.csv`, `results.png`, and `validation_metrics.json`. `runs/m3/environment.json` records the software and hardware environment. |
| `m3_artifacts/best.pt` and `selection.json` | Selected detector checkpoint and the validation-based selection record; the selected run is `lr0005_img640`. |
| `m3_artifacts/validation_comparison.csv` | Validation metrics for all three configurations. |
| `m3_artifacts/test_metrics.json` and `per_class_metrics.csv` | Overall and per-identity metrics on the held-out test split. |
| `m3_artifacts/per_image_review.json` and `test_predictions.json` | Per-image match counts and predicted boxes used to inspect detection outcomes. |
| [m3_gallery/gallery.md](m3_gallery/gallery.md) | Ten held-out test images with predicted identity boxes and captions describing correct detections and errors. |
| [requirements.txt](requirements.txt) and [requirements-m3.txt](requirements-m3.txt) | Classification and YOLOv8 Python dependencies, respectively. |

## Run the project

Use Python 3.10 and run commands from the repository root.

For Milestone 1, install `requirements.txt`, place `identity_CelebA.txt` at `data/raw/identity_CelebA.txt` and aligned images under `data/raw/img_align_celeba/`, then run notebooks 01–05 in order. For Milestone 2, the generated dataset is included; rebuilding it requires the original portraits under `data/shared_pool/<identity ID>/` and notebook 06.

For Milestone 3, install a PyTorch build compatible with your device, then install `requirements-m3.txt`. The command below checks the saved checkpoint and test result. In a fresh copy with the Milestone 2 dataset but without Milestone 3 results, it runs training, validation-based selection, and test evaluation:

```bash
python -m src.m3_reproduce --device 0
```

Run the single-image demo with a held-out test image:

```bash
python demo_m3.py data/m2_detection/images/test/test_0004.jpg --device 0
```

The demo prints identity IDs, confidence scores, and boxes, and saves an annotated image. Use `--device cpu` for demo inference without a GPU.
