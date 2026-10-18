# Celebrity Identification and Detection

This project covers celebrity classification, synthetic detection data, YOLOv8 training and integrated single/multi-person inference. Milestone 4 combines a 13-class ResNet50V2 classifier with the selected YOLOv8s detector.

## Dataset

Milestone 1 uses six identities; Milestones 2-4 use the 13-identity shared portrait pool. The generated detection images and labels are included in `data/m2_detection/`. Original portraits are supplied separately. [M2_DATASET.md](M2_DATASET.md) documents the source split, composition and annotations.

## Notebooks

| Notebook | Contents |
|---|---|
| 01-05; start with [05 - Results analysis](notebooks/05_results_analysis.ipynb) | Original classifier preparation, training and comparison |
| [06 - Detection data preparation](notebooks/06_detection_data_preparation.ipynb) | Synthetic data generation and audit |
| [07 - YOLOv8 detection](notebooks/07_yolov8_detection.ipynb) | Detector training, selection and evaluation |
| [08 - M4 classifier training](notebooks/08_m4_classifier_training.ipynb) | Thirteen-class training and validation |
| [09 - M4 cascade validation](notebooks/09_m4_cascade_validation.ipynb) | Classifier review of YOLO predicted crops |
| [10 - M4 final evaluation](notebooks/10_m4_final_evaluation.ipynb) | Integrated test results and ablations |

## Repository Structure

```text
notebooks/          Preparation, training and analysis (01-10)
src/                Data utilities, models, inference and evaluation
requirements*.txt   Classification, detection and notebook dependencies
data/m2_detection/  Generated detection images and YOLO labels
m2_artifacts/       Source/object manifests and identity mapping
m3_artifacts/       Selected detector and evaluation records
m3_gallery/         Detector prediction examples
m4_artifacts/       Frozen integration policy and final results
m4_gallery/         Correct/incorrect portraits and multi-person predictions
runs/               Training records and validation replay evidence
```

## Setup and Reproduction

Use Python 3.10. Run the Bash commands below from the server project root, `ie7615-project1/` (the directory containing `src/`). Install a device-compatible PyTorch build, then:

```bash
python -m pip install -r requirements.txt -r requirements-m3.txt
jupyter lab
```

For M1, place `identity_CelebA.txt` and aligned images under `data/raw/identity_CelebA.txt` and `data/raw/img_align_celeba/`, then run notebooks 01-05 in order. For M2, provide the shared portraits in `data/shared_pool/` and run notebook 06. Existing experiments and generated data should be preserved.

For M3, verify the saved experiment, or run all stages in a clean experiment directory:

```bash
python -m src.m3_reproduce --device 0
```

### M4 classifier training

[Notebook 08](notebooks/08_m4_classifier_training.ipynb) calls `src.m4_classifier.train()`. It uses the original portraits in `data/shared_pool/`, generated images in `data/m2_detection/` and manifests in `m2_artifacts/`: 217 training portraits and 715 ground-truth crops, with equal loss weight per domain. ResNet50V2 trains a classification head, then fine-tunes the backbone; the lowest weighted validation loss across both stages selects `best.weights.h5`. No test data is used for training or checkpoint selection.

For first-time training, use the same output directory as Notebook 08 and the inference code. Run this only when `runs/m4/classifier13/` does not exist; the training script preserves existing runs:

```bash
python -m src.m4_classifier \
  --source data/shared_pool \
  --output runs/m4/classifier13 \
  --batch-size 8 --warmup-epochs 15 --finetune-epochs 30 --seed 42
```

This generates `runs/m4/classifier13/best.weights.h5`, alongside `class_names.json`, `data_counts.json`, training logs and `selection.json`. These files already exist on the training server, so skip training there. A new training run may produce different weights; the saved integration policy verifies the original checkpoint hashes.

### M4 integrated inference

Single mode classifies the whole portrait; multi mode detects regions, classifies retained crops and applies the frozen identity rule. On the training server, the checkpoints already exist at `m3_artifacts/best.pt` and `runs/m4/classifier13/best.weights.h5`. Verify the classifier in place:

```bash
python -m src.m4_checkpoint
```

Use these existing server test images for a runnable example:

```bash
python -m src.m4_pipeline infer \
  --single data/shared_pool/3/003206.jpg \
  --multi data/m2_detection/images/test/test_0004.jpg \
  --output "runs/m4/demo_$(date +%Y%m%d_%H%M%S_%N)" \
  --device 0
```

Either input option may be omitted. Here `--output` saves predictions and annotated images only; inference loads the model from `runs/m4/classifier13/best.weights.h5`. The timestamp keeps each demo's results separate. For new inputs, use their actual project-relative filenames. The system supports 13 known identities and does not reject unknown people.

### M4 final evaluation

```bash
python -m src.m4_pipeline evaluate --source data/shared_pool
```

Evaluation additionally requires the original hash-matching portraits and complete `runs/m4/cascade_validation/` evidence. Completed results are verified and reused. See the [final gallery](m4_gallery/README.md) for prediction examples and error analysis.
