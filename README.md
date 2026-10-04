# Celebrity Identification and Detection

This project compares a custom CNN, a frozen ResNet50V2 feature extractor and a fine-tuned ResNet50V2 for classifying six CelebA identities using TensorFlow/Keras.

## Dataset

The selected CelebA IDs are **2619, 797, 2970, 4422, 3 and 2336**, chosen from the finalized class identity pool to include three male and three female identities. Each identity contributes 25 images: 17 for training, four for validation and four for testing, giving **102/24/24 images** overall.

Images are resized to 224 x 224 pixels and normalized for each model. Training uses horizontal flips and rotations. All models share the same data split, with seed 42.

## Notebooks

Notebooks 01–05 include saved outputs. Start with **05** for the model comparison, learning curves and class-level results.

| Notebook | Contents |
|---|---|
| [01 - Data preparation](notebooks/01_data_preparation.ipynb) | Identity selection, data split and image checks |
| [02 - Custom CNN](notebooks/02_train_custom_cnn.ipynb) | Model design, training and evaluation |
| [03 - Frozen ResNet50V2](notebooks/03_train_resnet_frozen.ipynb) | Training and evaluation with a frozen backbone |
| [04 - Fine-tuned ResNet50V2](notebooks/04_train_resnet_finetuned.ipynb) | Backbone fine-tuning and evaluation |
| [05 - Results analysis](notebooks/05_results_analysis.ipynb) | Model comparison, curves and confusion matrix |

Fine-tuned ResNet50V2 was selected by validation loss and achieved **23/24 correct test predictions (95.83%)**. Notebook 05 reports parameter counts, training time and accuracy for all three models.

## Milestone 2 Detection Data

[06 - Detection data preparation](notebooks/06_detection_data_preparation.ipynb) builds a 13-identity dataset from the shared CelebA portraits and shows the split, YOLO labels, counts, and annotated examples. It generates 234 synthetic images with 2–5 or 9 identities per frame and 1,065 labeled portrait regions. Four additional 3 × 3 grids are generated separately.

Notebook 06 reads the shared portraits from `data/shared_pool/` and writes the generated dataset to `data/m2_detection/`. The six-identity Milestone 1 dataset uses `data/selected_subset/`.

The repository includes the generated train/validation/test images and YOLO labels under `data/m2_detection/`, with manifests in `m2_artifacts/`. [M2_DATASET.md](M2_DATASET.md) describes the source split, composition, augmentations, annotations, and audit.

## Repository Structure

```text
notebooks/          Data preparation, training and result analysis
src/                Dataset utilities, models and result aggregation
utils.py            Random seed and reproducibility settings
requirements.txt    Python dependencies
m2_artifacts/       Detection dataset manifests and generation settings
data/m2_detection/  Generated train/val/test images and YOLO labels
```

Experiments write metrics, predictions, training logs and figures to `runs/<run_id>/`.

## Reproduce the Experiment

Use Python 3.10 and install the dependencies from the repository root:

```bash
pip install -r requirements.txt
jupyter lab
```

Place the CelebA source files at:

```text
data/raw/identity_CelebA.txt
data/raw/img_align_celeba/<image files>
```

Run notebooks **01-05 in order**. Notebook 01 prepares the split and creates the experiment directory; 02-04 train the models; 05 summarizes the saved results. Run 01 once per experiment. Save each notebook's outputs and shut down its kernel before starting the next training notebook to release GPU memory. Experiment settings are in `src/config.py`.

