# Milestone 2: Synthetic Detection Dataset

## Dataset and identity mapping

The detection dataset uses the celebrity portraits contributed to the course's shared Drive. It contains 319 source images across 13 CelebA identities: `3, 7, 797, 1212, 2336, 2619, 2970, 4422, 4428, 5695, 7007, 8335, 10002`. Each identity has 23–25 source images. YOLO class indices 0–12 follow this listed order; the exact mapping is in `m2_artifacts/identity_mapping.json`.

Source photographs are assigned to training, validation, or test **before** any synthetic images are composed. The six identities included in Milestone 1 retain their saved source-image assignments. The other seven use a deterministic split targeting 70/15/15. The resulting source counts are 217 training, 50 validation, and 52 test images. A source photograph can appear in several composites within its assigned split, but never in composites from another split. All 13 identities appear in all three splits.

## Image construction

The dataset has 234 synthetic 640 × 640 images: 156 training, 39 validation, and 39 test images (66.7%, 16.7%, and 16.7%). Frames contain 2–5 or 9 distinct identities; the nine-person counts are 31/7/7 across the three splits. The generator varies portrait scale, position, and tile proportions over a blurred background from a source photograph in the same split. Four separate 3 × 3 grids each contain nine identities and use 36 distinct test-source portraits in total.

Appearance augmentations are horizontal flip (probability 0.5), brightness scaling (0.85–1.15), and contrast scaling (0.90–1.10). Portraits are placed at 0.78–0.98 of their tile size with randomized offsets. These changes vary orientation, lighting, scale, and position; the seed and parameter ranges are saved in `m2_artifacts/generation_config.json`.

Tiling keeps the portraits separate, and each placed crop's boundaries determine its YOLO box.

## YOLO annotations and checks

Every image in `data/m2_detection/images/{train,val,test}/` has a same-named text file in `data/m2_detection/labels/{train,val,test}/`. Each row has the YOLO format `class_id x_center y_center width height`, with coordinates normalized by the final image dimensions. The 234 images contain 1,065 labeled portrait regions. Each region contains one face; its box follows the placed crop boundary, which can include hair, shoulders, and part of the source photograph. The displayed boxes visualize these generated annotations, not model predictions.

The preparation notebook, `notebooks/06_detection_data_preparation.ipynb`, shows the source split, class mapping, counts, label examples, and annotated images, including a nine-person frame. The audit checks image-label pairing, class IDs, coordinate bounds, identity coverage, and source-image separation across splits. The four additional grids are saved without boxes, with paired annotated previews for inspection.

## Repository files

The completed images and labels are included under `data/m2_detection/`. The notebook and `src/m2_sources.py` and `src/m2_detection.py` document the preparation process. After auditing the dataset, the notebook copies the source manifest, object manifest, identity mapping, and generation settings to `m2_artifacts/`. The source manifest records each original image's identity and split; the object manifest records the source image and pixel box for each placed portrait. The original shared-Drive portraits are not included in the repository.
