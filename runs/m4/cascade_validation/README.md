# M4 validation cascade review

Only the 39 synthetic validation frames (175 labeled portrait regions) are evaluated. A separate restoration check repeats the existing 50 portrait / 175 ground-truth-crop validation predictions. No training or test images are read.

YOLO-only and classifier-on-YOLO-crops use identical float predicted boxes and class-agnostic one-to-one matching at IoU >= 0.5. Identity micro precision/recall/F1 include missed and extra predictions; they are fixed-threshold metrics, not mAP. Wrong identity counts as both an identity FP and FN. Conditional identity accuracy uses only localization matches.

All 39 comparison images show predicted boxes only: green means localized and identity-correct; red means wrong identity or unmatched. Missing objects remain in per_image.csv and predictions.json. These are validation review images, not the final test gallery. YOLO confidence and classifier softmax confidence are different scores and should not be directly compared as calibrated probabilities.

predictions.json retains all 13 classifier probabilities in run_config.json class order, ground truth, crop bounds, predicted boxes and matching indices. input_snapshot/ records the actual code and manifests. Worker logs and separate environment files support debugging. No identity fusion policy is selected automatically.
