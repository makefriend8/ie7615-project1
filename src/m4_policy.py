"""Frozen M4 post-processing rules selected using validation predictions only."""

import copy
import math

from .m4_cascade_validation import box_iou


def suppress_duplicates(predictions, threshold=0.7):
    """Apply class-agnostic post-NMS to existing YOLO outputs, using YOLO scores.

    This is an additional pass after the original class-aware YOLO NMS; it does
    not change the detector's original predict settings or recover suppressed
    detections. It never reads ground truth or classifier identities.
    """
    if not 0 < threshold < 1:
        raise ValueError("Post-NMS IoU must lie between zero and one")
    kept = []
    for prediction in sorted(predictions, key=lambda p: -p["yolo_confidence"]):
        if all(box_iou(prediction["xyxy"], other["xyxy"]) <= threshold for other in kept):
            kept.append(copy.deepcopy(prediction))
    return kept


def choose_identity(prediction, threshold=0.9):
    """Resolve conflicts using only the classifier's own confidence threshold.

    Detector and classifier scores are not compared with each other. Single
    portraits always use the classifier directly and do not use this rule.
    """
    confidence = prediction["classifier_confidence"]
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid classifier confidence")
    if prediction["classifier_identity"] == prediction["yolo_identity"]:
        return prediction["classifier_identity"], "agreement"
    if confidence >= threshold:
        return prediction["classifier_identity"], "classifier_confident"
    return prediction["yolo_identity"], "yolo_fallback"


def apply_policy(frames, policy):
    """Replay a frozen rule on saved predictions without inference or truth use."""
    result = copy.deepcopy(frames)
    for frame in result:
        frame["predictions"] = suppress_duplicates(frame["predictions"], policy["post_nms_iou"])
        for prediction in frame["predictions"]:
            identity, reason = choose_identity(prediction, policy["classifier_conflict_threshold"])
            prediction["final_identity"] = identity
            prediction["identity_rule"] = reason
    return result
