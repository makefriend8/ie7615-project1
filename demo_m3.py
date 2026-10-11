"""Run the Milestone 3 YOLOv8 detector on one image and save its predictions."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    """Load selected weights and print every detected identity and confidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image, including a held-out test frame or separate sample")
    parser.add_argument("--output", type=Path, default=ROOT / "m3_demo_prediction.jpg")
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    args = parser.parse_args()

    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    selection = json.loads((ROOT / "m3_artifacts/selection.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / "m3_artifacts/best.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint differs from selection.json")
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    result = model.predict(source=str(image), imgsz=int(selection["settings"]["imgsz"]),
                           conf=args.confidence, iou=args.nms_iou,
                           device=args.device, verbose=False)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(args.output))
    print("Annotated image:", args.output.resolve())
    for class_id, confidence, box in zip(
            result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(),
            result.boxes.xyxy.cpu().tolist()):
        print(f"ID {model.names[int(class_id)]}: confidence={confidence:.3f}, "
              f"box={[round(value, 1) for value in box]}")
    if len(result.boxes) == 0:
        print("No detections above the confidence threshold")


if __name__ == "__main__":
    main()
