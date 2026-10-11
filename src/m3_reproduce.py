"""Run the Milestone 3 training sweep, model selection, and test evaluation."""

import argparse
import hashlib
import json
import subprocess
import sys

from .m3_protocol import ARTIFACTS, ROOT


def main():
    """Run the documented stages in order from a fresh or partial experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--model", default="yolov8s.pt")
    args = parser.parse_args()
    test_path = ARTIFACTS / "test_metrics.json"
    if test_path.is_file():
        selection = json.loads((ARTIFACTS / "selection.json").read_text(encoding="utf-8"))
        checkpoint = ARTIFACTS / "best.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
            raise ValueError("Saved checkpoint differs from selection.json")
        result = json.loads(test_path.read_text(encoding="utf-8"))
        if (result["split"] != "test" or result["images"] != 39 or
                len(result["per_class"]) != 13):
            raise ValueError("Saved test record is incomplete")
        gallery = ROOT / "m3_gallery"
        if not gallery.exists():
            subprocess.run([sys.executable, "-m", "src.m3_evaluation", "gallery",
                            "--device", args.device], check=True)
        if len(list(gallery.glob("*.jpg"))) < 10:
            raise ValueError("Saved prediction gallery needs at least ten images")
        print(f"Saved M3 result: {selection['selected_run']}; "
              f"test mAP50-95={result['map50_95']:.3f}")
        return
    stages = [
        ["-m", "src.m3_training", "train", "--run", "all",
         "--device", args.device, "--batch", str(args.batch),
         "--workers", str(args.workers), "--epochs", str(args.epochs),
         "--patience", str(args.patience), "--model", args.model],
        ["-m", "src.m3_training", "select"],
        ["-m", "src.m3_evaluation", "evaluate", "--device", args.device],
    ]
    for stage in stages:
        subprocess.run([sys.executable, *stage], check=True)


if __name__ == "__main__":
    main()
