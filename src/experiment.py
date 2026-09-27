"""Shared experiment identity and provenance for separate training notebooks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import uuid
import warnings
from .config import ROOT, DATA_DIR, SELECTED_IDS, SEED, BATCH_SIZE, EPOCHS
from .data_utils import audit_subset


def _source_hashes():
    """Hash executable sources so all notebooks use one implementation."""
    paths = sorted((ROOT / "src").glob("*.py")) + [ROOT / "utils.py", ROOT / "requirements.txt"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def start_experiment():
    """Audit data, create a unique run, and record its path for training notebooks."""
    manifest, classes = audit_subset(DATA_DIR)
    if set(classes) != set(map(str, SELECTED_IDS)):
        raise ValueError("Data identities do not match config.SELECTED_IDS")
    run = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    manifest.to_csv(run / "split_manifest.csv", index=False, lineterminator="\n")
    config = dict(seed=SEED, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  selected_ids=SELECTED_IDS, class_names=classes, source_hashes=_source_hashes())
    (run / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Archive executable code and notebook inputs before any model is trained.
    for relative in config["source_hashes"]:
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copytree(ROOT / "notebooks", run / "source" / "notebooks",
                    ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"))
    for name in ("selection_metadata.json", "split_config.json"):
        path = DATA_DIR / name
        if path.exists():
            shutil.copy2(path, run / name)
    (ROOT / "runs" / "active_run.txt").write_text(run.name, encoding="utf-8")
    return run


def get_active_run(*, verify_source=True):
    """Check training inputs; allow source revisions when reviewing saved results."""
    pointer = ROOT / "runs" / "active_run.txt"
    name = pointer.read_text(encoding="utf-8").strip()
    if Path(name).name != name:
        raise ValueError("Invalid experiment directory name")
    run = ROOT / "runs" / name
    config = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    if config["source_hashes"] != _source_hashes():
        if verify_source:
            raise ValueError("Training source changed; use a new experiment for new training")
        warnings.warn("Reviewing saved results with updated source; original training snapshot is retained.",
                      UserWarning, stacklevel=2)
    manifest, _ = audit_subset(DATA_DIR)
    if manifest.to_csv(index=False, lineterminator="\n") != (run / "split_manifest.csv").read_text(encoding="utf-8"):
        raise ValueError("Data changed after experiment creation")
    return run
