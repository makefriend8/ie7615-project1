"""Fixed settings used by the data and training notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "selected_subset"
SELECTED_IDS = [2619, 797, 2970, 4422, 3, 2336]
SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
MODEL_NAMES = ("Custom_CNN", "ResNet_Frozen", "ResNet_Finetuned")
