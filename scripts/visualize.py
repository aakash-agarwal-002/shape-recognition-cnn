import json
import random
import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path to allow imports from src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data.augment import augment_points
from src.utils.preprocessing import points_to_model_input

DATA_FILE = BASE_DIR / "data" / "processed" / "dataset.json"
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.json"
OUT_DIR = BASE_DIR / "data" / "viz"
MAX_SAMPLES_PER_LABEL = 20
APPLY_AUGMENTATION = True

# Load labels
if not LABELS_PATH.exists():
    print(f"Error: Labels file not found at {LABELS_PATH}")
    sys.exit(1)

with open(LABELS_PATH, "r") as f:
    LABEL_NAMES = json.load(f)

# Load data
if not DATA_FILE.exists():
    print(f"Error: Dataset file not found at {DATA_FILE}")
    sys.exit(1)

with open(DATA_FILE, "r") as f:
    data = json.load(f)

by_label = {label: [] for label in LABEL_NAMES}
for sample in data:
    label = sample["label"]
    label_name = LABEL_NAMES[label] if isinstance(label, int) else label
    by_label.setdefault(label_name, []).append(sample)

def plot_and_save(points, source, path):
    label_name = path.parent.name
    if APPLY_AUGMENTATION:
        points = augment_points(points, source, label_name)

    line_width = random.randint(1, 2) if APPLY_AUGMENTATION else 2
    img = points_to_model_input(points, line_width=line_width)[0]
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

for label_name, samples in by_label.items():
    if not samples:
        continue
        
    folder = OUT_DIR / label_name
    folder.mkdir(parents=True, exist_ok=True)

    chosen = random.sample(samples, min(len(samples), MAX_SAMPLES_PER_LABEL))

    for i, sample in enumerate(chosen):
        source = sample.get("source", "synthetic")
        path = folder / f"{i:03d}_{source}.png"
        plot_and_save(sample["points"], source, path)

print(f"Visualization saved in: {OUT_DIR}")
