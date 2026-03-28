import json
import sys
import argparse
from pathlib import Path

# Add project root to path to allow imports from scripts
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import the generation function
from scripts.generate_synthetic import generate_synthetic_data

# Paths relative to the project root
RAW_COLLECTED_DIR = BASE_DIR / "data" / "raw" / "collected"
RAW_SYNTHETIC_PATH = BASE_DIR / "data" / "raw" / "synthetic" / "synthetic_strokes.json"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "dataset.json"
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.json"

LABEL_NAMES = [
    "ellipse", "line", "triangle", "rectangle", "pentagon", "hexagon",
    "star", "zigzag", "arc", "heart", "diamond", "arrow",
    "double_arrow", "cloud", "message", "parallelogram",
]

LEGACY_LABEL_MAP = {
    "circle": "ellipse",
    "square": "rectangle",
}

def load_json(path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)

def sample_key(label, points):
    rounded_points = tuple(
        (round(float(x), 4), round(float(y), 4))
        for x, y in points
    )
    return (label, rounded_points)

def normalize_label(label):
    return LEGACY_LABEL_MAP.get(label, label)

def main():
    parser = argparse.ArgumentParser(description="Clean and prepare the dataset.")
    parser.add_argument("--delete", type=bool, default=False, help="Force recreation of synthetic data.")
    parser.add_argument("--n", type=int, default=200, help="Number of synthetic samples per shape.")
    args = parser.parse_args()

    # 1. Handle synthetic data generation
    if args.delete or not RAW_SYNTHETIC_PATH.exists():
        print(f"Generating new synthetic data (n={args.n} per shape)...")
        generate_synthetic_data(num_samples=args.n, output_path=RAW_SYNTHETIC_PATH)
    else:
        print(f"Synthetic data found at {RAW_SYNTHETIC_PATH}. Skipping generation.")
        print("Use --delete=True to force recreation.")

    # 2. Load and merge synthetic data
    print(f"Merging synthetic data from {RAW_SYNTHETIC_PATH}...")
    synthetic_data = load_json(RAW_SYNTHETIC_PATH)
    
    label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
    existing_keys = set()
    dataset = []

    for item in synthetic_data:
        label = item["label"]
        label_name = LABEL_NAMES[label] if isinstance(label, int) else normalize_label(label)
            
        key = sample_key(label_name, item["points"])
        if key in existing_keys:
            continue
        existing_keys.add(key)
        
        dataset.append({
            "points": item["points"],
            "label": label_to_idx[label_name],
            "source": "synthetic",
        })

    # 3. Consolidate browser samples into one file (only once)
    consolidated_browser_path = RAW_COLLECTED_DIR / "merged_browser_samples.json"
    
    if not consolidated_browser_path.exists():
        print(f"Consolidating individual browser samples into {consolidated_browser_path}...")
        browser_samples = []
        for child in sorted(RAW_COLLECTED_DIR.glob("browser*.json")):
            # Skip the merged file itself if it somehow existed
            if child.name == "merged_browser_samples.json":
                continue
            loaded = load_json(child)
            if isinstance(loaded, list):
                browser_samples.extend(loaded)
            else:
                browser_samples.append(loaded)
        
        if browser_samples:
            with open(consolidated_browser_path, "w") as f:
                json.dump(browser_samples, f)
            print(f"Consolidated {len(browser_samples)} samples into {consolidated_browser_path}")
        else:
            print("No individual browser samples found to consolidate.")
    else:
        print(f"Using consolidated browser samples from {consolidated_browser_path}")

    browser_samples = load_json(consolidated_browser_path)

    added = 0
    skipped_duplicates = 0
    for sample in browser_samples:
        label_name = normalize_label(sample["label"])
        points = sample["points"]
        
        if label_name not in label_to_idx:
            print(f"Skipping sample with unknown label '{label_name}'")
            continue
            
        key = sample_key(label_name, points)
        if key in existing_keys:
            skipped_duplicates += 1
            continue

        dataset.append({
            "points": points,
            "label": label_to_idx[label_name],
            "source": "browser",
        })
        existing_keys.add(key)
        added += 1

    # 4. Save processed dataset
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DATA_PATH, "w") as f:
        json.dump(dataset, f)

    with open(LABELS_PATH, "w") as f:
        json.dump(LABEL_NAMES, f)

    print(f"\nDataset preparation complete!")
    print(f"Total samples: {len(dataset)}")
    print(f"Added {added} browser samples.")
    print(f"Skipped {skipped_duplicates} duplicate browser samples.")
    print(f"Saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
