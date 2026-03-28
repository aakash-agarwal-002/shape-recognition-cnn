import json
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import os
import shutil
import sys
import argparse
import datetime
import csv
from pathlib import Path

# Add project root to path to allow imports from src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data.dataset import StrokeDataset
from src.models.tiny_cnn import TinyCNN
from src.utils.preprocessing import points_to_model_input

def main():
    parser = argparse.ArgumentParser(description="Train the Shape Classifier CNN.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--weight", type=float, default=12.0, help="Weight for browser samples in sampler.")
    parser.add_argument("--export_images", action="store_true", help="Export dataset images for debugging.")
    args = parser.parse_args()

    # ---------- session setup ----------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    LOG_DIR = BASE_DIR / "logs" / run_name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    CHECKPOINT_DIR = LOG_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    SAMPLE_VIS_DIR = LOG_DIR / "samples"
    SAMPLE_VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to both console and log file
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(LOG_DIR / "train.log", "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger()
    
    print(f"--- Training Session: {run_name} ---")
    print(f"Hyperparameters: {vars(args)}")

    # ---------- configuration ----------
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "dataset.json"
    LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.json"
    DATASET_IMAGE_DIR = LOG_DIR / "dataset_images"

    # ---------- load labels ----------
    with open(LABELS_PATH, "r") as f:
        LABEL_NAMES = json.load(f)

    label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
    idx_to_label = {i: name for i, name in enumerate(LABEL_NAMES)}

    # ---------- debugging tools ----------
    def save_model_input_image(img, path):
        plt.imsave(path, img, cmap="gray", vmin=0.0, vmax=1.0)

    def export_dataset_images(data, train_indices, test_indices, out_dir=DATASET_IMAGE_DIR):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        split_specs = [
            ("train_raw", train_indices, False),
            ("train_aug", train_indices, True),
            ("test", test_indices, False),
        ]

        for split_name, indices_for_split, use_augmentation in split_specs:
            split_dir = os.path.join(out_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for data_idx in indices_for_split:
                sample = data[data_idx]
                source = sample.get("source", "synthetic")
                label = sample["label"]
                label_name = LABEL_NAMES[label] if isinstance(label, int) else label
                points = sample["points"]

                if use_augmentation:
                    from src.data.augment import augment_points
                    points = augment_points(points, source, label_name)

                line_width = np.random.randint(1, 3) if use_augmentation else 2
                img = points_to_model_input(points, line_width=line_width)[0]

                label_dir = os.path.join(split_dir, label_name)
                os.makedirs(label_dir, exist_ok=True)

                filename = f"{data_idx:05d}_{source}.png"
                save_model_input_image(img, os.path.join(label_dir, filename))

    # ---------- load data ----------
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    with open(PROCESSED_DATA_PATH, "r") as f:
        raw_data = json.load(f)

    indices = np.random.permutation(len(raw_data))
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size].tolist()
    test_indices = indices[train_size:].tolist()

    train_ds = StrokeDataset(raw_data, LABEL_NAMES, label_to_idx, indices=train_indices, augment=True)
    test_ds = StrokeDataset(raw_data, LABEL_NAMES, label_to_idx, indices=test_indices, augment=False)

    train_weights = []
    browser_train_count = 0

    for idx in train_indices:
        sample = raw_data[idx]
        is_browser = sample.get("source") == "browser"
        train_weights.append(args.weight if is_browser else 1.0)
        browser_train_count += int(is_browser)

    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(train_weights),
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    test_source_counts = {"synthetic": 0, "browser": 0}
    for idx in test_indices:
        source = raw_data[idx].get("source", "synthetic")
        test_source_counts[source] = test_source_counts.get(source, 0) + 1

    if args.export_images:
        print(f"Exporting dataset images to {DATASET_IMAGE_DIR}...")
        export_dataset_images(raw_data, train_indices, test_indices)

    # ---------- device ----------
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)
    print("Train samples:", len(train_ds))
    print("Test samples:", len(test_ds))
    print("Browser samples in train split:", browser_train_count)
    print("Synthetic samples in test split:", test_source_counts.get("synthetic", 0))
    print("Browser samples in test split:", test_source_counts.get("browser", 0))

    # ---------- model ----------
    model = TinyCNN(n_classes=len(LABEL_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # ---------- utils ----------
    def save_sample_predictions(epoch, n=10):
        model.eval()
        images_collected = []
        labels_collected = []
        preds_collected = []
        sources_collected = []

        with torch.no_grad():
            for x, y, source in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                images_collected.append(x.cpu())
                labels_collected.append(y.cpu())
                preds_collected.append(preds.cpu())
                sources_collected.extend(source)

                if len(torch.cat(images_collected)) >= n:
                    break

        images = torch.cat(images_collected)[:n]
        labels = torch.cat(labels_collected)[:n]
        preds = torch.cat(preds_collected)[:n]
        sources = sources_collected[:n]

        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        for i in range(n):
            img = images[i][0].numpy()
            true_label = idx_to_label[labels[i].item()]
            pred_label = idx_to_label[preds[i].item()]
            source = sources[i]
            color = "green" if true_label == pred_label else "red"
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"T:{true_label}\nP:{pred_label}\nS:{source}", color=color, fontsize=8)
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(SAMPLE_VIS_DIR / f"epoch_{epoch+1}.png")
        plt.close()

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        n_classes = len(LABEL_NAMES)
        confusion = np.zeros((n_classes, n_classes))
        source_stats = {}

        with torch.no_grad():
            for x, y, source in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                for t, p, s in zip(y.cpu().numpy(), preds.cpu().numpy(), source):
                    confusion[t][p] += 1
                    if s not in source_stats:
                        source_stats[s] = {"correct": 0, "total": 0}
                    source_stats[s]["total"] += 1
                    source_stats[s]["correct"] += int(t == p)

        acc = correct / total
        source_acc = {s: stats["correct"] / stats["total"] for s, stats in source_stats.items() if stats["total"] > 0}
        return acc, confusion, source_acc

    # ---------- metrics logging ----------
    metrics_file = LOG_DIR / "metrics.csv"
    with open(metrics_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_acc", "test_acc", "test_synthetic_acc", "test_browser_acc"])

    # ---------- training ----------
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc, _, _ = evaluate(train_loader)
        test_acc, confusion, test_source_acc = evaluate(test_loader)
        test_synthetic_acc = test_source_acc.get("synthetic", "")
        test_browser_acc = test_source_acc.get("browser", "")
        synthetic_display = f"{test_synthetic_acc:.4f}" if test_synthetic_acc != "" else "n/a"
        browser_display = f"{test_browser_acc:.4f}" if test_browser_acc != "" else "n/a"
        
        save_sample_predictions(epoch)

        with open(metrics_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, train_acc, test_acc, test_synthetic_acc, test_browser_acc])

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")
        print(
            "  Test by source | "
            f"Synthetic: {synthetic_display} | "
            f"Browser: {browser_display}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "model_best.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "model_final.pth")

    print("\nPer-class accuracy (Final Epoch):")
    for i in range(len(LABEL_NAMES)):
        total = confusion[i].sum()
        correct = confusion[i][i]
        acc = correct / total if total > 0 else 0
        print(f"{idx_to_label[i]}: {acc:.3f}")

if __name__ == "__main__":
    main()
