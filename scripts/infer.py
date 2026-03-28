import json
import torch
import sys
from pathlib import Path

# Add project root to path to allow imports from src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.models.tiny_cnn import TinyCNN
from src.utils.preprocessing import points_to_model_input

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_label_names():
    labels_path = BASE_DIR / "data" / "processed" / "labels.json"
    with open(labels_path, "r") as f:
        return json.load(f)

def load_model(weights_path=BASE_DIR / "checkpoints" / "model_final.pth", device=None):
    device = device or get_device()
    label_names = load_label_names()

    model = TinyCNN(n_classes=len(label_names))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, label_names, device

def predict_points(points, model=None, label_names=None, device=None):
    if model is None or label_names is None:
        model, label_names, device = load_model(device=device)

    device = device or get_device()

    x = points_to_model_input(points)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    return {
        "label": label_names[pred_idx],
        "index": pred_idx,
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {
            label_names[i]: float(probs[i].item()) for i in range(len(label_names))
        },
    }

if __name__ == "__main__":
    sample_points = [
        [0, 0],
        [0.5, 1.0],
        [1.0, 0],
        [0, 0],
    ]

    result = predict_points(sample_points)
    print(json.dumps(result, indent=2))
