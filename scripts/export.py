import torch
import torch.onnx
import sys
from pathlib import Path

# Add project root to path to allow imports from src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.models.tiny_cnn import TinyCNN

def export_to_onnx(weights_path=BASE_DIR / "checkpoints" / "model_final.pth", onnx_path=BASE_DIR / "web" / "model.onnx"):
    # Load model (need n_classes, let's assume 16 from current SHAPE_NAMES or load from labels.json)
    import json
    labels_path = BASE_DIR / "data" / "processed" / "labels.json"
    with open(labels_path, "r") as f:
        label_names = json.load(f)
    
    n_classes = len(label_names)
    model = TinyCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 1, 64, 64)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
