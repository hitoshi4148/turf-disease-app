"""Export MobileNetV3-Small checkpoint to ONNX for browser inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = Path("models/mobilenet_v3_small_best.pth")
CLASS_NAMES_PATH = Path("class_names.json")
DEFAULT_OUTPUT = Path("model.onnx")


def load_class_names() -> list[str]:
    with CLASS_NAMES_PATH.open(encoding="utf-8") as f:
        return json.load(f).get("class_names", [])


def build_model(class_names: list[str]) -> nn.Module:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        loaded_class_names = checkpoint.get("class_names", [])
    else:
        state_dict = checkpoint
        loaded_class_names = []

    if not loaded_class_names:
        loaded_class_names = class_names
    if not loaded_class_names:
        raise RuntimeError("class_names が見つかりません。")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(loaded_class_names),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(output_path: Path) -> None:
    class_names = load_class_names()
    model = build_model(class_names)
    dummy = torch.randn(1, 3, 224, 224)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
    print(f"Exported ONNX model to {output_path} ({output_path.stat().st_size // 1024} KB)")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output ONNX path",
    )
    args = parser.parse_args()
    export_onnx(args.output)


if __name__ == "__main__":
    main()
