"""Compare PyTorch and ONNX inference on a sample image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from export_onnx import build_model, load_class_names

MODEL_PATH = Path("models/mobilenet_v3_small_best.pth")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def top_k(probs: np.ndarray, class_names: list[str], k: int = 3) -> list[tuple[str, float]]:
    idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idx]


def load_sample_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("model.onnx"),
        help="ONNX model path",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("ui_images/photo_good.jpg"),
        help="Sample image for comparison",
    )
    args = parser.parse_args()

    class_names = load_class_names()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = load_sample_image(args.image)
    tensor = transform(image).unsqueeze(0)

    model = build_model(class_names)
    with torch.inference_mode():
        pt_logits = model(tensor).numpy().squeeze()
    pt_probs = softmax(pt_logits)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(
        None,
        {"input": tensor.numpy()},
    )[0].squeeze()
    onnx_probs = softmax(onnx_logits)

    max_diff = float(np.max(np.abs(pt_probs - onnx_probs)))
    pt_top = top_k(pt_probs, class_names)
    onnx_top = top_k(onnx_probs, class_names)

    print(f"Image: {args.image}")
    print(f"Max probability diff: {max_diff:.6f}")
    print(f"PyTorch Top3: {pt_top}")
    print(f"ONNX    Top3: {onnx_top}")
    print(f"Top1 match: {pt_top[0][0] == onnx_top[0][0]}")

    if max_diff > 1e-4:
        raise SystemExit("ONNX validation failed: probability diff too large")
    if pt_top[0][0] != onnx_top[0][0]:
        raise SystemExit("ONNX validation failed: Top1 mismatch")


if __name__ == "__main__":
    main()
