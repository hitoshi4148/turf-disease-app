import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# ===== 設定 =====
BINARY_MODEL_PATH = "best_binary_model.pth"
MULTI_MODEL_PATH = "best_model.pth"
DATA_MULTI_DIR = "data_processed"
DATA_BINARY_DIR = "data_binary"
IMG_SIZE = 224
DEVICE = torch.device("cpu")

# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== クラス名取得 =====
from torchvision.datasets import ImageFolder

multi_dataset = ImageFolder(DATA_MULTI_DIR)
multi_class_names = multi_dataset.classes

binary_dataset = ImageFolder(DATA_BINARY_DIR)
binary_class_names = binary_dataset.classes

# ===== Binary Model =====
binary_model = models.resnet18(weights=None)
binary_model.fc = nn.Linear(binary_model.fc.in_features, 2)
binary_model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=DEVICE))
binary_model.to(DEVICE)
binary_model.eval()

# ===== Multi-class Model =====
multi_model = models.resnet18(weights=None)
multi_model.fc = nn.Linear(multi_model.fc.in_features, len(multi_class_names))
multi_model.load_state_dict(torch.load(MULTI_MODEL_PATH, map_location=DEVICE))
multi_model.to(DEVICE)
multi_model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    # ===== Step1: Binary =====
    with torch.no_grad():
        output_binary = binary_model(image)
        prob_binary = torch.softmax(output_binary, dim=1)
        conf_binary, pred_binary = torch.max(prob_binary, 1)

    pred_label_binary = binary_class_names[pred_binary.item()]
    conf_binary = conf_binary.item()

    print("\n===== Step1: Healthy判定 =====")
    print(f"予測: {pred_label_binary}")
    print(f"信頼度: {conf_binary:.2%}")

    if pred_label_binary == "healthy":
        print("\n🟢 診断結果: 健全芝の可能性が高いです。")
        return

    # ===== Step2: Disease分類 =====
    with torch.no_grad():
        output_multi = multi_model(image)
        prob_multi = torch.softmax(output_multi, dim=1)
        top_k = min(3, prob_multi.size(1))
        conf_multi, pred_multi = torch.topk(prob_multi, k=top_k, dim=1)

    top_confs = conf_multi.squeeze(0).tolist()
    top_preds = pred_multi.squeeze(0).tolist()
    top_labels = [multi_class_names[idx] for idx in top_preds]

    print("\n===== Step2: 病害分類 =====")
    for i, (label, conf) in enumerate(zip(top_labels, top_confs), start=1):
        print(f"{i}. {label:<16} {conf * 100:.1f}%")

    print("\n🔴 診断結果:")
    print("最も可能性が高い病害は")
    print(f"{top_labels[0]} ({top_confs[0] * 100:.1f}%)")
    print("です。")
    if len(top_labels) > 1:
        print("\nただし以下の可能性もあります")
        for label, conf in zip(top_labels[1:], top_confs[1:]):
            print(f"{label} ({conf * 100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python predict.py image.jpg")
    else:
        predict(sys.argv[1])