import torch
from torchvision import models
import torch.nn as nn

# ======================
# 設定
# ======================
MODEL_PATH = "models/disease_resnet18_best.pth"
ONNX_PATH = "models/disease_resnet18_best.onnx"
class_names = ["dollar_spot", "brown_patch", "leaf_spot", "pythium"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# モデル定義
# ======================
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(class_names))
)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ======================
# ダミー入力
# ======================
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# ======================
# ONNX 出力
# ======================
torch.onnx.export(
    model, dummy_input, ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)
print(f"ONNXモデル保存完了: {ONNX_PATH}")