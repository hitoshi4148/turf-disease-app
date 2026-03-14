import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import sys

# ======================
# 設定
# ======================
MODEL_PATH = "models/disease_resnet18_best.pth"
IMAGE_PATH = sys.argv[1]  # コマンドラインから画像指定

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# モデルロード
# ======================
checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(class_names))
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ======================
# 前処理
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ======================
# 推論
# ======================
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

predicted_class = class_names[predicted.item()]

print("Predicted class:", predicted_class)