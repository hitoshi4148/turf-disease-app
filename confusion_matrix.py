import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ===== 設定 =====
DATA_DIR = "data_processed"
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 16

# ===== transform =====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===== dataset =====
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes

# ===== モデル読み込み =====

import torchvision.models as models
import torch.nn as nn

checkpoint = torch.load(MODEL_PATH)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

# train.pyと同じモデル
model = models.efficientnet_b0(pretrained=False)

# 最終層を変更
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

y_true = []
y_pred = []

# ===== 推論 =====
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

# ===== 混同行列 =====
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)

fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix")
plt.show()
