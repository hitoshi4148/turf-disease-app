import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np

# ===== 設定 =====
DATA_DIR = "data_binary"
BATCH_SIZE = 16
EPOCHS = 8
LR = 0.0005
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = "best_binary_model.pth"

DEVICE = torch.device("cpu")

# ===== Transform =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== Dataset =====
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Total images:", len(dataset))

# クラス枚数取得
targets = dataset.targets
class_counts = np.bincount(targets)
print("Class counts:", class_counts)

# クラス重み（逆数）
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(DEVICE)

print("Class weights:", class_weights)

# Train/Val split
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== Model =====
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# ===== Loss / Optimizer =====
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# ===== Training Loop =====
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 20)

    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())

    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")

    print(classification_report(val_labels, val_preds, target_names=class_names))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Best binary model saved!")

print("\nTraining complete.")
print("Best Validation Accuracy:", best_val_acc)