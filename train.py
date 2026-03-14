import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ====== 設定 ======
DATA_DIR = "data_raw"
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.0002
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = "best_model.pth"
HARD_EXAMPLES_PATH = "hard_examples.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
torch.backends.cudnn.benchmark = True

print("Class folders:")
print(sorted(os.listdir(DATA_DIR)))

# ====== Transform ======
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ====== Dataset ======
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Total images:", len(dataset))

# ====== Train / Val split ======
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_split, val_split = random_split(dataset, [train_size, val_size])

# ====== Transform適用Dataset ======
train_dataset_base = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_dataset_base = datasets.ImageFolder(DATA_DIR, transform=val_transform)

train_dataset = Subset(train_dataset_base, train_split.indices)
print("Train dataset size:", len(train_dataset))
val_dataset = Subset(val_dataset_base, val_split.indices)
print("Validation dataset size:", len(val_dataset))

# ====== DataLoader ======
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ====== Model ======
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)

# ====== Loss ======
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ====== Optimizer ======
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ====== Scheduler ======
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

os.makedirs("models", exist_ok=True)

best_val_acc = 0.0
patience = 8
patience_counter = 0

# ====== Training Loop ======
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 20)

    # ---- Train ----
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

    # ---- Validation ----
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

    # ---- Save Best ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names
        }, MODEL_SAVE_PATH)

        print("Best model saved!")
    else:
        patience_counter += 1
        print(f"No improvement. EarlyStopping counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    scheduler.step()

print("\nTraining complete.")
print("Best Validation Accuracy:", best_val_acc)

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

# ====== Final Evaluation ======
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for inputs, labels in val_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n===== Classification Report =====")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ====== Hard Examples 保存 ======
hard_example_lines = []

for dataset_idx, true_label_idx, pred_label_idx in zip(val_split.indices, all_labels, all_preds):

    if pred_label_idx != true_label_idx:

        image_path = os.path.normpath(dataset.samples[dataset_idx][0])

        true_label = class_names[true_label_idx]
        pred_label = class_names[pred_label_idx]

        hard_example_lines.append(f"{image_path},{true_label},{pred_label}")

with open(HARD_EXAMPLES_PATH, "w", encoding="utf-8") as f:
    for line in hard_example_lines:
        f.write(f"{line}\n")

print(f"\nHard examples saved: {len(hard_example_lines)} -> {HARD_EXAMPLES_PATH}")