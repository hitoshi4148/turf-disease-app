import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


DATA_DIR = os.path.join("data_two_stage", "stage2_disease")
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.0002
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = os.path.join("models", "stage2_disease_model.pth")
CLASS_NAMES_JSON = "class_names.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    os.makedirs("models", exist_ok=True)
    dataset = datasets.ImageFolder(DATA_DIR)
    class_names = dataset.classes
    num_classes = len(class_names)

    with open(CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_split, val_split = random_split(dataset, [train_size, val_size])

    train_base = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_base = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    train_dataset = Subset(train_base, train_split.indices)
    val_dataset = Subset(val_base, val_split.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    patience = 8
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, MODEL_SAVE_PATH)
            print("Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step()

    print("Best Validation Accuracy:", best_val_acc)
    print(classification_report(val_labels, val_preds, target_names=class_names))


if __name__ == "__main__":
    main()
