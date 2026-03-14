import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# ===== 設定 =====
MODEL_PATH = "model_v0_1.keras"
VAL_DIR = "data_split/val"   # あなたのvalフォルダ構造に合わせて

# ===== モデル読み込み =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== データ読み込み =====
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = val_ds.class_names

# ===== 予測 =====
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ===== 混同行列 =====
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===== 詳細レポート =====
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))