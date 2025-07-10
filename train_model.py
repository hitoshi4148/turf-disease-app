import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# パス設定
dataset_dir = "dataset"
model_path = "model.h5"
labels_path = "turf_labels.json"

# ラベル取得と保存
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
with open(labels_path, "w") as f:
    json.dump(class_names, f, indent=2)

# データジェネレータ
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode="categorical",
    subset="validation"
)

# モデル構築
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(class_names), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# 転移学習（ベース層は固定）
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# 学習
model.fit(train_gen, validation_data=val_gen, epochs=10)

# 保存
model.save(model_path)
print(f"Saved model to {model_path}")
