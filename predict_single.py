import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 引数パース
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--turf_type", type=str, choices=["warm", "cool"], required=True, help="芝のタイプ: warm（暖地型）または cool（寒地型）")
args = parser.parse_args()

# ラベルとモデルの読み込み
with open("turf_labels.json") as f:
    labels = json.load(f)

model = tf.keras.models.load_model("model.h5")

# 画像の読み込みと前処理
img = image.load_img(args.image, target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# 芝タイプごとの病害リスト
type_both = ["Dollarspot", "FairyRing"]
type_warm = ["LargePatch"]
type_cool = ["BrownPatch", "SnowMold", "PythiumBlight", "DrechsleraleafSpot", "RedThread", "TakeAllPatch"]

def get_labels_by_turf_type(turf_type):
    if turf_type == "warm":
        return type_both + type_warm
    elif turf_type == "cool":
        return type_both + type_cool
    else:
        return type_both + type_warm + type_cool

# 推論
preds = model.predict(x)[0]

# 対象ラベルのみ抽出
target_labels = get_labels_by_turf_type(args.turf_type)
target_indices = [labels.index(label) for label in target_labels]
filtered_preds = [(labels[i], preds[i]) for i in target_indices]
filtered_preds.sort(key=lambda x: x[1], reverse=True)

print(f"\n分類結果（芝タイプ: {args.turf_type}）:")
for label, prob in filtered_preds:
    print(f"{label:<25}: {prob*100:.2f}%")
