import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse
import json

# 引数パース
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--turf_type", required=True, choices=["warm", "cool"])
args = parser.parse_args()

# ラベル一覧（必ずモデルと一致させること）
with open("turf_labels.json") as f:
    all_labels = json.load(f)

# 芝種に応じて分類対象を絞る
type_both = ["Dollarspot", "FairyRing"]
type_warm = ["LargePatch"]
type_cool = ["BrownPatch", "SnowMold", "PythiumBlight", "DrechsleraleafSpot", "RedThread", "TakeAllPatch"]

if args.turf_type == "warm":
    target_labels = type_both + type_warm
elif args.turf_type == "cool":
    target_labels = type_both + type_cool
else:
    target_labels = all_labels

target_indices = [all_labels.index(label) for label in target_labels]

# モデル読み込み
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# 画像前処理
img = Image.open(args.image).convert("RGB").resize((224, 224))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 推論
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: img_array})[0][0]

# 結果表示
sorted_idx = np.argsort([output[i] for i in target_indices])[::-1]

print("\n分類結果（上位）：")
for idx in sorted_idx:
    label = target_labels[idx]
    score = output[all_labels.index(label)] * 100
    print(f"{label:<25}: {score:.2f}%")
    