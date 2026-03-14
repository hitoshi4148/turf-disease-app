# 🌱 芝生病害分類アプリ Streamlit + ONNX 完全手順書

## 1️⃣ 環境準備

### 1-1. プロジェクトフォルダ作成
```powershell
mkdir C:\Users\hitos\disease_classification
cd C:\Users\hitos\disease_classification
1-2. 仮想環境作成（Python 3.12前提）
python -m venv venv
1-3. 仮想環境有効化
venv\Scripts\activate

表示が (venv) になれば OK

2️⃣ 必須パッケージのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit pillow scikit-learn onnx onnxruntime

GPU がある場合は CUDA バージョンに合わせて PyTorch をインストール

3️⃣ base 自動起動停止（オプション）

PowerShell 起動時に (base) が表示される場合：

conda config --set auto_activate_base false

PowerShell 再起動後、venv を再アクティベート

4️⃣ モデルファイル準備
4-1. 学習済み PyTorch モデル
models/disease_resnet18_best.pth

プロジェクト直下に models フォルダを作成し配置

5️⃣ PyTorch → ONNX 変換
import torch
from torchvision import models
import torch.nn as nn

# モデル定義
class_names = ["dollar_spot", "brown_patch", "leaf_spot", "pythium"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(class_names))
)
checkpoint = torch.load("models/disease_resnet18_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ダミー入力
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# ONNX 出力
torch.onnx.export(
    model, dummy_input, "models/disease_resnet18_best.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=13
)
print("ONNXモデル保存完了")
6️⃣ Streamlit アプリ作成（ONNX 推論版）
import streamlit as st
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np

MODEL_PATH = "models/disease_resnet18_best.onnx"

# ======================
# ONNX セッション
# ======================
@st.cache_resource
def load_model():
    ort_session = ort.InferenceSession(MODEL_PATH)
    class_names = ["dollar_spot", "brown_patch", "leaf_spot", "pythium"]
    return ort_session, class_names

ort_session, class_names = load_model()

# ======================
# 前処理
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# UI
# ======================
st.title("🌱 芝生病害診断AI (ONNX版)")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).numpy()
    input_tensor = np.transpose(input_tensor, (0,2,3,1))  # ONNX用にCHW→NHWCの場合は必要

    outputs = ort_session.run(None, {"input": input_tensor})
    probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))  # softmax
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_idx]
    confidence_score = probabilities[0][predicted_idx] * 100

    st.success(f"予測結果: {predicted_class}")
    st.write(f"信頼度: {confidence_score:.2f}%")

    st.subheader("クラス別確率")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probabilities[0][i]*100:.2f}%")
7️⃣ Streamlit 起動

venv を有効化：

venv\Scripts\activate

Streamlit 実行：

streamlit run app.py

ブラウザが自動で開く

スマホからも同一LAN内でアクセス可能
例：

http://PCのIPアドレス:8501
8️⃣ 注意点

ONNX化すると推論が高速化し、GPU活用も容易

Python 環境は必ず venv 内 で操作

クラス名 class_names はモデル保存時と順序を一致させること

9️⃣ 次のステップ

データ収集自動化でクラスを補強

UI改善（複数画像一括推論）

モデル最適化（量子化・軽量化）