import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import tempfile
import os
from PIL import Image

# モデルとラベルの読み込み
model = tf.keras.models.load_model("model.h5")
with open("turf_labels.json") as f:
    labels = json.load(f)

# 芝タイプによる病害グループ
type_both = ["Dollarspot", "FairyRing"]
type_warm = ["LargePatch"]
type_cool = ["BrownPatch", "SnowMold", "PythiumBlight", "DrechsleraleafSpot", "RedThread", "TakeAllPatch"]

def get_labels_by_turf_type(turf_type):
    if turf_type == "warm":
        return type_both + type_warm
    elif turf_type == "cool":
        return type_both + type_cool
    return labels

# Streamlit UI
st.title("芝生の病害分類AI")
st.write("暖地型／寒地型の芝生の種類を選んで、病害部分の画像をアップロードしてください。")

st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > label {
        margin-bottom: 0rem;
    }
    div[data-testid="stRadio"] {
        margin-top: -2.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

turf_type = st.radio("", ["warm", "cool"], format_func=lambda x: "暖地型" if x == "warm" else "寒地型")

uploaded_file = st.file_uploader("病害が発生している芝生の画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    w, h = img.size
    img_resized = img.resize((w // 2, h // 2))
    st.image(img_resized, caption="アップロード画像")
    # 以降、img_resizedを一時ファイルに保存して推論に使う
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_resized.save(tmp, format="JPEG")
        tmp_path = tmp.name

    # 推論処理
    img = image.load_img(tmp_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]

    # 芝種でフィルタ
    target_labels = get_labels_by_turf_type(turf_type)
    target_indices = [labels.index(lbl) for lbl in target_labels]
    filtered = [(labels[i], preds[i]) for i in target_indices]
    filtered.sort(key=lambda x: x[1], reverse=True)

    # --- ここで再正規化 ---
    total = sum([score for _, score in filtered])
    if total > 0:
        filtered = [(label, score / total) for label, score in filtered]
    else:
        filtered = [(label, 0) for label, _ in filtered]

    st.subheader(f"診断結果（芝種：{'暖地型' if turf_type == 'warm' else '寒地型'}）")
    for label, score in filtered:
        st.markdown(
            f"<div style='font-size:1.3em; font-weight:bold; margin-bottom:0.3em'>{label}: <span style='color:#0072B5'>{score*100:.2f}%</span></div>",
            unsafe_allow_html=True
        )
