import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import tempfile
import json
import os

# ラベル読み込み
with open("turf_labels.json") as f:
    labels = json.load(f)

type_both = ["Dollarspot", "FairyRing"]
type_warm = ["LargePatch"]
type_cool = ["BrownPatch", "SnowMold", "PythiumBlight", "DrechsleraleafSpot", "RedThread", "TakeAllPatch"]

def get_labels_by_turf_type(turf_type):
    if turf_type == "warm":
        return type_both + type_warm
    elif turf_type == "cool":
        return type_both + type_cool
    return labels

# モデル読み込み
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# UI
st.title("芝生の病害分類AI（ONNX版）")
turf_type = st.radio("芝の種類を選んでください", ["warm", "cool"], format_func=lambda x: "暖地型" if x == "warm" else "寒地型")
uploaded_file = st.file_uploader("病害画像をアップロードしてください", type=["jpg", "jpeg", "png"])

import base64

if uploaded_file:

    # img_bytes = uploaded_file.read()
    # OK（getvalueは複数回使ってもOK）
    img_bytes = uploaded_file.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    st.markdown(
        f"<img src='data:image/jpeg;base64,{img_base64}' "
        f"style='width:50%; height:auto; display:block; margin:auto;'/>",
        unsafe_allow_html=True
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # 画像前処理
    img = Image.open(tmp_path).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 推論
    output = session.run(None, {input_name: img_array})[0][0]
    os.remove(tmp_path)

    target_labels = get_labels_by_turf_type(turf_type)
    target_indices = [labels.index(lbl) for lbl in target_labels]
    results = [(lbl, output[labels.index(lbl)]) for lbl in target_labels]
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("分類結果")
    for lbl, score in results:

        st.markdown(f"<div style='font-size:20px; font-weight:bold'>{lbl}: {score * 100:.2f}%</div>", unsafe_allow_html=True)
