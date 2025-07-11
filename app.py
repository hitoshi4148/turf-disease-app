import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import tempfile
import json
import os

# ページ設定
st.set_page_config(
    page_title="芝生病害分類AI - グリーンキーパー向け診断ツール",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HTMLヘッダーにメタタグを追加
st.markdown("""
<head>
    <meta name="description" content="芝生の病害をAIで自動診断。暖地型・寒地型芝に対応した専門的な病害分類ツール。ダラースポット、ブラウンパッチ、雪腐病など9種類の病害を高精度で識別。">
    <meta name="keywords" content="芝生,病害,分類,AI,診断,グリーンキーパー,ダラースポット,ブラウンパッチ,雪腐病,暖地型,寒地型">
    <meta name="author" content="Growth and Progress">
    <meta property="og:title" content="芝生病害分類AI - グリーンキーパー向け診断ツール">
    <meta property="og:description" content="芝生の病害をAIで自動診断。専門的な病害分類ツール。">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary_large_image">
</head>
""", unsafe_allow_html=True)

# 構造化データを追加
st.markdown("""
<script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "芝生病害分類AI",
    "description": "芝生の病害をAIで自動診断する専門ツール",
    "applicationCategory": "農業・園芸アプリ",
    "operatingSystem": "Web",
    "offers": {
        "@type": "Offer",
        "price": "0",
        "priceCurrency": "JPY"
    },
    "author": {
        "@type": "Organization",
        "name": "Growth and Progress"
    }
}
</script>
""", unsafe_allow_html=True)

# モデルとラベルの読み込み
model = ort.InferenceSession("model.onnx")
with open("turf_labels.json") as f:
    labels = json.load(f)

# 病害名の日本語対応辞書
disease_names_jp = {
    "Dollarspot": "ダラースポット",
    "FairyRing": "フェアリーリング",
    "LargePatch": "ラージパッチ（葉腐病）",
    "BrownPatch": "ブラウンパッチ",
    "SnowMold": "雪腐病",
    "PythiumBlight": "ピシウムブライト（赤焼病）",
    "DrechsleraleafSpot": "ドレクスレラ葉枯病",
    "RedThread": "レッドスレッド（赤葉腐病）",
    "TakeAllPatch": "テイクオールパッチ（立枯病）"
}

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

# モデル読み込み
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# UI
st.markdown("<h1 style='text-align: center;'>芝生病害分類AI</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 1.2em;'>グリーンキーパーのための専門診断ツール</h2>", unsafe_allow_html=True)
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
        f"alt='芝生病害診断用画像' "
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
    # results = [(lbl, output[labels.index(lbl)]) for lbl in target_labels]
    # results.sort(key=lambda x: x[1], reverse=True)

    # 出力のスコアから対象病害のみ抽出
    raw_scores = [(lbl, output[labels.index(lbl)]) for lbl in target_labels]

    # 合計で100％になるように再スケーリング
    total = sum(score for _, score in raw_scores)
    results = [(lbl, score / total) for lbl, score in raw_scores] if total > 0 else raw_scores

    # 降順でソート
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("分類結果")
    for lbl, score in results:

        st.markdown(f"<div style='font-size:20px; font-weight:bold'>{disease_names_jp.get(lbl, lbl)}: {score * 100:.2f}%</div>", unsafe_allow_html=True)

# フッター
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>©2025 Growth and Progress</p>", unsafe_allow_html=True)
