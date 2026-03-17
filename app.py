import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import base64
import os
import gc
from urllib.parse import quote

st.set_page_config(
    page_title="芝しごと・芝生病害画像診断AI",
    layout="wide"
)

# ======================
# 設定
# ======================
MODEL_PATH = "models/mobilenet_v3_small_best.pth"
CLASS_NAMES_PATH = "class_names.json"
DISEASE_INFO_PATH = "disease_info.json"
BANNER_IMAGE_PATH = r"C:\Users\hitos\.cursor\projects\c-Users-hitos-disease-classification\assets\c__Users_hitos_AppData_Roaming_Cursor_User_workspaceStorage_06f2bd11c3ead2a302f748a2d89a9f59_images_banner_ad_recruitment_728x90-30f0f326-eb56-4988-892f-cad746e7e45b.png"
COOL_SEASON_DISEASES = {
    "anthracnose_decline",
    "brown_patch",
    "dollar_spot",
    "fairy_ring",
    "leaf_spot",
    "pythium",
    "red_thread",
    "snow_mold",
}
WARM_SEASON_DISEASES = {
    "large_patch",
    "take_all_patch",
    "pythium",
    "fairy_ring",
    "leaf_spot",
}
DISEASE_QUERY_NAME_MAP = {
    "anthracnose_decline": "炭疽病",
    "brown_patch": "ブラウンパッチ",
    "dollar_spot": "ダラースポット",
    "fairy_ring": "フェアリーリング",
    "large_patch": "ラージパッチ",
    "leaf_spot": "葉枯病",
    "pythium": "ピシウム",
    "red_thread": "赤葉腐病",
    "snow_mold": "雪腐病",
    "take_all_patch": "立枯病",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_required_files():
    missing = []
    if not os.path.exists(MODEL_PATH):
        missing.append(MODEL_PATH)

    if missing:
        st.error("推論に必要なファイルが不足しています。")
        for path in missing:
            st.write(f"- 未検出: `{path}`")
        st.write("- `python train.py` で単一モデルを再学習")
        st.write("- 学習後のモデルファイルを `models/` 配下に配置")
        st.stop()


# ======================
# モデル読み込み（キャッシュ）
# ======================
@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        return []
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("class_names", [])


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        loaded_class_names = checkpoint.get("class_names", [])
    else:
        state_dict = checkpoint
        loaded_class_names = []

    if not loaded_class_names:
        loaded_class_names = load_class_names()
    if not loaded_class_names:
        raise RuntimeError("class_names がモデルにも class_names.json にも存在しません。")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(loaded_class_names)
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, loaded_class_names


class_names = []


def load_disease_info():
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_banner_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError:
        return None


disease_info_data = load_disease_info()


def get_confidence_color(confidence_score):
    if confidence_score >= 80:
        return "green"
    if confidence_score >= 60:
        return "orange"
    return "red"


# 症状と芝種に応じて確率を軽く補正
def adjust_probabilities(probs, class_names, turf_type,
                         symptom_patch, symptom_thread,
                         symptom_water, symptom_ring):
    adjusted = np.asarray(probs, dtype=np.float64).copy()

    for i, name in enumerate(class_names):
        n = name.lower().replace("_", "")

        # 芝種補正
        if turf_type == "暖地型芝":
            if "snow" in n:
                adjusted[i] *= 0.2
        else:
            if "largepatch" in n:
                adjusted[i] *= 0.2

        # 症状補正
        if symptom_thread and "redthread" in n:
            adjusted[i] *= 2.0

        if symptom_ring and "fairy" in n:
            adjusted[i] *= 1.4

        if symptom_water and "pythium" in n:
            adjusted[i] *= 1.4

        if symptom_patch and "dollar" in n:
            adjusted[i] *= 1.2

    # 正規化
    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total

    return adjusted

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
st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 5rem;
        padding-right: 5rem;
    }
    div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        border-color: #2e7d32 !important;
    }
    div[role="radiogroup"] > label[data-baseweb="radio"][aria-checked="true"] {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 8px;
        padding: 6px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 style="text-align:center;">芝しごと・芝生病害画像診断AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:gray;">v1.0.0</p>', unsafe_allow_html=True)
st.write("芝生の病斑写真と葉写真をアップロードするとAIが病害を診断します。")

st.subheader("撮影方法")
col1, col2 = st.columns(2)
with col1:
    st.image("ui_images/photo_good.jpg")
    st.success("良い例：病斑がはっきり写っている")
with col2:
    st.image("ui_images/photo_bad.jpg")
    st.error("悪い例：遠すぎる / ピンぼけ / 影")

st.subheader("写真をアップロード")
uploaded_file = st.file_uploader(
    "芝生の写真をアップロードしてください",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)
patch_image = None

if uploaded_file is not None:
    patch_image = Image.open(uploaded_file).convert("RGB")
    st.image(patch_image, caption="アップロード画像", use_container_width=True)
st.caption("対応形式：JPG / JPEG / PNG")
st.caption("最大ファイルサイズ：200MB")

patch_name = uploaded_file.name if uploaded_file is not None else "未選択"
st.caption(f"現在使用中の patch画像: {patch_name}")

with st.form("diagnosis_form"):
    turf_type_label = st.radio(
        "芝種を選択してください",
        [
            "暖地型芝（野芝・高麗芝・バミューダ芝等）",
            "寒地型芝（ベントグラス・ライグラス・フェスク・ケンタッキーブルーグラス等）"
        ],
        horizontal=True
    )
    turf_type = "暖地型芝" if "暖地型芝" in turf_type_label else "寒地型芝"
    st.info(f"現在の芝種: {turf_type}")

    st.subheader("症状の特徴（わかる範囲で選択）")
    col1, col2 = st.columns(2)
    with col1:
        st.image("ui_images/symptom_patch.jpg", width=420)
        symptom_patch = st.checkbox("円形パッチ", key="symptom_patch")
        st.image("ui_images/symptom_thread.jpg", width=420)
        symptom_thread = st.checkbox("赤い糸", key="symptom_thread")
    with col2:
        st.image("ui_images/symptom_water.jpg", width=420)
        symptom_water = st.checkbox("水浸状", key="symptom_water")
        st.image("ui_images/symptom_ring.jpg", width=420)
        symptom_ring = st.checkbox("リング状", key="symptom_ring")

    diagnose_button = st.form_submit_button("AI診断を開始")

if diagnose_button:
    if patch_image is None:
        st.warning("病斑パッチの写真をアップロードしてください")
    else:
        validate_required_files()
        try:
            model, class_names = load_model()
        except RuntimeError as e:
            st.error("モデルの読み込みに失敗しました。学習時と推論時のモデル構造・クラス順が一致しているか確認してください。")
            st.code(str(e))
            st.write("- `train.py` 実行後の最新モデルを使用")
            st.write("- モデル内 `class_names` と `class_names.json` の整合を確認")
            st.stop()

        image = patch_image
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model(input_tensor)
            base_probs = torch.softmax(outputs, dim=1)
            final_probs = base_probs.clone()

            excluded_diseases = set()
            if turf_type == "寒地型芝":
                excluded_diseases = WARM_SEASON_DISEASES - COOL_SEASON_DISEASES
            elif turf_type == "暖地型芝":
                excluded_diseases = COOL_SEASON_DISEASES - WARM_SEASON_DISEASES

            for idx, cls in enumerate(class_names):
                if cls in excluded_diseases:
                    final_probs[0, idx] *= 0.2

            prob_sum = final_probs.sum(dim=1, keepdim=True)
            final_probs = final_probs / prob_sum.clamp(min=1e-12)

            final_probs_np = final_probs.squeeze(0).cpu().numpy()
            adjusted_probs = adjust_probabilities(
                final_probs_np,
                class_names,
                turf_type,
                symptom_patch,
                symptom_thread,
                symptom_water,
                symptom_ring
            )

            probs = torch.tensor(adjusted_probs, dtype=torch.float32).unsqueeze(0)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = class_names[pred_idx]
            confidence = probs[0][pred_idx].item()
            top_k = min(10, probs.size(1))
            top3_prob, top3_idx = torch.topk(probs, top_k)
            top3_classes = [
                class_names[top3_idx[0][rank].item()]
                for rank in range(top_k)
            ]
            del outputs, base_probs, final_probs

        probability_map = {class_name: float(adjusted_probs[i] * 100) for i, class_name in enumerate(class_names)}
        display_probabilities = {
            class_name: probability_map.get(class_name, 0.0)
            for class_name in class_names
        }

        predicted_class = pred_class
        disease_query_name = DISEASE_QUERY_NAME_MAP.get(predicted_class, predicted_class)
        confidence_score = confidence * 100
        confidence = f"{confidence_score:.0f}%"
        disease_info = disease_info_data.get(predicted_class, {})
        display_name = disease_info.get("name", predicted_class)

        st.subheader("診断結果")
        st.success(f"病名: {display_name} / 信頼度: {confidence}")
        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:
            st.markdown(f"### 🦠 {display_name}")
            st.metric("信頼度", confidence)
            st.markdown("<h3>症状</h3>", unsafe_allow_html=True)
            st.write(disease_info.get("symptom", ""))
            st.markdown("<h3>管理方法</h3>", unsafe_allow_html=True)
            st.write(disease_info.get("management", ""))
            st.markdown("<h3>推奨薬剤系統</h3>", unsafe_allow_html=True)
            st.write(disease_info.get("fungicide", ""))
            st.link_button(
                "この病害の防除農薬をみる",
                f"https://shigoto-raku-rac-rotate.onrender.com/?disease={quote(disease_query_name)}"
            )

            st.markdown("<h3>Top10予測</h3>", unsafe_allow_html=True)
            for rank in range(top_k):
                top_class = top3_classes[rank]
                top_display_name = disease_info_data.get(top_class, {}).get("name", top_class)
                top_percent = top3_prob[0][rank].item() * 100
                st.write(f"{rank + 1}位 {top_display_name} {top_percent:.0f}%")
                st.progress(int(round(top_percent)))

        with result_col2:
            class_key = predicted_class.lower().replace(" ", "_")
            image_path = None
            for ext in ["jpg", "jpeg", "png"]:
                path = os.path.join("images", f"{class_key}.{ext}")
                if os.path.exists(path):
                    image_path = path
                    break

            if image_path:
                st.image(image_path, caption="参考画像", use_container_width=True)
            else:
                st.write("参考画像は現在準備中です")

        del input_tensor, probs
        gc.collect()

st.divider()
mailto_url = f"mailto:growthandprogress4148.gmail.com?subject={quote('バナー広告について')}"
banner_base64 = load_banner_base64(BANNER_IMAGE_PATH)
if banner_base64:
    st.markdown(
        f"""
        <a href="{mailto_url}">
            <img src="data:image/png;base64,{banner_base64}" style="width:100%;max-width:728px;" />
        </a>
        """,
        unsafe_allow_html=True,
    )
else:
    st.link_button("バナー広告について", mailto_url)

st.markdown(
    """
    <div style="font-size:0.85rem; color:#666; line-height:1.5; margin-top:0.5rem;">
    本アプリは意思決定支援ツールです。最終判断は現場状況と専門家確認を推奨します。<br>
    - 撮影条件やデータ分布により診断精度は変動します。<br>
    - 推論モデルには学習済み単一分類モデルを使用しています。
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("©2025 Growth and Progress")
st.markdown("[グロウアンドプログレス](https://www.turf-tools.jp/)")