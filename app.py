import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
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
MAX_UPLOAD_MB = 12
TURF_CLASS_PRIORS = {
    # 暖地型芝: large_patch を強め、寒地型で多い病害を抑制
    "暖地型芝": {
        "large_patch": 2.2,
        "take_all_patch": 1.4,
        "snow_mold": 0.10,
        "dollar_spot": 0.20,
        "anthracnose_decline": 0.30,
        "leaf_spot": 0.60,
        "red_thread": 0.50,
    },
    # 寒地型芝: large_patch を強く抑制し、寒地型で多い病害を優遇
    "寒地型芝": {
        "large_patch": 0.05,
        "take_all_patch": 0.40,
        "snow_mold": 1.40,
        "dollar_spot": 1.25,
        "anthracnose_decline": 1.25,
        "leaf_spot": 1.15,
        "red_thread": 1.10,
    },
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
torch.set_num_threads(1)


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


@st.cache_data
def load_optimized_image_bytes(path, max_width=800, quality=72):
    # 低メモリ環境（Render free tier）ではサーバー側再エンコードを無効化
    # st.image(path) にフォールバックしてメモリスパイクを避ける
    return None


def prepare_uploaded_patch_image(uploaded_file, max_long_edge=1280):
    try:
        file_name = (uploaded_file.name or "").lower()
        if file_name.endswith(".heic") or file_name.endswith(".heif"):
            return None, (
                "HEIC/HEIF形式は現在の公開環境では不安定です。"
                "JPG/PNGに変換して再アップロードしてください。"
            )

        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            # スマホ撮影画像の向きをEXIFに従って補正
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")

            width, height = img.size
            long_edge = max(width, height)
            if long_edge > max_long_edge:
                scale = max_long_edge / float(long_edge)
                resized = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale))
                )
                img = img.resize(resized, Image.Resampling.LANCZOS)

            return img, None
    except Exception:
        return None, (
            "画像の読み込みに失敗しました。"
            "iPhoneのHEIC形式や破損ファイルの可能性があります。"
            "JPG/PNGに変換して再アップロードしてください。"
        )


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
    priors = TURF_CLASS_PRIORS.get(turf_type, {})

    for i, name in enumerate(class_names):
        n = name.lower().replace("_", "")

        # 芝種補正
        adjusted[i] *= priors.get(name, 1.0)

        # 症状補正
        if symptom_thread and "redthread" in n:
            adjusted[i] *= 1.6

        if symptom_ring and "fairy" in n:
            adjusted[i] *= 1.25

        if symptom_water and "pythium" in n:
            adjusted[i] *= 1.20

        if symptom_patch and "dollar" in n:
            adjusted[i] *= 1.15

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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
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
st.write("芝生の病斑写真をアップロードするとAIが病害を推定します。")
light_mode = st.checkbox("スマホ軽量モード（通信・メモリ節約）", value=True)

st.subheader("撮影方法")
if light_mode:
    st.info("軽量モードでは撮影参考画像を省略しています。病斑がはっきり見える近距離・明るい場所で撮影してください。")
else:
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
    accept_multiple_files=False
)
patch_image = None

if uploaded_file is not None:
    file_name = (uploaded_file.name or "").lower()
    allowed_ext = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")
    if not file_name.endswith(allowed_ext):
        st.error("対応形式は JPG / JPEG / PNG / WEBP です。HEIC/HEIFはJPG変換後にアップロードしてください。")
    else:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_MB:
            st.error(
                f"画像サイズが大きすぎます（{file_size_mb:.1f}MB）。"
                f"{MAX_UPLOAD_MB}MB以下の画像で再アップロードしてください。"
            )
        else:
            patch_image, image_error = prepare_uploaded_patch_image(uploaded_file, max_long_edge=1024)
            if image_error:
                st.error(image_error)
            elif patch_image is not None and not light_mode:
                st.image(patch_image, caption="アップロード画像", use_container_width=True)
st.caption("対応形式：JPG / JPEG / PNG / WEBP（推奨: JPG）")
st.caption("スマホの『カメラ起動』は端末負荷が高いため、撮影後の画像ファイル選択を推奨します。")
st.caption("iPhoneは『設定 > カメラ > フォーマット > 互換性優先』でJPG保存に変更できます。")
st.caption("高解像度画像は自動で縮小してから推論します（長辺1024px）。")
st.caption(f"最大ファイルサイズ：{MAX_UPLOAD_MB}MB")

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
        if not light_mode:
            st.image("ui_images/symptom_patch.jpg", width=420)
        symptom_patch = st.checkbox("円形パッチ", key="symptom_patch")
        if not light_mode:
            st.image("ui_images/symptom_thread.jpg", width=420)
        symptom_thread = st.checkbox("赤い糸", key="symptom_thread")
    with col2:
        if not light_mode:
            st.image("ui_images/symptom_water.jpg", width=420)
        symptom_water = st.checkbox("水浸状", key="symptom_water")
        if not light_mode:
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
            final_probs_np = base_probs.squeeze(0).cpu().numpy()
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
            healthy_override_used = False
            pred_idx = torch.argmax(probs, dim=1).item()

            # healthy過判定を抑える安全弁:
            # 症状入力があるのにhealthyが低信頼なら、次点の病害候補を優先する
            symptoms_selected = symptom_patch or symptom_thread or symptom_water or symptom_ring
            if "healthy" in class_names and probs.size(1) >= 2:
                healthy_idx = class_names.index("healthy")
                if pred_idx == healthy_idx:
                    healthy_prob = probs[0][healthy_idx].item()
                    top2_prob, top2_idx = torch.topk(probs, k=2)
                    second_idx = top2_idx[0][1].item()
                    if symptoms_selected and healthy_prob < 0.85:
                        pred_idx = second_idx
                        healthy_override_used = True

            pred_class = class_names[pred_idx]
            confidence = probs[0][pred_idx].item()
            top_k = min(10, probs.size(1))
            top3_prob, top3_idx = torch.topk(probs, top_k)
            top3_classes = [
                class_names[top3_idx[0][rank].item()]
                for rank in range(top_k)
            ]
            del outputs, base_probs

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
        if healthy_override_used:
            st.warning("症状入力と判定の整合を考慮し、healthy判定ではなく次点の病害候補を表示しています。")
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

            if image_path and not light_mode:
                st.image(image_path, caption="参考画像", use_container_width=True)
            elif image_path and light_mode:
                st.caption("軽量モードのため参考画像を省略しています。")
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