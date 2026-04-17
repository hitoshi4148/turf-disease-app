import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import json
import numpy as np
import base64
import os
from urllib.parse import quote

st.set_page_config(
    page_title="芝しごと・芝生病害画像診断AI",
    layout="wide"
)


def inject_google_analytics(measurement_id: str) -> None:
    if st.session_state.get("_ga_injected"):
        return

    ga_html = f"""
    <script>
      (function() {{
        try {{
          var MID = "{measurement_id}";
          var w = window.parent || window;
          var d = w.document;
          if (!d || !d.head) return;

          var existing = d.querySelector('script[src*="googletagmanager.com/gtag/js?id=' + MID + '"]');
          if (!existing) {{
            var s = d.createElement('script');
            s.async = true;
            s.src = 'https://www.googletagmanager.com/gtag/js?id=' + MID;
            d.head.appendChild(s);
          }}

          w.dataLayer = w.dataLayer || [];
          w.gtag = w.gtag || function() {{ w.dataLayer.push(arguments); }};
          w.gtag('js', new Date());
          w.gtag('config', MID, {{
            page_path: w.location.pathname + w.location.search
          }});
        }} catch (e) {{
          // no-op
        }}
      }})();
    </script>
    """
    components.html(ga_html, height=0, width=0)
    st.session_state["_ga_injected"] = True


inject_google_analytics("G-FT1B3ZCT2B")

PR_BANNER_IMAGE_PATH = "ui_images/banner_pr_size1.png"
PR_BANNER_LINK = "https://www.turf-tools.jp/services-4"
BLOG_BANNER_IMAGE_PATH = "ui_images/bloglink.png"
YOUTUBE_BANNER_IMAGE_PATH = "ui_images/youtubelink.png"
BLOG_BANNER_LINK = "https://www.turf-tools.jp/blog"
YOUTUBE_BANNER_LINK = (
    "https://www.youtube.com/channel/UCSRU0zk4Fj1ETWqMRlJDPJQ"
)

_MOBILE_BANNER_MAX_WIDTH_PX = 768


def inject_pr_banner_before_header(image_path: str, link_url: str) -> None:
    if st.session_state.get("_pr_banner_injected"):
        return

    b64 = load_banner_base64(image_path)
    if not b64:
        return

    b64_blog = load_banner_base64(BLOG_BANNER_IMAGE_PATH) or ""
    b64_youtube = load_banner_base64(YOUTUBE_BANNER_IMAGE_PATH) or ""

    banner_html = f"""
    <script>
      (function() {{
        try {{
          var doc = (window.parent && window.parent.document) || document;
          var win = window.parent || window;
          var MOBILE_MQ_STR = "(max-width: {_MOBILE_BANNER_MAX_WIDTH_PX}px)";
          var BLOG_B64 = {json.dumps(b64_blog)};
          var YT_B64 = {json.dumps(b64_youtube)};
          var BLOG_URL = {json.dumps(BLOG_BANNER_LINK)};
          var YT_URL = {json.dumps(YOUTUBE_BANNER_LINK)};

          function isMobileWidth() {{
            return win.matchMedia(MOBILE_MQ_STR).matches;
          }}

          function removeDesktopStack() {{
            var el = doc.getElementById("turf-banner-stack");
            if (el) el.remove();
          }}

          function fixedTopInsetPx() {{
            var selectors = [
              '[data-testid="stToolbar"]',
              '[data-testid="stHeader"]',
              '[data-testid="stDecoration"]',
              "header"
            ];
            var maxBottom = 0;
            for (var i = 0; i < selectors.length; i++) {{
              var el = doc.querySelector(selectors[i]);
              if (!el) continue;
              var cs = win.getComputedStyle(el);
              var pos = cs.position;
              var topPx = parseFloat(cs.top);
              if (pos !== "fixed" && pos !== "sticky" &&
                  !(pos === "absolute" && !isNaN(topPx) && topPx <= 2)) {{
                continue;
              }}
              var r = el.getBoundingClientRect();
              if (r.height <= 0) continue;
              maxBottom = Math.max(maxBottom, r.bottom);
            }}
            return Math.ceil(maxBottom);
          }}

          function applyBannerTopGap(wrap) {{
            var gap = fixedTopInsetPx();
            if (gap <= 0) {{
              var h =
                doc.querySelector('[data-testid="stHeader"]') ||
                doc.querySelector("header");
              if (h) {{
                var br = wrap.getBoundingClientRect();
                var hr = h.getBoundingClientRect();
                if (br.top < hr.bottom - 0.5 && hr.height > 0) {{
                  gap = Math.ceil(hr.bottom - br.top);
                }}
              }}
            }}
            if (gap > 0) wrap.style.marginTop = gap + "px";
          }}

          function scheduleDesktopInsert() {{
            var tries = 0;
            function tick() {{
              if (isMobileWidth()) {{
                removeDesktopStack();
                return;
              }}
              if (doc.getElementById("turf-banner-stack")) return;
              var app = doc.querySelector("section.stApp");
              var headerEl =
                doc.querySelector('[data-testid="stHeader"]') ||
                doc.querySelector("header");
              var mount = app || (headerEl && headerEl.parentNode);
              if (!mount || !headerEl) {{
                if (tries++ < 80) win.setTimeout(tick, 50);
                return;
              }}

            var stack = doc.createElement("div");
            stack.id = "turf-banner-stack";
            stack.setAttribute(
              "style",
              "margin:0;padding:0;line-height:0;display:flex;flex-direction:column;"
              + "align-items:stretch;width:100%;box-sizing:border-box;"
              + "overflow:visible;flex:0 0 auto;flex-shrink:0;min-height:min-content;"
              + "position:relative;z-index:1;background:#faf8f2;"
            );

            var prRow = doc.createElement("div");
            prRow.id = "turf-pr-banner-wrap";
            prRow.setAttribute(
              "style",
              "margin:0;padding:0;line-height:0;text-align:center;background:#faf8f2;"
              + "overflow:visible;width:100%;box-sizing:border-box;"
            );

            var a = doc.createElement("a");
            a.href = {json.dumps(link_url)};
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            a.setAttribute("style", "display:block;line-height:0;margin:0;padding:0;");

            var img = doc.createElement("img");
            img.src = "data:image/png;base64,{b64}";
            img.alt = "芝管理のプロにPRしませんか";
            img.setAttribute(
              "style",
              "width:auto;height:auto;max-width:100%;display:block;margin:0 auto;"
              + "vertical-align:top;"
            );
            img.onload = function() {{ applyBannerTopGap(stack); }};

            a.appendChild(img);
            prRow.appendChild(a);
            stack.appendChild(prRow);

            if (BLOG_B64 || YT_B64) {{
              var sub = doc.createElement("div");
              sub.id = "turf-sub-banners-wrap";
              sub.setAttribute(
                "style",
                "display:flex;flex-direction:row;justify-content:center;align-items:flex-start;"
                + "flex-wrap:wrap;gap:8px;width:100%;box-sizing:border-box;"
                + "margin:0;padding:4px 8px 8px;line-height:0;background:#faf8f2;"
              );

              function addSmallBanner(href, b64data, altText) {{
                if (!b64data) return;
                var la = doc.createElement("a");
                la.href = href;
                la.target = "_blank";
                la.rel = "noopener noreferrer";
                la.setAttribute(
                  "style",
                  "display:block;line-height:0;margin:0;padding:0;flex:0 0 auto;"
                );
                var im = doc.createElement("img");
                im.src = "data:image/png;base64," + b64data;
                im.alt = altText;
                im.setAttribute("width", "300");
                im.setAttribute("height", "100");
                im.setAttribute(
                  "style",
                  "width:300px;height:100px;object-fit:contain;display:block;"
                  + "box-sizing:border-box;vertical-align:top;"
                );
                la.appendChild(im);
                sub.appendChild(la);
              }}

              addSmallBanner(BLOG_URL, BLOG_B64, "芝管理技術ブログ");
              addSmallBanner(YT_URL, YT_B64, "芝管理ノウハウ YouTube");
              stack.appendChild(sub);
            }}

            if (app) {{
              app.insertBefore(stack, app.firstChild);
            }} else {{
              mount.insertBefore(stack, headerEl);
            }}

            applyBannerTopGap(stack);
            requestAnimationFrame(function() {{
              requestAnimationFrame(function() {{ applyBannerTopGap(stack); }});
            }});

            if (!doc.getElementById("turf-pr-banner-style")) {{
              var st = doc.createElement("style");
              st.id = "turf-pr-banner-style";
              st.textContent =
                "#turf-banner-stack,#turf-pr-banner-wrap,#turf-pr-banner-wrap a,"
                + "#turf-pr-banner-wrap img,#turf-sub-banners-wrap,"
                + "#turf-sub-banners-wrap a,#turf-sub-banners-wrap img{{"
                + "max-height:none!important;clip:auto!important;object-fit:contain;"
                + "}}";
              doc.head.appendChild(st);
            }}
            }}
            tick();
          }}

          function syncBannerMountMode() {{
            if (isMobileWidth()) {{
              removeDesktopStack();
              return;
            }}
            scheduleDesktopInsert();
          }}

          var mq = win.matchMedia(MOBILE_MQ_STR);
          if (mq.addEventListener) {{
            mq.addEventListener("change", syncBannerMountMode);
          }} else if (mq.addListener) {{
            mq.addListener(syncBannerMountMode);
          }}
          syncBannerMountMode();
        }} catch (e) {{}}
      }})();
    </script>
    """
    components.html(banner_html, height=0, width=0)
    st.session_state["_pr_banner_injected"] = True


# ======================
# 設定
# ======================
MODEL_PATH = "models/mobilenet_v3_small_best.pth"
CLASS_NAMES_PATH = "class_names.json"
DISEASE_INFO_PATH = "disease_info.json"
BANNER_IMAGE_PATH = r"C:\Users\hitos\.cursor\projects\c-Users-hitos-disease-classification\assets\c__Users_hitos_AppData_Roaming_Cursor_User_workspaceStorage_06f2bd11c3ead2a302f748a2d89a9f59_images_banner_ad_recruitment_728x90-30f0f326-eb56-4988-892f-cad746e7e45b.png"
MAX_UPLOAD_MB = 12
TURF_CLASS_PRIORS = {
    "暖地型芝": {
        "large_patch": 2.2,
        "take_all_patch": 1.4,
        "snow_mold": 0.10,
        "dollar_spot": 0.20,
        "anthracnose_decline": 0.30,
        "leaf_spot": 0.60,
        "red_thread": 0.50,
    },
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


def load_disease_info():
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_banner_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError:
        return None


def render_mobile_banners_inline() -> None:
    """狭い画面用: メインのスクロール領域内にバナーを置く（上半分固定を避ける）。"""
    pr_b64 = load_banner_base64(PR_BANNER_IMAGE_PATH)
    if not pr_b64:
        return
    b_blog = load_banner_base64(BLOG_BANNER_IMAGE_PATH) or ""
    b_yt = load_banner_base64(YOUTUBE_BANNER_IMAGE_PATH) or ""

    parts = [
        '<div class="turf-banner-inline-mobile-only">',
        '<a href="',
        PR_BANNER_LINK,
        '" target="_blank" rel="noopener noreferrer" '
        'style="display:block;line-height:0;margin:0;padding:0;">',
        '<img src="data:image/png;base64,',
        pr_b64,
        '" alt="芝管理のプロにPRしませんか" '
        'style="width:auto;height:auto;max-width:100%;display:block;margin:0 auto;"/>',
        "</a>",
    ]
    if b_blog or b_yt:
        parts.append(
            '<div class="turf-banner-subrow-mobile" '
            'style="display:flex;flex-wrap:wrap;justify-content:center;align-items:flex-start;'
            'gap:8px;margin:0;padding:8px 4px 4px;line-height:0;">'
        )
        if b_blog:
            parts.extend([
                '<a href="',
                BLOG_BANNER_LINK,
                '" target="_blank" rel="noopener noreferrer" '
                'style="display:block;line-height:0;flex:0 1 auto;max-width:100%;">',
                '<img src="data:image/png;base64,',
                b_blog,
                '" alt="芝管理技術ブログ" width="300" height="100" '
                'style="width:300px;max-width:100%;height:auto;aspect-ratio:3/1;'
                'object-fit:contain;display:block;box-sizing:border-box;"/>',
                "</a>",
            ])
        if b_yt:
            parts.extend([
                '<a href="',
                YOUTUBE_BANNER_LINK,
                '" target="_blank" rel="noopener noreferrer" '
                'style="display:block;line-height:0;flex:0 1 auto;max-width:100%;">',
                '<img src="data:image/png;base64,',
                b_yt,
                '" alt="芝管理ノウハウ YouTube" width="300" height="100" '
                'style="width:300px;max-width:100%;height:auto;aspect-ratio:3/1;'
                'object-fit:contain;display:block;box-sizing:border-box;"/>',
                "</a>",
            ])
        parts.append("</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


inject_pr_banner_before_header(PR_BANNER_IMAGE_PATH, PR_BANNER_LINK)


def prepare_uploaded_patch_image(uploaded_file, max_long_edge=1024):
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
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
        return None, "画像の読み込みに失敗しました。JPG/PNG/WEBP形式で再アップロードしてください。"


disease_info_data = load_disease_info()


def adjust_probabilities(
    probs,
    class_names,
    turf_type,
    symptom_patch,
    symptom_thread,
    symptom_water,
    symptom_ring,
):
    adjusted = np.asarray(probs, dtype=np.float64).copy()
    priors = TURF_CLASS_PRIORS.get(turf_type, {})

    for i, name in enumerate(class_names):
        n = name.lower().replace("_", "")
        adjusted[i] *= priors.get(name, 1.0)

        if symptom_thread and "redthread" in n:
            adjusted[i] *= 1.6
        if symptom_ring and "fairy" in n:
            adjusted[i] *= 1.25
        if symptom_water and "pythium" in n:
            adjusted[i] *= 1.20
        if symptom_patch and "dollar" in n:
            adjusted[i] *= 1.15

    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total
    return adjusted


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


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
    div.turf-banner-inline-mobile-only {
        margin: 0 0 0.75rem 0;
        padding: 0;
        line-height: 0;
        text-align: center;
        background: #faf8f2;
        width: 100%;
        box-sizing: border-box;
    }
    @media (min-width: 769px) {
        div.turf-banner-inline-mobile-only {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

render_mobile_banners_inline()

st.markdown('<h1 style="text-align:center;">芝しごと・芝生病害画像診断AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:gray;">v1.0.1</p>', unsafe_allow_html=True)
st.write("芝生の病斑写真をアップロードするとAIが病害を推定します。")
st.warning("v1.0.1 公開版はPCブラウザ推奨です。スマートフォンでは正常に動作しない場合があります。")

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
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)
patch_image = None

st.caption("対応形式：JPG / JPEG / PNG / WEBP（推奨: JPG）")
st.caption("スマホの『カメラ起動』は端末負荷が高いため、撮影後の画像ファイル選択を推奨します。")
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
    if uploaded_file is None:
        st.warning("病斑パッチの写真をアップロードしてください")
    else:
        file_name = (uploaded_file.name or "").lower()
        allowed_ext = (".jpg", ".jpeg", ".png", ".webp")
        if not file_name.endswith(allowed_ext):
            st.error("対応形式は JPG / JPEG / PNG / WEBP です。")
            st.stop()

        file_size_bytes = getattr(uploaded_file, "size", None)
        if isinstance(file_size_bytes, (int, float)):
            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb > MAX_UPLOAD_MB:
                st.error(
                    f"画像サイズが大きすぎます（{file_size_mb:.1f}MB）。"
                    f"{MAX_UPLOAD_MB}MB以下の画像で再アップロードしてください。"
                )
                st.stop()

        patch_image, image_error = prepare_uploaded_patch_image(uploaded_file, max_long_edge=1024)
        if image_error or patch_image is None:
            st.error(image_error or "画像の読み込みに失敗しました。")
            st.stop()

        st.image(patch_image, caption="アップロード画像", use_container_width=True)

        validate_required_files()
        try:
            model, class_names = load_model()
        except RuntimeError as e:
            st.error("モデルの読み込みに失敗しました。学習時と推論時のモデル構造・クラス順が一致しているか確認してください。")
            st.code(str(e))
            st.stop()

        input_tensor = transform(patch_image).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model(input_tensor)
            base_probs = torch.softmax(outputs, dim=1)
            adjusted_probs = adjust_probabilities(
                base_probs.squeeze(0).cpu().numpy(),
                class_names,
                turf_type,
                symptom_patch,
                symptom_thread,
                symptom_water,
                symptom_ring,
            )

            probs = torch.tensor(adjusted_probs, dtype=torch.float32).unsqueeze(0)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = class_names[pred_idx]
            confidence = probs[0][pred_idx].item()
            top_k = min(10, probs.size(1))
            top_prob, top_idx = torch.topk(probs, top_k)
            top_classes = [class_names[top_idx[0][rank].item()] for rank in range(top_k)]

        predicted_class = pred_class
        disease_query_name = DISEASE_QUERY_NAME_MAP.get(predicted_class, predicted_class)
        confidence_text = f"{confidence * 100:.0f}%"
        disease_info = disease_info_data.get(predicted_class, {})
        display_name = disease_info.get("name", predicted_class)

        st.subheader("診断結果")
        st.success(f"病名: {display_name} / 信頼度: {confidence_text}")
        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:
            st.markdown(f"### 🦠 {display_name}")
            st.metric("信頼度", confidence_text)
            st.markdown("<h3>症状</h3>", unsafe_allow_html=True)
            st.write(disease_info.get("symptom", ""))
            st.markdown("<h3>管理方法</h3>", unsafe_allow_html=True)
            st.write(disease_info.get("management", ""))
            st.markdown("<h3>推奨薬剤系統</h3>", unsafe_allow_html=True)
            st.write(disease_info.get("fungicide", ""))
            st.link_button(
                "この病害の防除農薬をみる",
                f"https://shigoto-raku-rac-rotate.onrender.com/?disease={quote(disease_query_name)}",
            )

            st.markdown("<h3>Top10予測</h3>", unsafe_allow_html=True)
            for rank in range(top_k):
                top_class = top_classes[rank]
                top_display_name = disease_info_data.get(top_class, {}).get("name", top_class)
                top_percent = top_prob[0][rank].item() * 100
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
    - 推論モデルには MobileNetV3-Small（単一モデル分類）を使用しています。<br>
    - v1.0.1公開版はPCブラウザ推奨です（スマートフォン非対応）。
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("©2025 Growth and Progress")
st.markdown("[グロウアンドプログレス](https://www.turf-tools.jp/)")
