import streamlit as st

st.set_page_config(page_title="Upload Probe", layout="centered")

st.title("Upload Probe (Minimal)")
st.write("スマホ側のアップロード動作を切り分けるための最小テスト画面です。")

uploaded_file = st.file_uploader(
    "画像ファイルを1枚選択してください",
    accept_multiple_files=False
)

if uploaded_file is None:
    st.info("まだファイル未選択です。")
else:
    st.success("ファイル選択イベントはサーバーに到達しました。")
    st.write("ファイル名:", uploaded_file.name)
    st.write("MIME:", uploaded_file.type or "不明")
    size = getattr(uploaded_file, "size", None)
    if isinstance(size, (int, float)):
        st.write("サイズ(MB):", f"{size / (1024 * 1024):.3f}")
    else:
        st.write("サイズ(MB): 不明")
