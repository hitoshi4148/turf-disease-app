# 芝しごと・芝生病害画像診断AI

芝生の病斑画像を使って病害を推定する、Streamlit ベースの診断アプリです。  
公開利用を想定し、**一般利用者向けの操作手順**と**開発者向けの学習手順**をまとめています。

---

## v0.92 → v1.0.1 アップデート要点

- これまでの病害分類AIから、**公開運用を優先した単一モデル診断**へ最適化
- 学習データを大幅に増やし、モデル自体の学習品質を改善
- 推論モデルを最新構成（MobileNetV3-Small）へ更新
- 芝種選択・症状特徴チェックによる重み付けで、病害分類精度の安定化を強化

---

## このアプリの診断方式（3段階）

本プロジェクトは、最終的な病害判定までを次の 3 段階で行います。

1. **AI分類（単一モデル）**  
   入力画像を MobileNetV3-Small で分類し、クラス確率を算出します。

2. **芝種ベース補正（暖地型 / 寒地型）**  
   発生しづらい病害クラスの確率を抑制します。

3. **知識ベース補正（芝種 + 症状チェック）**  
   - 芝種（暖地型 / 寒地型）情報で最終確率を安定化
   - 症状チェック（円形パッチ、赤い糸、水浸状、リング状）に応じて関連病害を加重
   - 最後に確率を再正規化して Top-N を表示

> 補正は「完全除外」ではなく、確率を重み付けして最終判定の安定化を狙う仕組みです。

---

## 主な機能

- 画像アップロード（病斑画像）
- 芝種選択（暖地型芝 / 寒地型芝）
- 症状チェックによる補助入力
- 単一モデル推論（MobileNetV3-Small）
- Top10 予測表示（確率バー付き）
- 病害説明表示（症状 / 管理方法 / 推奨薬剤系統）
- 参考画像表示（存在しない場合は案内表示）

---

## 推論モデル構成

- **分類モデル**: `MobileNetV3-Small`
- **推論フロー**: 単一モデル分類 + 芝種/症状による確率補正
- **表示**: Top10 候補（確率バー）

---

## 一般利用者向け 使い方

> v1.0.1 公開版は **PCブラウザ推奨** です。スマートフォンでは正常に動作しない場合があります。

1. アプリを起動する  
2. 「芝生の写真をアップロードしてください」から病斑画像を選ぶ  
3. 芝種を選択する（暖地型 / 寒地型）  
4. 症状の特徴を分かる範囲でチェックする  
5. 「AI診断を開始」を押す  
6. 診断結果カード、Top10予測、参考画像を確認する

### 診断結果の見方

- **病名 / 信頼度**: 最終判定の代表結果
- **Top10予測**: 近い候補を含む確率ランキング
- **症状 / 管理方法 / 推奨薬剤系統**: disease_info.json から表示

---

## 開発者向け セットアップ

## 1) 仮想環境と依存インストール（例）

```bash
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install torch torchvision streamlit scikit-learn tqdm pillow
```

## 2) データ準備

元データ（例）:

```text
data_raw/
  anthracnose_decline/
  brown_patch/
  dollar_spot/
  fairy_ring/
  healthy/
  large_patch/
  leaf_spot/
  pythium/
  red_thread/
  snow_mold/
  take_all_patch/
  ...
```

## 3) 学習

```bash
python train.py
```

出力（models）:

```text
best_model.pth
```

公開用配置（app.py 既定パス）:

```text
models/disease_resnet18_best.pth
```

`best_model.pth` を公開用に使う場合は、以下のいずれかで統一してください。

- `best_model.pth` を `models/disease_resnet18_best.pth` にコピー（またはリネーム）
- もしくは `app.py` の `MODEL_PATH` を `best_model.pth` に変更

出力（クラス順）:

```text
class_names.json
```

> `class_names.json` は app 側でフォールバック利用され、**学習時と推論時のクラス順ズレを防止**します。

## 4) アプリ起動

```bash
streamlit run app.py
```

---

## モデルと推論の整合ルール（重要）

精度劣化や誤判定を避けるため、以下を必ず守ってください。

- `app.py` の `MODEL_PATH` と実ファイル配置を一致させる
- `class_names.json` を最新学習結果で更新
- 学習後に旧モデルを残したまま動かさない
- 入力画像の撮影条件を統一（距離、明るさ、ピント）

---

## よくあるエラーと対処

### 1) モデルが見つからない

- `models/disease_resnet18_best.pth`（または `MODEL_PATH` で指定したファイル）
- `class_names.json`

上記の存在を確認し、なければ再学習してください。

### 2) state_dict 読み込みエラー

- 学習時と推論時でモデル構造が一致しているか確認
- 旧モデルを参照していないか確認

### 3) 予測が不安定

- 芝種選択、症状チェック、撮影条件を見直す
- 病斑が明確に写る画像を使用する

---

## 参考ファイル

- `app.py` : Streamlit アプリ本体
- `train.py` : 単一モデル学習（MobileNetV3-Small）
- `disease_info.json` : 病害説明データ
- `export_onnx.py` : PyTorch → ONNX 変換（ポータル公開用）
- `validate_onnx.py` : PyTorch / ONNX 推論照合

---

## ポータル公開用 ONNX エクスポート（tool-portal 連携）

本番 UI は [`tool-portal`](https://github.com/hitoshi4148/tool-portal) の `/portal/diagnosis/` でブラウザ内推論します。  
学習・変換はこのリポジトリ、公開物の配置は `tool-portal` 側です。

```bash
pip install onnx onnxscript onnxruntime
python export_onnx.py
python validate_onnx.py
```

`tool-portal/public/portal/diagnosis/` へコピーするファイル:

- `model.onnx`
- `class_names.json`
- `disease_info.json`
- `ui_images/`（撮影例・症状例）
- `images/`（病害参考画像）

学習用画像（`data_raw/` 等）は **tool-portal に含めない** でください。

---

## 注意事項

- 本アプリは意思決定支援ツールです。最終判断は現場状況と専門家確認を推奨します。
- 撮影条件やデータ分布により診断精度は変動します。
- 推論モデルには `MobileNetV3-Small`（単一モデル分類）を使用しています。
- v1.0.1 公開版は PCブラウザ推奨です（スマートフォン非対応）。

