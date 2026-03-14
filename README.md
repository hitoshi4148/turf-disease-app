# 芝しごと・芝生病害画像診断AI

芝生の病斑画像を使って病害を推定する、Streamlit ベースの診断アプリです。  
公開利用を想定し、**一般利用者向けの操作手順**と**開発者向けの学習手順**をまとめています。

---

## v0.92 → v1.0.0 アップデート要点

- これまでの病害分類AIから、**健全/病害の判定を含む2段階診断**へ拡張
- 学習データを大幅に増やし、モデル自体の学習品質を改善
- 推論モデルを最新構成（EfficientNetV2-SのTwo-Stage）へ更新
- 芝種選択・症状特徴チェックによる重み付けで、病害分類精度の安定化を強化

---

## このアプリの診断方式（3段階）

本プロジェクトは、最終的な病害判定までを次の 3 段階で行います。

1. **Stage1: 健全芝 / 病害芝の2値判定**  
   まず入力画像が healthy か disease かを判定します。

2. **Stage2: 病害の詳細分類**  
   Stage1 が disease のときだけ、10 病害クラスの詳細分類を実行します。

3. **知識ベース補正（芝種 + 症状チェック）**  
   - 芝種（暖地型 / 寒地型）に応じて、発生しづらい病害の確率を抑制
   - 症状チェック（円形パッチ、赤い糸、水浸状、リング状）に応じて関連病害を加重
   - 最後に確率を再正規化して Top-N を表示

> 補正は「完全除外」ではなく、確率を重み付けして最終判定の安定化を狙う仕組みです。

---

## 主な機能

- 画像アップロード（病斑画像）
- 芝種選択（暖地型芝 / 寒地型芝）
- 症状チェックによる補助入力
- 2段階推論（Stage1 → Stage2）
- Top10 予測表示（確率バー付き）
- 病害説明表示（症状 / 管理方法 / 推奨薬剤系統）
- 参考画像表示（存在しない場合は案内表示）

---

## 推論モデル構成

- **Stage1（二値分類）**: `EfficientNetV2-S`  
  healthy / disease を判定
- **Stage2（病害詳細分類）**: `EfficientNetV2-S`  
  10病害クラスを分類
- **推論フロー**: Two-Stage（Stage1 → Stage2）+ 芝種/症状による確率補正

---

## 一般利用者向け 使い方

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

## 2) データ準備（Two-Stage）

元データ（例）:

```text
data_raw/
  healthy/
  brown_patch/
  ...
```

Two-Stage 用データ生成:

```bash
python prepare_two_stage_dataset.py
```

生成先:

```text
data_two_stage/
  stage1_binary/
    healthy/
    disease/
  stage2_disease/
    anthracnose_decline/
    ...
```

## 3) 学習

Stage1 学習:

```bash
python train_stage1.py
```

Stage2 学習:

```bash
python train_stage2.py
```

出力（models）:

```text
models/stage1_binary_model.pth
models/stage2_disease_model.pth
```

出力（クラス順）:

```text
class_names_stage1.json
class_names.json
```

> `class_names.json` は app 側で読み込み、**学習時と推論時のクラス順ズレを防止**します。

## 4) アプリ起動

```bash
streamlit run app.py
```

---

## モデルと推論の整合ルール（重要）

精度劣化や誤判定を避けるため、以下を必ず守ってください。

- Stage1/Stage2 のモデルファイルを `models/` 配下に配置
- `class_names.json` を最新の Stage2 学習結果で更新
- 学習後に旧モデルを残したまま動かさない
- 入力画像の撮影条件を統一（距離、明るさ、ピント）

---

## よくあるエラーと対処

### 1) モデルが見つからない

- `models/stage1_binary_model.pth`
- `models/stage2_disease_model.pth`
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
- `prepare_two_stage_dataset.py` : Two-Stage 用データ生成
- `train_stage1.py` : healthy vs disease 学習
- `train_stage2.py` : disease 詳細分類学習
- `disease_info.json` : 病害説明データ

---

## 注意事項

- 本アプリは意思決定支援ツールです。最終判断は現場状況と専門家確認を推奨します。
- 撮影条件やデータ分布により診断精度は変動します。
- 推論モデルには `EfficientNetV2-S`（Two-Stage分類）を使用しています。

