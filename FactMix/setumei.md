# FactMix 実装の詳細解説

## 全体像

FactMixは **2種類のデータ拡張（context-level と entity-level）** を組み合わせたパイプラインです。
流れとしては：

1. **入力**：ラベル付き NER データセット（例: JSON形式で `text` + `entities`）
2. **Context-level Semi-fact Generation**：非エンティティ語（Oタグ）をMLMで置換
3. **Entity-level Semi-fact Generation**：エンティティ語を同タイプの他の語に置換
4. **フィルタリング**：拡張後のデータをNERモデルで再予測 → 正しくラベル付けできたものだけ残す
5. **Mix Up**：オリジナルと拡張サンプルを一定比率で混合して学習データを構成

---

## ステップ 1. データ前処理

NERデータを次の形式で扱うと便利です：

```python
{
  "text": "Germany imported 47,600 sheep from Britain last year.",
  "entities": [
      {"span": [0, 7], "type": "LOC", "name": "Germany"},
      {"span": [37, 44], "type": "LOC", "name": "Britain"}
  ]
}
```

* `entities` の位置情報をトークン化に合わせて調整する（BERT系なら WordPiece/SentencePiece のオフセット処理が必要）。
* **Oタグトークン（非エンティティ）とエンティティトークンを区別**できるようにする。

---

## ステップ 2. Context-level Semi-fact Generation

### 目的

非エンティティ語（O）をMLMで置換 → 文脈をゆるやかに変え、スプリアスな依存を崩す。

### 実装

1. 文からランダムに1つの O トークンを選択。

2. そのトークンを `[MASK]` に置換。

3. Hugging Face Transformers の `fill-mask` パイプラインを使って置換候補を生成。

   ```python
   from transformers import pipeline
   unmasker = pipeline("fill-mask", model="bert-base-cased")

   masked_text = "Germany imported 47,600 [MASK] from Britain last year."
   predictions = unmasker(masked_text, top_k=5)
   ```

4. 予測結果の1位を置換候補として使用。

   * 論文では「1〜2語置換」「top-1のみ」採用。
   * 多すぎるとノイズが増えるため制限する。

5. 生成例：

   * `sheep` → `coffee`
   * `sheep` → `machines`
     → これが **context-level semi-fact サンプル**。

---

## ステップ 3. Entity-level Semi-fact Generation

### 目的

エンティティ表記を置換して多様性を増やす。

### 実装

1. 学習データ全体から **Entity\_Base** を作成（タイプごとにリスト化）。

   ```python
   Entity_Base = {
     "LOC": ["Germany", "Britain", "France", "Japan"],
     "PER": ["Alice", "John", "Mary"],
     ...
   }
   ```
2. 文中のエンティティを1つ選び、同タイプの別のエンティティで置換。

   * `Germany` (LOC) → `Israel` (LOC)
3. ラベルはそのまま維持。

生成例：

```
Germany imported 47,600 sheep from Britain last year.
↓
Israel imported 47,600 sheep from Britain last year.
```

---

## ステップ 4. フィルタリング（ノイズ除去）

Context-level は特にノイズが多いため、フィルタリングが重要。
論文では次の方法：

1. オリジナルデータで学習したNERモデルを用意。
2. 拡張データをモデルに通す。
3. **全トークンのラベルが正しく予測されたサンプルだけ採用**。

   * Zeng et al. (2020) のように「エンティティだけ正しければOK」ではなく、より厳しい条件。

---

## ステップ 5. Mix Up（比率調整）

* 論文の推奨比率：

  * **Entity-level** augmentation: 最大 1:8
  * **Context-level** augmentation: 最大 1:5
* 実際には **開発セットでグリッドサーチ**して最適比率を選ぶ。

最終的に：

```
Training Set = Original + α * Entity-level + β * Context-level
```

---

## ステップ 6. 学習方法

FactMixは **モデル非依存** なので、通常のNER学習にそのまま使えます。

### Fine-tuning方式

* BERT / RoBERTa + 線形分類層
* 損失関数：クロスエントロピー
* optimizer: AdamW
* サンプリング：100-shot per class など few-shot 設定で実験

### Prompt-tuning方式

* EntLM (Ma et al., 2021b) などをベースに、拡張データを投入
* 5-shot per class で性能を比較

---

## ステップ 7. 評価

* 指標：Micro F1
* 設定：

  * In-domain: 学習・評価が同一コーパス
  * Out-of-domain: 異なるコーパス（例: Reutersで学習→Scienceでテスト）

---

## 実装のポイント

* **トークンオフセットの一貫性**を確保する（置換後の文字列でもエンティティspanがずれないように調整）。
* **フィルタリングが精度に直結**する（無条件で生成例を使うと性能悪化）。
* **augmentation ratio** は必ず調整。過剰に増やすと精度が落ちる。

