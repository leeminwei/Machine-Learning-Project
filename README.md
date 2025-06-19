# 模器學習系列作業專案

本專案包含四個關聯作業，分別是決策樹分析、分類比較、純素神經網路回歸預測、CNN 情緒辨識與錯誤分析，適合學生深入了解算法效能、樣本特性處理與誤差分析。

---

## 🔹 A1: 決策樹測試和設定 min\_samples\_leaf

* 使用 sklearn 實作 DecisionTreeClassifier
* 2 個 OpenML 資料集，使用 entropy 為觀察基準
* 使用10-fold 交叉驗證，調整 `min_samples_leaf`
* 繪製 AUC 最佳值線圖與 ROC Curve

供使用者:

* Python 3.12
* pip install numpy pandas matplotlib scikit-learn
* OpenML 資料集瀏覽: [https://www.openml.org/search?type=data](https://www.openml.org/search?type=data)

---

## 🔹 A2: 分類機器比較 (決策樹, KNN, NB, LR, Dummy)

* 選用2 個具有名目 target + 名目 feature 的資料集
* 5 個分類法: Decision Tree, kNN, MultinomialNB, LogisticRegression, DummyClassifier
* 使用 GridSearchCV 調優待比較參數：

  * Decision Tree: min\_samples\_leaf
  * kNN: n\_neighbors
* 最終用 AUC (or weighted AUC) 評估

系統環境:

* Python 3.12
* pip install numpy pandas scikit-learn

---

## 🔹 A3: Keras 純素神經網路回歸預測

* 2 個 OpenML 數值回歸資料集 (>=1000 examples)
* 範本建構 3 種純素網路:

  * 少量細胞
  * 適中細胞
  * 大量細胞
* 傳回 Training / Validation / Test set
* 繪製 Training 和 Validation loss 變化圖
* 評估 MSE (Mean Squared Error)

系統環境:

* Python 3.11
* pip install numpy pandas matplotlib tensorflow scikit-learn

---

## 🔹 A4: CNN 情緒分類與誤差分析

* 資料集: 下載 [6 Emotions for Image Classification](https://www.kaggle.com/datasets)
* Task 1: 自建 2 種 CNN 組合比較
* Task 2: 展開 Fine-tuned VGG16 預訓練網路
* Task 3: 使用 10 張外部情緒圖片進行誤差分析
* 輸出檔案: .keras 檔、每樣機型正確率和預測結果

系統環境:

* Python 3.12
* pip install numpy pandas matplotlib tensorflow scikit-learn
* 須準備自行採集或搜尋的 10 張情緒圖片 (jpeg / png)

---

## 💾 文件設計和基礎構成

```
ML_Assignments/
├── project1
├── project2
├── project3
├── project4

```

---

## 🎓 推薦執行流程

1. 確認 Python 環境 (3.11 或 3.12)
2. 安裝相關套件
3. 自 OpenML 下載符合條件的資料集
4. 直接執行 `.py` 檔即可產生結果與圖表

---
