
---

# 📊 使用 scikit-learn 進行決策樹 AUC 分析

本專案旨在利用 `scikit-learn` 套件，針對兩組不同的二元分類資料集，分析決策樹中 `min_samples_leaf` 參數對 AUC（曲線下面積）表現的影響，並透過繪製 ROC 曲線與 AUC 變化圖，觀察模型的過擬合與欠擬合情形。

---

## 📁 專案內容

* `ML1.py` – 主程式檔，包含：

  * 從 OpenML 載入兩筆資料集（Bioresponse 與 USPS）
  * 訓練不同 `min_samples_leaf` 值的決策樹模型
  * 使用 10-fold 交叉驗證評估模型
  * 繪製 AUC 對參數變化圖
  * 顯示最佳模型的 ROC 曲線

---

## 🔍 資料來源說明

本專案使用的資料集來自 [OpenML 資料庫](https://www.openml.org/search?type=data)。

挑選條件如下：

* 二元分類任務（Binary classification）
* 所有特徵為數值型（Numeric features）
* 至少包含 1000 筆資料（Instances ≥ 1000）
* 無缺失值（No missing values）

🔗 推薦使用篩選條件：

* 點選頁面右上角 **「Filter results」**

  * 選擇 `Target` → `Binary classification`
  * 選擇 `Instances` → `1000s`
* 點選右上角 **「Sort results」**，排序方式選擇 `Numeric Features`，以優先顯示所有特徵為數值型的資料集。

---

## 🐍 執行環境與相依套件

### ✅ Python 版本

```bash
Python 3.12 或以上版本
```

### ✅ 安裝所需套件

請執行以下指令安裝所有需要的 Python 套件：

```bash
pip install scikit-learn matplotlib numpy pandas openml
```

---

## ▶️ 執行方式

打開終端機並輸入：

```bash
python ML1.py
```

執行後會自動執行以下步驟：

1. 從 OpenML 載入兩組資料集
2. 訓練數個不同 `min_samples_leaf` 參數值的決策樹模型
3. 使用 10-fold 交叉驗證計算 AUC 分數
4. 繪製每個參數下的測試 AUC 曲線圖，並標註過擬合與欠擬合區域
5. 顯示最佳模型的 ROC 曲線圖

---

## 📈 評估方式說明

* 採用 10-fold cross-validation
* 測試指標：`ROC AUC Score`
* `DecisionTreeClassifier` 使用資訊熵作為切割準則 (`criterion="entropy"`)
* 繪製：

  * AUC vs min\_samples\_leaf 的趨勢圖
  * ROC 曲線（最佳參數下）

---

## 📬 注意事項

* 此程式碼不需額外手動下載資料，會自動從 OpenML 載入。
* 圖片將使用 `matplotlib` 顯示，若無法跳出視窗，請確認使用的是支援 GUI 的環境。
* 若在 Jupyter Notebook 中執行，請使用 `%matplotlib inline` 顯示圖表。

