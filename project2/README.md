
---

# 📊 機器學習分類模型比較實驗（使用 scikit-learn）

## 🧾 專案簡介

本專案是機器學習課程的作業，目標是比較多種分類模型在不同資料集上的效能，並使用 `scikit-learn` 套件進行實作與交叉驗證分析。我們使用兩筆來自 [OpenML](https://www.openml.org/) 的資料集，並針對每個資料集：

* 對分類器進行超參數調整
* 使用 10-fold 交叉驗證
* 比較 AUC（Area Under Curve）分數的平均值與標準差
* 評估五種不同分類模型的效能

## 📦 使用的資料集

* Dataset 1：OpenML ID `41283`（包含分類目標與名目變數）
* Dataset 2：OpenML ID `41335`（同樣符合分類與名目變數條件）

這兩個資料集皆符合以下條件：

* 目標欄位為類別（nominal target）
* 至少包含一個類別特徵（categorical feature）
* 至少 1000 筆資料
* 無缺失值或特徵數量過多的問題

你可以自行更換資料集，只需修改 `fetch_openml(data_id=...)` 中的 ID 即可。
👉 可透過 [OpenML 搜尋介面](https://www.openml.org/search?type=data) 篩選符合條件的資料集。

## 🧠 模型與調參說明

程式中評估以下五種模型：

| 模型名稱                     | 調參項目                     |
| ------------------------ | ------------------------ |
| Decision Tree Classifier | `min_samples_leaf`（5 組值） |
| K-Nearest Neighbors      | `n_neighbors`（5 組值）      |
| Multinomial Naive Bayes  | 無                        |
| Logistic Regression      | 無                        |
| Dummy Classifier         | 無（最常出現類別）                |

使用 `GridSearchCV` 對有調參需求的模型做參數搜尋，並使用 `roc_auc` 做為效能評估指標（若為多類別則使用 `roc_auc_ovr_weighted`）。

## 📈 輸出結果格式

每組模型會輸出：

* 平均 AUC（Mean AUC）
* AUC 標準差（Standard Deviation）

範例輸出如下：

```
Results for Dataset 1:
               Model  Mean AUC  Standard Deviation AUC
0      Decision Tree     0.891                   0.024
1                KNN     0.883                   0.030
2       Naive Bayes     0.825                   0.038
3  Logistic Regression  0.902                   0.020
4              Dummy     0.495                   0.005
```

## 🔧 安裝與執行方式

### ✅ 環境需求

* Python 3.12+
* 套件依賴如下：

```bash
pip install scikit-learn pandas numpy
```

### ▶️ 執行方法

直接執行 `ML2.py` 檔案即可：

```bash
python ML2.py
```

程式將依序對兩筆資料集進行特徵編碼、建模與交叉驗證，並印出模型效能比較結果。

---

## 📚 檔案說明

| 檔名       | 說明                      |
| -------- | ----------------------- |
| `ML2.py` | 主程式，載入資料、處理特徵、建模與比較所有結果 |

