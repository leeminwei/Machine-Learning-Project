
---

# 🔍 情緒圖片分類任務 — 自建 CNN 與遷移學習模型比較

本專案為機器學習課程期末作業的一部分，目標是使用 Keras 深度學習框架，對六種人類情緒圖片進行分類，並比較不同 CNN 模型架構與預訓練模型（VGG16）的效能。

## 🧪 Python 環境需求

* Python 版本：**3.12**
* 套件需求（可使用 pip 安裝）：

  ```bash
  pip install numpy pandas matplotlib tensorflow
  ```

> ✅ 建議使用 [virtualenv](https://virtualenv.pypa.io/en/latest/) 或 Conda 建立乾淨的虛擬環境進行測試與執行。

---

## 📁 資料說明

請下載資料集 **6 Emotions for Image Classification**(在A4投影片中有附連結)，並解壓縮後放置於專案根目錄下，資料夾名稱請保持為：

```
./6 Emotions for image classification
```

此外，請準備 **10 張自選情緒圖片** 放入資料夾 `./emotion` 供 Task 3 錯誤分析使用。

---

## 🚀 執行方式

### 1️⃣ 模型訓練與測試（包含錯誤分析）

```bash
python ML4_include_training.py
```

此腳本將執行以下流程：

* 載入訓練資料與測試資料（80/20 切分）
* 建立與訓練兩個自建 CNN 模型（模型一無 Dropout/Pooling，模型二有）
* 微調 VGG16 預訓練模型（Fine-tune）
* 儲存訓練歷程與測試結果到 CSV 檔案
* 對 10 張情緒圖片進行模型預測並產出預測比較結果表

### 2️⃣ 載入 `.keras` 模型檔案並重新評估測試集準確率

```bash
python ML4_load_keras.py
```

此腳本會：

* 載入三個訓練完成的模型
* 對原始測試集重新計算準確率
* 對 10 張自選圖片進行預測
* 輸出 `test_accuracy_comparison.csv` 與 `emotion_results.csv`

---

## 🧠 模型架構與功能說明

### Task 1: 自建 CNN 模型比較

* **Model 1：** 無 Dropout / MaxPooling（參考基本 CNN 結構）
* **Model 2：** 加入 Dropout 與 MaxPooling 以避免過擬合
* 使用 EarlyStopping 機制避免過度訓練（連續 10 epochs validation accuracy 無提升即停止）

### Task 2: 微調預訓練模型 VGG16

* 凍結 VGG16 的卷積層權重
* 自訂新的全連接分類層（Dense+Dropout+Softmax）
* 經微調後取得測試準確率與 Task 1 模型進行比較

### Task 3: 錯誤分析

* 測試自選的 10 張情緒圖片
* 比較 Task 1 與 Fine-tuned 模型的分類結果
* 產出圖片視覺化與預測標籤差異分析報告

---

## 📄 產出檔案

* `cnn_model1.keras`, `cnn_model2.keras`, `fine_tuned_model.keras`：三個已訓練完成的模型檔
* `model1_accuracy.csv`, `model2_accuracy.csv`, `fine_tuned_accuracy.csv`：各模型訓練歷程記錄
* `test_accuracy_comparison.csv`：三個模型的測試準確率比較表
* `prediction_results.csv`：自選圖片的預測結果（Task 3）
* `emotion_results.csv`：載入模型後重新測試的結果

---

## 🗂 專案結構建議

```
emotion_cnn_project/
├── ML4_include_training.py
├── ML4_load_keras.py
├── cnn_model1.keras
├── cnn_model2.keras
├── fine_tuned_model.keras
├── 6 Emotions for image classification/
│   ├── happy/
│   ├── sad/
│   └── ...
├── emotion/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── *.csv
└── README.md
```

---

## 💬 備註

* 本任務目標並非追求極高準確率，因資料量有限，重點在於不同架構對分類結果的影響與錯誤分析的深度。
* 自選圖片的真實標籤請於程式中 `correct_classes` 字典處設定。

---
