
---

# 📊 機器學習作業三：使用 Keras 建立回歸神經網路模型

## 🧠 專案說明

本專案為課程《Machine Learning》第三次作業，目的是使用 Keras 建立三種不同的神經網路模型，對兩組回歸任務的資料集進行訓練與評估，並探討不同模型架構對於模型效能的影響。

本作業所使用的兩個資料集皆來自 [OpenML](https://www.openml.org/)，分別為：

* Dataset 1：`cpu_act` (data\_id=42369)
* Dataset 2：`cpu_small` (data\_id=503)

## 🔧 執行環境與依賴套件

### ✅ Python 環境版本

本專案使用 Python **3.11**

### ✅ 所需套件安裝

請先使用 pip 安裝以下必要套件：

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

若使用的是 Jupyter Notebook，建議搭配以下指令載入圖形：

```python
%matplotlib inline
```

## 📂 檔案說明

* `ML3.py`：主程式，包含兩組資料集的前處理、三種神經網路模型訓練、繪圖與測試。
* 輸出內容：

  * 每個模型的訓練與驗證誤差（MSE）隨 Epochs 變化圖
  * 測試資料的預測散佈圖
  * 模型比較表（測試 MSE）

## 📈 模型架構介紹

對每組資料集皆使用以下三種神經網路架構：

1. **Model 1：少量神經元**

   * 隱藏層神經元數量：5
2. **Model 2：適量神經元**

   * 隱藏層神經元數量：30
3. **Model 3：過多神經元**

   * 隱藏層神經元數量：200

所有模型皆使用：

* 激活函數：`sigmoid`
* 損失函數：`mse`
* 最佳化器：`adam`
* 輸出層：1 個節點，無 activation（針對回歸）

## 📊 訓練與評估流程

1. 使用 `train_test_split` 將資料分為訓練集與測試集（80%/20%）。
2. 再將訓練集切分為訓練與驗證集（80%/20%）。
3. 對 `x` 與 `y` 特徵皆進行標準化處理。
4. 模型訓練 200 個 epochs。
5. 根據驗證誤差選出最佳 epoch，可進行 Early Stopping。
6. 在測試集上評估每個模型的 MSE。

## 🧪 執行方式

直接執行 `ML3.py` 即可：

```bash
python ML3.py
```
