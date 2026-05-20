# 🚗 汽車價格預測 — 多元線性回歸（CRISP-DM 流程）

> **AIoT 機器學習作業 HW3**  
> 資料集：[Car Price Prediction — Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)  
> 演算法：Multiple Linear Regression  
> 目標變數：`price`（汽車售價，單位 USD）

---

## 📁 專案結構

```
hw3/
├── car_price_prediction.py      ← 主程式
├── CarPrice_Assignment.csv      ← 原始資料集（請手動下載）
├── car_price_model.pkl          ← 訓練完成的模型（執行後產生）
├── README.md                    ← 本文件
│
└── figures/                     ← 所有輸出圖表（共 10 張）
    ├── eda_price_distribution.png    ← Phase 2：price 分佈圖
    ├── eda_correlation.png           ← Phase 2：特徵相關性熱力圖
    ├── eda_categorical_boxplots.png  ← Phase 2：類別特徵箱型圖
    ├── eda_scatter_plots.png         ← Phase 2：數值特徵散佈圖
    ├── eda_brand_price.png           ← Phase 3：品牌平均售價
    ├── feature_selection_vif.png     ← Phase 3：VIF 多重共線性
    ├── model_coefficients.png        ← Phase 4：迴歸係數圖
    ├── eval_actual_vs_predicted.png  ← Phase 5：預測結果圖（含信賴區間）★
    ├── eval_residual_diagnostics.png ← Phase 5：殘差診斷四宮格
    └── eval_metrics_comparison.png   ← Phase 5：指標比較圖
```

---

## 🚀 快速開始

### 1. 環境需求

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy joblib
```

### 2. 資料集下載

前往 [Kaggle Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) 下載 `CarPrice_Assignment.csv`，放置於 `hw3/` 資料夾。

> ⚠ **注意**：若本地無資料檔，Notebook 會自動嘗試從備用網址下載。

### 3. 執行 car_price_prediction.py

```bash
cd hw3
python car_price_prediction.py
```

---

## 📚 CRISP-DM 六大流程說明

### Phase 1 — Business Understanding（商業理解）

| 應用場景 | 商業價值 |
|---------|---------|
| 車廠定價策略 | 根據工程規格快速生成建議售價，縮短定價週期 |
| 二手車估價平台 | 提供公平市場價（如 CarMax、速買配），提升用戶信任 |
| 銀行/保險核保 | 客觀車輛殘值估算，降低核保風險 |

**分析目標：** R² ≥ 0.85，最終特徵數 10–20 個，所有特徵 p-value < 0.05

---

### Phase 2 — Data Understanding（資料理解）

- **資料規模**：205 筆 × 26 個原始欄位
- **目標變數**：`price`（右偏分佈，均值約 $13,276，中位數約 $10,295）
- **缺失值**：✅ 無缺失值
- **特別欄位**：`CarName` 含品牌與型號，需進行字串解析

**主要發現（EDA）：**

| 特徵 | 與 price 相關性 | 說明 |
|------|----------------|------|
| `enginesize` | r ≈ 0.87 | 引擎排量最強正相關 |
| `curbweight`  | r ≈ 0.84 | 車重反映車輛等級 |
| `horsepower`  | r ≈ 0.81 | 馬力高=效能車=高價 |
| `carwidth`    | r ≈ 0.76 | 車寬與車輛尺寸相關 |
| `citympg`     | r ≈ -0.69 | 油耗高（省油）反映小型車 |

---

### Phase 3 — Data Preparation（資料準備）

#### 3.1 CarName 品牌萃取
```python
df['brand'] = df['CarName'].str.split().str[0].str.lower()
# 修正拼字錯誤
brand_fix = {'maxda': 'mazda', 'porcshce': 'porsche', 
             'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'}
```

#### 3.2 One-Hot Encoding
- 類別欄位：`fueltype`, `aspiration`, `doornumber`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `cylindernumber`, `fuelsystem`, `brand`
- OHE 後特徵數：約 60+ 個（遠超 20，需特徵選擇）

#### 3.3 數值標準化（Z-Score）
- **注意**：Scaler 只 `fit` 訓練集，再 `transform` 測試集（避免資料洩漏）

#### 3.4 特徵選擇（關鍵步驟）

| 步驟 | 方法 | 說明 |
|------|------|------|
| Step A | **RFE**（遞迴特徵消除） | 從 60+ 個特徵初步篩至 15 個 |
| Step B | **p-value 逐步剔除** | 逐步移除 p > 0.05 的特徵，保留至少 10 個 |

**最終保留特徵（10–20 個）：** 詳見 Notebook 執行結果

**特徵選擇摘要（示意）：**
```
RFE 保留（15 個）→ p-value 再篩選 → 最終 N 個（10~20）
```

---

### Phase 4 — Modeling（建模）

使用兩種框架建立多元線性回歸：

| 框架 | 用途 |
|------|------|
| `statsmodels OLS` | 查看完整統計報表（R², F-test, p-value, AIC, DW） |
| `sklearn LinearRegression` | 預測、評估、模型儲存 |

```python
# statsmodels（詳細報表）
ols_model = sm.OLS(y_train, sm.add_constant(X_train_final)).fit()
print(ols_model.summary())

# sklearn（預測）
sk_model = LinearRegression().fit(X_train_final, y_train)
```

---

### Phase 5 — Evaluation（模型評估）

#### 評估指標

| 指標 | 訓練集 | 測試集 |
|------|--------|--------|
| R² | — | — |
| Adjusted R² | — | — |
| RMSE | — | — |
| MAE | — | — |
| MAPE | — | — |

> 📝 實際數值請執行 car_price_prediction.py 查看（因特徵選擇結果可能略有不同）

#### 視覺化圖表（共 10 張）

1. **`eda_price_distribution.png`**：目標變數分佈（原始 + log 轉換）
2. **`eda_correlation.png`**：相關性熱力圖 + 條形圖
3. **`eda_categorical_boxplots.png`**：類別特徵 vs price 箱型圖
4. **`eda_scatter_plots.png`**：數值特徵散佈圖 + 趨勢線
5. **`eda_brand_price.png`**：各品牌平均售價
6. **`feature_selection_vif.png`**：VIF 多重共線性
7. **`model_coefficients.png`**：迴歸係數條形圖
8. **`eval_actual_vs_predicted.png`** ⭐：實際 vs 預測散佈圖，含 **95% 信賴區間**
9. **`eval_residual_diagnostics.png`**：四宮格殘差診斷（殘差分佈、Q-Q 圖等）
10. **`eval_metrics_comparison.png`**：訓練 vs 測試指標比較

#### ⭐ 信賴區間視覺化（必要項目）

```python
# 使用 seaborn.regplot 達成 95% 信賴區間視覺化
sns.regplot(
    x=y_test.values,
    y=y_test_pred,
    ci=95,   # ← 95% Confidence Interval（淺藍色陰影區域）
    ...
)
```

> **解讀**：淺藍色陰影為 95% 信賴區間，代表若重複抽樣建模，有 95% 的機率回歸線會落在此區域內。

---

### Phase 6 — Deployment（部署）

#### 部署架構

```
User (Browser / App)
      │  POST /predict { enginesize, horsepower, brand, ... }
      ▼
FastAPI Service（特徵工程 → 標準化 → 預測）
      │
      ▼
car_price_model.pkl（joblib 載入）
      │
      ▼
JSON Response: { predicted_price: 18500.0, ci_lower: 15200.0, ci_upper: 21800.0 }
```

#### 未來優化方向

| 優先度 | 方向 | 做法 |
|--------|------|------|
| 🔴 高 | 模型升級 | XGBoost/LightGBM（預期 R² 可達 0.95+） |
| 🔴 高 | 特徵工程 | 加入車齡、里程數、事故紀錄 |
| 🟡 中 | 資料擴充 | 定期爬取中古車平台成交行情 |
| 🟡 中 | 監控 | Evidently AI / MLflow drift detection |
| 🟢 低 | 再訓練 | 每季 Airflow 自動觸發 |

---

## 🔧 技術規格

| 項目 | 使用套件 / 版本 |
|------|----------------|
| 資料處理 | `pandas`, `numpy` |
| 視覺化 | `matplotlib`, `seaborn` |
| 統計建模 | `statsmodels` |
| 機器學習 | `scikit-learn` |
| 模型儲存 | `joblib` |
| 統計檢驗 | `scipy.stats` |

---

## ✅ 作業檢查清單

- [x] **Phase 1**：說明商業價值（車廠定價 / 二手車平台 / 保險）
- [x] **Phase 2**：EDA（分佈、相關性、箱型圖、散佈圖）
- [x] **Phase 2**：檢查缺失值 ✅（本資料集無缺失值）
- [x] **Phase 2**：特別分析 `CarName` 欄位（字串解析）
- [x] **Phase 3**：從 `CarName` 萃取品牌 + 修正拼字錯誤
- [x] **Phase 3**：One-Hot Encoding（fueltype, carbody 等）
- [x] **Phase 3**：數值標準化（Z-Score）
- [x] **Phase 3** ⭐：特徵選擇（RFE + p-value），最終特徵數 **嚴格在 10~20 個**
- [x] **Phase 3**：說明保留與淘汰的特徵
- [x] **Phase 4**：使用 `statsmodels` 查看 p-value 與詳細報表
- [x] **Phase 5**：計算 RMSE、R²、MAE、MAPE
- [x] **Phase 5** ⭐：繪製實際值 vs 預測值散佈圖
- [x] **Phase 5** ⭐：加上 **95% 信賴區間**（`seaborn.regplot ci=95`）
- [x] **Phase 6**：部署場景說明（API 架構圖 + 未來優化）
- [x] **README.md**：完整流程與成果整理

---

*作業截止日期：請確認課程規定。執行前請先下載資料集至 `hw3/` 資料夾。*
