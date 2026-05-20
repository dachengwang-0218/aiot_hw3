# 📋 開發日誌 — 汽車價格預測作業 (HW3)

> **日期：** 2026-05-20  
> **會話類型：** AI 輔助開發（Antigravity × Claude Sonnet）  
> **目標：** 完成 CRISP-DM 多元線性回歸作業並部署至 GitHub

---

## 對話流程紀錄

---

### 🟢 Step 1 — 初始任務建立

**使用者需求：**
> 作為資深資料科學家，依照 CRISP-DM 六大流程完成汽車價格預測作業：
> - 資料集：Kaggle Car Price Prediction
> - 演算法：多元線性回歸 (Multiple Linear Regression)
> - 目標變數：`price`

**關鍵強制要求：**
- 特徵選擇後特徵數量嚴格控制在 **10~20 個**
- 評估圖必須包含 **信賴區間 (Confidence Interval)**
- 提供完整 Python 程式碼並搭配繁體中文解說
- 整理 README.md

**AI 執行動作：**
- 建立 [`car_price_prediction.py`](car_price_prediction.py)（878 行，含完整 CRISP-DM 六階段）
- 建立 [`car_price_prediction.ipynb`](car_price_prediction.ipynb)（28 個 Cell）
- 建立 [`README.md`](README.md)

**產出結果：**
```
car_price_prediction.py   ✅
car_price_prediction.ipynb ✅
README.md                  ✅
```

---

### 🟢 Step 2 — 確認可執行性

**使用者需求：**
> 我不需要 ipynb 檔案，我可以直接執行 car_price_prediction.py 嗎？

**AI 執行動作：**
1. 檢查 Python 套件安裝狀況 → 發現缺少 `seaborn`
2. 執行 `pip3 install seaborn` 安裝
3. 直接執行 `python3 car_price_prediction.py`

**執行結果（模型效能）：**

| 指標 | 訓練集 | 測試集 |
|------|--------|--------|
| R² | 0.9400 | **0.8779** ✅ |
| Adjusted R² | 0.9339 | 0.8046 |
| RMSE | $1,891 | $3,105 |
| MAE | $1,402 | $2,302 |
| MAPE | 11.85% | 18.62% |

**最終選定特徵（15 個，符合 10~20 要求）：**

| 特徵 | 係數 | 意義 |
|------|------|------|
| `curbweight` | +4,602 | 車重越高，定價越高 |
| `enginelocation_rear` | +1,877 | 後置引擎（跑車）大幅溢價 |
| `enginesize` | +1,576 | 引擎排量正向影響售價 |
| `cylindernumber_six` | -1,562 | 相對基準的負向效應 |
| `brand_bmw` | +1,486 | BMW 品牌溢價顯著 |
| `carbody_wagon` | -1,448 | 旅行車相對便宜 |
| `carbody_hatchback` | -1,362 | 掀背車相對便宜 |
| `carwidth` | +1,347 | 車寬正向影響（尺寸溢價） |
| `carbody_sedan` | -1,167 | 轎車相對基準便宜 |
| `cylindernumber_four` | -1,161 | 四缸相對便宜 |
| `carlength` | -1,139 | 車長（含控制其他變數） |
| `brand_peugeot` | -1,078 | Peugeot 品牌折扣 |
| `wheelbase` | +979 | 軸距越長定價越高 |
| `brand_porsche` | +888 | Porsche 品牌溢價 |
| `boreratio` | -711 | 缸徑比的負向效應 |

**產出圖表（9 張 PNG）：**
```
eda_price_distribution.png      ✅
eda_correlation.png             ✅
eda_categorical_boxplots.png    ✅
eda_scatter_plots.png           ✅
feature_selection_vif.png       ✅
model_coefficients.png          ✅
eval_actual_vs_predicted.png    ✅  ← 含 95% Confidence Interval
eval_residual_diagnostics.png   ✅
eval_metrics_comparison.png     ✅
car_price_model.pkl             ✅
```

---

### 🟡 Step 3 — 圖表中文顯示問題修正

**使用者需求：**
> 圖片中的中文無法顯示出來，請幫我把所有圖片的標題改成英文

**問題原因：**
系統使用 `DejaVu Sans` 字型，不支援 CJK 中文字元，導致標題顯示為方塊。

**AI 執行動作：**
使用 `multi_replace_file_content` 工具批次替換 **24 處**中文圖表文字：

| 原始中文 | 替換英文 |
|---------|---------|
| `原始 price 直方圖` | `Price Histogram (Original)` |
| `數值特徵相關性熱力圖` | `Numeric Feature Correlation Heatmap` |
| `類別特徵 vs Price 箱型圖` | `Categorical Features vs Price (Box Plot)` |
| `VIF 多重共線性檢查（最終特徵）` | `VIF Multicollinearity Check (Final Features)` |
| `迴歸係數（標準化後）` | `Regression Coefficients (Standardized Features)` |
| `實際值 vs 預測值（測試集）` | `Actual vs Predicted Price (Test Set)` |
| `殘差診斷圖（測試集）` | `Residual Diagnostics (Test Set)` |
| `訓練集 / 測試集` legend | `Train Set / Test Set` |
| … | … |

**重新執行結果：** ✅ 所有圖表英文標題正常顯示

---

### 🟢 Step 4 — 上傳至 GitHub

**使用者需求：**
> 幫我用 SSH 上傳到 GitHub：`git@github.com:dachengwang-0218/aiot_hw3.git`

**AI 執行動作：**
```bash
git init
git add -A           # 含 py、ipynb、README、CSV、pkl、9 張 PNG
git commit -m "Add Car Price Prediction (CRISP-DM, Multiple Linear Regression)"
git remote add origin git@github.com:dachengwang-0218/aiot_hw3.git
git push -u origin main
```

**推送結果：**
- ✅ 15 個檔案成功推送
- Commit hash：`1e2b1c9`
- Repository：https://github.com/dachengwang-0218/aiot_hw3

---

### 🟢 Step 5 — 重建 Jupyter Notebook

**使用者需求：**
> 請幫我再寫一個 ipynb 檔，我不小心刪掉了

**AI 執行動作：**
- 依照最新版 `.py`（已改為英文圖表）重建完整 notebook
- JSON 格式驗證：28 個 Cell，nbformat 4.4
- 同步推送至 GitHub（commit `eb75016`）

---

### 🟢 Step 6 — 圖片整理至 figures/ 資料夾

**使用者需求：**
> 幫我把圖片都放到一個資料夾中，並且更新 README.md 中的專案結構

**AI 執行動作：**
1. 建立 `figures/` 資料夾，移入所有 9 張 PNG
2. 批次更新 `.py` 中所有 `plt.savefig()` 路徑 → `figures/xxx.png`
3. 更新 `ipynb` 中所有 `plt.savefig()` 路徑（共 10 處）
4. 更新 README.md 專案結構區塊

---

### 🟢 Step 7 — 建立開發日誌

**使用者需求：**
> 請寫一個 log.md 把我們的對話過程寫進去，並更新 README.md 架構

**AI 執行動作：**
- 建立本檔案 [`log.md`](log.md)
- 更新 README.md 加入 `log.md` 至專案結構
- 推送至 GitHub

---

## 📊 最終成果摘要

### 模型效能

```
R²（測試集）     = 0.8779  ✅ 達成目標 (≥ 0.85)
Adjusted R²      = 0.8046
RMSE（測試集）   = $3,105.22
MAE（測試集）    = $2,302.24
MAPE（測試集）   = 18.62%
特徵數量         = 15 個    ✅ 符合 10~20 要求
過擬合差距       = 0.0622   ⚠ 輕度（線性回歸正常現象）
```

### CRISP-DM 完成度

| Phase | 完成狀態 | 重要細節 |
|-------|---------|---------|
| 1. Business Understanding | ✅ | 車廠定價、二手車平台、保險核保三大場景 |
| 2. Data Understanding | ✅ | EDA 4 張圖、相關性分析、缺失值確認（無缺失） |
| 3. Data Preparation | ✅ | 品牌萃取 + OHE + 標準化 + RFE→p-value 選出 15 特徵 |
| 4. Modeling | ✅ | statsmodels OLS（含 p-value 報表）+ sklearn LR |
| 5. Evaluation | ✅ | RMSE/R²/MAE + **95% CI 信賴區間視覺化** |
| 6. Deployment | ✅ | FastAPI 架構說明 + 5 項未來優化方向 |

---

## 🔧 開發環境

| 項目 | 版本/說明 |
|------|----------|
| Python | 3.14 (macOS) |
| pandas | 最新版 |
| scikit-learn | 最新版 |
| statsmodels | 最新版 |
| seaborn | 0.13.2（本次新安裝）|
| 圖表語言 | 英文（DejaVu Sans 字型不支援 CJK，故改英文）|
| Git remote | SSH (`git@github.com:dachengwang-0218/aiot_hw3.git`) |

---

*Log generated by Antigravity AI on 2026-05-20*
