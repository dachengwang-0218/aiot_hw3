# =============================================================================
# 汽車價格預測 - 多元線性回歸 (CRISP-DM 流程)
# Car Price Prediction - Multiple Linear Regression (CRISP-DM Process)
# =============================================================================
# 資料集：Car Price Prediction (Kaggle)
# https://www.kaggle.com/datasets/hellbuoy/car-price-prediction
# =============================================================================

# ── 安裝必要套件（若尚未安裝）──
# pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 設定繪圖風格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_theme(style='whitegrid', palette='muted')

print("✅ 套件載入完成")
print("=" * 70)


# =============================================================================
# ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗     ██╗
# ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ███║
# ██████╔╝███████║███████║███████╗█████╗      ╚██║
# ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝       ██║
# ██║     ██║  ██║██║  ██║███████║███████╗     ██║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝     ╚═╝
# Business Understanding（商業理解）
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         PHASE 1 ── Business Understanding（商業理解）               ║
╚══════════════════════════════════════════════════════════════════════╝

【專案背景】
  汽車定價是一個複雜的決策過程，涉及工程規格、市場定位、品牌溢價等多個維度。
  對於汽車製造商、二手車平台及消費者而言，準確預測車價具有重要的商業價值。

【商業目標】
  1. 車廠定價策略：幫助車廠根據車型規格（引擎排量、馬力、車身尺寸等）
     制定具競爭力的出廠定價，避免訂價過低導致利潤流失或過高導致市占率下降。

  2. 二手車估價平台：二手車網站（如 CarMax、Carvana）可將此模型整合至估價
     系統，根據車齡、品牌、規格快速給出公平市場價，提升用戶信任度。

  3. 保險核保輔助：保險公司可參考模型輸出的市場公允價值，作為車輛保額的
     客觀依據，降低核保風險。

【分析目標（Data Mining Goal）】
  使用多元線性回歸建立一個可解釋性強的預測模型，目標：
  - R² ≥ 0.85（解釋超過 85% 的價格變異）
  - RMSE 盡可能低，且模型特徵符合汽車領域常識

【成功標準】
  - 模型 R² ≥ 0.85，RMSE 合理
  - 最終模型特徵數量控制在 10–20 個之間（避免過擬合）
  - 所有留用特徵的 p-value < 0.05（統計顯著）
""")

# =============================================================================
# ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ██████╗
# ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ╚════██╗
# ██████╔╝███████║███████║███████╗█████╗       █████╔╝
# ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝      ██╔═══╝
# ██║     ██║  ██║██║  ██║███████║███████╗    ███████╗
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚══════╝
# Data Understanding（資料理解）
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         PHASE 2 ── Data Understanding（資料理解）                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ── 2.1 載入資料 ──────────────────────────────────────────────────────────
print("▶ 2.1 載入資料集")
print("-" * 60)

# 嘗試從本地讀取；若無則從 GitHub 備用連結下載
DATA_URL = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/CarPrice_Assignment.csv"
)

try:
    df = pd.read_csv("CarPrice_Assignment.csv")
    print("  ✅ 從本地檔案載入成功")
except FileNotFoundError:
    print(f"  ⚠ 本地檔案不存在，嘗試從網路載入...")
    df = pd.read_csv(DATA_URL)
    print("  ✅ 從網路載入成功")

print(f"\n  資料維度：{df.shape[0]} 筆資料 × {df.shape[1]} 個欄位")

# ── 2.2 基本資訊 ──────────────────────────────────────────────────────────
print("\n▶ 2.2 資料基本資訊")
print("-" * 60)
print(df.info())

# ── 2.3 統計摘要 ──────────────────────────────────────────────────────────
print("\n▶ 2.3 數值欄位統計摘要")
print("-" * 60)
print(df.describe().round(2).to_string())

# ── 2.4 缺失值檢查 ────────────────────────────────────────────────────────
print("\n▶ 2.4 缺失值檢查")
print("-" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失數量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失數量'] > 0]
if missing_df.empty:
    print("  ✅ 無任何缺失值！")
else:
    print(missing_df)

# ── 2.5 類別欄位檢視 ──────────────────────────────────────────────────────
print("\n▶ 2.5 類別欄位唯一值")
print("-" * 60)
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    vals = df[col].unique()
    print(f"  {col:20s} ({len(vals):3d} 種): {list(vals[:8])}{'...' if len(vals) > 8 else ''}")

# ── 2.6 目標變數分佈 ──────────────────────────────────────────────────────
print("\n▶ 2.6 目標變數 price 分佈分析")
print("-" * 60)
print(df['price'].describe().round(2))
print(f"  偏度 (Skewness): {df['price'].skew():.4f}")
print(f"  峰度 (Kurtosis): {df['price'].kurt():.4f}")

# ── 2.7 EDA 視覺化 ────────────────────────────────────────────────────────
print("\n▶ 2.7 繪製 EDA 視覺化圖表...")

# --- 圖 1：目標變數分佈 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 2 -- Price Distribution (Target Variable)", fontsize=15, fontweight='bold', y=1.02)

ax = axes[0]
ax.hist(df['price'], bins=30, color='#4C72B0', edgecolor='white', alpha=0.85)
ax.set_title("Price Histogram (Original)", fontsize=12)
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Frequency")

ax2 = axes[1]
log_price = np.log1p(df['price'])
ax2.hist(log_price, bins=30, color='#DD8452', edgecolor='white', alpha=0.85)
ax2.set_title("log(price+1) Histogram (Skewness Correction)", fontsize=12)
ax2.set_xlabel("log(Price + 1)")
ax2.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("figures/eda_price_distribution.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：eda_price_distribution.png")

# --- 圖 2：數值特徵相關性熱力圖 ---
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[num_cols].corr()
price_corr = corr_matrix['price'].drop('price').sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Phase 2 -- Feature Correlation Analysis", fontsize=15, fontweight='bold')

# 熱力圖
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, ax=axes[0], cmap='coolwarm',
            vmin=-1, vmax=1, center=0, annot=False, fmt='.2f',
            linewidths=0.3, cbar_kws={'shrink': 0.8})
axes[0].set_title("Numeric Feature Correlation Heatmap", fontsize=12)
axes[0].tick_params(axis='x', rotation=45, labelsize=7)
axes[0].tick_params(axis='y', rotation=0, labelsize=7)

# 與 price 的相關係數條形圖
colors = ['#c0392b' if v > 0 else '#2980b9' for v in price_corr.values]
axes[1].barh(range(len(price_corr)), price_corr.values, color=colors, alpha=0.8)
axes[1].set_yticks(range(len(price_corr)))
axes[1].set_yticklabels(price_corr.index, fontsize=9)
axes[1].axvline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_title("Pearson Correlation with Price", fontsize=12)
axes[1].set_xlabel("Pearson r")

plt.tight_layout()
plt.savefig("figures/eda_correlation.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：eda_correlation.png")

# --- 圖 3：類別變數 vs price 箱型圖 ---
cat_to_plot = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Phase 2 -- Categorical Features vs Price (Box Plot)", fontsize=15, fontweight='bold')
axes = axes.flatten()

palette = sns.color_palette("Set2")
for i, col in enumerate(cat_to_plot):
    order = df.groupby(col)['price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=col, y='price', order=order, ax=axes[i],
                palette=palette, linewidth=1.2)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel("")
    axes[i].tick_params(axis='x', rotation=30, labelsize=8)

plt.tight_layout()
plt.savefig("figures/eda_categorical_boxplots.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：eda_categorical_boxplots.png")

# --- 圖 4：數值特徵 vs price 散佈圖 ---
num_features = ['enginesize', 'horsepower', 'curbweight', 'carwidth', 'wheelbase']
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Phase 2 -- Key Numeric Features vs Price (Scatter Plot)", fontsize=14, fontweight='bold')
colors_list = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
for i, feat in enumerate(num_features):
    axes[i].scatter(df[feat], df['price'], alpha=0.35, s=20, color=colors_list[i])
    z = np.polyfit(df[feat], df['price'], 1)
    p = np.poly1d(z)
    xline = np.linspace(df[feat].min(), df[feat].max(), 200)
    axes[i].plot(xline, p(xline), color='red', linewidth=1.5, linestyle='--')
    axes[i].set_title(feat, fontsize=10, fontweight='bold')
    axes[i].set_xlabel(feat, fontsize=8)
    axes[i].set_ylabel("price" if i == 0 else "", fontsize=8)

plt.tight_layout()
plt.savefig("figures/eda_scatter_plots.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：eda_scatter_plots.png")


# =============================================================================
# ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ██████╗
# ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ╚════██╗
# ██████╔╝███████║███████║███████╗█████╗       █████╔╝
# ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝       ╚═══██╗
# ██║     ██║  ██║██║  ██║███████║███████╗    ██████╔╝
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚═════╝
# Data Preparation（資料準備）
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         PHASE 3 ── Data Preparation（資料準備）                     ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ── 3.1 複製資料，保留原始 ────────────────────────────────────────────────
df_clean = df.copy()

# ── 3.2 從 CarName 萃取品牌（Brand） ─────────────────────────────────────
print("▶ 3.2 從 CarName 萃取汽車品牌（Brand）")
print("-" * 60)

df_clean['brand'] = df_clean['CarName'].str.split().str[0].str.lower()

# 修正拼字錯誤（資料集中有一些拼字錯誤）
brand_fix = {
    'maxda':    'mazda',
    'porcshce': 'porsche',
    'toyouta':  'toyota',
    'vokswagen': 'volkswagen',
    'vw':       'volkswagen',
}
df_clean['brand'] = df_clean['brand'].replace(brand_fix)

brand_counts = df_clean['brand'].value_counts()
print(f"  品牌總數：{df_clean['brand'].nunique()} 個")
print(f"  各品牌數量：\n{brand_counts.to_string()}")

# 計算每個品牌的平均定價（作為品牌溢價特徵）
brand_price_avg = df_clean.groupby('brand')['price'].mean().sort_values(ascending=False)
print(f"\n  各品牌平均售價（Top 10）：")
print(brand_price_avg.head(10).round(0).to_string())

# ── 3.3 刪除不需要的欄位 ──────────────────────────────────────────────────
print("\n▶ 3.3 刪除冗餘欄位")
print("-" * 60)
drop_cols = ['car_ID', 'CarName']
df_clean.drop(columns=drop_cols, inplace=True)
print(f"  已刪除：{drop_cols}")
print(f"  目前維度：{df_clean.shape}")

# ── 3.4 One-Hot Encoding ──────────────────────────────────────────────────
print("\n▶ 3.4 類別變數 One-Hot Encoding")
print("-" * 60)

cat_encode = [
    'fueltype', 'aspiration', 'doornumber', 'carbody',
    'drivewheel', 'enginelocation', 'enginetype',
    'cylindernumber', 'fuelsystem', 'brand'
]

# 確認欄位存在
cat_encode = [c for c in cat_encode if c in df_clean.columns]
print(f"  進行 OHE 的欄位：{cat_encode}")

df_encoded = pd.get_dummies(df_clean, columns=cat_encode, drop_first=True)

# 確保所有欄位為數值型（布林 → int）
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print(f"  OHE 後維度：{df_encoded.shape}")
print(f"  特徵數量（不含 price）：{df_encoded.shape[1] - 1}")

# ── 3.5 特徵與目標分離 ────────────────────────────────────────────────────
print("\n▶ 3.5 分離特徵（X）與目標（y）")
print("-" * 60)

target = 'price'
feature_cols = [c for c in df_encoded.columns if c != target]
X = df_encoded[feature_cols]
y = df_encoded[target]

print(f"  X 維度：{X.shape}")
print(f"  y 維度：{y.shape}")
print(f"  特徵總數（進入選擇前）：{X.shape[1]}")

# ── 3.6 訓練集 / 測試集分割 ───────────────────────────────────────────────
print("\n▶ 3.6 訓練集 / 測試集分割（80% / 20%）")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  訓練集：{X_train.shape[0]} 筆")
print(f"  測試集：{X_test.shape[0]} 筆")

# ── 3.7 數值特徵標準化（Standardization）─────────────────────────────────
print("\n▶ 3.7 數值特徵標準化（Z-Score Standardization）")
print("-" * 60)

num_features_to_scale = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[num_features_to_scale] = scaler.fit_transform(X_train[num_features_to_scale])
X_test_scaled[num_features_to_scale]  = scaler.transform(X_test[num_features_to_scale])

print(f"  已標準化欄位數：{len(num_features_to_scale)}")
print("  ✅ 標準化完成（fit 於訓練集，transform 於測試集）")

# ── 3.8 特徵選擇（Feature Selection） ────────────────────────────────────
print("""
▶ 3.8 特徵選擇（Feature Selection）── 目標：10~20 個特徵
---------------------------------------------------------
  策略：先以 RFE 初步篩選，再用 statsmodels p-value 精細調整
""")

TARGET_FEATURES = 15  # 目標保留 15 個特徵（在 10~20 之間）

# ── 步驟 A：RFE 初步選擇 ──
print("  [步驟 A] RFE（Recursive Feature Elimination）")
lr_rfe = LinearRegression()
rfe = RFE(estimator=lr_rfe, n_features_to_select=TARGET_FEATURES, step=1)
rfe.fit(X_train_scaled, y_train)

rfe_selected = X_train_scaled.columns[rfe.support_].tolist()
rfe_eliminated = X_train_scaled.columns[~rfe.support_].tolist()

print(f"\n  ✅ RFE 保留（{len(rfe_selected)} 個）：")
for f in rfe_selected:
    print(f"     ├─ {f}")
print(f"\n  ❌ RFE 淘汰（{len(rfe_eliminated)} 個）：")
for f in rfe_eliminated[:20]:
    print(f"     ├─ {f}")
if len(rfe_eliminated) > 20:
    print(f"     └─ ... 及其他 {len(rfe_eliminated)-20} 個")

X_train_rfe = X_train_scaled[rfe_selected]
X_test_rfe  = X_test_scaled[rfe_selected]

# ── 步驟 B：statsmodels p-value 精細篩選 ──
print("\n  [步驟 B] p-value 篩選（逐步剔除 p > 0.05 的特徵）")
X_sm = sm.add_constant(X_train_rfe)
eliminated_pvalue = []
iteration = 0

while True:
    model_ols = sm.OLS(y_train, X_sm).fit()
    pvalues = model_ols.pvalues.drop('const')
    max_pval = pvalues.max()

    if max_pval > 0.05:
        worst_feature = pvalues.idxmax()
        current_count = len(pvalues)

        # 如果再刪就低於 10 個，則停止
        if current_count - 1 < 10:
            print(f"\n  ⚠ 停止：再刪除會低於 10 個特徵（目前 {current_count} 個）")
            break

        eliminated_pvalue.append(worst_feature)
        X_sm = X_sm.drop(columns=[worst_feature])
        iteration += 1
        print(f"  迭代 {iteration:2d}：刪除 '{worst_feature}' (p={max_pval:.4f})")
    else:
        print(f"\n  ✅ 所有剩餘特徵 p-value < 0.05，停止迭代（迭代 {iteration} 次）")
        break

# 最終選定特徵
final_features = [c for c in X_sm.columns if c != 'const']
final_count    = len(final_features)

print(f"\n  📋 最終選定特徵數量：{final_count} 個（在 10~20 之間）")
print(f"\n  ✅ 最終保留特徵：")
for f in final_features:
    print(f"     ├─ {f}")
print(f"\n  ❌ p-value 額外淘汰特徵（{len(eliminated_pvalue)} 個）：")
for f in eliminated_pvalue:
    print(f"     ├─ {f}")

assert 10 <= final_count <= 20, f"特徵數量 {final_count} 不在 10~20 範圍內！"
print(f"\n  ✅ 特徵數量驗證通過：{final_count} 個（符合 10~20 的要求）")

# 準備最終訓練 / 測試資料
X_train_final = X_train_scaled[final_features]
X_test_final  = X_test_scaled[final_features]

# ── 3.9 VIF 多重共線性檢查 ────────────────────────────────────────────────
print("\n▶ 3.9 VIF（變異數膨脹因子）多重共線性檢查")
print("-" * 60)

X_vif = sm.add_constant(X_train_final)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF']     = [variance_inflation_factor(X_vif.values, i)
                       for i in range(X_vif.shape[1])]
vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))
print("\n  （VIF > 10 通常視為嚴重共線性，VIF > 5 為中度共線性）")

# --- 特徵選擇結果視覺化 ---
fig, ax = plt.subplots(figsize=(10, 6))
colors_vif = ['#c0392b' if v > 10 else '#e67e22' if v > 5 else '#27ae60'
              for v in vif_data['VIF']]
bars = ax.barh(vif_data['Feature'], vif_data['VIF'], color=colors_vif, alpha=0.85)
ax.axvline(5,  color='orange', linestyle='--', linewidth=1.5, label='VIF=5 (Moderate)')
ax.axvline(10, color='red',    linestyle='--', linewidth=1.5, label='VIF=10 (Severe)')
ax.set_title("Phase 3 -- VIF Multicollinearity Check (Final Features)", fontsize=13, fontweight='bold')
ax.set_xlabel("VIF Score", fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig("figures/feature_selection_vif.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：feature_selection_vif.png")


# =============================================================================
# ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ██╗  ██╗
# ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ██║  ██║
# ██████╔╝███████║███████║███████╗█████╗      ███████║
# ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝      ╚════██║
# ██║     ██║  ██║██║  ██║███████║███████╗         ██║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝         ╚═╝
# Modeling（建模）
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         PHASE 4 ── Modeling（建模）                                 ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ── 4.1 statsmodels OLS（詳細統計報表）────────────────────────────────────
print("▶ 4.1 使用 statsmodels OLS 建立最終模型")
print("-" * 60)

X_train_sm = sm.add_constant(X_train_final)
X_test_sm  = sm.add_constant(X_test_final)

ols_model = sm.OLS(y_train, X_train_sm).fit()
print(ols_model.summary())

# ── 4.2 scikit-learn LinearRegression（預測用）────────────────────────────
print("\n▶ 4.2 使用 scikit-learn LinearRegression 建立預測模型")
print("-" * 60)

sk_model = LinearRegression()
sk_model.fit(X_train_final, y_train)

# 係數表
coef_df = pd.DataFrame({
    'Feature':     ['intercept'] + final_features,
    'Coefficient': [sk_model.intercept_] + sk_model.coef_.tolist()
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n  回歸係數（按絕對值排序）：")
print(coef_df.to_string(index=False))

# --- 係數視覺化 ---
plot_coef = coef_df[coef_df['Feature'] != 'intercept'].sort_values('Coefficient')
colors_coef = ['#c0392b' if v > 0 else '#2980b9' for v in plot_coef['Coefficient']]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(plot_coef['Feature'], plot_coef['Coefficient'],
        color=colors_coef, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title("Phase 4 -- Regression Coefficients (Standardized Features)", fontsize=13, fontweight='bold')
ax.set_xlabel("Coefficient Value (Standardized)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/model_coefficients.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：model_coefficients.png")


# =============================================================================
# ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ███████╗
# ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ██╔════╝
# ██████╔╝███████║███████║███████╗█████╗      ███████╗
# ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝      ╚════██║
# ██║     ██║  ██║██║  ██║███████║███████╗    ███████║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚══════╝
# Evaluation（模型評估）
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         PHASE 5 ── Evaluation（模型評估）                           ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ── 5.1 預測 ──────────────────────────────────────────────────────────────
y_train_pred = sk_model.predict(X_train_final)
y_test_pred  = sk_model.predict(X_test_final)

# ── 5.2 評估指標 ──────────────────────────────────────────────────────────
print("▶ 5.1 模型評估指標")
print("-" * 60)

def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - len(final_features) - 1)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n  {'='*40}")
    print(f"  {label}")
    print(f"  {'='*40}")
    print(f"  R²            = {r2:.4f}")
    print(f"  Adjusted R²   = {adj_r2:.4f}")
    print(f"  RMSE          = {rmse:,.2f}")
    print(f"  MAE           = {mae:,.2f}")
    print(f"  MAPE          = {mape:.2f}%")
    return {'R2': r2, 'Adj_R2': adj_r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

train_metrics = evaluate(y_train, y_train_pred, "訓練集 (Train Set)")
test_metrics  = evaluate(y_test,  y_test_pred,  "測試集 (Test Set)")

# ── 5.3 殘差分析 ──────────────────────────────────────────────────────────
print("\n▶ 5.2 殘差分析（Residual Analysis）")
print("-" * 60)

residuals_train = y_train - y_train_pred
residuals_test  = y_test  - y_test_pred

print(f"  訓練集殘差統計：")
print(f"    均值：{residuals_train.mean():.4f}（應接近 0）")
print(f"    標準差：{residuals_train.std():.2f}")
print(f"    偏度：{stats.skew(residuals_train):.4f}（應接近 0）")
print(f"    Shapiro-Wilk p-value：{stats.shapiro(residuals_train)[1]:.4f}")

# ── 5.4 評估視覺化 ────────────────────────────────────────────────────────
print("\n▶ 5.3 繪製評估視覺化圖表...")

# =====================================================================
# 圖 A：實際值 vs 預測值（含信賴區間）── 使用 seaborn.regplot
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# 測試集散佈圖 + 回歸線 + 信賴區間
sns.regplot(
    x=y_test.values,
    y=y_test_pred,
    ax=ax,
    scatter_kws={
        'alpha': 0.55,
        's': 60,
        'color': '#4C72B0',
        'edgecolors': 'white',
        'linewidths': 0.5,
    },
    line_kws={
        'color': '#c0392b',
        'linewidth': 2.5,
        'label': 'Regression Line'
    },
    ci=95,            # 95% 信賴區間
    color='#c0392b'
)

# 理想預測線（y = x）
lim_min = min(y_test.min(), y_test_pred.min()) * 0.9
lim_max = max(y_test.max(), y_test_pred.max()) * 1.1
ax.plot([lim_min, lim_max], [lim_min, lim_max],
        'k--', linewidth=1.5, alpha=0.7, label='Perfect Prediction (y=x)')

ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_title(
    f"Phase 5 -- Actual vs Predicted Price (Test Set)\n"
    f"R² = {test_metrics['R2']:.4f}  |  RMSE = {test_metrics['RMSE']:,.0f}  |  "
    f"with 95% Confidence Interval",
    fontsize=12, fontweight='bold'
)
ax.set_xlabel("Actual Price (USD)", fontsize=12)
ax.set_ylabel("Predicted Price (USD)", fontsize=12)

# 加入文字標注說明信賴區間
ax.text(
    0.03, 0.97,
    "Shaded area = 95% Confidence Interval\n(uncertainty range of regression line)",
    transform=ax.transAxes, fontsize=9,
    va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
              edgecolor='gray', alpha=0.85)
)

ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("figures/eval_actual_vs_predicted.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：eval_actual_vs_predicted.png")

# =====================================================================
# 圖 B：四宮格殘差診斷
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 5 -- Residual Diagnostics (Test Set)", fontsize=14, fontweight='bold')

# B1：殘差 vs 預測值
axes[0, 0].scatter(y_test_pred, residuals_test, alpha=0.5,
                   color='#4C72B0', s=40, edgecolors='white', linewidths=0.3)
axes[0, 0].axhline(0, color='red', linewidth=1.5, linestyle='--')
axes[0, 0].set_title("Residuals vs Fitted Values", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel("Predicted Price")
axes[0, 0].set_ylabel("Residuals")

# B2：殘差直方圖
axes[0, 1].hist(residuals_test, bins=20, color='#55A868',
                edgecolor='white', alpha=0.8)
axes[0, 1].set_title("Residuals Histogram", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("Residuals")
axes[0, 1].set_ylabel("Frequency")

# B3：Q-Q 圖（檢驗常態性）
sm.qqplot(residuals_test, line='s', ax=axes[1, 0], alpha=0.5,
          markerfacecolor='#C44E52', markersize=5)
axes[1, 0].set_title("Q-Q Plot (Normality Check)", fontsize=11, fontweight='bold')

# B4：尺度-位置圖
std_resid = np.sqrt(np.abs(residuals_test / residuals_test.std()))
axes[1, 1].scatter(y_test_pred, std_resid, alpha=0.5,
                   color='#8172B2', s=40, edgecolors='white', linewidths=0.3)
axes[1, 1].set_title("Scale-Location Plot", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Predicted Price")
axes[1, 1].set_ylabel("sqrt(|Standardized Residuals|)")

plt.tight_layout()
plt.savefig("figures/eval_residual_diagnostics.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：eval_residual_diagnostics.png")

# =====================================================================
# 圖 C：指標比較（訓練 vs 測試）
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 5 -- Model Performance Metrics Comparison", fontsize=13, fontweight='bold')

metrics_labels = ['R²', 'Adj R²']
train_vals = [train_metrics['R2'], train_metrics['Adj_R2']]
test_vals  = [test_metrics['R2'], test_metrics['Adj_R2']]
x = np.arange(len(metrics_labels))
width = 0.35

axes[0].bar(x - width/2, train_vals, width, label='Train Set', color='#4C72B0', alpha=0.85)
axes[0].bar(x + width/2, test_vals,  width, label='Test Set',  color='#DD8452', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_labels, fontsize=11)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("R-squared and Adjusted R-squared", fontsize=11)
axes[0].legend()
for i, (tv, dv) in enumerate(zip(train_vals, test_vals)):
    axes[0].text(i - width/2, tv + 0.01, f'{tv:.3f}', ha='center', va='bottom', fontsize=9)
    axes[0].text(i + width/2, dv + 0.01, f'{dv:.3f}', ha='center', va='bottom', fontsize=9)

error_labels = ['RMSE', 'MAE']
train_err = [train_metrics['RMSE'], train_metrics['MAE']]
test_err  = [test_metrics['RMSE'],  test_metrics['MAE']]
x2 = np.arange(len(error_labels))

axes[1].bar(x2 - width/2, train_err, width, label='Train Set', color='#55A868', alpha=0.85)
axes[1].bar(x2 + width/2, test_err,  width, label='Test Set',  color='#C44E52', alpha=0.85)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(error_labels, fontsize=11)
axes[1].set_title("RMSE and MAE (USD)", fontsize=11)
axes[1].legend()
for i, (tv, dv) in enumerate(zip(train_err, test_err)):
    axes[1].text(i - width/2, tv + 50, f'{tv:,.0f}', ha='center', va='bottom', fontsize=8)
    axes[1].text(i + width/2, dv + 50, f'{dv:,.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("figures/eval_metrics_comparison.png", bbox_inches='tight')
plt.show()
print("  💾 儲存：eval_metrics_comparison.png")

# ── 5.5 評估摘要 ──────────────────────────────────────────────────────────
print(f"""
▶ 5.4 模型評估摘要
{'-'*60}
  指標              訓練集        測試集
  ─────────────────────────────────────
  R²                {train_metrics['R2']:.4f}        {test_metrics['R2']:.4f}
  Adjusted R²       {train_metrics['Adj_R2']:.4f}        {test_metrics['Adj_R2']:.4f}
  RMSE              {train_metrics['RMSE']:>10,.2f}  {test_metrics['RMSE']:>10,.2f}
  MAE               {train_metrics['MAE']:>10,.2f}  {test_metrics['MAE']:>10,.2f}
  MAPE              {train_metrics['MAPE']:>9.2f}%  {test_metrics['MAPE']:>9.2f}%

  📊 模型診斷：
  {'✅' if test_metrics['R2'] >= 0.85 else '⚠'} R² {'≥ 0.85（達成目標）' if test_metrics['R2'] >= 0.85 else '< 0.85（未達目標）'}
  {'✅' if abs(train_metrics['R2'] - test_metrics['R2']) < 0.05 else '⚠'} 訓練/測試 R² 差距：{abs(train_metrics['R2'] - test_metrics['R2']):.4f}（{'無明顯過擬合' if abs(train_metrics['R2'] - test_metrics['R2']) < 0.05 else '可能有過擬合'})
  ✅ 最終特徵數量：{final_count} 個（符合 10~20 的要求）
""")


# =============================================================================
# ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗     ██████╗
# ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ██╔════╝
# ██████╔╝███████║███████║███████╗█████╗      ███████╗
# ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝      ██╔══╝██╗
# ██║     ██║  ██║██║  ██║███████║███████╗    ╚██████╔╝
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝     ╚═════╝
# Deployment（部署）
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         PHASE 6 ── Deployment（部署）                               ║
╚══════════════════════════════════════════════════════════════════════╝

【實際應用場景】

  1. 汽車買賣網站即時估價 API
     ─────────────────────────
     將訓練好的模型包裝為 RESTful API（使用 FastAPI 或 Flask），
     整合至二手車交易平台（如 Yahoo 拍賣、速買配）。
     用戶只需輸入：品牌、排氣量、馬力、車體型式等規格，
     系統即時回傳預測市場公允價，並顯示「95% 信賴區間」，
     讓買賣雙方都有價格參考依據。

  2. 車廠定價決策支援系統
     ─────────────────────────
     新車開發完成後，工程師輸入車輛規格，系統自動比較競品定價，
     建議出廠建議售價（MSRP），縮短定價決策週期。

  3. 銀行 / 保險核保輔助
     ─────────────────────────
     結合車齡折舊模型，提供車輛殘值估算，
     協助銀行汽車貸款成數設定與保險理賠金額計算。

【部署架構建議】

  ┌──────────────────────────────────────────────────────────┐
  │  User (Browser/App)                                      │
  │        │  HTTP Request (車輛規格)                         │
  │        ▼                                                  │
  │  API Gateway (FastAPI/Flask)                             │
  │        │  特徵工程 & 標準化                                │
  │        ▼                                                  │
  │  ML Model (joblib 載入 .pkl)                             │
  │        │  回傳預測值 & 信賴區間                            │
  │        ▼                                                  │
  │  JSON Response → 前端顯示                                  │
  └──────────────────────────────────────────────────────────┘

【未來優化方向】

  1. 模型升級：
     - 嘗試 Ridge / Lasso / ElasticNet 正則化線性模型
     - 使用 XGBoost / LightGBM 非線性模型，可能提升 R² 至 0.95+
     - Ensemble 模型（堆疊多個模型）

  2. 特徵工程：
     - 加入車齡（year）資訊
     - 油耗效率（city_mpg × highway_mpg 交互項）
     - 品牌溢價指數（Brand Premium Index）

  3. 資料擴充：
     - 定期爬取中古車平台最新成交行情
     - 加入車況評分（里程數、事故紀錄）
     - 整合通貨膨脹指數校正歷史價格

  4. 監控與再訓練：
     - 建立 Model Drift 偵測（Evidently AI / MLflow）
     - 每季自動觸發再訓練流程（Airflow / Prefect）
     - A/B 測試新舊模型效能
""")

# ── 儲存模型 ────────────────────────────────────────────────────────────────
import joblib

model_artifacts = {
    'sk_model':       sk_model,
    'ols_model':      ols_model,
    'scaler':         scaler,
    'final_features': final_features,
    'feature_cols':   feature_cols,
}
joblib.dump(model_artifacts, 'car_price_model.pkl')
print("  💾 模型已儲存：car_price_model.pkl")
print("\n  快速載入示範：")
print("  >>> import joblib")
print("  >>> artifacts = joblib.load('car_price_model.pkl')")
print("  >>> model = artifacts['sk_model']")

# =============================================================================
# 完成摘要
# =============================================================================
print(f"""
{'='*70}
  🎉 CRISP-DM 全流程完成！
{'='*70}

  ┌─ 各階段產出摘要 ────────────────────────────────────────────────┐
  │  Phase 1 │ 商業理解：定義定價預測的商業價值與成功標準            │
  │  Phase 2 │ 資料理解：EDA + 相關性分析（4 張圖表）                │
  │  Phase 3 │ 資料準備：品牌萃取、OHE、標準化、RFE + p-value 選擇  │
  │          │   → 最終保留 {final_count:2d} 個特徵（符合 10~20 要求）          │
  │  Phase 4 │ 建模：statsmodels OLS（含統計報表）+ sklearn LR       │
  │  Phase 5 │ 評估：RMSE/R²/MAE + 實際 vs 預測圖（含信賴區間）     │
  │  Phase 6 │ 部署：API 架構說明 + 優化路線圖                       │
  └─────────────────────────────────────────────────────────────────┘

  📈 最終模型效能：
     R²（測試集）= {test_metrics['R2']:.4f}
     RMSE（測試集）= {test_metrics['RMSE']:,.2f} USD
     特徵數量 = {final_count} 個

  📁 產出檔案：
     ├─ eda_price_distribution.png
     ├─ eda_correlation.png
     ├─ eda_categorical_boxplots.png
     ├─ eda_scatter_plots.png
     ├─ feature_selection_vif.png
     ├─ model_coefficients.png
     ├─ eval_actual_vs_predicted.png
     ├─ eval_residual_diagnostics.png
     ├─ eval_metrics_comparison.png
     └─ car_price_model.pkl
{'='*70}
""")
