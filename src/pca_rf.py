import pandas as pd 
import numpy as np   
import pickle        
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from auto_preprocess import AutoPreprocess
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 設定分析資料
data = pd.read_csv(r"C:\Users\evanl\tibame_ttp_env\Taiwanese_Bankruptcy_Data.csv")

# 刪除 price為空值的資料列
data = data.dropna(subset=['Bankrupt?'])

# 資料預處理
ap = AutoPreprocess()
ap.fit(data, [' ROA(C) before interest and depreciation before interest',
       ' ROA(A) before interest and % after tax',
       ' ROA(B) before interest and depreciation after tax',
       ' Operating Gross Margin', ' Realized Sales Gross Margin',
       ' Operating Profit Rate', ' Pre-tax net Interest Rate',
       ' After-tax net Interest Rate',
       ' Non-industry income and expenditure/revenue',
       ' Continuous interest rate (after tax)', ' Operating Expense Rate',
       ' Research and development expense rate', ' Cash flow rate',
       ' Interest-bearing debt interest rate', ' Tax rate (A)',
       ' Net Value Per Share (B)', ' Net Value Per Share (A)',
       ' Net Value Per Share (C)', ' Persistent EPS in the Last Four Seasons',
       ' Cash Flow Per Share', ' Revenue Per Share (Yuan ¥)',
       ' Operating Profit Per Share (Yuan ¥)',
       ' Per Share Net profit before tax (Yuan ¥)',
       ' Realized Sales Gross Profit Growth Rate',
       ' Operating Profit Growth Rate', ' After-tax Net Profit Growth Rate',
       ' Regular Net Profit Growth Rate', ' Continuous Net Profit Growth Rate',
       ' Total Asset Growth Rate', ' Net Value Growth Rate',
       ' Total Asset Return Growth Rate Ratio', ' Cash Reinvestment %',
       ' Current Ratio', ' Quick Ratio', ' Interest Expense Ratio',
       ' Total debt/Total net worth', ' Debt ratio %', ' Net worth/Assets',
       ' Long-term fund suitability ratio (A)', ' Borrowing dependency',
       ' Contingent liabilities/Net worth',
       ' Operating profit/Paid-in capital',
       ' Net profit before tax/Paid-in capital',
       ' Inventory and accounts receivable/Net value', ' Total Asset Turnover',
       ' Accounts Receivable Turnover', ' Average Collection Days',
       ' Inventory Turnover Rate (times)', ' Fixed Assets Turnover Frequency',
       ' Net Worth Turnover Rate (times)', ' Revenue per person',
       ' Operating profit per person', ' Allocation rate per person',
       ' Working Capital to Total Assets', ' Quick Assets/Total Assets',
       ' Current Assets/Total Assets', ' Cash/Total Assets',
       ' Quick Assets/Current Liability', ' Cash/Current Liability',
       ' Current Liability to Assets', ' Operating Funds to Liability',
       ' Inventory/Working Capital', ' Inventory/Current Liability',
       ' Current Liabilities/Liability', ' Working Capital/Equity',
       ' Current Liabilities/Equity', ' Long-term Liability to Current Assets',
       ' Retained Earnings to Total Assets', ' Total income/Total expense',
       ' Total expense/Assets', ' Current Asset Turnover Rate',
       ' Quick Asset Turnover Rate', ' Working capitcal Turnover Rate',
       ' Cash Turnover Rate', ' Cash Flow to Sales', ' Fixed Assets to Assets',
       ' Current Liability to Liability', ' Current Liability to Equity',
       ' Equity to Long-term Liability', ' Cash Flow to Total Assets',
       ' Cash Flow to Liability', ' CFO to Assets', ' Cash Flow to Equity',
       ' Current Liability to Current Assets', ' Liability-Assets Flag',
       ' Net Income to Total Assets', ' Total assets to GNP price',
       ' No-credit Interval', ' Gross Profit to Sales',
       " Net Income to Stockholder's Equity", ' Liability to Equity',
       ' Degree of Financial Leverage (DFL)',
       ' Interest Coverage Ratio (Interest expense to EBIT)',
       ' Net Income Flag', ' Equity to Liability'])
ap.save("preprocess.bin")

# 設定預測目標
X = ap.transform(data)
y = data['Bankrupt?']

# 檢查標籤分佈
print("標籤分佈:")
print(y.value_counts())
print("破產比例:", y.mean())

# 分割數據
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=None)

# 決定要保留的主成分數量 (可以調整)
n_components = 20  # 例如保留20個主成分，你可以根據需要調整

# 創建PCA和RandomForest的管道
pipeline = Pipeline([
    ('pca', PCA(n_components=n_components, random_state=0)),
    ('rf', RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_split=10, 
                                min_samples_leaf=4, max_features=0.3, random_state=0))
])

# 使用管道擬合數據
pipeline.fit(X_train, y_train)

# 評估模型
y_pred_train = pipeline.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)
print("訓練組 R2:", r2_train)

y_pred_valid = pipeline.predict(X_valid)
r2_valid = r2_score(y_valid, y_pred_valid)
print("驗證組 R2:", r2_valid)

# 檢查PCA解釋方差比例
pca = pipeline.named_steps['pca']
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# 繪製解釋方差圖
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, color='b', label='個別解釋方差')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='累積解釋方差')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 解釋方差閾值')
plt.xlabel('主成分數量')
plt.ylabel('解釋方差比例')
plt.title('PCA解釋方差')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 繪製散布圖
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(y_train, y_pred_train, alpha=0.6, color='b')
axs[0].set_title('訓練集預測 vs 實際')
axs[0].set_xlabel('實際值')
axs[0].set_ylabel('預測值')

axs[1].scatter(y_valid, y_pred_valid, alpha=0.6, color='r')
axs[1].set_title('驗證集預測 vs 實際')
axs[1].set_xlabel('實際值')
axs[1].set_ylabel('預測值')

plt.tight_layout()
plt.show()