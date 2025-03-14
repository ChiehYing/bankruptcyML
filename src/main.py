import pandas as pd 
import numpy as np   
import pickle        
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from auto_preprocess import AutoPreprocess
from regression_model_valid import regression_validation

# 設定分析資料
data = pd.read_csv("../data/train_data_0313.csv")
data = data.dropna(subset=["Bankrupt?"])

# 設定驗證資料
data_val = pd.read_csv("../data/test_data_0313.csv")
data_val = data_val.dropna(subset=["Bankrupt?"])

# 資料預處理
ap = AutoPreprocess()
ap.fit(data, 
       ["ROA(C) before interest and depreciation before interest",
       "ROA(A) before interest and % after tax",
       "ROA(B) before interest and depreciation after tax",
       "Operating Gross Margin", "Realized Sales Gross Margin",
       "Operating Profit Rate", "Pre-tax net Interest Rate",
       "After-tax net Interest Rate",
       "Non-industry income and expenditure/revenue",
       "Continuous interest rate (after tax)", "Cash flow rate",
       "Interest-bearing debt interest rate", "Tax rate (A)",
       "Net Value Per Share (B)", "Net Value Per Share (A)",
       "Net Value Per Share (C)", "Persistent EPS in the Last Four Seasons",
       "Cash Flow Per Share", "Revenue Per Share (Yuan ¥)",
       "Operating Profit Per Share (Yuan ¥)",
       "Per Share Net profit before tax (Yuan ¥)",
       "Realized Sales Gross Profit Growth Rate",
       "Operating Profit Growth Rate", "After-tax Net Profit Growth Rate",
       "Regular Net Profit Growth Rate", "Continuous Net Profit Growth Rate",
       "Net Value Growth Rate", "Total Asset Return Growth Rate Ratio",
       "Cash Reinvestment %", "Current Ratio", "Quick Ratio",
       "Interest Expense Ratio", "Total debt/Total net worth", "Debt ratio %",
       "Net worth/Assets", "Long-term fund suitability ratio (A)",
       "Borrowing dependency", "Contingent liabilities/Net worth",
       "Operating profit/Paid-in capital",
       "Net profit before tax/Paid-in capital",
       "Inventory and accounts receivable/Net value", "Total Asset Turnover",
       "Accounts Receivable Turnover", "Average Collection Days",
       "Net Worth Turnover Rate (times)", "Revenue per person",
       "Operating profit per person", "Allocation rate per person",
       "Working Capital to Total Assets", "Quick Assets/Total Assets",
       "Current Assets/Total Assets", "Cash/Total Assets",
       "Quick Assets/Current Liability", "Cash/Current Liability",
       "Current Liability to Assets", "Operating Funds to Liability",
       "Inventory/Working Capital", "Inventory/Current Liability",
       "Current Liabilities/Liability", "Working Capital/Equity",
       "Current Liabilities/Equity", "Long-term Liability to Current Assets",
       "Retained Earnings to Total Assets", "Total income/Total expense",
       "Total expense/Assets", "Working capitcal Turnover Rate",
       "Cash Flow to Sales", "Fixed Assets to Assets",
       "Current Liability to Liability", "Current Liability to Equity",
       "Equity to Long-term Liability", "Cash Flow to Total Assets",
       "Cash Flow to Liability", "CFO to Assets", "Cash Flow to Equity",
       "Current Liability to Current Assets", "Liability-Assets Flag",
       "Net Income to Total Assets", "Total assets to GNP price",
       "No-credit Interval", "Gross Profit to Sales",
       "Net Income to Stockholder's Equity", "Liability to Equity",
       "Degree of Financial Leverage (DFL)",
       "Interest Coverage Ratio (Interest expense to EBIT)", "Net Income Flag",
       "Equity to Liability"])
ap.save("../preprocess_log/preprocess.bin")

# 設定預測目標
X_train = ap.transform(data)
y_train = data["Bankrupt?"]

# 設定驗證資料
ap_fitted = AutoPreprocess.load("../preprocess_log/preprocess.bin")
ap_fitted.transform(data_val)
X_valid = ap.transform(data_val)
y_valid = data_val["Bankrupt?"]

# 定義與訓練模型
model = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_split=10, min_samples_leaf=4, max_features=0.3)
model.fit(X_train, y_train)

# 驗證模型
regression_validation(model, X_train, y_train)




# # 使用RFE進行特徵選擇
# rfe = RFE(estimator=model, n_features_to_select=10)
# rfe.fit(X_train, y_train)

# # 選擇的重要特徵
# selected_features = X.columns[rfe.support_]
# print(selected_features)



