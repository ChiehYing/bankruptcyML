import pandas as pd 
import numpy as np   
import pickle        
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import fbeta_score, make_scorer, classification_report, f1_score
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from auto_preprocess import AutoPreprocess
from regression_model_valid import regression_validation
from feature_select import feature_selection, feature_select_enhanced
from multi_model_evaluate import compare_models, tune_models, find_best_model
from stacking_models import StackingModel
from evaluate_saved_models import evaluate_saved_model, evaluate_multiple_models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



# 加載已調參的回歸模型
def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# 設定分析資料
data = pd.read_csv("data/train_data_0313.csv")
data = data.dropna(subset=["Bankrupt?"])

# 設定驗證資料
data_val = pd.read_csv("data/test_data_0313.csv")
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
ap.save("preprocess_log/preprocess.bin")

# 設定預測目標
X_train = ap.transform(data)
y_train = data["Bankrupt?"]

# # 若需要，設定驗證集
# X = ap.transform(data)
# y = data["Bankrupt?"]
# X_train, y_train, X_valid, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵篩選
X_train, feature_result = feature_select_enhanced(
       X_train, 
       y_train, 
       variance_threshold=0.1, 
       correlation_threshold=0.8,
       p_value_threshold=0.05
) 

# 將選取的特徵儲存
feature_result["Original_Order"] = feature_result["Feature"].apply(
    lambda x: list(X_train.columns).index(x) if x in X_train.columns else -1
)
feature_result.to_csv("preprocess_log/selected_features.csv", encoding="utf-8")

# 設定驗證資料
ap_fitted = AutoPreprocess.load("preprocess_log/preprocess.bin")
ap_fitted.transform(data_val)
X_test = ap.transform(data_val)
y_test = data_val["Bankrupt?"]

# 讀取選取的特徵
feature_culumns = pd.read_csv("preprocess_log/selected_features.csv", index_col=0)
selected_features = feature_culumns[feature_culumns["Selected"] == True]
selected_features = selected_features.sort_values('Original_Order')['Feature'].tolist()
X_test = X_test[selected_features]


##############################################################################################


# # 選擇模型
# models = {
#     "LinearRegression": LinearRegression(),
#     "SVR": SVR(kernel="rbf"),  
#     "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5),
#     "RandomForestRegressor": RandomForestRegressor(n_estimators=100),
#     "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100),
#     "XGBRegressor": XGBRegressor(n_estimators=100),
#     "LGBMRegressor": LGBMRegressor(n_estimators=100),
#     "CatBoostRegressor": CatBoostRegressor(iterations=100, verbose=False),
#     "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
# }

# # 模型比較
# model_result = compare_models(models, X_train, y_train, task_type="prob_classification", threshold=None, save_figures=True)

# # 儲存模型比較結果
# df_clean = model_result.drop(columns=["cv_curve_data","train_curve_data"])
# df_clean.to_csv("data/model_result.csv", encoding="utf-8")

# # 找出分數最高的模型
# find_best_model(model_result, task_type="prob_classification")


##############################################################################################


# # 設定模型參數
# model = {
#        "GradientBoosting": GradientBoostingRegressor()
# }

# params = {
#        "GradientBoosting":{
#               'learning_rate': [0.01],
#               'n_estimators': [300, 400],
#               'max_depth': [2, 3],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4],
#               'subsample': [0.8, 0.9, 1.0],
#               'loss': ['squared_error', 'huber'],
#               'alpha': [0.9]
#        }
# }

# # 調整模型參數
# tuned_models, tuned_result = tune_models(model, X_train, y_train, param_grids=params, task_type="prob_classification", scoring=None, threshold=None, save_figures=True)

# # 儲存模型與結果
# with open("model/tuned_model.pkl", "wb") as f:
       # pickle.dump(tuned_models["GradientBoosting"]["model"], f)

# tuned_result.to_csv("data/tuned_result.csv", encoding="utf-8")


##############################################################################################


# # 設定模型參數
# model = {
#        "GradientBoosting": GradientBoostingRegressor(),
#        "LGBMRegressor": LGBMRegressor(),
#        "XGBRegressor": XGBRegressor(),
#        "CatBoostRegressor": CatBoostRegressor(),
#        "RandomForestRegressor": RandomForestRegressor()
       
# }

# param_grids = {
#     'GradientBoosting': {
#         'learning_rate': [0.005, 0.01],
#         'n_estimators': [300, 400],
#         'max_depth': [2, 3],
#         'min_samples_split': [10],
#         'min_samples_leaf': [6, 8],
#         'subsample': [0.6, 0.7],
#         'max_features': ['sqrt'],
#         'loss': ['squared_error', 'huber']
#     },
       
#        'CatBoostRegressor': {
#        'learning_rate': [0.005],
#        'iterations': [350, 400],
#        'depth': [2],
#        'l2_leaf_reg': [10, 15],
#        'random_strength': [2, 3],
#        'bootstrap_type': ['Bernoulli'],
#        'subsample': [0.4],  # 注意：在 Bernoulli 引導類型中，這個參數控制抽樣率
#        'leaf_estimation_iterations': [5],
#        'border_count': [32]
#    },
    
#     'XGBRegressor': {
#         'learning_rate': [0.005],
#         'n_estimators': [400],
#         'max_depth': [2],
#         'min_child_weight': [5, 7],
#         'gamma': [0.2, 0.3],
#         'subsample': [0.5],
#         'colsample_bytree': [0.5],
#         'colsample_bylevel': [0.7],
#         'reg_alpha': [1.5, 2.0],
#         'reg_lambda': [1.5, 2.0],
#         'scale_pos_weight': [30]
#     },
    
#     'RandomForestRegressor': {
#         'n_estimators': [350, 400],
#         'max_depth': [3],
#         'min_samples_split': [20],
#         'min_samples_leaf': [10, 12],
#         'max_features': [0.5, 'sqrt'],
#         'bootstrap': [True],
#         'oob_score': [True],
#         'max_samples': [0.6],
#         'ccp_alpha': [0.01]
#     },
    
#     'LGBMRegressor': {
#         'learning_rate': [0.005],
#         'n_estimators': [400],
#         'num_leaves': [7, 10],
#         'min_child_samples': [30, 40],
#         'min_child_weight': [7, 10],
#         'subsample': [0.5],
#         'colsample_bytree': [0.5],
#         'reg_alpha': [1.5, 2.0],
#         'reg_lambda': [1.5, 2.0],
#         'path_smooth': [1],
#         'extra_trees': [True],
#         'max_bin': [64]
#     }
# }

# # 調整模型參數
# tuned_models, tuned_result = tune_models(model, X_train, y_train, param_grids=param_grids, task_type="prob_classification", scoring=None, threshold=None, save_figures=True)

# # 儲存模型與結果
# for model_name in tuned_models.keys():
#     with open(f"model/tuned_{model_name}.pkl", "wb") as f:
#         pickle.dump(tuned_models[model_name]["model"], f)

# tuned_result.to_csv("data/tuned_result.csv", encoding="utf-8")


##############################################################################################


# # 定義與訓練模型
# model = RandomForestRegressor(
#        n_estimators=500, 
#        max_depth=5, 
#        min_samples_split=10, 
#        min_samples_leaf=4, 
#        max_features=0.3
# )
# model.fit(X_train, y_train)

# # 驗證模型
# regression_validation(model, X_train, y_train, X_valid, y_valid)


# # 使用RFE進行特徵選擇
# rfe = RFE(estimator=model, n_features_to_select=10)
# rfe.fit(X_train, y_train)

# # 選擇的重要特徵
# selected_features = X.columns[rfe.support_]
# print(selected_features)


##############################################################################################




# # 載入所有已調參的基礎回歸模型
# lgbm_model = load_model("model/tuned_LGBMRegressor.pkl")
# gbr_model = load_model("model/tuned_GradientBoosting.pkl")
# rf_model = load_model("model/tuned_RandomForestRegressor.pkl") 
# catboost_model = load_model("model/tuned_CatBoostRegressor.pkl")
# xgb_model = load_model("model/tuned_XGBRegressor.pkl")

# base_models = [lgbm_model, gbr_model, rf_model, catboost_model, xgb_model]
# meta_model = LinearRegression()

# # 堆疊模型
# stacking = StackingModel(
#     base_models=base_models,
#     meta_model = LinearRegression(),
#     task_type="prob_classification",
#     threshold=0.5
# )

# stacking.fit(X_train, y_train)

# # 5. 優化閾值
# best_threshold, best_score = stacking.optimize_threshold(X_test, y_test)
# print(f"最佳閾值: {best_threshold:.4f}, F1分數: {best_score:.4f}")

# # 6. 測試集評估
# predictions = stacking.predict(X_test)
# print(classification_report(y_test, predictions))
# print(f"F1分數: {f1_score(y_test, predictions):.4f}")

# # 7. 保存最終模型
# stacking.save_model("model/final_stacking_model.pkl")


##############################################################################################


# # 驗證已保存的模型
# model = "model/final_stacking_model.pkl"
# model_evaluate = evaluate_saved_model(model, X_train, y_train, X_test, y_test, task_type="prob_classification", threshold=None)
# model_evaluate.to_csv("data/model_evaluate.csv", encoding="utf-8")


##############################################################################################


# # 驗證已保存的多個模型

# models = [
#        "model/tuned_LGBMRegressor.pkl", 
#        "model/tuned_GradientBoostingRegressor.pkl", 
#        "model/tuned_RandomForestRegressor.pkl", 
#        "model/tuned_CatBoostRegressor.pkl", 
#        "model/tuned_XGBRegressor.pkl"
# ]

# model_evaluate = evaluate_multiple_models(
#        models, 
#        X_train, y_train, 
#        X_test, y_test, 
#        task_type="prob_classification", 
#        threshold=None
# )


##############################################################################################


# 模型
model = load_model("model/final_stacking_model.pkl")

y_train_pred = model.predict_proba_for_regression(X_train)
y_test_pred = model.predict_proba_for_regression(X_test)

# regression_validation(model, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred)


# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)
# classification_validation(model,)


pred_df = pd.DataFrame({"pred": y_train_pred, "true": y_train})
# print(pred_df.head())

pos = pred_df["true"] == 1
pos_pred_value = pred_df[pos]["pred"]

neg = pred_df["true"] == 0
neg_pred_value = pred_df[neg]["pred"]

pos_percentiles = np.percentile(pos_pred_value, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("正類的10分位數(0-100) (10th percentile):", pos_percentiles)

neg_percentiles = np.percentile(neg_pred_value, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 95, 99])
print("負類的10分位數(0-100), 95分, 99分 (10th percentile):", neg_percentiles)


fil = pred_df["pred"] < 0.0228424
print("低風險(無正類)，佔全體:", pred_df[fil]["pred"].count() / pred_df["pred"].count())

fil = (pred_df["pred"] >= 0.0228424) & (pred_df["pred"] <= 0.09959364)
print("中度風險(含有10%正類)，佔全體:", pred_df[fil]["pred"].count() / pred_df["pred"].count())

fil = (pred_df["pred"] > 0.09959364) 
print("高風險(含有90%正類)，佔全體:", pred_df[fil]["pred"].count() / pred_df["pred"].count())



