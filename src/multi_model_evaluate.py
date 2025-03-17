import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import log_loss, brier_score_loss, precision_recall_curve, auc, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.font_manager as fm
import seaborn as sns
import os
from datetime import datetime
from model_compare_plot import plot_model_results


# 設定支援中文的字體（例如微軟正黑體）
plt.rcParams["font.family"] = "Microsoft JhengHei"


# 創建資料夾的輔助函數
def ensure_dir_exists(directory):
    """確保指定的目錄存在，如果不存在則創建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# 模型比較
def compare_models(models, X_train, y_train, task_type="classification", threshold=None, average="binary", cv=5, save_figures=False, figure_dir="picture"):
    """
    比較多個機器學習模型的性能，使用交叉驗證和訓練集評估。
    
    Parameters:
    -----------
    models : dict
        包含多個模型的字典，格式為 {模型名稱: 模型實例}
    X_train : array-like
        訓練集特徵
    y_train : array-like
        訓練集標籤
    task_type : str, optional (default='classification')
        任務類型，可選值:
        - "classification": 標準分類問題評估
        - "regression": 標準回歸問題評估
        - "prob_classification": 使用回歸模型進行概率預測的分類問題評估
    threshold : float, optional (default=None)
        概率分類任務中，將概率轉換為類別的閾值
        若為None，則會自動尋找最佳閾值（僅適用於prob_classification）
    average : str, optional (default='binary')
        多分類問題中的平均方式，可選值: 'micro', 'macro', 'weighted', 'binary'
    cv : int, optional (default=5)
        交叉驗證的折數
    save_figures : bool, optional (default=False)
        是否保存圖表
    figure_dir : str, optional (default="picture")
        圖表保存目錄
    
    Returns:
    --------
    pandas.DataFrame
        包含所有模型評估結果的DataFrame
    """
    # 選擇評分指標 - 根據任務類型調整，確保回歸分類可用
    if task_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "balanced_acc": "balanced_accuracy"
        }
        
        # 檢查是否為二分類問題且可能使用predict_proba
        if len(np.unique(y_train)) == 2:
            # 預設評分不包含需要predict_proba的指標
            base_scoring = scoring.copy()
            # 在迴圈中對每個模型單獨檢查是否可以添加
    elif task_type == "prob_classification":
        # 使用不依賴predict_proba的基本評分指標
        scoring = {
            "mse": "neg_mean_squared_error", 
            "r2": "r2"
        }
    else:  # 回歸任務
        scoring = {
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        }
    
    # 評估結果
    results = {}
    
    # 對每個模型進行評估
    for name, model in models.items():
        print(f"評估 {name}...")
        start_time = time()
        
        # 根據模型類型調整評分指標
        current_scoring = scoring.copy()
        
        # 如果是分類問題，且模型支援predict_proba，添加AUC評分
        if task_type == "classification" and len(np.unique(y_train)) == 2:
            if hasattr(model, "predict_proba"):
                current_scoring["auc"] = "roc_auc"
        
        # 進行交叉驗證
        cv_results = cross_validate(
            model, X_train, y_train, 
            scoring=current_scoring, cv=cv, 
            return_train_score=False
        )
        
        # 在訓練集上擬合模型
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        
        # 計算指標
        model_results = {}
        
        # 記錄交叉驗證結果
        for metric, scores in cv_results.items():
            metric_name = metric.replace("test_", "")
            
            # 處理負值指標
            if metric.startswith("test_neg_"):
                model_results[metric_name.replace("neg_", "")] = -np.mean(scores)
            else:
                model_results[metric_name] = np.mean(scores)
        
        # 添加結果 - 根據任務類型分別處理
        if task_type == "classification":
            model_results["train_accuracy"] = accuracy_score(y_train, y_train_pred)
            model_results["train_precision"] = precision_score(y_train, y_train_pred, average=average)
            model_results["train_recall"] = recall_score(y_train, y_train_pred, average=average)
            model_results["train_f1"] = f1_score(y_train, y_train_pred, average=average)
            model_results["train_balanced_acc"] = balanced_accuracy_score(y_train, y_train_pred)
            
            # 對於二分類問題，添加AUC（如果模型支援）
            if len(np.unique(y_train)) == 2 and hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                model_results["train_auc"] = roc_auc_score(y_train, y_train_proba)
                
                # 添加Brier分數
                model_results["train_brier_score"] = brier_score_loss(y_train, y_train_proba)
                
                # 添加對數損失
                model_results["train_log_loss"] = log_loss(y_train, y_train_proba)
                
                # 計算校準圖
                prob_true, prob_pred = calibration_curve(y_train, y_train_proba, n_bins=10)
                model_results["train_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
                
                # 計算期望校準誤差 (Expected Calibration Error)
                bins = np.linspace(0, 1, 11)
                binned = np.digitize(y_train_proba, bins) - 1
                bin_counts = np.bincount(binned, minlength=10)
                bin_sums = np.bincount(binned, weights=y_train_proba, minlength=10)
                bin_true = np.bincount(binned, weights=y_train.astype(float), minlength=10)
                
                # 避免除以零
                bin_props = np.zeros(10)
                mask = bin_counts > 0
                bin_props[mask] = bin_sums[mask] / bin_counts[mask]
                bin_true_props = np.zeros(10)
                bin_true_props[mask] = bin_true[mask] / bin_counts[mask]
                
                # 計算ECE
                weights = bin_counts / bin_counts.sum()
                ece = np.sum(weights * np.abs(bin_props - bin_true_props))
                model_results["train_ece"] = ece
                
                # 儲存曲線資料供視覺化使用
                model_results["curve_data"] = {
                    "roc": {
                        "fpr": roc_curve(y_train, y_train_proba)[0],
                        "tpr": roc_curve(y_train, y_train_proba)[1]
                    },
                    "calibration": {
                        "prob_true": prob_true,
                        "prob_pred": prob_pred
                    }
                }
            
        elif task_type == "prob_classification":  # 回歸分類
            # 回歸評估指標
            model_results["train_mse"] = mean_squared_error(y_train, y_train_pred)
            model_results["train_rmse"] = sqrt(model_results["train_mse"])
            model_results["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            model_results["train_r2"] = r2_score(y_train, y_train_pred)
            
            # 確保預測值在[0,1]範圍內
            if np.any((y_train_pred < 0) | (y_train_pred > 1)):
                print(f"警告: {name} 產生了超出[0,1]範圍的預測值")
                # 使用Sigmoid函數重新縮放，而非簡單裁剪
                y_train_pred_scaled = 1 / (1 + np.exp(-(y_train_pred * 4 - 2)))  # 縮放因子4和偏移-2適用於大多數情況
                print(f"已使用Sigmoid函數重新縮放預測值到[0,1]範圍")
            else:
                y_train_pred_scaled = y_train_pred
            
            # 閾值優化 (如果未指定閾值)
            if threshold is None:
                # 計算不同閾值下的F1分數
                thresholds = np.linspace(0.1, 0.9, 81)  # 從0.1到0.9，步長0.01
                f1_scores = []
                
                for thresh in thresholds:
                    y_binary = (y_train_pred_scaled >= thresh).astype(int)
                    f1 = f1_score(y_train, y_binary, average=average)
                    f1_scores.append(f1)
                
                # 找出最大F1分數對應的閾值
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                
                # 記錄最佳閾值
                model_results["threshold"] = best_threshold
                model_results["threshold_f1"] = f1_scores[best_idx]
                print(f"{name}的最佳閾值: {best_threshold:.3f}, F1: {f1_scores[best_idx]:.3f}")
                
                # 使用最佳閾值計算分類指標
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
            else:
                # 使用指定閾值
                best_threshold = threshold
                model_results["threshold"] = best_threshold
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
            
            # 計算基於閾值的分類指標
            model_results["train_accuracy"] = accuracy_score(y_train, y_train_binary)
            model_results["train_precision"] = precision_score(y_train, y_train_binary, average=average)
            model_results["train_recall"] = recall_score(y_train, y_train_binary, average=average)
            model_results["train_f1"] = f1_score(y_train, y_train_binary, average=average)
            model_results["train_balanced_acc"] = balanced_accuracy_score(y_train, y_train_binary)
            
            # 計算Brier分數（直接使用縮放後的預測值）
            model_results["train_brier_score"] = brier_score_loss(y_train, y_train_pred_scaled)
            
            # 計算對數損失（使用縮放後的預測值）
            model_results["train_log_loss"] = log_loss(y_train, y_train_pred_scaled)
            
            # 計算ROC AUC和PR AUC（使用縮放後的預測值）
            model_results["train_auc_roc"] = roc_auc_score(y_train, y_train_pred_scaled)
            
            precision, recall, _ = precision_recall_curve(y_train, y_train_pred_scaled)
            model_results["train_auc_pr"] = auc(recall, precision)
            
            # 校準評估
            prob_true, prob_pred = calibration_curve(y_train, y_train_pred_scaled, n_bins=10)
            model_results["train_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
            
            # 計算期望校準誤差 (Expected Calibration Error)
            bins = np.linspace(0, 1, 11)  # 創建 11 個點，形成 10 個區間
            binned = np.clip(np.digitize(y_train_pred_scaled, bins) - 1, 0, 9)  # 確保索引在0-9之間
            bin_counts = np.bincount(binned, minlength=10)  # 明確指定長度為10
            bin_sums = np.bincount(binned, weights=y_train_pred_scaled, minlength=10)
            bin_true = np.bincount(binned, weights=y_train.astype(float), minlength=10)

            # 避免除以零
            bin_props = np.zeros(10)  # 明確設置長度為10
            mask = bin_counts > 0  # 這樣mask的長度就是10了
            bin_props[mask] = bin_sums[mask] / bin_counts[mask]
            bin_true_props = np.zeros(10)
            bin_true_props[mask] = bin_true[mask] / bin_counts[mask]
            
            # 計算ECE
            weights = bin_counts / bin_counts.sum()
            ece = np.sum(weights * np.abs(bin_props - bin_true_props))
            model_results["train_ece"] = ece
            
            # 儲存曲線資料供視覺化使用
            fpr, tpr, _ = roc_curve(y_train, y_train_pred_scaled)
            model_results["curve_data"] = {
                "roc": {
                    "fpr": fpr,
                    "tpr": tpr
                },
                "pr": {
                    "precision": precision,
                    "recall": recall
                },
                "calibration": {
                    "prob_true": prob_true,
                    "prob_pred": prob_pred
                },
                "y_pred": y_train_pred_scaled
            }
            
        else:  # regression
            model_results["train_mse"] = mean_squared_error(y_train, y_train_pred)
            model_results["train_rmse"] = sqrt(model_results["train_mse"])
            model_results["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            model_results["train_r2"] = r2_score(y_train, y_train_pred)
            
            # 儲存預測資料供視覺化使用
            model_results["regression_data"] = {
                "y_true": y_train,
                "y_pred": y_train_pred
            }

        # 添加訓練時間
        model_results["train_time"] = time() - start_time
        
        # 儲存結果
        results[name] = model_results
    
    # 創建結果DataFrame
    results_df = pd.DataFrame(results).T
    
    # 顯示結果
    print("\n模型比較結果:")
    print(results_df)
    
    # 使用統一視覺化函數
    plot_model_results(results_df, task_type, is_tuned=False, save_figures=save_figures, figure_dir=figure_dir)
    
    return results_df


# 模型參數調優
def tune_models(models, X_train, y_train, param_grids, task_type="classification", threshold=None, average="binary", cv=5, save_figures=False, figure_dir="picture"):
    """
    使用網格搜索對多個模型進行參數調優，並評估調優後模型的性能。
    
    Parameters:
    -----------
    models : dict
        包含多個模型的字典，格式為 {模型名稱: 模型實例}
    X_train : array-like
        訓練集特徵
    y_train : array-like
        訓練集標籤
    param_grids : dict
        包含每個模型參數網格的字典，格式為 {模型名稱: 參數網格字典}
    task_type : str, optional (default='classification')
        任務類型，可選值:
        - "classification": 標準分類問題評估
        - "regression": 標準回歸問題評估
        - "prob_classification": 使用回歸模型進行概率預測的分類問題評估
    threshold : float, optional (default=None)
        概率分類任務中，將概率轉換為類別的閾值
        若為None，則會自動尋找最佳閾值（僅適用於prob_classification）
    average : str, optional (default='binary')
        多分類問題中的平均方式，可選值: 'micro', 'macro', 'weighted', 'binary'
    cv : int, optional (default=5)
        交叉驗證的折數
    save_figures : bool, optional (default=False)
        是否保存圖表
    figure_dir : str, optional (default="picture")
        圖表保存目錄
    
    Returns:
    --------
    tuple
        包含兩個元素:
        - dict: 調優後的模型字典，包含最佳模型實例和參數
        - pandas.DataFrame: 調優後模型的評估結果摘要
    """
    # 選擇主要評分標準
    if task_type == "classification":
        scoring = "f1_weighted"
    elif task_type == "prob_classification":
        # 對於回歸分類，使用不依賴predict_proba的指標
        scoring = "neg_mean_squared_error"
    else:
        scoring = "neg_mean_squared_error"
    
    # 調優結果
    tuned_models = {}
    results_summary = {}
    
    # 對每個模型進行調優
    for name, model in models.items():
        if name not in param_grids:
            print(f"跳過 {name}，沒有找到參數網格")
            continue
            
        print(f"調優 {name}...")
        start_time = time()
        
        # 為分類器使用不同的評分標準
        current_scoring = scoring
        if task_type == "classification" and hasattr(model, "predict_proba") and len(np.unique(y_train)) == 2:
            current_scoring = "roc_auc"
        
        # 創建網格搜索
        grid = GridSearchCV(
            model,
            param_grids[name],
            scoring=current_scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        # 進行參數搜索
        grid.fit(X_train, y_train)
        
        # 獲取最佳模型
        best_model = grid.best_estimator_
        y_train_pred = best_model.predict(X_train)
        
        # 創建結果摘要
        model_results = {
            "best_params": grid.best_params_,
            "train_time": time() - start_time
        }
        
        # 添加結果
        if task_type == "classification":
            model_results["train_accuracy"] = accuracy_score(y_train, y_train_pred)
            model_results["train_precision"] = precision_score(y_train, y_train_pred, average=average)
            model_results["train_recall"] = recall_score(y_train, y_train_pred, average=average)
            model_results["train_f1"] = f1_score(y_train, y_train_pred, average=average)
            model_results["train_balanced_acc"] = balanced_accuracy_score(y_train, y_train_pred)
            
            # 對於二分類問題，添加AUC（如果模型支援）
            if len(np.unique(y_train)) == 2 and hasattr(best_model, "predict_proba"):
                y_train_proba = best_model.predict_proba(X_train)[:, 1]
                model_results["train_auc"] = roc_auc_score(y_train, y_train_proba)
                
                # 添加Brier分數
                model_results["train_brier_score"] = brier_score_loss(y_train, y_train_proba)
                
                # 添加對數損失
                model_results["train_log_loss"] = log_loss(y_train, y_train_proba)
                
                # 計算校準圖
                prob_true, prob_pred = calibration_curve(y_train, y_train_proba, n_bins=10)
                model_results["train_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
                
                # 計算期望校準誤差 (Expected Calibration Error)
                bins = np.linspace(0, 1, 11)
                binned = np.digitize(y_train_proba, bins) - 1
                bin_counts = np.bincount(binned, minlength=10)
                bin_sums = np.bincount(binned, weights=y_train_proba, minlength=10)
                bin_true = np.bincount(binned, weights=y_train.astype(float), minlength=10)
                
                # 避免除以零
                bin_props = np.zeros(10)
                mask = bin_counts > 0
                bin_props[mask] = bin_sums[mask] / bin_counts[mask]
                bin_true_props = np.zeros(10)
                bin_true_props[mask] = bin_true[mask] / bin_counts[mask]
                
                # 計算ECE
                weights = bin_counts / bin_counts.sum()
                ece = np.sum(weights * np.abs(bin_props - bin_true_props))
                model_results["train_ece"] = ece
                
                # 儲存曲線資料供視覺化使用
                model_results["curve_data"] = {
                    "roc": {
                        "fpr": roc_curve(y_train, y_train_proba)[0],
                        "tpr": roc_curve(y_train, y_train_proba)[1]
                    },
                    "calibration": {
                        "prob_true": prob_true,
                        "prob_pred": prob_pred
                    }
                }
            
        elif task_type == "prob_classification":  # 回歸分類
            # 回歸評估指標
            model_results["train_mse"] = mean_squared_error(y_train, y_train_pred)
            model_results["train_rmse"] = sqrt(model_results["train_mse"])
            model_results["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            model_results["train_r2"] = r2_score(y_train, y_train_pred)
            
            # 確保預測值在[0,1]範圍內
            if np.any((y_train_pred < 0) | (y_train_pred > 1)):
                print(f"警告: {name} 產生了超出[0,1]範圍的預測值")
                # 使用Sigmoid函數重新縮放，而非簡單裁剪
                y_train_pred_scaled = 1 / (1 + np.exp(-(y_train_pred * 4 - 2)))  # 縮放因子4和偏移-2適用於大多數情況
                print(f"已使用Sigmoid函數重新縮放預測值到[0,1]範圍")
            else:
                y_train_pred_scaled = y_train_pred
            
            # 閾值優化 (如果未指定閾值)
            if threshold is None:
                # 計算不同閾值下的F1分數
                thresholds = np.linspace(0.1, 0.9, 81)  # 從0.1到0.9，步長0.01
                f1_scores = []
                
                for thresh in thresholds:
                    y_binary = (y_train_pred_scaled >= thresh).astype(int)
                    f1 = f1_score(y_train, y_binary, average=average)
                    f1_scores.append(f1)
                
                # 找出最大F1分數對應的閾值
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                
                # 記錄最佳閾值
                model_results["threshold"] = best_threshold
                model_results["threshold_f1"] = f1_scores[best_idx]
                print(f"{name}的最佳閾值: {best_threshold:.3f}, F1: {f1_scores[best_idx]:.3f}")
                
                # 使用最佳閾值計算分類指標
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
            else:
                # 使用指定閾值
                best_threshold = threshold
                model_results["threshold"] = best_threshold
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
            
            # 計算基於閾值的分類指標
            model_results["train_accuracy"] = accuracy_score(y_train, y_train_binary)
            model_results["train_precision"] = precision_score(y_train, y_train_binary, average=average)
            model_results["train_recall"] = recall_score(y_train, y_train_binary, average=average)
            model_results["train_f1"] = f1_score(y_train, y_train_binary, average=average)
            model_results["train_balanced_acc"] = balanced_accuracy_score(y_train, y_train_binary)
            
            # 計算Brier分數（直接使用縮放後的預測值）
            model_results["train_brier_score"] = brier_score_loss(y_train, y_train_pred_scaled)
            
            # 計算對數損失（使用縮放後的預測值）
            model_results["train_log_loss"] = log_loss(y_train, y_train_pred_scaled)
            
            # 計算ROC AUC和PR AUC（使用縮放後的預測值）
            model_results["train_auc_roc"] = roc_auc_score(y_train, y_train_pred_scaled)
            
            precision, recall, _ = precision_recall_curve(y_train, y_train_pred_scaled)
            model_results["train_auc_pr"] = auc(recall, precision)
            
            # 校準評估
            prob_true, prob_pred = calibration_curve(y_train, y_train_pred_scaled, n_bins=10)
            model_results["train_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
            
            # 計算期望校準誤差 (Expected Calibration Error)
            bins = np.linspace(0, 1, 11)  # 創建 11 個點，形成 10 個區間
            binned = np.clip(np.digitize(y_train_pred_scaled, bins) - 1, 0, 9)  # 確保索引在0-9之間
            bin_counts = np.bincount(binned, minlength=10)  # 明確指定長度為10
            bin_sums = np.bincount(binned, weights=y_train_pred_scaled, minlength=10)
            bin_true = np.bincount(binned, weights=y_train.astype(float), minlength=10)

            # 避免除以零
            bin_props = np.zeros(10)  # 明確設置長度為10
            mask = bin_counts > 0  # 這樣mask的長度就是10了
            bin_props[mask] = bin_sums[mask] / bin_counts[mask]
            bin_true_props = np.zeros(10)
            bin_true_props[mask] = bin_true[mask] / bin_counts[mask]
            
            # 計算ECE
            weights = bin_counts / bin_counts.sum()
            ece = np.sum(weights * np.abs(bin_props - bin_true_props))
            model_results["train_ece"] = ece
            
            # 儲存曲線資料供視覺化使用
            fpr, tpr, _ = roc_curve(y_train, y_train_pred_scaled)
            model_results["curve_data"] = {
                "roc": {
                    "fpr": fpr,
                    "tpr": tpr
                },
                "pr": {
                    "precision": precision,
                    "recall": recall
                },
                "calibration": {
                    "prob_true": prob_true,
                    "prob_pred": prob_pred
                },
                "y_pred": y_train_pred_scaled
            }
        
        else:  # regression
            model_results["train_mse"] = mean_squared_error(y_train, y_train_pred)
            model_results["train_rmse"] = sqrt(model_results["train_mse"])
            model_results["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            model_results["train_r2"] = r2_score(y_train, y_train_pred)
            
            # 儲存預測資料供視覺化使用
            model_results["regression_data"] = {
                "y_true": y_train,
                "y_pred": y_train_pred
            }
        
        # 儲存結果
        tuned_models[name] = {
            "model": best_model,
            "best_params": grid.best_params_,
            "cv_score": grid.best_score_
        }
        results_summary[name] = model_results
    
    # 創建結果摘要DataFrame
    summary_df = pd.DataFrame(results_summary).T
    
    # 顯示結果
    print("\n調優後的模型比較:")
    print(summary_df)
    
    # 使用統一視覺化函數
    plot_model_results(summary_df, task_type, is_tuned=True, save_figures=save_figures, figure_dir=figure_dir)
    
    return tuned_models, summary_df


# 找出最佳模型
def find_best_model(results, task_type="classification", metric=None):
    """
    根據指定的評估指標找出最佳模型。
    
    Parameters:
    -----------
    results : pandas.DataFrame
        模型比較結果DataFrame
    task_type : str, optional (default='classification')
        任務類型，可選值:
        - "classification": 分類問題
        - "regression": 回歸問題
        - "prob_classification": 概率分類問題
    metric : str, optional (default=None)
        用於評估模型的指標名稱。若為None，則根據任務類型自動選擇合適的指標。
    
    Returns:
    --------
    str
        最佳模型的名稱
    """
    # 使用默認指標
    if metric is None:
        if task_type == "classification":
            for m in ["train_f1", "f1", "train_accuracy", "accuracy"]:
                if m in results.columns:
                    metric = m
                    break
        elif task_type == "prob_classification":
            for m in ["train_auc_roc", "auc_roc", "train_f1", "f1"]:
                if m in results.columns:
                    metric = m
                    break
        else:
            for m in ["train_r2", "r2", "train_mse", "mse"]:
                if m in results.columns:
                    metric = m
                    break
    
    if metric not in results.columns:
        print(f"找不到指標 {metric}，可用指標: {results.columns.tolist()}")
        return None
    
    # 根據指標找出最佳模型
    if task_type == "regression" and metric in ["train_mse", "train_rmse", "train_mae", "mse", "rmse", "mae"]:
        # 對於這些指標，越小越好
        best_model = results[metric].idxmin()
        best_score = results.loc[best_model, metric]
        print(f"根據 {metric} 的最佳模型: {best_model} (分數: {best_score:.4f})")
    else:
        # 對於其他指標，越大越好
        best_model = results[metric].idxmax()
        best_score = results.loc[best_model, metric]
        print(f"根據 {metric} 的最佳模型: {best_model} (分數: {best_score:.4f})")
    
    return best_model