import numpy as np
import pandas as pd
from math import sqrt
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_predict
from sklearn.metrics import log_loss, brier_score_loss, precision_recall_curve, auc, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.base import clone
import os
from datetime import datetime
from model_compare_plot import plot_model_results


# 創建資料夾的輔助函數
def ensure_dir_exists(directory):
    """確保指定的目錄存在，如果不存在則創建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# 模型比較
def compare_models(models, X_train, y_train, X_val=None, y_val=None, task_type="classification", threshold=None, average="binary", cv=5, save_figures=False, figure_dir="picture"):
    """
    比較多個機器學習模型的性能，使用訓練集和驗證集評估。
    若提供驗證集，則使用驗證集評估；否則使用交叉驗證結果作為驗證集。
    
    Parameters:
    -----------
    models : dict
        包含多個模型的字典，格式為 {模型名稱: 模型實例}
    X_train : array-like
        訓練集特徵
    y_train : array-like
        訓練集標籤
    X_val : array-like, optional (default=None)
        驗證集特徵，如果提供則會評估驗證集性能
    y_val : array-like, optional (default=None)
        驗證集標籤，如果提供則會評估驗證集性能
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
    # 檢查是否提供了驗證集
    has_validation = X_val is not None and y_val is not None
    
    # 選擇評分指標 - 根據任務類型調整
    if task_type == "classification":
        scoring = {
            "cv_accuracy": "accuracy",
            "cv_precision": "precision_weighted",
            "cv_recall": "recall_weighted",
            "cv_f1": "f1_weighted",
            "cv_balanced_acc": "balanced_accuracy"
        }
        
        # 檢查是否為二分類問題且可能使用predict_proba
        if len(np.unique(y_train)) == 2:
            pass  # 不需要定義base_scoring
    elif task_type == "prob_classification":
        # 使用不依賴predict_proba的基本評分指標
        scoring = {
            "cv_mse": "neg_mean_squared_error", 
            "cv_r2": "r2"
        }
    else:  # 回歸任務
        scoring = {
            "cv_mse": "neg_mean_squared_error",
            "cv_mae": "neg_mean_absolute_error",
            "cv_r2": "r2"
        }
    
    # 評估結果
    results = {}
    
    # 對每個模型進行評估
    for name, model in models.items():
        print(f"評估 {name}...")
        start_time = time()
        
        # 初始化模型結果字典
        model_results = {}
        
        # 如果沒有驗證集，則執行交叉驗證
        if not has_validation:
            # 根據模型類型調整評分指標
            current_scoring = scoring.copy()
            
            # 如果是分類問題，且模型支援predict_proba，添加AUC評分
            if task_type == "classification" and len(np.unique(y_train)) == 2:
                if hasattr(model, "predict_proba"):
                    current_scoring["cv_auc"] = "roc_auc"
            
            # 進行交叉驗證
            cv_results = cross_validate(
                model, X_train, y_train, 
                scoring=current_scoring, cv=cv, 
                return_train_score=False
            )
            
            # 記錄交叉驗證結果
            for metric, scores in cv_results.items():
                metric_name = metric.replace("test_", "")
                
                # 處理負值指標
                if metric.startswith("test_neg_"):
                    model_results[metric_name.replace("neg_", "")] = -np.mean(scores)
                else:
                    model_results[metric_name] = np.mean(scores)
            
            # 特別處理回歸分類任務的交叉驗證
            if task_type == "prob_classification":
                # 獲取交叉驗證預測值
                cv_predictions = cross_val_predict(model, X_train, y_train, cv=cv)
                
                # 確保預測值在[0,1]範圍內
                if np.any((cv_predictions < 0) | (cv_predictions > 1)):
                    print(f"警告: {name} 交叉驗證產生了超出[0,1]範圍的預測值")
                    cv_predictions_scaled = 1 / (1 + np.exp(-(cv_predictions * 4 - 2)))
                    print(f"已使用Sigmoid函數重新縮放預測值到[0,1]範圍")
                else:
                    cv_predictions_scaled = cv_predictions
                    
                # 閾值處理
                if threshold is None:
                    # 尋找最佳閾值 (對於交叉驗證集)
                    thresholds = np.linspace(0.1, 0.9, 81)  # 從0.1到0.9，步長0.01
                    cv_f1_scores = []
                    
                    for thresh in thresholds:
                        y_binary = (cv_predictions_scaled >= thresh).astype(int)
                        f1 = f1_score(y_train, y_binary, average=average)
                        cv_f1_scores.append(f1)
                    
                    # 找出最大F1分數對應的閾值
                    best_idx = np.argmax(cv_f1_scores)
                    cv_best_threshold = thresholds[best_idx]
                    
                    # 記錄交叉驗證的最佳閾值
                    model_results["cv_threshold"] = cv_best_threshold
                    model_results["cv_threshold_f1"] = cv_f1_scores[best_idx]
                    print(f"{name}的交叉驗證最佳閾值: {cv_best_threshold:.3f}, F1: {cv_f1_scores[best_idx]:.3f}")
                    
                    # 使用最佳閾值進行二值化
                    cv_binary = (cv_predictions_scaled >= cv_best_threshold).astype(int)
                else:
                    # 使用指定閾值
                    cv_binary = (cv_predictions_scaled >= threshold).astype(int)
                    model_results["cv_threshold"] = threshold
                    
                    # 記錄閾值對應的F1分數 (新增)
                    cv_threshold_f1 = f1_score(y_train, cv_binary, average=average)
                    model_results["cv_threshold_f1"] = cv_threshold_f1
                    print(f"{name}使用指定閾值: {threshold:.3f}, F1: {cv_threshold_f1:.3f}")
                
                # 計算交叉驗證分類指標
                model_results["cv_accuracy"] = accuracy_score(y_train, cv_binary)
                model_results["cv_precision"] = precision_score(y_train, cv_binary, average=average)
                model_results["cv_recall"] = recall_score(y_train, cv_binary, average=average)
                model_results["cv_f1"] = f1_score(y_train, cv_binary, average=average)
                model_results["cv_balanced_acc"] = balanced_accuracy_score(y_train, cv_binary)
                
                # 計算交叉驗證的Brier分數和對數損失
                model_results["cv_brier_score"] = brier_score_loss(y_train, cv_predictions_scaled)
                model_results["cv_log_loss"] = log_loss(y_train, cv_predictions_scaled)
                
                # 計算交叉驗證的ROC AUC和PR AUC
                model_results["cv_auc_roc"] = roc_auc_score(y_train, cv_predictions_scaled)
                
                precision_cv, recall_cv, _ = precision_recall_curve(y_train, cv_predictions_scaled)
                model_results["cv_auc_pr"] = auc(recall_cv, precision_cv)
                
                # 保存交叉驗證曲線數據
                fpr_cv, tpr_cv, _ = roc_curve(y_train, cv_predictions_scaled)
                model_results["cv_curve_data"] = {
                    "roc": {
                        "fpr": fpr_cv,
                        "tpr": tpr_cv
                    },
                    "pr": {
                        "precision": precision_cv,
                        "recall": recall_cv
                    },
                    "y_pred": cv_predictions_scaled,
                    "y_true": y_train
                }
        
        # 在訓練集上擬合模型
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        
        # 添加訓練集結果 - 根據任務類型分別處理
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
                model_results["train_curve_data"] = {
                    "roc": {
                        "fpr": roc_curve(y_train, y_train_proba)[0],
                        "tpr": roc_curve(y_train, y_train_proba)[1]
                    },
                    "calibration": {
                        "prob_true": prob_true,
                        "prob_pred": prob_pred
                    },
                    "y_pred": y_train_proba,
                    "y_true": y_train
                }
                
                # 如果提供了驗證集，評估驗證集性能
                if has_validation:
                    y_val_pred = model.predict(X_val)
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                    
                    # 添加驗證集評估指標
                    model_results["val_accuracy"] = accuracy_score(y_val, y_val_pred)
                    model_results["val_precision"] = precision_score(y_val, y_val_pred, average=average)
                    model_results["val_recall"] = recall_score(y_val, y_val_pred, average=average)
                    model_results["val_f1"] = f1_score(y_val, y_val_pred, average=average)
                    model_results["val_balanced_acc"] = balanced_accuracy_score(y_val, y_val_pred)
                    model_results["val_auc"] = roc_auc_score(y_val, y_val_proba)
                    model_results["val_brier_score"] = brier_score_loss(y_val, y_val_proba)
                    model_results["val_log_loss"] = log_loss(y_val, y_val_proba)
                    
                    # 計算驗證集校準圖
                    val_prob_true, val_prob_pred = calibration_curve(y_val, y_val_proba, n_bins=10)
                    model_results["val_calibration_slope"] = np.polyfit(val_prob_pred, val_prob_true, 1)[0]
                    
                    # 計算驗證集ECE
                    val_binned = np.digitize(y_val_proba, bins) - 1
                    val_bin_counts = np.bincount(val_binned, minlength=10)
                    val_bin_sums = np.bincount(val_binned, weights=y_val_proba, minlength=10)
                    val_bin_true = np.bincount(val_binned, weights=y_val.astype(float), minlength=10)
                    
                    # 避免除以零
                    val_bin_props = np.zeros(10)
                    val_mask = val_bin_counts > 0
                    val_bin_props[val_mask] = val_bin_sums[val_mask] / val_bin_counts[val_mask]
                    val_bin_true_props = np.zeros(10)
                    val_bin_true_props[val_mask] = val_bin_true[val_mask] / val_bin_counts[val_mask]
                    
                    # 計算ECE
                    val_weights = val_bin_counts / val_bin_counts.sum()
                    val_ece = np.sum(val_weights * np.abs(val_bin_props - val_bin_true_props))
                    model_results["val_ece"] = val_ece
                    
                    # 儲存驗證集曲線資料
                    val_fpr, val_tpr, _ = roc_curve(y_val, y_val_proba)
                    model_results["val_curve_data"] = {
                        "roc": {
                            "fpr": val_fpr,
                            "tpr": val_tpr
                        },
                        "calibration": {
                            "prob_true": val_prob_true,
                            "prob_pred": val_prob_pred
                        },
                        "y_pred": y_val_proba,
                        "y_true": y_val
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
                y_train_pred_scaled = 1 / (1 + np.exp(-(y_train_pred * 4 - 2)))
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
                model_results["train_threshold"] = best_threshold
                model_results["train_threshold_f1"] = f1_scores[best_idx]
                print(f"{name}的訓練集最佳閾值: {best_threshold:.3f}, F1: {f1_scores[best_idx]:.3f}")
                
                # 使用最佳閾值計算分類指標
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
            else:
                # 使用指定閾值
                best_threshold = threshold
                model_results["train_threshold"] = best_threshold
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
                
                # 計算並記錄指定閾值的F1分數 (新增)
                train_threshold_f1 = f1_score(y_train, y_train_binary, average=average)
                model_results["train_threshold_f1"] = train_threshold_f1
                print(f"{name}使用指定閾值: {threshold:.3f}, F1: {train_threshold_f1:.3f}")
            
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
            try:
                model_results["train_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
            except np.linalg.LinAlgError:
                print(f"警告: {name} 模型的校準曲線擬合失敗，將使用預設值1.0")
                model_results["train_calibration_slope"] = 1.0  # 使用預設值
            
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
            model_results["train_curve_data"] = {
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
                "y_pred": y_train_pred_scaled,
                "y_true": y_train
            }
            
            # 如果提供了驗證集，評估驗證集性能
            if has_validation:
                y_val_pred = model.predict(X_val)
                
                # 驗證集回歸評估
                model_results["val_mse"] = mean_squared_error(y_val, y_val_pred)
                model_results["val_rmse"] = sqrt(model_results["val_mse"])
                model_results["val_mae"] = mean_absolute_error(y_val, y_val_pred)
                model_results["val_r2"] = r2_score(y_val, y_val_pred)
                
                # 確保預測值在[0,1]範圍內
                if np.any((y_val_pred < 0) | (y_val_pred > 1)):
                    print(f"警告: {name} 在驗證集產生了超出[0,1]範圍的預測值")
                    y_val_pred_scaled = 1 / (1 + np.exp(-(y_val_pred * 4 - 2)))
                    print(f"已使用Sigmoid函數重新縮放驗證集預測值到[0,1]範圍")
                else:
                    y_val_pred_scaled = y_val_pred
                
                # 使用訓練集的最佳閾值或指定閾值
                used_threshold = model_results.get("train_threshold", best_threshold)
                y_val_binary = (y_val_pred_scaled >= used_threshold).astype(int)
                
                # 記錄驗證集閾值評估
                model_results["val_threshold"] = used_threshold
                model_results["val_threshold_f1"] = f1_score(y_val, y_val_binary, average=average)
                
                # 驗證集分類評估
                model_results["val_accuracy"] = accuracy_score(y_val, y_val_binary)
                model_results["val_precision"] = precision_score(y_val, y_val_binary, average=average)
                model_results["val_recall"] = recall_score(y_val, y_val_binary, average=average)
                model_results["val_f1"] = f1_score(y_val, y_val_binary, average=average)
                model_results["val_balanced_acc"] = balanced_accuracy_score(y_val, y_val_binary)
                
                # 驗證集概率評估
                model_results["val_brier_score"] = brier_score_loss(y_val, y_val_pred_scaled)
                model_results["val_log_loss"] = log_loss(y_val, y_val_pred_scaled)
                model_results["val_auc_roc"] = roc_auc_score(y_val, y_val_pred_scaled)
                
                val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_pred_scaled)
                model_results["val_auc_pr"] = auc(val_recall, val_precision)
                
                # 驗證集校準評估
                val_prob_true, val_prob_pred = calibration_curve(y_val, y_val_pred_scaled, n_bins=10)
                try:
                    model_results["val_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
                except np.linalg.LinAlgError:
                    print(f"警告: {name} 模型的校準曲線擬合失敗，將使用預設值1.0")
                    model_results["val_calibration_slope"] = 1.0  # 使用預設值
                
                # 驗證集ECE計算
                val_binned = np.clip(np.digitize(y_val_pred_scaled, bins) - 1, 0, 9)
                val_bin_counts = np.bincount(val_binned, minlength=10)
                val_bin_sums = np.bincount(val_binned, weights=y_val_pred_scaled, minlength=10)
                val_bin_true = np.bincount(val_binned, weights=y_val.astype(float), minlength=10)
                
                val_bin_props = np.zeros(10)
                val_mask = val_bin_counts > 0
                val_bin_props[val_mask] = val_bin_sums[val_mask] / val_bin_counts[val_mask]
                val_bin_true_props = np.zeros(10)
                val_bin_true_props[val_mask] = val_bin_true[val_mask] / val_bin_counts[val_mask]
                
                val_weights = val_bin_counts / val_bin_counts.sum()
                val_ece = np.sum(val_weights * np.abs(val_bin_props - val_bin_true_props))
                model_results["val_ece"] = val_ece
                
                # 儲存驗證集曲線數據
                val_fpr, val_tpr, _ = roc_curve(y_val, y_val_pred_scaled)
                model_results["val_curve_data"] = {
                    "roc": {
                        "fpr": val_fpr,
                        "tpr": val_tpr
                    },
                    "pr": {
                        "precision": val_precision,
                        "recall": val_recall
                    },
                    "calibration": {
                        "prob_true": val_prob_true,
                        "prob_pred": val_prob_pred
                    },
                    "y_pred": y_val_pred_scaled,
                    "y_true": y_val
                }
            
        else:  # regression
            model_results["train_mse"] = mean_squared_error(y_train, y_train_pred)
            model_results["train_rmse"] = sqrt(model_results["train_mse"])
            model_results["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            model_results["train_r2"] = r2_score(y_train, y_train_pred)
            
            # 儲存預測資料供視覺化使用
            model_results["train_regression_data"] = {
                "y_true": y_train,
                "y_pred": y_train_pred
            }
            
            # 如果提供了驗證集，評估驗證集性能
            if has_validation:
                y_val_pred = model.predict(X_val)
                
                model_results["val_mse"] = mean_squared_error(y_val, y_val_pred)
                model_results["val_rmse"] = sqrt(model_results["val_mse"])
                model_results["val_mae"] = mean_absolute_error(y_val, y_val_pred)
                model_results["val_r2"] = r2_score(y_val, y_val_pred)
                
                # 儲存驗證集預測資料供視覺化使用
                model_results["val_regression_data"] = {
                    "y_true": y_val,
                    "y_pred": y_val_pred
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
def tune_models(models, X_train, y_train, param_grids, X_val=None, y_val=None, task_type="classification", scoring=None, threshold=None, average="binary", cv=5, save_figures=False, figure_dir="picture"):
    """
    使用網格搜索對多個模型進行參數調優，並評估調優後模型的性能。
    若提供驗證集，則使用驗證集評估；否則使用交叉驗證結果作為驗證集。
    
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
    X_val : array-like, optional (default=None)
        驗證集特徵，如果提供則會評估驗證集性能
    y_val : array-like, optional (default=None)
        驗證集標籤，如果提供則會評估驗證集性能
    task_type : str, optional (default='classification')
        任務類型，可選值:
        - "classification": 標準分類問題評估
        - "regression": 標準回歸問題評估
        - "prob_classification": 使用回歸模型進行概率預測的分類問題評估
    scoring : str, optional (default=None)
        網格搜尋時主要評分標準，可自行選值
        如果為None，則根據任務類型自動選擇
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
    # 檢查是否提供了驗證集
    has_validation = X_val is not None and y_val is not None
    
    # 選擇主要評分標準
    if scoring is None:
        if task_type == "classification":
            scoring = "f1_weighted"
        elif task_type == "prob_classification":
            # 對於回歸分類，使用不依賴predict_proba的指標
            scoring = "neg_mean_squared_error"
        else:
            scoring = "neg_mean_squared_error"
    else:
        scoring = scoring
    
    # 調優結果
    tuned_models = {}
    model_results = {}
    
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
        
        # 儲存調優後的模型
        best_model = grid.best_estimator_
        tuned_models[name] = {
            "model": best_model,
            "best_params": grid.best_params_,
            "cv_score": grid.best_score_
        }
        
        # 確保模型結果字典已初始化
        model_results[name] = {
            "best_params": grid.best_params_,
            "train_time": time() - start_time
        }
        
        # 如果沒有驗證集，則執行交叉驗證評估
        if not has_validation and task_type == "prob_classification":
            # 獲取交叉驗證預測值
            cv_predictions = cross_val_predict(best_model, X_train, y_train, cv=cv)
            
            # 添加交叉驗證回歸指標 - 新增
            model_results[name]["cv_mse"] = mean_squared_error(y_train, cv_predictions)
            model_results[name]["cv_rmse"] = sqrt(model_results[name]["cv_mse"])
            model_results[name]["cv_mae"] = mean_absolute_error(y_train, cv_predictions)
            model_results[name]["cv_r2"] = r2_score(y_train, cv_predictions)
            
            # 確保預測值在[0,1]範圍內
            if np.any((cv_predictions < 0) | (cv_predictions > 1)):
                print(f"警告: {name} 交叉驗證產生了超出[0,1]範圍的預測值")
                cv_predictions_scaled = 1 / (1 + np.exp(-(cv_predictions * 4 - 2)))
                print(f"已使用Sigmoid函數重新縮放預測值到[0,1]範圍")
            else:
                cv_predictions_scaled = cv_predictions
                
            # 閾值處理
            if threshold is None:
                # 尋找最佳閾值 (對於交叉驗證集)
                thresholds = np.linspace(0.1, 0.9, 81)  # 從0.1到0.9，步長0.01
                cv_f1_scores = []
                
                for thresh in thresholds:
                    y_binary = (cv_predictions_scaled >= thresh).astype(int)
                    f1 = f1_score(y_train, y_binary, average=average)
                    cv_f1_scores.append(f1)
                
                # 找出最大F1分數對應的閾值
                best_idx = np.argmax(cv_f1_scores)
                cv_best_threshold = thresholds[best_idx]
                
                # 記錄交叉驗證的最佳閾值
                model_results[name]["cv_threshold"] = cv_best_threshold
                model_results[name]["cv_threshold_f1"] = cv_f1_scores[best_idx]
                print(f"{name}的交叉驗證最佳閾值: {cv_best_threshold:.3f}, F1: {cv_f1_scores[best_idx]:.3f}")
                
                # 使用最佳閾值進行二值化
                cv_binary = (cv_predictions_scaled >= cv_best_threshold).astype(int)
            else:
                # 使用指定閾值
                cv_binary = (cv_predictions_scaled >= threshold).astype(int)
                model_results[name]["cv_threshold"] = threshold
                
                # 記錄閾值對應的F1分數 (新增)
                cv_threshold_f1 = f1_score(y_train, cv_binary, average=average)
                model_results[name]["cv_threshold_f1"] = cv_threshold_f1
                print(f"{name}使用指定閾值: {threshold:.3f}, F1: {cv_threshold_f1:.3f}")
            
            # 計算交叉驗證分類指標
            model_results[name]["cv_accuracy"] = accuracy_score(y_train, cv_binary)
            model_results[name]["cv_precision"] = precision_score(y_train, cv_binary, average=average)
            model_results[name]["cv_recall"] = recall_score(y_train, cv_binary, average=average)
            model_results[name]["cv_f1"] = f1_score(y_train, cv_binary, average=average)
            model_results[name]["cv_balanced_acc"] = balanced_accuracy_score(y_train, cv_binary)
            
            # 計算交叉驗證的Brier分數和對數損失
            model_results[name]["cv_brier_score"] = brier_score_loss(y_train, cv_predictions_scaled)
            model_results[name]["cv_log_loss"] = log_loss(y_train, cv_predictions_scaled)
            
            # 計算交叉驗證的ROC AUC和PR AUC
            model_results[name]["cv_auc_roc"] = roc_auc_score(y_train, cv_predictions_scaled)
            
            precision_cv, recall_cv, _ = precision_recall_curve(y_train, cv_predictions_scaled)
            model_results[name]["cv_auc_pr"] = auc(recall_cv, precision_cv)
            
            # 保存交叉驗證曲線數據
            fpr_cv, tpr_cv, _ = roc_curve(y_train, cv_predictions_scaled)
            model_results[name]["cv_curve_data"] = {
                "roc": {
                    "fpr": fpr_cv,
                    "tpr": tpr_cv
                },
                "pr": {
                    "precision": precision_cv,
                    "recall": recall_cv
                },
                "y_pred": cv_predictions_scaled,
                "y_true": y_train
            }
        
        # 在訓練集上評估
        y_train_pred = best_model.predict(X_train)
        
        # 添加結果 - 根據任務類型分別處理
        if task_type == "classification":
            model_results[name]["train_accuracy"] = accuracy_score(y_train, y_train_pred)
            model_results[name]["train_precision"] = precision_score(y_train, y_train_pred, average=average)
            model_results[name]["train_recall"] = recall_score(y_train, y_train_pred, average=average)
            model_results[name]["train_f1"] = f1_score(y_train, y_train_pred, average=average)
            model_results[name]["train_balanced_acc"] = balanced_accuracy_score(y_train, y_train_pred)
            
            # 對於二分類問題，添加AUC（如果模型支援）
            if len(np.unique(y_train)) == 2 and hasattr(best_model, "predict_proba"):
                y_train_proba = best_model.predict_proba(X_train)[:, 1]
                model_results[name]["train_auc"] = roc_auc_score(y_train, y_train_proba)
                
                # 添加Brier分數
                model_results[name]["train_brier_score"] = brier_score_loss(y_train, y_train_proba)
                
                # 添加對數損失
                model_results[name]["train_log_loss"] = log_loss(y_train, y_train_proba)
                
                # 計算校準圖
                prob_true, prob_pred = calibration_curve(y_train, y_train_proba, n_bins=10)
                model_results[name]["train_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
                
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
                model_results[name]["train_ece"] = ece
                
                # 儲存曲線資料供視覺化使用
                model_results[name]["train_curve_data"] = {
                    "roc": {
                        "fpr": roc_curve(y_train, y_train_proba)[0],
                        "tpr": roc_curve(y_train, y_train_proba)[1]
                    },
                    "calibration": {
                        "prob_true": prob_true,
                        "prob_pred": prob_pred
                    },
                    "y_pred": y_train_proba,
                    "y_true": y_train
                }
                
                # 如果提供了驗證集，評估驗證集性能
                if has_validation:
                    y_val_pred = best_model.predict(X_val)
                    y_val_proba = best_model.predict_proba(X_val)[:, 1]
                    
                    # 添加驗證集評估指標
                    model_results[name]["val_accuracy"] = accuracy_score(y_val, y_val_pred)
                    model_results[name]["val_precision"] = precision_score(y_val, y_val_pred, average=average)
                    model_results[name]["val_recall"] = recall_score(y_val, y_val_pred, average=average)
                    model_results[name]["val_f1"] = f1_score(y_val, y_val_pred, average=average)
                    model_results[name]["val_balanced_acc"] = balanced_accuracy_score(y_val, y_val_pred)
                    model_results[name]["val_auc"] = roc_auc_score(y_val, y_val_proba)
                    model_results[name]["val_brier_score"] = brier_score_loss(y_val, y_val_proba)
                    model_results[name]["val_log_loss"] = log_loss(y_val, y_val_proba)
                    
                    # 計算驗證集校準圖
                    val_prob_true, val_prob_pred = calibration_curve(y_val, y_val_proba, n_bins=10)
                    model_results[name]["val_calibration_slope"] = np.polyfit(val_prob_pred, val_prob_true, 1)[0]
                    
                    # 計算驗證集ECE
                    val_binned = np.digitize(y_val_proba, bins) - 1
                    val_bin_counts = np.bincount(val_binned, minlength=10)
                    val_bin_sums = np.bincount(val_binned, weights=y_val_proba, minlength=10)
                    val_bin_true = np.bincount(val_binned, weights=y_val.astype(float), minlength=10)
                    
                    # 避免除以零
                    val_bin_props = np.zeros(10)
                    val_mask = val_bin_counts > 0
                    val_bin_props[val_mask] = val_bin_sums[val_mask] / val_bin_counts[val_mask]
                    val_bin_true_props = np.zeros(10)
                    val_bin_true_props[val_mask] = val_bin_true[val_mask] / val_bin_counts[val_mask]
                    
                    # 計算ECE
                    val_weights = val_bin_counts / val_bin_counts.sum()
                    val_ece = np.sum(val_weights * np.abs(val_bin_props - val_bin_true_props))
                    model_results[name]["val_ece"] = val_ece
                    
                    # 儲存驗證集曲線資料
                    val_fpr, val_tpr, _ = roc_curve(y_val, y_val_proba)
                    model_results[name]["val_curve_data"] = {
                        "roc": {
                            "fpr": val_fpr,
                            "tpr": val_tpr
                        },
                        "calibration": {
                            "prob_true": val_prob_true,
                            "prob_pred": val_prob_pred
                        },
                        "y_pred": y_val_proba,
                        "y_true": y_val
                    }
                
        elif task_type == "prob_classification":  # 回歸分類
            # 回歸評估指標
            model_results[name]["train_mse"] = mean_squared_error(y_train, y_train_pred)
            model_results[name]["train_rmse"] = sqrt(model_results[name]["train_mse"])
            model_results[name]["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            model_results[name]["train_r2"] = r2_score(y_train, y_train_pred)
            
            # 確保預測值在[0,1]範圍內
            if np.any((y_train_pred < 0) | (y_train_pred > 1)):
                print(f"警告: {name} 產生了超出[0,1]範圍的預測值")
                # 使用Sigmoid函數重新縮放，而非簡單裁剪
                y_train_pred_scaled = 1 / (1 + np.exp(-(y_train_pred * 4 - 2)))
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
                model_results[name]["train_threshold"] = best_threshold
                model_results[name]["train_threshold_f1"] = f1_scores[best_idx]
                print(f"{name}的訓練集最佳閾值: {best_threshold:.3f}, F1: {f1_scores[best_idx]:.3f}")
                
                # 使用最佳閾值計算分類指標
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
            else:
                # 使用指定閾值
                best_threshold = threshold
                model_results[name]["train_threshold"] = best_threshold
                y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
                
                # 計算並記錄指定閾值的F1分數 (新增)
                train_threshold_f1 = f1_score(y_train, y_train_binary, average=average)
                model_results[name]["train_threshold_f1"] = train_threshold_f1
                print(f"{name}使用指定閾值: {threshold:.3f}, F1: {train_threshold_f1:.3f}")
            
            # 計算基於閾值的分類指標
            model_results[name]["train_accuracy"] = accuracy_score(y_train, y_train_binary)
            model_results[name]["train_precision"] = precision_score(y_train, y_train_binary, average=average)
            model_results[name]["train_recall"] = recall_score(y_train, y_train_binary, average=average)
            model_results[name]["train_f1"] = f1_score(y_train, y_train_binary, average=average)
            model_results[name]["train_balanced_acc"] = balanced_accuracy_score(y_train, y_train_binary)
            
            # 計算Brier分數（直接使用縮放後的預測值）
            model_results[name]["train_brier_score"] = brier_score_loss(y_train, y_train_pred_scaled)
            
            # 計算對數損失（使用縮放後的預測值）
            model_results[name]["train_log_loss"] = log_loss(y_train, y_train_pred_scaled)
            
            # 計算ROC AUC和PR AUC（使用縮放後的預測值）
            model_results[name]["train_auc_roc"] = roc_auc_score(y_train, y_train_pred_scaled)
            
            precision, recall, _ = precision_recall_curve(y_train, y_train_pred_scaled)
            model_results[name]["train_auc_pr"] = auc(recall, precision)
            
            # 校準評估
            prob_true, prob_pred = calibration_curve(y_train, y_train_pred_scaled, n_bins=10)
            try:
                model_results[name]["train_calibration_slope"] = np.polyfit(prob_pred, prob_true, 1)[0]
            except np.linalg.LinAlgError:
                print(f"警告: {name} 模型的校準曲線擬合失敗，將使用預設值1.0")
                model_results[name]["train_calibration_slope"] = 1.0  # 使用預設值
            
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
            model_results[name]["train_ece"] = ece
            
            # 儲存曲線資料供視覺化使用
            fpr, tpr, _ = roc_curve(y_train, y_train_pred_scaled)
            model_results[name]["train_curve_data"] = {
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
                "y_pred": y_train_pred_scaled,
                "y_true": y_train
            }
            
            # 如果提供了驗證集，評估驗證集性能
            if has_validation:
                y_val_pred = best_model.predict(X_val)
                
                # 驗證集回歸評估
                model_results[name]["val_mse"] = mean_squared_error(y_val, y_val_pred)
                model_results[name]["val_rmse"] = sqrt(model_results[name]["val_mse"])
                model_results[name]["val_mae"] = mean_absolute_error(y_val, y_val_pred)
                model_results[name]["val_r2"] = r2_score(y_val, y_val_pred)
                
                # 確保預測值在[0,1]範圍內
                if np.any((y_val_pred < 0) | (y_val_pred > 1)):
                    print(f"警告: {name} 在驗證集產生了超出[0,1]範圍的預測值")
                    y_val_pred_scaled = 1 / (1 + np.exp(-(y_val_pred * 4 - 2)))
                    print(f"已使用Sigmoid函數重新縮放驗證集預測值到[0,1]範圍")
                else:
                    y_val_pred_scaled = y_val_pred
                
                # 使用訓練集的最佳閾值或指定閾值
                used_threshold = model_results[name].get("train_threshold", best_threshold)
                y_val_binary = (y_val_pred_scaled >= used_threshold).astype(int)
                
                # 記錄驗證集閾值評估
                model_results[name]["val_threshold"] = used_threshold
                model_results[name]["val_threshold_f1"] = f1_score(y_val, y_val_binary, average=average)
                
                # 驗證集分類評估
                model_results[name]["val_accuracy"] = accuracy_score(y_val, y_val_binary)
                model_results[name]["val_precision"] = precision_score(y_val, y_val_binary, average=average)
                model_results[name]["val_recall"] = recall_score(y_val, y_val_binary, average=average)
                model_results[name]["val_f1"] = f1_score(y_val, y_val_binary, average=average)
                model_results[name]["val_balanced_acc"] = balanced_accuracy_score(y_val, y_val_binary)
                
                # 驗證集概率評估
                model_results[name]["val_brier_score"] = brier_score_loss(y_val, y_val_pred_scaled)
                model_results[name]["val_log_loss"] = log_loss(y_val, y_val_pred_scaled)
                model_results[name]["val_auc_roc"] = roc_auc_score(y_val, y_val_pred_scaled)
                
                val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_pred_scaled)
                model_results[name]["val_auc_pr"] = auc(val_recall, val_precision)
                
                # 驗證集校準評估
                val_prob_true, val_prob_pred = calibration_curve(y_val, y_val_pred_scaled, n_bins=10)
                try:
                    model_results[name]["val_calibration_slope"] = np.polyfit(val_prob_pred, val_prob_true, 1)[0]
                except np.linalg.LinAlgError:
                    print(f"警告: {name} 模型的校準曲線擬合失敗，將使用預設值1.0")
                    model_results[name]["val_calibration_slope"] = 1.0  # 使用預設值
                
                # 驗證集ECE計算
                val_binned = np.clip(np.digitize(y_val_pred_scaled, bins) - 1, 0, 9)
                val_bin_counts = np.bincount(val_binned, minlength=10)
                val_bin_sums = np.bincount(val_binned, weights=y_val_pred_scaled, minlength=10)
                val_bin_true = np.bincount(val_binned, weights=y_val.astype(float), minlength=10)
                
                val_bin_props = np.zeros(10)
                val_mask = val_bin_counts > 0
                val_bin_props[val_mask] = val_bin_sums[val_mask] / val_bin_counts[val_mask]
                val_bin_true_props = np.zeros(10)
                val_bin_true_props[val_mask] = val_bin_true[val_mask] / val_bin_counts[val_mask]
                
                val_weights = val_bin_counts / val_bin_counts.sum()
                val_ece = np.sum(val_weights * np.abs(val_bin_props - val_bin_true_props))
                model_results[name]["val_ece"] = val_ece
                
                # 儲存驗證集曲線數據
                val_fpr, val_tpr, _ = roc_curve(y_val, y_val_pred_scaled)
                model_results[name]["val_curve_data"] = {
                    "roc": {
                        "fpr": val_fpr,
                        "tpr": val_tpr
                    },
                    "pr": {
                        "precision": val_precision,
                        "recall": val_recall
                    },
                    "calibration": {
                        "prob_true": val_prob_true,
                        "prob_pred": val_prob_pred
                    },
                    "y_pred": y_val_pred_scaled,
                    "y_true": y_val
                }
            
        else:  # regression
            model_results[name]["train_mse"] = mean_squared_error(y_train, y_train_pred)
            model_results[name]["train_rmse"] = sqrt(model_results[name]["train_mse"])
            model_results[name]["train_mae"] = mean_absolute_error(y_train, y_train_pred)
            model_results[name]["train_r2"] = r2_score(y_train, y_train_pred)
            
            # 儲存預測資料供視覺化使用
            model_results[name]["train_regression_data"] = {
                "y_true": y_train,
                "y_pred": y_train_pred
            }
            
            # 如果提供了驗證集，評估驗證集性能
            if has_validation:
                y_val_pred = best_model.predict(X_val)
                
                model_results[name]["val_mse"] = mean_squared_error(y_val, y_val_pred)
                model_results[name]["val_rmse"] = sqrt(model_results[name]["val_mse"])
                model_results[name]["val_mae"] = mean_absolute_error(y_val, y_val_pred)
                model_results[name]["val_r2"] = r2_score(y_val, y_val_pred)
                
                # 儲存驗證集預測資料供視覺化使用
                model_results[name]["val_regression_data"] = {
                    "y_true": y_val,
                    "y_pred": y_val_pred
                }
    
    # 創建結果摘要DataFrame
    summary_df = pd.DataFrame(model_results).T
    
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
    優先使用驗證集指標，其次是交叉驗證指標。
    
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
    # 檢查列名前綴以確定是否有真正的驗證集
    has_real_validation = any(col.startswith("val_") for col in results.columns)
    
    # 使用默認指標
    if metric is None:
        # 選擇指標優先順序：驗證集 > 交叉驗證 > 訓練集
        if task_type == "classification":
            if has_real_validation:
                for m in ["val_f1", "val_accuracy", "val_auc"]:
                    if m in results.columns:
                        metric = m
                        break
            if metric is None:  # 如果沒找到驗證集指標，尋找交叉驗證指標
                for m in ["cv_f1", "cv_accuracy", "cv_auc"]:
                    if m in results.columns:
                        metric = m
                        break
            if metric is None:  # 如果沒找到交叉驗證指標，使用訓練集指標
                for m in ["train_f1", "train_accuracy", "train_auc"]:
                    if m in results.columns:
                        metric = m
                        break
        elif task_type == "prob_classification":
            if has_real_validation:
                for m in ["val_auc_roc", "val_f1", "val_accuracy"]:
                    if m in results.columns:
                        metric = m
                        break
            if metric is None:
                for m in ["cv_auc_roc", "cv_f1", "cv_accuracy"]:
                    if m in results.columns:
                        metric = m
                        break
            if metric is None:
                for m in ["train_auc_roc", "train_f1", "train_accuracy"]:
                    if m in results.columns:
                        metric = m
                        break
        else:  # 回歸任務
            if has_real_validation:
                for m in ["val_r2", "val_rmse", "val_mse"]:
                    if m in results.columns:
                        metric = m
                        break
            if metric is None:
                for m in ["cv_r2", "cv_rmse", "cv_mse"]:
                    if m in results.columns:
                        metric = m
                        break
            if metric is None:
                for m in ["train_r2", "train_rmse", "train_mse"]:
                    if m in results.columns:
                        metric = m
                        break
    
    if metric not in results.columns:
        print(f"找不到指標 {metric}，可用指標: {results.columns.tolist()}")
        return None
    
    # 根據指標特性找出最佳模型
    lower_is_better = any(name in metric for name in ["mse", "rmse", "mae", "log_loss", "brier_score", "ece"])
    
    if lower_is_better:
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


