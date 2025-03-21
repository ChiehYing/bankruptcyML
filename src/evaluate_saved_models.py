import numpy as np
import pandas as pd
import pickle
import os
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, log_loss, brier_score_loss
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from math import sqrt
import matplotlib.pyplot as plt
from model_compare_plot import plot_model_results, ensure_dir_exists

def evaluate_saved_model(model_path, X_train, y_train, X_val=None, y_val=None, 
                         task_type="classification", threshold=None, average="binary", 
                         save_figures=True, figure_dir="picture"):
    """
    評估已保存的模型並生成視覺化圖形。
    
    Parameters:
    -----------
    model_path : str
        模型的.pkl檔案路徑
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
    save_figures : bool, optional (default=True)
        是否保存圖表
    figure_dir : str, optional (default="picture")
        圖表保存目錄
    
    Returns:
    --------
    pandas.DataFrame
        模型評估結果
    """
    # 載入模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 獲取模型名稱
    model_name = os.path.basename(model_path).replace('.pkl', '')
    
    # 初始化結果字典
    results = {}
    start_time = time()
    
    # 檢查是否提供了驗證集
    has_validation = X_val is not None and y_val is not None
    
    # 在訓練集上測試模型
    model_results = {}
    y_train_pred = model.predict(X_train)
    
    # 根據任務類型分別處理
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
            fpr, tpr, _ = roc_curve(y_train, y_train_proba)
            model_results["train_curve_data"] = {
                "roc": {
                    "fpr": fpr,
                    "tpr": tpr
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
                
                val_bin_props = np.zeros(10)
                val_mask = val_bin_counts > 0
                val_bin_props[val_mask] = val_bin_sums[val_mask] / val_bin_counts[val_mask]
                val_bin_true_props = np.zeros(10)
                val_bin_true_props[val_mask] = val_bin_true[val_mask] / val_bin_counts[val_mask]
                
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
            print(f"警告: {model_name} 產生了超出[0,1]範圍的預測值")
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
            print(f"{model_name}的訓練集最佳閾值: {best_threshold:.3f}, F1: {f1_scores[best_idx]:.3f}")
            
            # 使用最佳閾值計算分類指標
            y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
        else:
            # 使用指定閾值
            best_threshold = threshold
            model_results["train_threshold"] = best_threshold
            y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
            
            # 計算並記錄指定閾值的F1分數
            train_threshold_f1 = f1_score(y_train, y_train_binary, average=average)
            model_results["train_threshold_f1"] = train_threshold_f1
            print(f"{model_name}使用指定閾值: {threshold:.3f}, F1: {train_threshold_f1:.3f}")
        
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
            print(f"警告: {model_name} 模型的校準曲線擬合失敗，將使用預設值1.0")
            model_results["train_calibration_slope"] = 1.0  # 使用預設值
        
        # 計算期望校準誤差 (Expected Calibration Error)
        bins = np.linspace(0, 1, 11)
        binned = np.clip(np.digitize(y_train_pred_scaled, bins) - 1, 0, 9)
        bin_counts = np.bincount(binned, minlength=10)
        bin_sums = np.bincount(binned, weights=y_train_pred_scaled, minlength=10)
        bin_true = np.bincount(binned, weights=y_train.astype(float), minlength=10)

        bin_props = np.zeros(10)
        mask = bin_counts > 0
        bin_props[mask] = bin_sums[mask] / bin_counts[mask]
        bin_true_props = np.zeros(10)
        bin_true_props[mask] = bin_true[mask] / bin_counts[mask]
        
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
                print(f"警告: {model_name} 在驗證集產生了超出[0,1]範圍的預測值")
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
                model_results["val_calibration_slope"] = np.polyfit(val_prob_pred, val_prob_true, 1)[0]
            except np.linalg.LinAlgError:
                print(f"警告: {model_name} 模型的校準曲線擬合失敗，將使用預設值1.0")
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
    
    # 將結果放入字典
    results[model_name] = model_results
    
    # 創建結果DataFrame
    results_df = pd.DataFrame(results).T
    
    # 顯示結果
    print("\n模型評估結果:")
    print(results_df)
    
    # 使用視覺化函數
    if save_figures:
        ensure_dir_exists(figure_dir)
    
    plot_model_results(results_df, task_type, is_tuned=False, save_figures=save_figures, figure_dir=figure_dir)
    
    return results_df


def evaluate_multiple_models(model_paths, X_train, y_train, X_val=None, y_val=None, 
                            task_type="classification", threshold=None, average="binary", 
                            save_figures=True, figure_dir="picture"):
    """
    評估多個已保存的模型並生成比較視覺化。
    
    Parameters:
    -----------
    model_paths : list
        模型的.pkl檔案路徑列表
    其他參數與evaluate_saved_model函數相同
    
    Returns:
    --------
    pandas.DataFrame
        所有模型的評估結果
    """
    # 初始化結果字典
    all_results = {}
    
    # 檢查是否提供了驗證集
    has_validation = X_val is not None and y_val is not None
    
    # 評估每個模型
    for model_path in model_paths:
        print(f"開始嘗試處理模型: {model_path}")
        try:
            # 載入模型
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 獲取模型名稱
            model_name = os.path.basename(model_path).replace('.pkl', '')
            
            print(f"評估模型: {model_name}...")
            start_time = time()
            
            # 初始化結果字典
            model_results = {}
            
            # 在訓練集上測試模型
            y_train_pred = model.predict(X_train)
            
            # 根據任務類型分別處理
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
                    fpr, tpr, _ = roc_curve(y_train, y_train_proba)
                    model_results["train_curve_data"] = {
                        "roc": {
                            "fpr": fpr,
                            "tpr": tpr
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
                        
                        val_bin_props = np.zeros(10)
                        val_mask = val_bin_counts > 0
                        val_bin_props[val_mask] = val_bin_sums[val_mask] / val_bin_counts[val_mask]
                        val_bin_true_props = np.zeros(10)
                        val_bin_true_props[val_mask] = val_bin_true[val_mask] / val_bin_counts[val_mask]
                        
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
                    print(f"警告: {model_name} 產生了超出[0,1]範圍的預測值")
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
                    print(f"{model_name}的訓練集最佳閾值: {best_threshold:.3f}, F1: {f1_scores[best_idx]:.3f}")
                    
                    # 使用最佳閾值計算分類指標
                    y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
                else:
                    # 使用指定閾值
                    best_threshold = threshold
                    model_results["train_threshold"] = best_threshold
                    y_train_binary = (y_train_pred_scaled >= best_threshold).astype(int)
                    
                    # 計算並記錄指定閾值的F1分數
                    train_threshold_f1 = f1_score(y_train, y_train_binary, average=average)
                    model_results["train_threshold_f1"] = train_threshold_f1
                    print(f"{model_name}使用指定閾值: {threshold:.3f}, F1: {train_threshold_f1:.3f}")
                
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
                    print(f"警告: {model_name} 模型的校準曲線擬合失敗，將使用預設值1.0")
                    model_results["train_calibration_slope"] = 1.0  # 使用預設值
                
                # 計算期望校準誤差 (Expected Calibration Error)
                bins = np.linspace(0, 1, 11)
                binned = np.clip(np.digitize(y_train_pred_scaled, bins) - 1, 0, 9)
                bin_counts = np.bincount(binned, minlength=10)
                bin_sums = np.bincount(binned, weights=y_train_pred_scaled, minlength=10)
                bin_true = np.bincount(binned, weights=y_train.astype(float), minlength=10)

                bin_props = np.zeros(10)
                mask = bin_counts > 0
                bin_props[mask] = bin_sums[mask] / bin_counts[mask]
                bin_true_props = np.zeros(10)
                bin_true_props[mask] = bin_true[mask] / bin_counts[mask]
                
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
                        print(f"警告: {model_name} 在驗證集產生了超出[0,1]範圍的預測值")
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
                        model_results["val_calibration_slope"] = np.polyfit(val_prob_pred, val_prob_true, 1)[0]
                    except np.linalg.LinAlgError:
                        print(f"警告: {model_name} 模型的校準曲線擬合失敗，將使用預設值1.0")
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
            
            # 將此模型的結果添加到總結果
            all_results[model_name] = model_results
            
        except Exception as e:
            print(f"評估模型 {model_path} 時發生錯誤: {str(e)}")
    
    # 創建結果DataFrame
    results_df = pd.DataFrame(all_results).T
    
    # 顯示結果
    print("\n模型比較結果:")
    print(results_df)
    
    # 使用視覺化函數
    if save_figures:
        ensure_dir_exists(figure_dir)
    
    plot_model_results(results_df, task_type, is_tuned=False, save_figures=save_figures, figure_dir=figure_dir)
    
    return results_df


# 創建一個實用程式函數，根據儲存的模型直接生成評估報告
def generate_model_evaluation_report(model_path, data_dict, task_type="classification", 
                                     threshold=None, average="binary", 
                                     save_figures=True, figure_dir="picture"):
    """
    根據已保存的模型和提供的數據生成完整的評估報告。
    
    Parameters:
    -----------
    model_path : str
        模型的.pkl檔案路徑，或包含多個模型的目錄路徑
    data_dict : dict
        包含數據的字典，格式為:
        {
            'X_train': 訓練集特徵,
            'y_train': 訓練集標籤,
            'X_val': 驗證集特徵 (可選),
            'y_val': 驗證集標籤 (可選)
        }
    task_type : str, optional (default='classification')
        任務類型，可選值:
        - "classification": 標準分類問題評估
        - "regression": 標準回歸問題評估
        - "prob_classification": 使用回歸模型進行概率預測的分類問題評估
    threshold : float, optional (default=None)
        概率分類任務中，將概率轉換為類別的閾值
    average : str, optional (default='binary')
        多分類問題中的平均方式
    save_figures : bool, optional (default=True)
        是否保存圖表
    figure_dir : str, optional (default="picture")
        圖表保存目錄
    
    Returns:
    --------
    pandas.DataFrame
        模型評估結果
    """
    # 提取數據
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict.get('X_val', None)
    y_val = data_dict.get('y_val', None)
    
    # 檢查是否為目錄或單一模型文件
    if os.path.isdir(model_path):
        # 如果是目錄，獲取所有.pkl文件
        model_files = [os.path.join(model_path, f) for f in os.listdir(model_path) 
                      if f.endswith('.pkl')]
        
        if not model_files:
            raise ValueError(f"在目錄 {model_path} 中沒有找到任何.pkl模型文件")
        
        print(f"找到 {len(model_files)} 個模型檔案，進行比較評估...")
        return evaluate_multiple_models(
            model_paths=model_files,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=task_type,
            threshold=threshold,
            average=average,
            save_figures=save_figures,
            figure_dir=figure_dir
        )
    else:
        # 如果是單一文件
        print(f"評估單一模型: {model_path}")
        return evaluate_saved_model(
            model_path=model_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=task_type,
            threshold=threshold,
            average=average,
            save_figures=save_figures,
            figure_dir=figure_dir
        )


# 用法示例
if __name__ == "__main__":
    '''
    # 範例 1: 評估單一已保存模型
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # 載入示例數據集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 準備數據字典
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_test,
        'y_val': y_test
    }
    
    # 生成評估報告
    results = generate_model_evaluation_report(
        model_path="my_model.pkl",
        data_dict=data_dict,
        task_type="classification",
        save_figures=True,
        figure_dir="model_evaluation"
    )
    '''
    
    '''
    # 範例 2: 評估目錄中的多個模型
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    
    # 載入示例數據集
    data = load_boston()
    X = data.data
    y = data.target
    
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 準備數據字典
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_test,
        'y_val': y_test
    }
    
    # 生成評估報告 - 評估models目錄下的所有模型
    results = generate_model_evaluation_report(
        model_path="models/",
        data_dict=data_dict,
        task_type="regression",
        save_figures=True,
        figure_dir="regression_model_comparison"
    )
    '''