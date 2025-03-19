import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import roc_curve, precision_recall_curve


# 設定支援中文的字體（例如微軟正黑體）
plt.rcParams["font.family"] = "Microsoft JhengHei"


# 創建資料夾的輔助函數
def ensure_dir_exists(directory):
    """確保指定的目錄存在，如果不存在則創建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


# 獲取模型排序
def get_model_order(df, metrics, ascending=False):
    """
    根據指標的平均性能對模型進行排序
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含模型性能指標的DataFrame
    metrics : list
        要考慮的指標列表
    ascending : bool, optional (default=False)
        是否升序排序 (True為小值優先，False為大值優先)
        
    Returns:
    --------
    list
        排序後的模型名稱列表
    """
    if len(metrics) > 1:
        # 標準化數值以便計算平均排名
        normalized_df = df.copy()
        for col in metrics:
            if col in df.columns:
                col_range = df[col].max() - df[col].min()
                if col_range > 0:  # 避免零除錯誤
                    if ascending:  # 對於越小越好的指標
                        normalized_df[col] = (df[col].max() - df[col]) / col_range
                    else:  # 對於越大越好的指標
                        normalized_df[col] = (df[col] - df[col].min()) / col_range
        
        avg_performance = normalized_df[metrics].mean(axis=1)
        return avg_performance.sort_values(ascending=False).index.tolist()
    elif len(metrics) == 1 and metrics[0] in df.columns:
        return df[metrics[0]].sort_values(ascending=ascending).index.tolist()
    else:
        return df.index.tolist()  # 如果沒有有效指標，返回原始順序


# 獲取指標分組
def get_metric_groups(results_df, task_type, tuned=False):
    """
    根據任務類型獲取分組的指標列表
    如果沒有驗證集，將交叉驗證指標視為驗證集指標
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        模型比較結果DataFrame
    task_type : str
        任務類型
    tuned : bool, optional (default=False)
        是否是調優後的模型
        
    Returns:
    --------
    dict
        包含分組指標的字典
    """
    # 分類指標 - 越大越好
    accuracy_metrics = ["accuracy", "precision", "recall", "f1", "balanced_acc", "auc"]
    # 概率指標 - 越大越好
    prob_metrics = ["auc_roc", "auc_pr", "calibration_slope"]
    # 誤差指標 - 越小越好
    error_metrics = ["log_loss", "brier_score", "ece"]
    # 回歸指標 - 部分越大越好，部分越接近0越好
    regression_metrics_higher = ["r2"]
    regression_metrics_lower = ["mse", "rmse", "mae"]
    # 閾值相關指標
    threshold_metrics = ["threshold_f1"]
    
    # 根據任務類型選擇指標
    if task_type == "classification":
        metrics = accuracy_metrics
    elif task_type == "prob_classification":
        metrics = prob_metrics + error_metrics + accuracy_metrics + regression_metrics_higher + regression_metrics_lower + threshold_metrics
    else:  # 標準回歸
        metrics = regression_metrics_higher + regression_metrics_lower
    
    # 篩選CV指標、訓練集指標和驗證集指標
    cv_metrics = []
    train_metrics = []
    val_metrics = []
    
    # 檢查是否有驗證集指標
    has_validation = any(col.startswith("val_") for col in results_df.columns)
    
    # 檢查所有可能的前綴
    for m in metrics:
        # 訓練集指標
        if f"train_{m}" in results_df.columns:
            train_metrics.append(f"train_{m}")
            
        # 驗證集指標 - 優先使用
        if f"val_{m}" in results_df.columns:
            val_metrics.append(f"val_{m}")
        # 如果沒有驗證集指標，但有交叉驗證指標，將其作為驗證集指標
        elif not has_validation and f"cv_{m}" in results_df.columns:
            # 將交叉驗證指標記錄到val_metrics中，以便生成相同的視覺化
            val_metrics.append(f"cv_{m}")
    
    # 對指標進行分組
    train_metrics_higher = []
    train_metrics_lower = []
    val_metrics_higher = []
    val_metrics_lower = []
    
    # 訓練集指標分組
    for m in train_metrics:
        base_metric = m.replace("train_", "")
        if base_metric in error_metrics + regression_metrics_lower:
            train_metrics_lower.append(m)
        else:
            train_metrics_higher.append(m)
    
    # 驗證集/交叉驗證指標分組
    for m in val_metrics:
        # 提取指標基礎名稱，移除前綴
        if m.startswith("val_"):
            base_metric = m.replace("val_", "")
        else:  # 以cv_開頭
            base_metric = m.replace("cv_", "")
        
        if base_metric in error_metrics + regression_metrics_lower:
            val_metrics_lower.append(m)
        else:
            val_metrics_higher.append(m)
    
    return {
        "train_metrics": {"higher": train_metrics_higher, "lower": train_metrics_lower},
        "val_metrics": {"higher": val_metrics_higher, "lower": val_metrics_lower}
    }



# 將相似指標分組
def group_similar_metrics(metrics):
    """
    將相似的指標分組
    
    Parameters:
    -----------
    metrics : list
        指標列表
        
    Returns:
    --------
    dict
        分組後的指標字典
    """
    groups = {}
    
    # 機率相關指標
    prob_metrics = [m for m in metrics if any(x in m for x in ["auc", "log_loss", "brier", "ece", "calibration"])]
    if prob_metrics:
        groups["機率預測"] = prob_metrics
    
    # 分類相關指標
    class_metrics = [m for m in metrics if any(x in m for x in ["accuracy", "precision", "recall", "f1", "balanced"])]
    if class_metrics:
        groups["分類性能"] = class_metrics
    
    # 回歸相關指標
    reg_metrics = [m for m in metrics if any(x in m for x in ["r2", "mse", "rmse", "mae"])]
    if reg_metrics:
        groups["回歸性能"] = reg_metrics
    
    # 如果沒有找到任何分組，使用一個默認組
    if not groups:
        groups["性能指標"] = metrics
    
    return groups


# 繪製單個模型的曲線（專用於分類和概率分類）
def plot_model_curves(name, model_results, task_type="prob_classification", save_figure=False, figure_dir="picture"):
    """
    繪製單個模型的評估曲線
    
    Parameters:
    -----------
    name : str
        模型名稱
    model_results : dict
        模型的評估結果字典
    task_type : str, optional (default="prob_classification")
        任務類型
    save_figure : bool, optional (default=False)
        是否保存圖表
    figure_dir : str, optional (default="picture")
        圖表保存目錄
    """
    # 優先檢查是否有驗證集曲線數據，其次檢查訓練集曲線數據
    curve_data = None
    
    # 首先檢查驗證集/交叉驗證曲線數據
    if "val_curve_data" in model_results:
        curve_data = model_results["val_curve_data"]
        source_label = "驗證集"
    elif "cv_curve_data" in model_results:
        curve_data = model_results["cv_curve_data"]
        source_label = "交叉驗證"
    # 其次檢查訓練集曲線數據
    elif "train_curve_data" in model_results:
        curve_data = model_results["train_curve_data"]
        source_label = "訓練集"
    elif "curve_data" in model_results:
        curve_data = model_results["curve_data"]
        source_label = "訓練集"
    
    if task_type not in ["classification", "prob_classification"] or curve_data is None:
        return  # 只針對有曲線數據的分類模型
    
    # 確保目錄存在
    if save_figure:
        ensure_dir_exists(figure_dir)
    
    # 創建子圖
    n_plots = 0
    if "roc" in curve_data:
        n_plots += 1
    if "pr" in curve_data:
        n_plots += 1
    if "calibration" in curve_data:
        n_plots += 1
    
    if n_plots == 0:
        return  # 沒有可繪製的曲線
    
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 6, 5))
    
    # 如果只有一個子圖，確保axes是列表
    if n_plots == 1:
        axes = [axes]
    
    # 繪製ROC曲線
    ax_idx = 0
    if "roc" in curve_data:
        fpr = curve_data["roc"]["fpr"]
        tpr = curve_data["roc"]["tpr"]
        
        # 嘗試獲取AUC值
        auc_value = 0
        for key in ["train_auc_roc", "train_auc", "val_auc_roc", "val_auc", "cv_auc_roc", "cv_auc"]:
            if key in model_results:
                auc_value = model_results[key]
                break
        
        axes[ax_idx].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_value:.3f})')
        axes[ax_idx].plot([0, 1], [0, 1], 'k--')
        axes[ax_idx].set_xlim([0.0, 1.0])
        axes[ax_idx].set_ylim([0.0, 1.05])
        axes[ax_idx].set_xlabel('False Positive Rate')
        axes[ax_idx].set_ylabel('True Positive Rate')
        axes[ax_idx].set_title(f'{name} - ROC Curve ({source_label})')
        axes[ax_idx].legend(loc="lower right")
        ax_idx += 1
    
    # 繪製PR曲線（如果有）
    if "pr" in curve_data:
        precision = curve_data["pr"]["precision"]
        recall = curve_data["pr"]["recall"]
        
        # 嘗試獲取PR AUC值
        auc_pr_value = 0
        for key in ["train_auc_pr", "val_auc_pr", "cv_auc_pr"]:
            if key in model_results:
                auc_pr_value = model_results[key]
                break
        
        axes[ax_idx].plot(recall, precision, label=f'PR curve (AUC = {auc_pr_value:.3f})')
        axes[ax_idx].set_xlim([0.0, 1.0])
        axes[ax_idx].set_ylim([0.0, 1.05])
        axes[ax_idx].set_xlabel('Recall')
        axes[ax_idx].set_ylabel('Precision')
        axes[ax_idx].set_title(f'{name} - Precision-Recall Curve ({source_label})')
        axes[ax_idx].legend(loc="lower left")
        ax_idx += 1
    
    # 繪製校準曲線
    if "calibration" in curve_data:
        prob_pred = curve_data["calibration"]["prob_pred"]
        prob_true = curve_data["calibration"]["prob_true"]
        
        # 嘗試獲取ECE值
        ece_value = 0
        for key in ["train_ece", "val_ece", "cv_ece"]:
            if key in model_results:
                ece_value = model_results[key]
                break
        
        axes[ax_idx].plot(prob_pred, prob_true, marker='o', label='Calibration curve')
        axes[ax_idx].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        axes[ax_idx].set_xlim([0.0, 1.0])
        axes[ax_idx].set_ylim([0.0, 1.0])
        axes[ax_idx].set_xlabel('Mean predicted probability')
        axes[ax_idx].set_ylabel('Fraction of positives')
        axes[ax_idx].set_title(f'{name} - Calibration Curve ({source_label}, ECE = {ece_value:.3f})')
        axes[ax_idx].legend(loc="lower right")
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(f"{figure_dir}/{name}_curves.png", dpi=300, bbox_inches='tight')
    
    plt.show()


# 繪製單個回歸模型的散點圖
def plot_regression_result(name, model_results, save_figure=False, figure_dir="picture"):
    """
    繪製回歸模型的實際值與預測值對比圖，同時顯示訓練集和驗證集（如果有）
    
    Parameters:
    -----------
    name : str
        模型名稱
    model_results : dict
        模型的評估結果字典
    save_figure : bool, optional (default=False)
        是否保存圖表
    figure_dir : str, optional (default="picture")
        圖表保存目錄
    """
    # 檢查是否存在回歸數據
    has_train_regression = "train_regression_data" in model_results
    has_val_regression = "val_regression_data" in model_results
    
    # 如果沒有回歸數據，則使用舊版本的兼容性檢查
    if not (has_train_regression or has_val_regression):
        # 嘗試使用舊版本的回歸數據
        if "regression_data" in model_results:
            has_train_regression = True
            model_results["train_regression_data"] = model_results["regression_data"]
        else:
            return  # 沒有數據可視化
    
    # 確保目錄存在
    if save_figure:
        ensure_dir_exists(figure_dir)
    
    # 配置子圖數量
    n_plots = 1
    if has_train_regression and has_val_regression:
        n_plots = 3  # 訓練集、驗證集和組合圖
    
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 6, 5))
    
    # 如果只有一個子圖，確保axes是一個列表
    if n_plots == 1:
        axes = [axes]
    
    # 繪製訓練集散點圖
    if has_train_regression:
        ax = axes[0]
        
        train_regression_data = model_results["train_regression_data"]
        y_true = train_regression_data["y_true"]
        y_pred = train_regression_data["y_pred"]
        
        # 計算訓練集評估指標
        train_r2 = model_results.get("train_r2", r2_score(y_true, y_pred))
        train_rmse = model_results.get("train_rmse", sqrt(mean_squared_error(y_true, y_pred)))
        
        # 繪製散點圖
        ax.scatter(y_true, y_pred, alpha=0.5, label=f'Train: R² = {train_r2:.3f}, RMSE = {train_rmse:.3f}')
        
        # 計算最小和最大值用於畫對角線
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('實際值')
        ax.set_ylabel('預測值')
        ax.set_title(f'{name} - 訓練集實際值 vs 預測值')
        ax.legend()
        
        # 如果只有訓練集數據，則保存並返回
        if not has_val_regression and n_plots == 1:
            if save_figure:
                plt.savefig(f"{figure_dir}/{name}_train_regression.png", dpi=300, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            return
    
    # 繪製驗證集散點圖
    if has_val_regression:
        if n_plots == 3:
            ax = axes[1]
        else:
            ax = axes[0]
        
        val_regression_data = model_results["val_regression_data"]
        val_y_true = val_regression_data["y_true"]
        val_y_pred = val_regression_data["y_pred"]
        
        # 計算驗證集評估指標
        val_r2 = model_results.get("val_r2", r2_score(val_y_true, val_y_pred))
        val_rmse = model_results.get("val_rmse", sqrt(mean_squared_error(val_y_true, val_y_pred)))
        
        # 繪製散點圖
        ax.scatter(val_y_true, val_y_pred, alpha=0.5, color='orange', 
                  label=f'Validation: R² = {val_r2:.3f}, RMSE = {val_rmse:.3f}')
        
        # 計算最小和最大值用於畫對角線
        min_val = min(min(val_y_true), min(val_y_pred))
        max_val = max(max(val_y_true), max(val_y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('實際值')
        ax.set_ylabel('預測值')
        ax.set_title(f'{name} - 驗證集實際值 vs 預測值')
        ax.legend()
        
        # 如果只有驗證集數據，則保存並返回
        if not has_train_regression and n_plots == 1:
            if save_figure:
                plt.savefig(f"{figure_dir}/{name}_val_regression.png", dpi=300, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            return
    
    # 如果同時有訓練集和驗證集，繪製組合圖
    if has_train_regression and has_val_regression and n_plots == 3:
        ax = axes[2]
        
        train_regression_data = model_results["train_regression_data"]
        y_true = train_regression_data["y_true"]
        y_pred = train_regression_data["y_pred"]
        
        val_regression_data = model_results["val_regression_data"]
        val_y_true = val_regression_data["y_true"]
        val_y_pred = val_regression_data["y_pred"]
        
        # 計算評估指標
        train_r2 = model_results.get("train_r2", r2_score(y_true, y_pred))
        val_r2 = model_results.get("val_r2", r2_score(val_y_true, val_y_pred))
        
        # 繪製訓練集和驗證集散點圖
        ax.scatter(y_true, y_pred, alpha=0.5, label=f'Train: R² = {train_r2:.3f}')
        ax.scatter(val_y_true, val_y_pred, alpha=0.5, color='orange', label=f'Validation: R² = {val_r2:.3f}')
        
        # 計算整體最小和最大值用於畫對角線
        all_true = np.concatenate([y_true, val_y_true])
        all_pred = np.concatenate([y_pred, val_y_pred])
        min_val = min(min(all_true), min(all_pred))
        max_val = max(max(all_true), max(all_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('實際值')
        ax.set_ylabel('預測值')
        ax.set_title(f'{name} - 訓練集 vs 驗證集')
        ax.legend()
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(f"{figure_dir}/{name}_regression.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 繪製驗證集散點圖
    if has_val_regression:
        if n_plots == 3:
            ax = axes[1]
        else:
            ax = axes[0]
        
        val_regression_data = model_results["val_regression_data"]
        val_y_true = val_regression_data["y_true"]
        val_y_pred = val_regression_data["y_pred"]
        
        # 計算驗證集評估指標
        val_r2 = model_results.get("val_r2", r2_score(val_y_true, val_y_pred))
        val_rmse = model_results.get("val_rmse", sqrt(mean_squared_error(val_y_true, val_y_pred)))
        
        # 繪製散點圖
        ax.scatter(val_y_true, val_y_pred, alpha=0.5, color='orange', 
                  label=f'Validation: R² = {val_r2:.3f}, RMSE = {val_rmse:.3f}')
        
        # 計算最小和最大值用於畫對角線
        min_val = min(min(val_y_true), min(val_y_pred))
        max_val = max(max(val_y_true), max(val_y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('實際值')
        ax.set_ylabel('預測值')
        ax.set_title(f'{name} - 驗證集實際值 vs 預測值')
        ax.legend()
        
        # 如果只有驗證集數據，則保存並返回
        if not has_train_regression and n_plots == 1:
            if save_figure:
                plt.savefig(f"{figure_dir}/{name}_val_regression.png", dpi=300, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            return
    
    # 如果同時有訓練集和驗證集，繪製組合圖
    if has_train_regression and has_val_regression and n_plots == 3:
        ax = axes[2]
        
        train_regression_data = model_results["train_regression_data"]
        y_true = train_regression_data["y_true"]
        y_pred = train_regression_data["y_pred"]
        
        val_regression_data = model_results["val_regression_data"]
        val_y_true = val_regression_data["y_true"]
        val_y_pred = val_regression_data["y_pred"]
        
        # 計算評估指標
        train_r2 = model_results.get("train_r2", r2_score(y_true, y_pred))
        val_r2 = model_results.get("val_r2", r2_score(val_y_true, val_y_pred))
        
        # 繪製訓練集和驗證集散點圖
        ax.scatter(y_true, y_pred, alpha=0.5, label=f'Train: R² = {train_r2:.3f}')
        ax.scatter(val_y_true, val_y_pred, alpha=0.5, color='orange', label=f'Validation: R² = {val_r2:.3f}')
        
        # 計算整體最小和最大值用於畫對角線
        all_true = np.concatenate([y_true, val_y_true])
        all_pred = np.concatenate([y_pred, val_y_pred])
        min_val = min(min(all_true), min(all_pred))
        max_val = max(max(all_true), max(all_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('實際值')
        ax.set_ylabel('預測值')
        ax.set_title(f'{name} - 訓練集 vs 驗證集')
        ax.legend()
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(f"{figure_dir}/{name}_regression.png", dpi=300, bbox_inches='tight')
    
    plt.show()


# 繪製指標組
def plot_metric_groups(results_df, metrics, title, is_train=True, return_fig=False):
    """
    分組繪製指標
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        模型比較結果DataFrame
    metrics : dict
        分組指標字典，包含"higher"和"lower"兩組
    title : str
        圖表標題
    is_train : bool, optional (default=True)
        是否是訓練集指標
    return_fig : bool, optional (default=False)
        是否返回圖表對象，用於保存
    
    Returns:
    --------
    matplotlib.figure.Figure or None
        如果return_fig為True，返回圖表對象，否則返回None
    """
    # 獲取分組的指標
    higher_groups = {}
    if metrics["higher"]:
        higher_groups = group_similar_metrics(metrics["higher"])
        
    lower_groups = {}
    if metrics["lower"]:
        lower_groups = group_similar_metrics(metrics["lower"])
    
    # 計算總分組數
    total_groups = len(higher_groups) + len(lower_groups)
    
    if total_groups == 0:
        return None if return_fig else None
    
    # 創建足夠的子圖
    fig, axes = plt.subplots(total_groups, 1, figsize=(15, 5 * total_groups))
    # 確保axes是列表，即使只有一個子圖
    if total_groups == 1:
        axes = [axes]
    
    # 繪製「越高越好」的指標
    group_idx = 0
    if metrics["higher"]:
        for group_name, group_metrics in higher_groups.items():
            # 使用預先創建的子圖
            ax = axes[group_idx]
            group_idx += 1
            
            # 創建包含所選指標的DataFrame
            plot_df = results_df[group_metrics].copy()
            
            # 對模型按平均性能排序
            model_order = get_model_order(plot_df, group_metrics, ascending=False)
            
            # 重新排序DataFrame
            plot_df = plot_df.loc[model_order]
            
            # 使用不同顏色繪製每個指標
            colors = plt.cm.tab10(np.linspace(0, 1, len(group_metrics)))
            
            for j, metric in enumerate(group_metrics):
                plot_df[metric].plot(kind="barh", color=colors[j], alpha=0.7, label=metric.replace("train_", ""), ax=ax)
                
                # 在柱狀圖上添加數值標籤
                for i, v in enumerate(plot_df[metric]):
                    ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
            
            ax.set_title(f"{group_name} 指標 (越高越好)")
            ax.set_xlabel("Score")
            ax.legend()
            ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # 繪製「越低越好」的指標
    if metrics["lower"]:
        for group_name, group_metrics in lower_groups.items():
            # 使用預先創建的子圖
            ax = axes[group_idx]
            group_idx += 1
            
            # 創建包含所選指標的DataFrame
            plot_df = results_df[group_metrics].copy()
            
            # 對模型按平均性能排序 (注意這裡是越小越好，所以ascending=True)
            model_order = get_model_order(plot_df, group_metrics, ascending=True)
            
            # 重新排序DataFrame
            plot_df = plot_df.loc[model_order]
            
            # 使用不同顏色繪製每個指標
            colors = plt.cm.tab10(np.linspace(0, 1, len(group_metrics)))
            
            for j, metric in enumerate(group_metrics):
                plot_df[metric].plot(kind="barh", color=colors[j], alpha=0.7, label=metric.replace("train_", ""), ax=ax)
                
                # 在柱狀圖上添加數值標籤
                for i, v in enumerate(plot_df[metric]):
                    ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
            
            ax.set_title(f"{group_name} 指標 (越接近0越好)")
            ax.set_xlabel("Score")
            ax.legend()
            ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if return_fig:
        return fig
    plt.show()
    return None


# 繪製熱圖
def plot_heatmap(results_df, metrics, show_avg_performance=True, return_fig=False):
    """
    使用熱圖可視化模型性能比較
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        模型比較結果DataFrame
    metrics : dict
        包含"higher"和"lower"兩組的指標字典
    show_avg_performance : bool, optional (default=True)
        是否顯示平均性能排名條形圖
    return_fig : bool, optional (default=False)
        是否返回圖表對象，用於保存
    
    Returns:
    --------
    matplotlib.figure.Figure or None
        如果return_fig為True，返回圖表對象，否則返回None
    """
    # 合併所有指標
    all_metrics = metrics["higher"] + metrics["lower"]
    if not all_metrics:
        return None if return_fig else None
    
    # 篩選出需要的列並複製
    plot_df = results_df[all_metrics].copy()
    
    # 過濾掉非數值列
    numeric_columns = []
    for col in plot_df.columns:
        try:
            plot_df[col] = pd.to_numeric(plot_df[col])
            numeric_columns.append(col)
        except:
            print(f"警告: 跳過非數值列 '{col}'")
    
    # 如果沒有有效的數值列，無法繪製熱圖
    if not numeric_columns:
        print("錯誤: 沒有數值列可用於繪製熱圖")
        return None
    
    plot_df = plot_df[numeric_columns]
    
    # 標準化和處理數據
    normalized_df = plot_df.copy()
    for col in normalized_df.columns:
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        col_range = col_max - col_min
        
        if col_range > 0:  # 避免零除錯誤
            if col in metrics["higher"]:
                # 對於越大越好的指標，標準化為0-1，越大越好
                normalized_df[col] = (normalized_df[col] - col_min) / col_range
            else:
                # 對於越小越好的指標，標準化為0-1，越小越好（翻轉）
                normalized_df[col] = (col_max - normalized_df[col]) / col_range
    
    # 計算平均性能並排序
    normalized_df["平均性能"] = normalized_df.mean(axis=1)
    model_order = normalized_df["平均性能"].sort_values(ascending=False).index
    
    # 繪製熱圖
    fig = plt.figure(figsize=(12, 8))
    
    # 準備熱圖數據
    heatmap_df = normalized_df.drop(columns=["平均性能"]).loc[model_order]
    
    # 使用原始值作為註釋
    annot_df = plot_df.loc[model_order]
    annot_df = annot_df.applymap(lambda x: f"{x:.3f}")
    
    # 繪製熱圖
    ax = sns.heatmap(
        heatmap_df, 
        annot=annot_df, 
        cmap="YlGnBu", 
        linewidths=.5, 
        fmt="", 
        vmin=0, 
        vmax=1,
        cbar_kws={"label": "標準化性能 (越高越好)"}
    )
    
    # 添加標籤
    plt.title("模型性能比較熱圖", fontsize=14)
    plt.tight_layout()
    
    heatmap_fig = fig if return_fig else None

    if not return_fig:
        plt.show()
    
    # 是否顯示平均性能排名
    avg_fig = None
    if show_avg_performance:
        avg_fig = plt.figure(figsize=(10, 4))
        avg_performance = normalized_df["平均性能"].loc[model_order]
        ax = avg_performance.plot(kind="barh", color="teal")
        
        # 添加標籤
        for i, v in enumerate(avg_performance):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center")
        
        plt.title("模型綜合性能排名 (標準化後)", fontsize=14)
        plt.xlabel("標準化平均性能分數")
        plt.ylabel("模型")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        if not return_fig:
            plt.show()
    
    # 返回兩個圖形對象
    if return_fig:
        return (heatmap_fig, avg_fig)
    return None


# 新增: 繪製訓練集與驗證集性能比較圖
def plot_train_val_comparison(df, metric_groups, save_figures=False, figure_dir="picture", prefix="", timestamp=None):
    """
    繪製訓練集與驗證集/交叉驗證的性能比較圖，用於檢測過擬合
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含模型性能指標的DataFrame
    metric_groups : dict
        包含分組指標的字典
    save_figures : bool, optional (default=False)
        是否保存圖表
    figure_dir : str, optional (default="picture")
        圖表保存目錄
    prefix : str, optional (default="")
        圖表文件名前綴
    timestamp : str, optional (default=None)
        時間戳，用於圖表文件名
    """
    # 創建時間戳（如果未提供）
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 獲取訓練集和驗證集/交叉驗證的指標
    train_metrics = metric_groups["train_metrics"]["higher"] + metric_groups["train_metrics"]["lower"]
    val_metrics = metric_groups["val_metrics"]["higher"] + metric_groups["val_metrics"]["lower"]
    
    # 檢查是否有真正的驗證集指標
    has_real_validation = any(m.startswith("val_") for m in val_metrics)
    val_label = "驗證集" if has_real_validation else "交叉驗證"
    
    # 尋找訓練集指標對應的驗證集/交叉驗證指標
    metric_pairs = []
    for train_metric in train_metrics:
        base_metric = train_metric.replace("train_", "")
        
        # 尋找對應的驗證集/交叉驗證指標
        matched_val_metric = None
        for val_metric in val_metrics:
            if (val_metric.startswith("val_") and val_metric.replace("val_", "") == base_metric) or \
               (val_metric.startswith("cv_") and val_metric.replace("cv_", "") == base_metric):
                matched_val_metric = val_metric
                break
        
        if matched_val_metric:
            metric_pairs.append((train_metric, matched_val_metric, base_metric))
    
    # 如果沒有匹配的指標對，則返回
    if not metric_pairs:
        print(f"沒有找到訓練集和{val_label}共同的指標，無法繪製比較圖")
        return
    
    # 對每個模型繪製比較圖
    for model_name in df.index:
        # 準備數據
        train_val_diff = []
        metric_names = []
        diff_colors = []
        
        # 遍歷指標對
        for train_metric, val_metric, base_metric in metric_pairs:
            # 檢查這些列是否存在且為數值
            if train_metric in df.columns and val_metric in df.columns:
                train_val = df.loc[model_name, train_metric]
                val_val = df.loc[model_name, val_metric]
                
                # 檢查值是否為數字
                if not (isinstance(train_val, (int, float)) and isinstance(val_val, (int, float))):
                    continue
                
                # 計算差異（訓練集 - 驗證集/交叉驗證）
                diff = train_val - val_val
                
                # 判斷這個指標是否越大越好
                is_higher_better = any(base_metric == m.replace("train_", "").replace("val_", "").replace("cv_", "") 
                                      for m in metric_groups["train_metrics"]["higher"] + 
                                              metric_groups["val_metrics"]["higher"])
                
                # 確定顏色：如果差異顯示過擬合，則為紅色；否則為綠色
                if (is_higher_better and diff > 0) or (not is_higher_better and diff < 0):
                    # 這種情況表示訓練集比驗證集表現更好，可能有過擬合
                    color = 'orangered' if abs(diff) > 0.05 else 'orange'  # 差異較大時用深紅色
                else:
                    # 這種情況表示驗證集表現不差甚至更好，模型泛化能力好
                    color = 'green'
                
                train_val_diff.append(diff)
                metric_names.append(base_metric)
                diff_colors.append(color)
        
        # 如果沒有有效的比較數據，跳過
        if not train_val_diff:
            continue
        
        # 繪製比較圖
        fig, ax = plt.subplots(figsize=(10, max(4, len(metric_names) * 0.4)))
        bars = ax.barh(metric_names, train_val_diff, color=diff_colors)
        
        # 添加標題和標籤
        ax.set_title(f"{model_name} - 訓練集與{val_label}性能差異 (Train - {val_label.split('集')[0]})")
        ax.set_xlabel("差異值")
        
        # 添加垂直線於零點
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # 添加註解
        for i, v in enumerate(train_val_diff):
            ax.text(v + (0.01 if v >= 0 else -0.02), i, f"{v:.3f}", 
                   va='center', ha='left' if v >= 0 else 'right')
        
        # 添加說明文字
        ax.text(0.02, -0.15, "注意: 紅色/橘色條表示可能的過擬合", transform=ax.transAxes, 
               color='gray', fontsize=9)
        
        plt.tight_layout()
        
        # 保存圖表
        if save_figures:
            fig_path = os.path.join(figure_dir, f"{prefix}{model_name}_train_val_diff_{timestamp}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"訓練{val_label}差異圖已保存至: {fig_path}")
        
        plt.show()
    
    # 繪製所有模型的綜合比較
    # 選擇一些關鍵指標進行比較
    key_metrics = ['accuracy', 'f1', 'auc', 'auc_roc', 'r2']
    key_metric_pairs = []
    
    for train_metric, val_metric, base_metric in metric_pairs:
        if base_metric in key_metrics:
            key_metric_pairs.append((train_metric, val_metric, base_metric))
    
    # 如果沒有找到關鍵指標，使用前3個指標對
    if not key_metric_pairs and len(metric_pairs) > 0:
        key_metric_pairs = metric_pairs[:min(3, len(metric_pairs))]
    
    for train_metric, val_metric, base_metric in key_metric_pairs:
        # 檢查這些列是否存在
        if train_metric in df.columns and val_metric in df.columns:
            # 準備數據
            models = []
            train_vals = []
            val_vals = []
            
            for model_name in df.index:
                train_val = df.loc[model_name, train_metric]
                val_val = df.loc[model_name, val_metric]
                
                # 檢查值是否為數字
                if isinstance(train_val, (int, float)) and isinstance(val_val, (int, float)):
                    models.append(model_name)
                    train_vals.append(train_val)
                    val_vals.append(val_val)
            
            # 如果沒有有效的比較數據，跳過
            if not models:
                continue
            
            # 判斷指標是否越大越好
            is_higher_better = any(base_metric == m.replace("train_", "").replace("val_", "").replace("cv_", "") 
                                  for m in metric_groups["train_metrics"]["higher"] + 
                                          metric_groups["val_metrics"]["higher"])
            
            # 按照驗證集/交叉驗證性能對模型進行排序
            sort_idx = np.argsort(val_vals)
            if is_higher_better:
                # 如果是越大越好的指標，則降序排序
                sort_idx = sort_idx[::-1]
            
            # 重新排序數據
            models = [models[i] for i in sort_idx]
            train_vals = [train_vals[i] for i in sort_idx]
            val_vals = [val_vals[i] for i in sort_idx]
            
            # 繪製對比圖
            fig, ax = plt.subplots(figsize=(12, max(5, len(models) * 0.4)))
            
            # 設定條形寬度和位置
            bar_width = 0.35
            r1 = np.arange(len(models))
            r2 = [x + bar_width for x in r1]
            
            # 繪製條形圖
            ax.barh(r1, train_vals, bar_width, label='訓練集', color='royalblue', alpha=0.7)
            ax.barh(r2, val_vals, bar_width, label=val_label, color='orange', alpha=0.7)
            
            # 添加標題和標籤
            ax.set_title(f"模型比較 - {base_metric}")
            ax.set_xlabel(base_metric)
            ax.set_yticks([r + bar_width/2 for r in range(len(models))])
            ax.set_yticklabels(models)
            ax.legend()
            
            # 為每個條形添加數值標籤
            for i, v in enumerate(train_vals):
                ax.text(v + 0.01, i, f"{v:.3f}", va='center')
            for i, v in enumerate(val_vals):
                ax.text(v + 0.01, i + bar_width, f"{v:.3f}", va='center')
            
            # 添加網格線
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存圖表
            if save_figures:
                fig_path = os.path.join(figure_dir, f"{prefix}all_models_{base_metric}_compare_{timestamp}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"模型{base_metric}比較圖已保存至: {fig_path}")
            
            plt.show()
    
    # 繪製過擬合程度排名圖
    overfitting_scores = {}
    
    for model_name in df.index:
        # 計算所有指標的過擬合得分
        score = 0
        count = 0
        
        for train_metric, val_metric, base_metric in metric_pairs:
            # 檢查這些列是否存在
            if train_metric in df.columns and val_metric in df.columns:
                train_val = df.loc[model_name, train_metric]
                val_val = df.loc[model_name, val_metric]
                
                # 檢查值是否為數字
                if not (isinstance(train_val, (int, float)) and isinstance(val_val, (int, float))):
                    continue
                
                # 計算差異（訓練集 - 驗證集/交叉驗證）
                diff = train_val - val_val
                
                # 判斷這個指標是否越大越好
                is_higher_better = any(base_metric == m.replace("train_", "").replace("val_", "").replace("cv_", "") 
                                      for m in metric_groups["train_metrics"]["higher"] + 
                                              metric_groups["val_metrics"]["higher"])
                
                # 標準化差異（對於越小越好的指標，取反）
                if not is_higher_better:
                    diff = -diff
                
                # 累加過擬合得分
                score += diff
                count += 1
        
        # 計算平均過擬合得分
        if count > 0:
            overfitting_scores[model_name] = score / count
    
    # 如果有過擬合得分，繪製排名圖
    if overfitting_scores:
        # 按過擬合得分排序
        sorted_models = sorted(overfitting_scores.items(), key=lambda x: x[1], reverse=True)
        models = [m[0] for m in sorted_models]
        scores = [m[1] for m in sorted_models]
        
        # 設置顏色
        colors = ['orangered' if s > 0.05 else 'orange' if s > 0 else 'green' for s in scores]
        
        # 繪製排名圖
        fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.4)))
        bars = ax.barh(models, scores, color=colors)
        
        # 添加標題和標籤
        ax.set_title("模型過擬合程度排名")
        ax.set_xlabel(f"過擬合得分 (訓練集性能 - {val_label}性能)")
        
        # 添加垂直線於零點
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # 添加註解
        for i, v in enumerate(scores):
            ax.text(v + (0.01 if v >= 0 else -0.02), i, f"{v:.3f}", 
                   va='center', ha='left' if v >= 0 else 'right')
        
        # 添加說明文字
        ax.text(0.02, -0.1, "注意: 紅色/橘色條表示可能的過擬合，綠色表示良好的泛化能力", 
               transform=ax.transAxes, color='gray', fontsize=9)
        
        plt.tight_layout()
        
        # 保存圖表
        if save_figures:
            fig_path = os.path.join(figure_dir, f"{prefix}overfitting_ranking_{timestamp}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"過擬合排名圖已保存至: {fig_path}")
        
        plt.show()


# 統一視覺化函數
def plot_model_results(results_df, task_type="classification", is_tuned=False, show_avg_performance=True, save_figures=False, figure_dir="picture"):
    """
    統一的模型結果可視化函數
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        模型比較結果DataFrame
    task_type : str, optional (default="classification")
        任務類型
    is_tuned : bool, optional (default=False)
        是否是調優後的模型
    show_avg_performance : bool, optional (default=True)
        是否顯示平均性能排名條形圖
    save_figures : bool, optional (default=False)
        是否保存圖表到文件
    figure_dir : str, optional (default="picture")
        圖表保存的目錄
    """
    # 如果需要保存圖表，確保目錄存在
    if save_figures:
        ensure_dir_exists(figure_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "tuned_" if is_tuned else ""
        
    # 保存原始DataFrame以用於繪製曲線圖
    original_df = results_df.copy()
    
    # 清理 DataFrame，移除非數值列，僅用於熱力圖和條形圖
    cleaned_df = results_df.copy()
    for col in cleaned_df.columns:
        if cleaned_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            cleaned_df = cleaned_df.drop(columns=[col])
    
    # 獲取指標分類
    metric_groups = get_metric_groups(cleaned_df, task_type, tuned=is_tuned)
    
    # 檢查是否有真正的驗證集指標
    has_real_validation = any(m.startswith("val_") for m in 
                             metric_groups["val_metrics"]["higher"] + metric_groups["val_metrics"]["lower"])
    val_label = "驗證集" if has_real_validation else "交叉驗證"
    
    # 繪製訓練集指標條形圖
    if metric_groups["train_metrics"]["higher"] + metric_groups["train_metrics"]["lower"]:
        title = "調優後的訓練集結果" if is_tuned else "訓練集結果比較"
        fig = plot_metric_groups(cleaned_df, metric_groups["train_metrics"], title, is_train=True, return_fig=True)
        if save_figures and fig:
            fig_path = os.path.join(figure_dir, f"{prefix}train_metrics_{timestamp}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {fig_path}")
        plt.show()
    
    # 繪製驗證集/交叉驗證指標條形圖
    if metric_groups["val_metrics"]["higher"] + metric_groups["val_metrics"]["lower"]:
        title = f"調優後的{val_label}結果" if is_tuned else f"{val_label}結果比較"
        fig = plot_metric_groups(cleaned_df, metric_groups["val_metrics"], title, is_train=False, return_fig=True)
        if save_figures and fig:
            fig_path = os.path.join(figure_dir, f"{prefix}{val_label.split('集')[0]}_metrics_{timestamp}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {fig_path}")
        plt.show()
    
    # 繪製熱圖比較 - 訓練集指標
    if metric_groups["train_metrics"]["higher"] + metric_groups["train_metrics"]["lower"]:
        figs = plot_heatmap(cleaned_df, metric_groups["train_metrics"], show_avg_performance=show_avg_performance, return_fig=True)
        
        if save_figures and figs:
            heatmap_fig, avg_fig = figs
            
            # 儲存熱力圖
            if heatmap_fig:
                fig_path = os.path.join(figure_dir, f"{prefix}train_heatmap_{timestamp}.png")
                heatmap_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"訓練集熱力圖已保存至: {fig_path}")
            
            # 儲存性能排名圖
            if avg_fig:
                fig_path = os.path.join(figure_dir, f"{prefix}train_performance_ranking_{timestamp}.png")
                avg_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"訓練集性能排名圖已保存至: {fig_path}")
        
        # 顯示圖表
        if isinstance(figs, tuple) and figs[0]:
            plt.figure(figs[0].number)
            plt.show()
        if isinstance(figs, tuple) and figs[1]:
            plt.figure(figs[1].number)
            plt.show()
    
    # 繪製熱圖比較 - 驗證集/交叉驗證指標
    if metric_groups["val_metrics"]["higher"] + metric_groups["val_metrics"]["lower"]:
        figs = plot_heatmap(cleaned_df, metric_groups["val_metrics"], show_avg_performance=show_avg_performance, return_fig=True)
        
        if save_figures and figs:
            heatmap_fig, avg_fig = figs
            
            # 儲存熱力圖
            if heatmap_fig:
                fig_path = os.path.join(figure_dir, f"{prefix}{val_label.split('集')[0]}_heatmap_{timestamp}.png")
                heatmap_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"{val_label}熱力圖已保存至: {fig_path}")
            
            # 儲存性能排名圖
            if avg_fig:
                fig_path = os.path.join(figure_dir, f"{prefix}{val_label.split('集')[0]}_performance_ranking_{timestamp}.png")
                avg_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"{val_label}性能排名圖已保存至: {fig_path}")
        
        # 顯示圖表
        if isinstance(figs, tuple) and figs[0]:
            plt.figure(figs[0].number)
            plt.show()
        if isinstance(figs, tuple) and figs[1]:
            plt.figure(figs[1].number)
            plt.show()
    
    # 繪製訓練集與驗證集/交叉驗證性能比較圖
    if metric_groups["train_metrics"]["higher"] + metric_groups["train_metrics"]["lower"] and \
       metric_groups["val_metrics"]["higher"] + metric_groups["val_metrics"]["lower"]:
        plot_train_val_comparison(cleaned_df, metric_groups, save_figures=save_figures, figure_dir=figure_dir, prefix=prefix, timestamp=timestamp)
    
     # 繪製每個模型的詳細曲線圖 - 使用原始DataFrame
    for name, row in original_df.iterrows():
        # 跳過無法轉換為字典的行
        if not isinstance(row, (dict, pd.Series)):
            continue
            
        model_results = row.to_dict() if isinstance(row, pd.Series) else row
        
        # 處理可能的特殊情況 - 未嵌套的數據
        # 對於驗證集指標，也檢查交叉驗證數據
        for field in ["curve_data", "train_curve_data", "val_curve_data", "cv_curve_data",
                     "regression_data", "train_regression_data", "val_regression_data", "cv_regression_data"]:
            if field in original_df.columns:
                model_results[field] = original_df.loc[name, field]
        
        # ======= 這裡是重要的修改 =======
        # 如果沒有驗證集數據，但有交叉驗證數據，將交叉驗證數據作為驗證集數據
        if "val_curve_data" not in model_results and "cv_curve_data" in model_results:
            model_results["val_curve_data"] = model_results["cv_curve_data"]
            
        if "val_regression_data" not in model_results and "cv_regression_data" in model_results:
            model_results["val_regression_data"] = model_results["cv_regression_data"]
            
        # 如果沒有交叉驗證曲線數據，但可以從其他字段中構建，嘗試構建
        if task_type in ["classification", "prob_classification"]:
            # 檢查是否需要構建交叉驗證曲線數據
            if "val_curve_data" not in model_results and "cv_curve_data" not in model_results:
                # 檢查是否有必要的交叉驗證預測數據
                cv_pred_key = None
                cv_true_key = None
                
                # 查找可能包含交叉驗證預測的字段
                for key in model_results:
                    if key.startswith("cv_") and "_pred" in key:
                        cv_pred_key = key
                    if key.startswith("cv_") and "_true" in key:
                        cv_true_key = key
                
                # 如果找到必要的數據，嘗試構建曲線
                if cv_pred_key and cv_true_key and isinstance(model_results[cv_pred_key], (list, np.ndarray)) and isinstance(model_results[cv_true_key], (list, np.ndarray)):
                    cv_pred = model_results[cv_pred_key]
                    cv_true = model_results[cv_true_key]
                    
                    try:
                        # 構建ROC曲線數據
                        fpr, tpr, _ = roc_curve(cv_true, cv_pred)
                        
                        # 如果是概率分類，還可以構建PR曲線
                        if task_type == "prob_classification":
                            precision, recall, _ = precision_recall_curve(cv_true, cv_pred)
                            
                            # 構建完整的曲線數據
                            model_results["val_curve_data"] = {
                                "roc": {"fpr": fpr, "tpr": tpr},
                                "pr": {"precision": precision, "recall": recall},
                                "y_pred": cv_pred,
                                "y_true": cv_true
                            }
                        else:
                            # 對於普通分類，只構建ROC曲線
                            model_results["val_curve_data"] = {
                                "roc": {"fpr": fpr, "tpr": tpr},
                                "y_pred": cv_pred,
                                "y_true": cv_true
                            }
                    except Exception as e:
                        print(f"警告: 無法為模型 {name} 構建交叉驗證曲線: {str(e)}")
        
        # 繪製分類或概率分類的曲線
        if task_type in ["classification", "prob_classification"]:
            plot_model_curves(name, model_results, task_type, save_figures, figure_dir)
        
        # 繪製回歸模型的散點圖
        if task_type == "regression":
            plot_regression_result(name, model_results, save_figures, figure_dir)
    
    # 僅對調優後的模型額外繪製訓練時間圖
    if is_tuned and "train_time" in cleaned_df.columns:
        fig = plt.figure(figsize=(10, 4))
        train_times = cleaned_df["train_time"].sort_values()
        ax = train_times.plot(kind="barh", color="teal")
        for i, v in enumerate(train_times):
            ax.text(v + 0.1, i, f"{v:.2f}s", va="center")
        plt.title("訓練時間 (秒)")
        plt.tight_layout()
        
        if save_figures:
            fig_path = os.path.join(figure_dir, f"{prefix}train_time_{timestamp}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {fig_path}")
        plt.show()



# 簡化的視覺化接口函數
def plot_model_comparison(results_df, task_type="classification", save_figures=False, figure_dir="picture"):
    """
    可視化模型比較結果
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        模型比較結果DataFrame
    task_type : str, optional (default="classification")
        任務類型
    save_figures : bool, optional (default=False)
        是否保存圖表到文件
    figure_dir : str, optional (default="picture")
        圖表保存的目錄
    """
    plot_model_results(results_df, task_type, is_tuned=False, save_figures=save_figures, figure_dir=figure_dir)


def plot_tuned_models(results_df, task_type="classification", save_figures=False, figure_dir="picture"):
    """
    可視化調優後的模型比較結果
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        調優後的模型比較結果DataFrame
    task_type : str, optional (default="classification")
        任務類型
    save_figures : bool, optional (default=False)
        是否保存圖表到文件
    figure_dir : str, optional (default="picture")
        圖表保存的目錄
    """
    plot_model_results(results_df, task_type, is_tuned=True, save_figures=save_figures, figure_dir=figure_dir)