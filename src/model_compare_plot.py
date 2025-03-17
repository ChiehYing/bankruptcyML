import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


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
    # 回歸指標 - 部分越大越好，部分越小越好
    regression_metrics_higher = ["r2"]
    regression_metrics_lower = ["mse", "rmse", "mae"]
    
    # 根據任務類型選擇指標
    if task_type == "classification":
        metrics = accuracy_metrics
    elif task_type == "prob_classification":
        metrics = prob_metrics + error_metrics + accuracy_metrics + regression_metrics_higher + regression_metrics_lower
    else:  # 標準回歸
        metrics = regression_metrics_higher + regression_metrics_lower
    
    # 篩選CV指標和訓練集指標
    cv_metrics = [m for m in metrics if m in results_df.columns]
    train_metrics = [f"train_{m}" for m in metrics if f"train_{m}" in results_df.columns]
    
    # 對指標進行分組
    train_metrics_higher = []
    train_metrics_lower = []
    cv_metrics_higher = []
    cv_metrics_lower = []
    
    for m in train_metrics:
        if any(x in m for x in ["train_" + metric for metric in error_metrics + regression_metrics_lower]):
            train_metrics_lower.append(m)
        else:
            train_metrics_higher.append(m)
    
    for m in cv_metrics:
        if any(x == m for x in error_metrics + regression_metrics_lower):
            cv_metrics_lower.append(m)
        else:
            cv_metrics_higher.append(m)
    
    return {
        "cv_metrics": {"higher": cv_metrics_higher, "lower": cv_metrics_lower},
        "train_metrics": {"higher": train_metrics_higher, "lower": train_metrics_lower}
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
    if task_type not in ["classification", "prob_classification"] or "curve_data" not in model_results:
        return  # 只針對有曲線數據的分類模型
    
    curve_data = model_results["curve_data"]
    
    # 確保目錄存在
    if save_figure:
        ensure_dir_exists(figure_dir)
    
    # 創建子圖
    n_plots = 3 if "pr" in curve_data else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 6, 5))
    
    # 繪製ROC曲線
    ax_idx = 0
    if "roc" in curve_data:
        fpr = curve_data["roc"]["fpr"]
        tpr = curve_data["roc"]["tpr"]
        auc_value = model_results.get("train_auc_roc", model_results.get("train_auc", 0))
        
        axes[ax_idx].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_value:.3f})')
        axes[ax_idx].plot([0, 1], [0, 1], 'k--')
        axes[ax_idx].set_xlim([0.0, 1.0])
        axes[ax_idx].set_ylim([0.0, 1.05])
        axes[ax_idx].set_xlabel('False Positive Rate')
        axes[ax_idx].set_ylabel('True Positive Rate')
        axes[ax_idx].set_title(f'{name} - ROC Curve')
        axes[ax_idx].legend(loc="lower right")
        ax_idx += 1
    
    # 繪製PR曲線（如果有）
    if "pr" in curve_data:
        precision = curve_data["pr"]["precision"]
        recall = curve_data["pr"]["recall"]
        auc_pr_value = model_results.get("train_auc_pr", 0)
        
        axes[ax_idx].plot(recall, precision, label=f'PR curve (AUC = {auc_pr_value:.3f})')
        axes[ax_idx].set_xlim([0.0, 1.0])
        axes[ax_idx].set_ylim([0.0, 1.05])
        axes[ax_idx].set_xlabel('Recall')
        axes[ax_idx].set_ylabel('Precision')
        axes[ax_idx].set_title(f'{name} - Precision-Recall Curve')
        axes[ax_idx].legend(loc="lower left")
        ax_idx += 1
    
    # 繪製校準曲線
    if "calibration" in curve_data:
        prob_pred = curve_data["calibration"]["prob_pred"]
        prob_true = curve_data["calibration"]["prob_true"]
        ece_value = model_results.get("train_ece", 0)
        
        axes[ax_idx].plot(prob_pred, prob_true, marker='o', label='Calibration curve')
        axes[ax_idx].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        axes[ax_idx].set_xlim([0.0, 1.0])
        axes[ax_idx].set_ylim([0.0, 1.0])
        axes[ax_idx].set_xlabel('Mean predicted probability')
        axes[ax_idx].set_ylabel('Fraction of positives')
        axes[ax_idx].set_title(f'{name} - Calibration Curve (ECE = {ece_value:.3f})')
        axes[ax_idx].legend(loc="lower right")
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(f"{figure_dir}/{name}_curves.png", dpi=300, bbox_inches='tight')
    
    plt.show()


# 繪製單個回歸模型的散點圖
def plot_regression_result(name, model_results, save_figure=False, figure_dir="picture"):
    """
    繪製回歸模型的實際值與預測值對比圖
    
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
    if "regression_data" not in model_results:
        return  # 只針對有回歸數據的模型
    
    regression_data = model_results["regression_data"]
    
    # 確保目錄存在
    if save_figure:
        ensure_dir_exists(figure_dir)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(regression_data["y_true"], regression_data["y_pred"], alpha=0.5)
    
    # 計算最小和最大值用於畫對角線
    y_true = regression_data["y_true"]
    y_pred = regression_data["y_pred"]
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('實際值')
    plt.ylabel('預測值')
    plt.title(f'{name} - 實際值 vs 預測值')
    
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
            
            ax.set_title(f"{group_name} 指標 (越低越好)")
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
        
    # 清理 DataFrame，移除非數值列
    cleaned_df = results_df.copy()
    for col in cleaned_df.columns:
        if cleaned_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            cleaned_df = cleaned_df.drop(columns=[col])
    
    # 獲取指標分類
    metric_groups = get_metric_groups(cleaned_df, task_type, tuned=is_tuned)
    
    # 繪製指標條形圖
    if not is_tuned and metric_groups["cv_metrics"]:
        fig = plot_metric_groups(cleaned_df, metric_groups["cv_metrics"], "交叉驗證結果比較", is_train=False, return_fig=True)
        if save_figures and fig:
            fig_path = os.path.join(figure_dir, f"{prefix}cv_metrics_{timestamp}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {fig_path}")
        plt.show()
    
    if metric_groups["train_metrics"]:
        title = "調優後的模型比較" if is_tuned else "訓練集結果比較"
        fig = plot_metric_groups(cleaned_df, metric_groups["train_metrics"], title, is_train=True, return_fig=True)
        if save_figures and fig:
            fig_path = os.path.join(figure_dir, f"{prefix}train_metrics_{timestamp}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {fig_path}")
        plt.show()
    
    # 繪製熱圖比較
    if metric_groups["train_metrics"]:
        figs = plot_heatmap(cleaned_df, metric_groups["train_metrics"], show_avg_performance=show_avg_performance, return_fig=True)
        
        if save_figures and figs:
            heatmap_fig, avg_fig = figs
            
            # 儲存熱力圖
            if heatmap_fig:
                fig_path = os.path.join(figure_dir, f"{prefix}heatmap_{timestamp}.png")
                heatmap_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"熱力圖已保存至: {fig_path}")
            
            # 儲存性能排名圖
            if avg_fig:
                fig_path = os.path.join(figure_dir, f"{prefix}performance_ranking_{timestamp}.png")
                avg_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"性能排名圖已保存至: {fig_path}")
        
        # 顯示圖表
        if isinstance(figs, tuple) and figs[0]:
            plt.figure(figs[0].number)
            plt.show()
        if isinstance(figs, tuple) and figs[1]:
            plt.figure(figs[1].number)
            plt.show()
    
    # 繪製每個模型的詳細曲線圖
    for name, row in cleaned_df.iterrows():
        # 跳過無法轉換為字典的行
        if not isinstance(row, (dict, pd.Series)):
            continue
            
        model_results = row.to_dict() if isinstance(row, pd.Series) else row
        
        # 繪製分類或概率分類的曲線
        if task_type in ["classification", "prob_classification"] and "curve_data" in model_results:
            plot_model_curves(name, model_results, task_type, save_figures, figure_dir)
        
        # 繪製回歸模型的散點圖
        elif task_type == "regression" and "regression_data" in model_results:
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