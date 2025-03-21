from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 驗證模型並繪製圖表 (主要調用執行的函式)
def regression_validation(model, X_train, y_train, X_valid=None, y_valid=None, y_train_pred=None, y_valid_pred=None, plot_residuals=True):
    # 計算指標並獲取預測值
    y_pred_train, y_pred_valid, train_metrics, valid_metrics = model_validation(model, X_train, y_train, X_valid, y_valid, y_train_pred, y_valid_pred)
    
    # 繪製散布圖
    scatter_plot(y_train, y_pred_train, y_valid, y_pred_valid)
    
    # 繪製殘差圖
    if plot_residuals:
        residual_plot(y_train, y_pred_train, y_valid, y_pred_valid)
    
    # 返回指標的字典，以便進一步使用
    if valid_metrics is not None:
        return {"train": train_metrics, "valid": valid_metrics}
    else:
        return {"train": train_metrics}

# 驗證模型
def model_validation(model, X_train, y_train, X_valid=None, y_valid=None ,y_train_pred=None, y_valid_pred=None):
    # 訓練集指標
    if y_train_pred is not None:
        y_pred_train = y_train_pred
        train_metrics = calculate_regression_metrics(y_train, y_pred_train)
        
    else:
        y_pred_train = model.predict(X_train)
        train_metrics = calculate_regression_metrics(y_train, y_pred_train)
    
    print("訓練集指標:")
    print_metrics(train_metrics)
    
    if y_valid_pred is not None:
        y_pred_valid = y_valid_pred
        valid_metrics = calculate_regression_metrics(y_valid, y_pred_valid)
        
        print("\n驗證集指標:")
        print_metrics(valid_metrics)

    else:
        if X_valid is not None and y_valid is not None:
            # 驗證集指標
            y_pred_valid = model.predict(X_valid)
            valid_metrics = calculate_regression_metrics(y_valid, y_pred_valid)
            
            print("\n驗證集指標:")
            print_metrics(valid_metrics)
        else:
            y_pred_valid = None
            valid_metrics = None
        
    return y_pred_train, y_pred_valid, train_metrics, valid_metrics

# 計算回歸指標
def calculate_regression_metrics(y_true, y_pred):
    metrics = {}
    
    # R平方 (決定係數)
    metrics["R2"] = r2_score(y_true, y_pred)
    
    # 平均絕對誤差 (MAE)
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    
    # 均方誤差 (MSE)
    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    
    # 均方根誤差 (RMSE)
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    
    # 解釋方差分數
    metrics["Explained_Variance"] = explained_variance_score(y_true, y_pred)
    
    # 平均絕對百分比誤差 (MAPE)
    # 避免除以零，加入小數值
    metrics["MAPE"] = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    
    # 最大誤差
    metrics["Max_Error"] = np.max(np.abs(y_true - y_pred))
    
    return metrics

# 打印指標
def print_metrics(metrics):
    print(f"R2 (決定係數): {metrics['R2']:.4f}")
    print(f"MAE (平均絕對誤差): {metrics['MAE']:.4f}")
    print(f"MSE (均方誤差): {metrics['MSE']:.4f}")
    print(f"RMSE (均方根誤差): {metrics['RMSE']:.4f}")
    print(f"Explained Variance (解釋方差分數): {metrics['Explained_Variance']:.4f}")
    print(f"MAPE (平均絕對百分比誤差): {metrics['MAPE']:.2f}%")
    print(f"Max Error (最大誤差): {metrics['Max_Error']:.4f}")

# 繪製散布圖
def scatter_plot(y1_true, y1_pred, y2_true=None, y2_pred=None):
    if y2_true is not None and y2_pred is not None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # 訓練集散布圖
        axs[0].scatter(y1_true, y1_pred, alpha=0.6, color="b")
        axs[0].set_xlabel("Actual Values")
        axs[0].set_ylabel("Predicted Values")
        axs[0].set_title("Training Set: Predicted vs Actual")
        # 添加理想線
        min_val = min(min(y1_true), min(y1_pred))
        max_val = max(max(y1_true), max(y1_pred))
        axs[0].plot([min_val, max_val], [min_val, max_val], "k--")
        
        # 驗證集散布圖
        axs[1].scatter(y2_true, y2_pred, alpha=0.6, color="r")
        axs[1].set_xlabel("Actual Values")
        axs[1].set_ylabel("Predicted Values")
        axs[1].set_title("Validation Set: Predicted vs Actual")
        # 添加理想線
        min_val = min(min(y2_true), min(y2_pred))
        max_val = max(max(y2_true), max(y2_pred))
        axs[1].plot([min_val, max_val], [min_val, max_val], "k--")
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(y1_true, y1_pred, alpha=0.6, color="b")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Training Set: Predicted vs Actual")
        # 添加理想線
        min_val = min(min(y1_true), min(y1_pred))
        max_val = max(max(y1_true), max(y1_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "k--")
    
    plt.tight_layout()
    plt.show()

# 繪製殘差圖
def residual_plot(y1_true, y1_pred, y2_true=None, y2_pred=None):
    if y2_true is not None and y2_pred is not None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # 訓練集殘差圖
        residuals1 = y1_true - y1_pred
        axs[0].scatter(y1_pred, residuals1, alpha=0.6, color="b")
        axs[0].axhline(y=0, color="k", linestyle="--")
        axs[0].set_xlabel("Predicted Values")
        axs[0].set_ylabel("Residuals")
        axs[0].set_title("Training Set Residuals")
        
        # 驗證集殘差圖
        residuals2 = y2_true - y2_pred
        axs[1].scatter(y2_pred, residuals2, alpha=0.6, color="r")
        axs[1].axhline(y=0, color="k", linestyle="--")
        axs[1].set_xlabel("Predicted Values")
        axs[1].set_ylabel("Residuals")
        axs[1].set_title("Validation Set Residuals")
    else:
        plt.figure(figsize=(8, 6))
        residuals = y1_true - y1_pred
        plt.scatter(y1_pred, residuals, alpha=0.6, color="b")
        plt.axhline(y=0, color="k", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Training Set Residuals")
    
    plt.tight_layout()
    plt.show()