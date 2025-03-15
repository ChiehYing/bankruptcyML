from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve, auc)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 驗證模型並繪製混淆矩陣 (主要調用函數)
def classification_validation(model, X_train, y_train, X_valid=None, y_valid=None, average="macro"):
    """
    average: Averaging method for multi-class metrics ("macro", "micro", "weighted")
      - "macro": Calculate metrics for each class and take the average (ignores class imbalance)
      - "micro": Calculate metrics by aggregating TP, FP etc. across all classes (handles class imbalance)
      - "weighted": Calculate metrics for each class and take weighted average based on class frequency
    """
    y_pred_train, y_pred_valid = model_validation(model, X_train, y_train, X_valid, y_valid, average)
    
    print("\nTraining Set Classification Report:")
    print(classification_report(y_train, y_pred_train))
    
    if y_valid is not None and y_pred_valid is not None:
        print("\nValidation Set Classification Report:")
        print(classification_report(y_valid, y_pred_valid))
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training set confusion matrix
        cm_train = confusion_matrix(y_train, y_pred_train)
        sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axs[0])
        axs[0].set_title("Training Set Confusion Matrix")
        axs[0].set_ylabel("Actual Class")
        axs[0].set_xlabel("Predicted Class")
        
        # Validation set confusion matrix
        cm_valid = confusion_matrix(y_valid, y_pred_valid)
        sns.heatmap(cm_valid, annot=True, fmt="d", cmap="Reds", ax=axs[1])
        axs[1].set_title("Validation Set Confusion Matrix")
        axs[1].set_ylabel("Actual Class")
        axs[1].set_xlabel("Predicted Class")
    
    else:
        # Only training set confusion matrix
        confusion_matrix_plot(y_train, y_pred_train, title="Training Set Confusion Matrix")
    
    plt.tight_layout()
    plt.show()

# 驗證模型
def model_validation(model, X_train, y_train, X_valid=None, y_valid=None, average="macro"):

    # 訓練集評估
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average=average)
    recall_train = recall_score(y_train, y_pred_train, average=average)
    f1_train = f1_score(y_train, y_pred_train, average=average)
    
    print("Training Accuracy:", round(accuracy_train, 4))
    print("Training Precision:", round(precision_train, 4))
    print("Training Recall:", round(recall_train, 4))
    print("Training F1 Score:", round(f1_train, 4))
    
    # 驗證集評估 (如果有)
    if X_valid is not None and y_valid is not None:
        y_pred_valid = model.predict(X_valid)
        accuracy_valid = accuracy_score(y_valid, y_pred_valid)
        precision_valid = precision_score(y_valid, y_pred_valid, average=average)
        recall_valid = recall_score(y_valid, y_pred_valid, average=average)
        f1_valid = f1_score(y_valid, y_pred_valid, average=average)
        
        print("\nValidation Accuracy:", round(accuracy_valid, 4))
        print("Validation Precision:", round(precision_valid, 4))
        print("Validation Recall:", round(recall_valid, 4))
        print("Validation F1 Score:", round(f1_valid, 4))
    else:
        y_pred_valid = None
        
    return y_pred_train, y_pred_valid

# 繪製混淆矩陣
def confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix", cmap="Blues"):
    """
    Plot confusion matrix as a heatmap
    
    Parameters:
    y_true: Actual labels
    y_pred: Predicted labels
    title: Chart title
    cmap: Color theme
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.show()

# 增加一個ROC曲線繪製函數（適用於二元分類）
def plot_roc_curve(model, X_train, y_train, X_valid=None, y_valid=None, display_auc=True):
    
    # 單獨計算和顯示AUC值
    if display_auc:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_prob)
        print(f"Training AUC: {auc_train:.4f}")
        
        if X_valid is not None and y_valid is not None:
            y_valid_prob = model.predict_proba(X_valid)[:, 1]
            auc_valid = roc_auc_score(y_valid, y_valid_prob)
            print(f"Validation AUC: {auc_valid:.4f}")
    
    # 確保模型有predict_proba方法
    if not hasattr(model, "predict_proba"):
        print("Error: This model does not have a predict_proba method, cannot plot ROC curve")
        return
    
    # 獲取預測機率
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    # 計算ROC曲線
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    plt.figure(figsize=(10, 8))
    
    # 繪製訓練集ROC曲線
    plt.plot(fpr_train, tpr_train, color="blue", lw=2, 
             label=f"Training ROC (AUC = {roc_auc_train:.3f})")
    
    # 如果有驗證集
    if X_valid is not None and y_valid is not None:
        y_valid_prob = model.predict_proba(X_valid)[:, 1]
        fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_valid_prob)
        roc_auc_valid = auc(fpr_valid, tpr_valid)
        
        # 繪製驗證集ROC曲線
        plt.plot(fpr_valid, tpr_valid, color="red", lw=2, 
                 label=f"Validation ROC (AUC = {roc_auc_valid:.3f})")
    
    # 繪製對角線
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    
    # 設定圖表
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()