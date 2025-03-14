from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

    # 驗證模型
def model_validation(model, X_train, y_train, X_valid=None, y_valid=None, average="macro"):
    """
    average參數說明:
    - "macro": 計算每個類別的指標，然後取平均值（不考慮類別不平衡）
    - "micro": 將所有類別的真陽性、假陽性等加總後計算指標（適合類別不平衡）
    - "weighted": 計算每個類別的指標，然後根據該類別的樣本數量進行加權平均
    - "binary": 僅適用於二元分類（只計算陽性類別）
    - None: 多類別預設為None，會傳回每個類別的指標陣列
    """
    """
    驗證分類模型並返回預測結果
    
    參數:
    model: 分類模型
    X_train: 訓練資料特徵
    y_train: 訓練資料標籤
    X_valid: 驗證資料特徵
    y_valid: 驗證資料標籤
    average: 多分類評估指標的平均方式 ("macro", "micro", "weighted")
    
    返回:
    y_pred_train: 訓練資料預測結果
    y_pred_valid: 驗證資料預測結果
    """
    if X_valid is not None and y_valid is not None:
        y_pred_train = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train, average=average)
        recall_train = recall_score(y_train, y_pred_train, average=average)
        f1_train = f1_score(y_train, y_pred_train, average=average)
        
        print("訓練組準確率 (Accuracy):", round(accuracy_train, 4))
        print("訓練組精確率 (Precision):", round(precision_train, 4))
        print("訓練組召回率 (Recall):", round(recall_train, 4))
        print("訓練組 F1 分數:", round(f1_train, 4))
        
        y_pred_valid = model.predict(X_valid)
        accuracy_valid = accuracy_score(y_valid, y_pred_valid)
        precision_valid = precision_score(y_valid, y_pred_valid, average=average)
        recall_valid = recall_score(y_valid, y_pred_valid, average=average)
        f1_valid = f1_score(y_valid, y_pred_valid, average=average)
        
        print("驗證組準確率 (Accuracy):", round(accuracy_valid, 4))
        print("驗證組精確率 (Precision):", round(precision_valid, 4))
        print("驗證組召回率 (Recall):", round(recall_valid, 4))
        print("驗證組 F1 分數:", round(f1_valid, 4))
        
        return y_pred_train, y_pred_valid
        
    else:
        y_pred_train = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train, average=average)
        recall_train = recall_score(y_train, y_pred_train, average=average)
        f1_train = f1_score(y_train, y_pred_train, average=average)
        
        print("訓練組準確率 (Accuracy):", round(accuracy_train, 4))
        print("訓練組精確率 (Precision):", round(precision_train, 4))
        print("訓練組召回率 (Recall):", round(recall_train, 4)) 
        print("訓練組 F1 分數:", round(f1_train, 4))

        y_pred_valid = None
        
        return y_pred_train, y_pred_valid


# 繪製混淆矩陣
def confusion_matrix_plot(y_true, y_pred, title="混淆矩陣", cmap="Blues"):
    """
    繪製混淆矩陣的熱力圖
    
    參數:
    y_true: 實際標籤
    y_pred: 預測標籤
    title: 圖表標題
    cmap: 顏色主題
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.ylabel("實際類別")
    plt.xlabel("預測類別")
    plt.tight_layout()
    plt.show()


# 驗證模型並繪製混淆矩陣
def classification_validation(model, X_train, y_train, X_valid=None, y_valid=None, average="macro"):
    """
    驗證分類模型並繪製混淆矩陣
    
    參數:
    model: 分類模型
    X_train: 訓練資料特徵
    y_train: 訓練資料標籤
    X_valid: 驗證資料特徵
    y_valid: 驗證資料標籤
    average: 多分類評估指標的平均方式 ("macro", "micro", "weighted")
      - "macro": 計算每個類別的指標，然後取平均值（不考慮類別不平衡）
      - "micro": 將所有類別的真陽性、假陽性等加總後計算指標（適合類別不平衡）
      - "weighted": 計算每個類別的指標，然後根據該類別的樣本數量進行加權平均
    """
    y_pred_train, y_pred_valid = model_validation(model, X_train, y_train, X_valid, y_valid, average)
    
    print("\n訓練組分類報告:")
    print(classification_report(y_train, y_pred_train))
    
    if y_valid is not None and y_pred_valid is not None:
        print("\n驗證組分類報告:")
        print(classification_report(y_valid, y_pred_valid))
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # 訓練組混淆矩陣
        cm_train = confusion_matrix(y_train, y_pred_train)
        sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axs[0])
        axs[0].set_title("訓練組混淆矩陣")
        axs[0].set_ylabel("實際類別")
        axs[0].set_xlabel("預測類別")
        
        # 驗證組混淆矩陣
        cm_valid = confusion_matrix(y_valid, y_pred_valid)
        sns.heatmap(cm_valid, annot=True, fmt="d", cmap="Reds", ax=axs[1])
        axs[1].set_title("驗證組混淆矩陣")
        axs[1].set_ylabel("實際類別")
        axs[1].set_xlabel("預測類別")
    
    else:
        # 只有訓練組的混淆矩陣
        confusion_matrix_plot(y_train, y_pred_train, title="訓練組混淆矩陣")
    
    plt.tight_layout()
    plt.show()


# 增加一個ROC曲線繪製函數（適用於二元分類）
def plot_roc_curve(model, X_train, y_train, X_valid=None, y_valid=None, display_auc=True):
    """
    繪製ROC曲線（僅適用於二元分類）並計算AUC值
    
    參數:
    model: 分類模型，需要有predict_proba方法
    X_train: 訓練資料特徵
    y_train: 訓練資料標籤
    X_valid: 驗證資料特徵
    y_valid: 驗證資料標籤
    display_auc: 是否在圖表中顯示AUC值
    
    注意: AUC (Area Under Curve) 是ROC曲線下的面積，數值介於0~1之間
          - AUC=1: 完美分類
          - AUC>0.9: 優秀
          - AUC>0.8: 良好
          - AUC>0.7: 尚可
          - AUC>0.6: 勉強
          - AUC=0.5: 與隨機猜測相同
    """
    from sklearn.metrics import roc_curve, auc
    
    # 單獨計算和顯示AUC值
    if display_auc:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_prob)
        print(f"訓練組 AUC: {auc_train:.4f}")
        
        if X_valid is not None and y_valid is not None:
            y_valid_prob = model.predict_proba(X_valid)[:, 1]
            auc_valid = roc_auc_score(y_valid, y_valid_prob)
            print(f"驗證組 AUC: {auc_valid:.4f}")
    
    # 確保模型有predict_proba方法
    if not hasattr(model, "predict_proba"):
        print("錯誤：此模型沒有predict_proba方法，無法繪製ROC曲線")
        return
    
    # 獲取預測機率
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    # 計算ROC曲線
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    plt.figure(figsize=(10, 8))
    
    # 繪製訓練集ROC曲線
    plt.plot(fpr_train, tpr_train, color="blue", lw=2, 
             label=f"訓練組 ROC曲線 (AUC = {roc_auc_train:.3f})")
    
    # 如果有驗證集
    if X_valid is not None and y_valid is not None:
        y_valid_prob = model.predict_proba(X_valid)[:, 1]
        fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_valid_prob)
        roc_auc_valid = auc(fpr_valid, tpr_valid)
        
        # 繪製驗證集ROC曲線
        plt.plot(fpr_valid, tpr_valid, color="red", lw=2, 
                 label=f"驗證組 ROC曲線 (AUC = {roc_auc_valid:.3f})")
    
    # 繪製對角線
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    
    # 設定圖表
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("假陽性率 (False Positive Rate)")
    plt.ylabel("真陽性率 (True Positive Rate)")
    plt.title("接收者操作特徵 (ROC) 曲線")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()