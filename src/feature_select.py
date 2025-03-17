import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


# 進行特徵篩選，包括低方差特徵移除和高相關性特徵移除
def feature_selection(X, y=None, variance_threshold=0.01, correlation_threshold=0.8, show_plots=True):
    results = {
        "original_shape": X.shape,
        "original_features": list(X.columns),
        "low_variance_features": [],
        "high_correlation_pairs": [],
        "dropped_features": [],
        "kept_features": []
    }
    
    # 1. 低方差特徵篩選
    print("\n=== 進行低方差特徵篩選 ===\n")
    # 使用方差閾值過濾器，特徵需要標準化後再計算方差
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    var_selector = VarianceThreshold(threshold=variance_threshold)
    var_selector.fit(X_scaled_df)
    low_variance_mask = var_selector.get_support()
    low_variance_features = X.columns[~low_variance_mask].tolist()
    
    print(f"使用閾值 {variance_threshold} 識別的低方差特徵:")
    for feature in low_variance_features:
        print(f"- {feature} (標準化後方差: {X_scaled_df[feature].var():.6f})")
    
    # 移除低方差特徵
    X_reduced = X.drop(columns=low_variance_features)
    print(f"\n移除低方差特徵後的資料維度: {X_reduced.shape}")
    print(f"移除的特徵: {low_variance_features}")
    
    results["low_variance_features"] = low_variance_features
    results["after_variance_filtering_shape"] = X_reduced.shape
    
    # 2. 高相關性特徵篩選
    print("\n\n=== 進行高相關性特徵篩選 ===\n")
    
    # 計算相關性矩陣
    correlation_matrix = X_reduced.corr().abs()
    
    # 只顯示相關性較高的部分
    if show_plots:
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="coolwarm", 
                    fmt=".2f", linewidths=0.5, vmin=0, vmax=1)
        plt.title("特徵相關性矩陣")
        plt.tight_layout()
        plt.show()
    
    # 獲取上三角矩陣 (避免重複對)
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # 找出高相關特徵對
    high_corr_pairs = []
    for col in upper_tri.columns:
        high_corr_cols = upper_tri.index[upper_tri[col] > correlation_threshold].tolist()
        for high_corr_col in high_corr_cols:
            high_corr_pairs.append((col, high_corr_col, correlation_matrix.loc[col, high_corr_col]))
    
    print("\n高相關性特徵對 (相關係數 > {})：".format(correlation_threshold))
    for pair in high_corr_pairs:
        print(f"- {pair[0]} 和 {pair[1]}, 相關係數: {pair[2]:.4f}")
    
    results["high_correlation_pairs"] = [(p[0], p[1], float(p[2])) for p in high_corr_pairs]
    
    # 若沒有提供目標變數，則根據特徵名稱或特徵方差選擇
    features_to_drop = set()
    
    if high_corr_pairs:
        if y is not None:
            print("\n基於與目標變量的相關性選擇特徵:")
            # 計算每個特徵與目標變量的相關性
            target_correlations = {}
            for feature in X_reduced.columns:
                corr = np.corrcoef(X_reduced[feature], y)[0, 1]
                target_correlations[feature] = abs(corr)  # 使用絕對值
            
            for feat1, feat2, _ in high_corr_pairs:
                # 保留與目標相關性較高的特徵
                if target_correlations[feat1] < target_correlations[feat2]:
                    features_to_drop.add(feat1)
                    print(f"  保留 {feat2} (與目標相關性: {target_correlations[feat2]:.4f}), 移除 {feat1} (與目標相關性: {target_correlations[feat1]:.4f})")
                else:
                    features_to_drop.add(feat2)
                    print(f"  保留 {feat1} (與目標相關性: {target_correlations[feat1]:.4f}), 移除 {feat2} (與目標相關性: {target_correlations[feat2]:.4f})")
        else:
            print("\n無目標變量，根據特徵方差選擇保留方差較大的特徵:")
            feature_variances = X_reduced.var()
            
            for feat1, feat2, _ in high_corr_pairs:
                # 保留方差較大的特徵
                if feature_variances[feat1] < feature_variances[feat2]:
                    features_to_drop.add(feat1)
                    print(f"  保留 {feat2} (方差: {feature_variances[feat2]:.4f}), 移除 {feat1} (方差: {feature_variances[feat1]:.4f})")
                else:
                    features_to_drop.add(feat2)
                    print(f"  保留 {feat1} (方差: {feature_variances[feat1]:.4f}), 移除 {feat2} (方差: {feature_variances[feat2]:.4f})")
        
        # 移除選定的高相關特徵
        X_final = X_reduced.drop(columns=list(features_to_drop))
        print(f"\n移除高相關特徵後的資料維度: {X_final.shape}")
        print(f"移除的特徵: {list(features_to_drop)}")
    else:
        X_final = X_reduced
        print("\n沒有發現高相關特徵，保持當前特徵集。")
    
    # 3. 結果摘要
    all_dropped = low_variance_features + list(features_to_drop)
    print("\n\n=== 特徵篩選最終結果 ===\n")
    print(f"原始特徵集 ({X.shape[1]} 個特徵): {list(X.columns)}")
    print(f"最終特徵集 ({X_final.shape[1]} 個特徵): {list(X_final.columns)}")
    print(f"被移除的特徵 ({len(all_dropped)} 個特徵): {all_dropped}")
    
    # 視覺化最終相關性矩陣
    if show_plots and X_final.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(X_final.corr().abs(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("篩選後特徵集的相關性矩陣")
        plt.tight_layout()
        plt.show()
    
    # 更新結果
    results["final_shape"] = X_final.shape
    results["kept_features"] = list(X_final.columns)
    results["dropped_features"] = all_dropped
    
    return X_final, results