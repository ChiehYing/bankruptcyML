import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from statsmodels.stats.multitest import multipletests


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


# 增加F檢定，沒有圖例
def feature_select_enhanced(X, y=None, variance_threshold=0.2, correlation_threshold=0.7, 
                            p_value_threshold=0.05, adjust_p_values=True, 
                            adjustment_method='fdr_bh', target_correlation=True,
                            max_features=None, verbose=True):
    """
    綜合特徵選擇函數，結合方差閾值、相關性篩選和統計顯著性測試。
    
    Parameters:
    -----------
    X : pandas DataFrame
        特徵矩陣
    y : pandas Series or numpy array, optional (default=None)
        目標變數，用於統計顯著性測試和目標相關性篩選
    variance_threshold : float, optional (default=0.2)
        方差閾值，低於此值的特徵將被移除
    correlation_threshold : float, optional (default=0.7)
        相關性閾值，高於此值的特徵對將被視為高度相關
    p_value_threshold : float, optional (default=0.05)
        p值閾值，高於此值的特徵將被視為統計不顯著
    adjust_p_values : bool, optional (default=True)
        是否對p值進行多重比較調整
    adjustment_method : str, optional (default='fdr_bh')
        p值調整方法，可選：'bonferroni', 'fdr_bh'(Benjamini-Hochberg)
    target_correlation : bool, optional (default=True)
        在相關性篩選中，是否基於與目標變數的相關性決定保留哪個特徵
    max_features : int or None, optional (default=None)
        最終保留的最大特徵數量，如果為None則保留所有通過篩選的特徵
    verbose : bool, optional (default=True)
        是否顯示詳細信息
        
    Returns:
    --------
    X_selected : pandas DataFrame
        篩選後的特徵矩陣
    feature_info : pandas DataFrame
        包含所有特徵及其篩選信息的DataFrame
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X必須是pandas DataFrame")
    
    original_features = X.columns.tolist()
    n_original = len(original_features)
    feature_info = pd.DataFrame({'Feature': original_features})
    
    if verbose:
        print(f"原始特徵數量: {n_original}")
    
    # 步驟1: 方差閾值篩選
    if variance_threshold > 0:
        # 標準化數據計算方差
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 計算每個特徵的方差
        variances = X_scaled_df.var().to_dict()
        feature_info['Variance'] = feature_info['Feature'].map(variances)
        
        # 應用方差閾值
        var_selector = VarianceThreshold(threshold=variance_threshold)
        var_selector.fit(X_scaled_df)
        var_selected_features = X.columns[var_selector.get_support()].tolist()
        
        # 更新特徵信息
        feature_info['Pass_Variance'] = feature_info['Feature'].isin(var_selected_features)
        
        if verbose:
            n_var_selected = len(var_selected_features)
            print(f"方差閾值篩選後保留的特徵數量: {n_var_selected} ({n_var_selected/n_original:.1%})")
            
            # 顯示被移除的低方差特徵
            removed_features = [f for f in original_features if f not in var_selected_features]
            if removed_features:
                print("移除的低方差特徵:")
                for feature in removed_features:
                    print(f"- {feature} (標準化後方差: {variances[feature]:.6f})")
        
        # 更新X為方差篩選後的數據
        X = X[var_selected_features]
    else:
        feature_info['Variance'] = X.var().values
        feature_info['Pass_Variance'] = True
        var_selected_features = original_features
    
    # 步驟2: 相關性篩選
    if correlation_threshold < 1.0 and len(X.columns) > 1:
        # 計算相關矩陣
        corr_matrix = X.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        
        # 獲取上三角矩陣
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找出高度相關的特徵對
        high_corr_pairs = []
        for col in upper_tri.columns:
            high_corr_cols = upper_tri.index[upper_tri[col] > correlation_threshold].tolist()
            for high_corr_col in high_corr_cols:
                high_corr_pairs.append((col, high_corr_col, corr_matrix.loc[col, high_corr_col]))
        
        if verbose and high_corr_pairs:
            print("\n高相關性特徵對 (相關係數 > {})：".format(correlation_threshold))
            for pair in high_corr_pairs:
                print(f"- {pair[0]} 和 {pair[1]}, 相關係數: {pair[2]:.4f}")
        
        # 決定要移除的特徵
        to_drop = set()
        
        if high_corr_pairs:
            if target_correlation and y is not None:
                # 計算每個特徵與目標變數的相關性
                if verbose:
                    print("\n基於與目標變量的相關性選擇特徵:")
                    
                target_corr = {}
                for col in X.columns:
                    if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                        target_corr[col] = abs(np.corrcoef(X[col], y)[0, 1])
                    else:
                        target_corr[col] = 0
                
                # 添加到特徵信息
                feature_info['Target_Correlation'] = feature_info['Feature'].map(
                    lambda x: target_corr.get(x, 0) if x in var_selected_features else 0
                )
                
                # 基於目標相關性決定要移除的特徵
                for feat1, feat2, _ in high_corr_pairs:
                    if feat1 in to_drop or feat2 in to_drop:
                        continue
                        
                    if target_corr[feat1] < target_corr[feat2]:
                        to_drop.add(feat1)
                        if verbose:
                            print(f"  保留 {feat2} (與目標相關性: {target_corr[feat2]:.4f}), 移除 {feat1} (與目標相關性: {target_corr[feat1]:.4f})")
                    else:
                        to_drop.add(feat2)
                        if verbose:
                            print(f"  保留 {feat1} (與目標相關性: {target_corr[feat1]:.4f}), 移除 {feat2} (與目標相關性: {target_corr[feat2]:.4f})")
            else:
                # 基於特徵方差決定要移除的特徵
                if verbose:
                    print("\n無目標變量或未啟用目標相關性篩選，根據特徵方差選擇保留方差較大的特徵:")
                    
                feature_variances = X.var()
                
                for feat1, feat2, _ in high_corr_pairs:
                    if feat1 in to_drop or feat2 in to_drop:
                        continue
                        
                    if feature_variances[feat1] < feature_variances[feat2]:
                        to_drop.add(feat1)
                        if verbose:
                            print(f"  保留 {feat2} (方差: {feature_variances[feat2]:.4f}), 移除 {feat1} (方差: {feature_variances[feat1]:.4f})")
                    else:
                        to_drop.add(feat2)
                        if verbose:
                            print(f"  保留 {feat1} (方差: {feature_variances[feat1]:.4f}), 移除 {feat2} (方差: {feature_variances[feat2]:.4f})")
            
            # 更新特徵信息
            feature_info['Pass_Correlation'] = feature_info['Feature'].apply(
                lambda x: x not in to_drop if x in var_selected_features else False
            )
            
            # 移除高度相關的特徵
            corr_selected_features = [f for f in X.columns if f not in to_drop]
            
            if verbose:
                n_corr_selected = len(corr_selected_features)
                print(f"\n相關性篩選後保留的特徵數量: {n_corr_selected} ({n_corr_selected/n_original:.1%})")
                print(f"移除的特徵: {list(to_drop)}")
            
            # 更新X為相關性篩選後的數據
            X = X[corr_selected_features]
        else:
            feature_info['Pass_Correlation'] = feature_info['Pass_Variance']
            corr_selected_features = var_selected_features
            if verbose:
                print("\n沒有發現高相關特徵，保持當前特徵集。")
    else:
        feature_info['Pass_Correlation'] = feature_info['Pass_Variance']
        feature_info['Target_Correlation'] = 0
        if y is not None and target_correlation:
            for col in X.columns:
                if X[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    feature_info.loc[feature_info['Feature'] == col, 'Target_Correlation'] = \
                        abs(np.corrcoef(X[col], y)[0, 1])
        corr_selected_features = X.columns.tolist()
    
    # 步驟3: 統計顯著性測試（t檢定）
    if y is not None and p_value_threshold < 1.0:
        if verbose:
            print("\n\n=== 進行統計顯著性測試 ===\n")
            
        # 應用t檢定
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        # 獲取特徵分數和p值
        scores = selector.scores_
        p_values = selector.pvalues_
        
        # 調整p值（如果需要）
        if adjust_p_values:
            adjusted_p_values = multipletests(p_values, method=adjustment_method)[1]
        else:
            adjusted_p_values = p_values
        
        # 建立特徵與分數、p值的映射
        score_dict = {X.columns[i]: scores[i] for i in range(len(X.columns))}
        p_dict = {X.columns[i]: p_values[i] for i in range(len(X.columns))}
        adj_p_dict = {X.columns[i]: adjusted_p_values[i] for i in range(len(X.columns))}
        
        # 更新特徵信息
        feature_info['F_Score'] = feature_info['Feature'].map(
            lambda x: score_dict.get(x, 0) if x in corr_selected_features else 0
        )
        feature_info['P_Value'] = feature_info['Feature'].map(
            lambda x: p_dict.get(x, 1) if x in corr_selected_features else 1
        )
        feature_info['Adjusted_P_Value'] = feature_info['Feature'].map(
            lambda x: adj_p_dict.get(x, 1) if x in corr_selected_features else 1
        )
        
        # 根據調整後的p值確定統計顯著性
        feature_info['Pass_Significance'] = feature_info['Feature'].apply(
            lambda x: adj_p_dict.get(x, 1) < p_value_threshold if x in corr_selected_features else False
        )
        
        # 保留統計顯著的特徵
        stat_selected_features = [f for f in X.columns if adj_p_dict[f] < p_value_threshold]
        
        if verbose:
            not_significant = [f for f in X.columns if adj_p_dict[f] >= p_value_threshold]
            n_stat_selected = len(stat_selected_features)
            print(f"統計顯著性測試信息:")
            for feat in X.columns:
                print(f"- {feat}: F分數={score_dict[feat]:.4f}, 原始p值={p_dict[feat]:.6f}, 調整後p值={adj_p_dict[feat]:.6f}")
            
            print(f"\n統計顯著性篩選後保留的特徵數量: {n_stat_selected} ({n_stat_selected/n_original:.1%})")
            if not_significant:
                print(f"移除的非顯著特徵: {not_significant}")
        
        # 更新X為統計顯著性篩選後的數據
        X = X[stat_selected_features]
    else:
        feature_info['F_Score'] = 0
        feature_info['P_Value'] = 1
        feature_info['Adjusted_P_Value'] = 1
        feature_info['Pass_Significance'] = feature_info['Pass_Correlation']
        stat_selected_features = X.columns.tolist()
    
    # 最終選定特徵
    selected_features = list(X.columns)
    
    # 如果指定了最大特徵數量，按F分數排序選擇前N個特徵
    if max_features is not None and len(selected_features) > max_features and y is not None:
        feature_scores = feature_info[feature_info['Feature'].isin(selected_features)].sort_values('F_Score', ascending=False)
        selected_features = feature_scores.head(max_features)['Feature'].tolist()
        
        if verbose:
            print(f"\n限制最大特徵數量後保留的特徵數量: {len(selected_features)} ({len(selected_features)/n_original:.1%})")
            print(f"最終選定的特徵: {selected_features}")
        
        # 更新X
        X = X[selected_features]
    
    # 更新特徵信息，標記最終選定的特徵
    feature_info['Selected'] = feature_info['Feature'].isin(selected_features)
    
    # 按照篩選通過情況排序特徵信息
    feature_info = feature_info.sort_values(
        by=['Selected', 'Pass_Significance', 'Pass_Correlation', 'Pass_Variance', 'F_Score'], 
        ascending=[False, False, False, False, False]
    )
    
    # 結果摘要
    if verbose:
        print("\n\n=== 特徵篩選最終結果 ===\n")
        print(f"原始特徵集 ({n_original} 個特徵): {original_features}")
        print(f"最終特徵集 ({len(selected_features)} 個特徵): {selected_features}")
        print(f"被移除的特徵 ({n_original - len(selected_features)} 個特徵): {[f for f in original_features if f not in selected_features]}")
    
    return X, feature_info