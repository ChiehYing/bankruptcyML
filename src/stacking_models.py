import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score


class StackingModel:
    """
    通用 Stacking 模型實現，適用於分類與回歸任務
    """
    
    def __init__(self, base_models=None, meta_model=None, 
                 task_type='regression', cv=5, use_proba=False, 
                 threshold=0.5, random_state=42):
        """
        初始化 Stacking 模型
        
        參數:
            base_models: 列表，基礎模型列表
            meta_model: 元學習器，默認為 LinearRegression (回歸) 或 LogisticRegression (分類)
            task_type: 字符串，'regression' 或 'classification' 或 'reg_to_class'
            cv: 整數，交叉驗證折數
            use_proba: 布爾值，是否使用 predict_proba（僅用於分類）
            threshold: 浮點數，二元分類閾值（僅用於回歸轉分類）
            random_state: 整數，隨機種子
        """
        self.base_models = base_models if base_models else []
        self.meta_model = meta_model
        self.task_type = task_type
        self.cv = cv
        self.use_proba = use_proba
        self.threshold = threshold
        self.random_state = random_state
        self.trained_base_models = []
        
        # 如果未提供元學習器，則根據任務類型設置默認值
        if self.meta_model is None:
            if task_type == 'classification':
                self.meta_model = LogisticRegression(random_state=random_state)
            else:  # 回歸或回歸轉分類
                self.meta_model = LinearRegression()
    
    def fit(self, X, y):
        """
        訓練 Stacking 模型
        
        參數:
            X: 特徵矩陣
            y: 目標變量
        
        返回:
            self: 訓練好的模型
        """
        # 確定使用的交叉驗證策略
        if self.task_type == 'classification' or self.task_type == 'prob_classification':
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            fold_splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            fold_splits = list(kf.split(X))
        
        # 為元學習器創建特徵
        n_models = len(self.base_models)
        
        if self.task_type == 'classification' and self.use_proba:
            # 對於多分類問題，獲取類別數量
            if len(np.unique(y)) > 2:  # 多分類
                n_classes = len(np.unique(y))
                meta_features = np.zeros((X.shape[0], n_models * n_classes))
            else:  # 二分類
                meta_features = np.zeros((X.shape[0], n_models * 2))
        else:
            meta_features = np.zeros((X.shape[0], n_models))
        
    # 使用交叉驗證生成元特徵
    def fit(self, X, y):
        """
        訓練 Stacking 模型
        
        參數:
            X: 特徵矩陣
            y: 目標變量
        
        返回:
            self: 訓練好的模型
        """
        # 將X和y轉換為numpy數組以確保兼容性
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # 確定使用的交叉驗證策略
        if self.task_type == 'classification' or self.task_type == 'prob_classification':
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            fold_splits = list(kf.split(X_array, y_array))
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            fold_splits = list(kf.split(X_array))
        
        # 為元學習器創建特徵
        n_models = len(self.base_models)
        
        if self.task_type == 'classification' and self.use_proba:
            # 對於多分類問題，獲取類別數量
            if len(np.unique(y_array)) > 2:  # 多分類
                n_classes = len(np.unique(y_array))
                meta_features = np.zeros((X_array.shape[0], n_models * n_classes))
            else:  # 二分類
                meta_features = np.zeros((X_array.shape[0], n_models * 2))
        else:
            meta_features = np.zeros((X_array.shape[0], n_models))
        
        # 使用交叉驗證生成元特徵
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in fold_splits:
                # 深度複製模型避免修改原始模型
                model_clone = pickle.loads(pickle.dumps(model))
                
                # 訓練模型 - 使用numpy數組進行索引
                model_clone.fit(X_array[train_idx], y_array[train_idx])
                
                # 生成預測
                if self.task_type == 'classification' and self.use_proba:
                    if len(np.unique(y_array)) > 2:  # 多分類
                        preds = model_clone.predict_proba(X_array[val_idx])
                        for j, proba in enumerate(preds.T):
                            meta_features[val_idx, i * n_classes + j] = proba
                    else:  # 二分類，只使用正類的概率
                        preds = model_clone.predict_proba(X_array[val_idx])[:, 1]
                        meta_features[val_idx, i * 2] = 1 - preds  # 負類概率
                        meta_features[val_idx, i * 2 + 1] = preds  # 正類概率
                else:
                    preds = model_clone.predict(X_array[val_idx])
                    meta_features[val_idx, i] = preds
        
        # 訓練元學習器
        self.meta_model.fit(meta_features, y_array)
        
        # 使用全部數據重新訓練基礎模型
        self.trained_base_models = []
        for model in self.base_models:
            model_clone = pickle.loads(pickle.dumps(model))
            model_clone.fit(X_array, y_array)
            self.trained_base_models.append(model_clone)
        
        return self
    
    def predict(self, X):
        """
        使用訓練好的 Stacking 模型進行預測
        
        參數:
            X: 特徵矩陣
        
        返回:
            預測結果
        """
        # 轉換X為numpy數組      
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # 確保模型已經訓練
        if not self.trained_base_models:
            raise ValueError("模型尚未訓練，請先調用 fit 方法")
        
        # 生成元特徵
        n_models = len(self.trained_base_models)
        n_samples = X.shape[0]
        
        if self.task_type == 'classification' and self.use_proba:
            # 對於分類問題，獲取類別數量
            if hasattr(self.meta_model, 'classes_') and len(self.meta_model.classes_) > 2:
                n_classes = len(self.meta_model.classes_)
                meta_features = np.zeros((n_samples, n_models * n_classes))
            else:  # 二分類
                meta_features = np.zeros((n_samples, n_models * 2))
        else:
            meta_features = np.zeros((n_samples, n_models))
        
        # 獲取基礎模型預測
        for i, model in enumerate(self.trained_base_models):
            if self.task_type == 'classification' and self.use_proba:
                if hasattr(self.meta_model, 'classes_') and len(self.meta_model.classes_) > 2:
                    preds = model.predict_proba(X)
                    for j, proba in enumerate(preds.T):
                        meta_features[:, i * n_classes + j] = proba
                else:  # 二分類
                    preds = model.predict_proba(X)[:, 1]
                    meta_features[:, i * 2] = 1 - preds  # 負類概率
                    meta_features[:, i * 2 + 1] = preds  # 正類概率
            else:
                preds = model.predict(X)
                meta_features[:, i] = preds
        
        # 使用元學習器進行預測
        predictions = self.meta_model.predict(meta_features)
        
        # 對於概率分類任務，應用閾值
        if self.task_type == 'prob_classification':
            predictions = (predictions >= self.threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        對於分類任務，預測概率
        
        參數:
            X: 特徵矩陣
        
        返回:
            預測概率
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba 僅適用於分類任務")
            
        # 確保模型已經訓練
        if not self.trained_base_models:
            raise ValueError("模型尚未訓練，請先調用 fit 方法")
        
        # 生成元特徵
        n_models = len(self.trained_base_models)
        n_samples = X.shape[0]
        
        if self.use_proba:
            # 獲取類別數量
            if hasattr(self.meta_model, 'classes_') and len(self.meta_model.classes_) > 2:
                n_classes = len(self.meta_model.classes_)
                meta_features = np.zeros((n_samples, n_models * n_classes))
            else:  # 二分類
                meta_features = np.zeros((n_samples, n_models * 2))
                
            # 獲取基礎模型預測
            for i, model in enumerate(self.trained_base_models):
                if hasattr(self.meta_model, 'classes_') and len(self.meta_model.classes_) > 2:
                    preds = model.predict_proba(X)
                    for j, proba in enumerate(preds.T):
                        meta_features[:, i * n_classes + j] = proba
                else:  # 二分類
                    preds = model.predict_proba(X)[:, 1]
                    meta_features[:, i * 2] = 1 - preds  # 負類概率
                    meta_features[:, i * 2 + 1] = preds  # 正類概率
        else:
            # 使用離散預測
            meta_features = np.zeros((n_samples, n_models))
            for i, model in enumerate(self.trained_base_models):
                meta_features[:, i] = model.predict(X)
        
        # 使用元學習器預測概率
        return self.meta_model.predict_proba(meta_features)
    
    def get_params(self):
        """
        獲取模型參數
        
        返回:
            模型參數字典
        """
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'task_type': self.task_type,
            'cv': self.cv,
            'use_proba': self.use_proba,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'trained_base_models': self.trained_base_models
        }
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        參數:
            filepath: 文件路徑
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        從文件載入模型
        
        參數:
            filepath: 文件路徑
            
        返回:
            載入的模型
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def optimize_threshold(self, X_val, y_val, metric=f1_score, 
                           threshold_range=np.arange(0.1, 0.9, 0.01)):
        """
        在驗證集上優化分類閾值（用於回歸轉分類任務）
        
        參數:
            X_val: 驗證集特徵
            y_val: 驗證集標籤
            metric: 評估指標函數
            threshold_range: 閾值範圍
            
        返回:
            最佳閾值
        """
        if self.task_type != 'prob_classification':
            raise ValueError("閾值優化僅適用於概率分類任務")
            
        # 獲取驗證集上的預測
        val_predictions = self._get_raw_predictions(X_val)
        
        # 尋找最佳閾值
        best_score = -float('inf')
        best_threshold = 0.5
        
        for threshold in threshold_range:
            val_pred_class = (val_predictions >= threshold).astype(int)
            score = metric(y_val, val_pred_class)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # 更新模型的閾值
        self.threshold = best_threshold
        
        return best_threshold, best_score
    
    def _get_raw_predictions(self, X):
        """
        獲取元學習器的原始預測（不應用閾值）
        
        參數:
            X: 特徵矩陣
            
        返回:
            原始預測值
        """
        # 確保模型已經訓練
        if not self.trained_base_models:
            raise ValueError("模型尚未訓練，請先調用 fit 方法")
        
        # 生成元特徵
        n_models = len(self.trained_base_models)
        meta_features = np.zeros((X.shape[0], n_models))
        
        # 獲取基礎模型預測
        for i, model in enumerate(self.trained_base_models):
            meta_features[:, i] = model.predict(X)
        
        # 使用元學習器進行預測
        return self.meta_model.predict(meta_features)
    
    # 在 StackingModel 類中添加一個方法來獲取概率形式的預測
    def predict_proba_for_regression(self, X):
        """
        對於回歸轉分類任務，返回轉換為 [0,1] 範圍的概率估計
        
        參數:
            X: 特徵矩陣
            
        返回:
            轉換後的概率估計
        """
        if self.task_type != 'prob_classification':
            raise ValueError("此方法僅適用於概率分類任務")
            
        # 獲取原始預測
        raw_predictions = self._get_raw_predictions(X)
        
        # 將預測值轉換為 [0,1] 範圍（簡單方法）
        # 可以使用 sigmoid 函數: 1 / (1 + np.exp(-raw_predictions))
        # 或者使用最小-最大縮放
        min_pred = np.min(raw_predictions)
        max_pred = np.max(raw_predictions)
        
        if max_pred > min_pred:
            scaled_predictions = (raw_predictions - min_pred) / (max_pred - min_pred)
        else:
            scaled_predictions = np.zeros_like(raw_predictions)
            
        return scaled_predictions


# 使用範例
def example_classification():
    """分類任務範例"""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    # 載入數據
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42)
    
    # 創建基礎模型
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # 創建並訓練 stacking 模型
    stacking = StackingModel(
        base_models=base_models,
        task_type='classification',
        use_proba=True
    )
    stacking.fit(X_train, y_train)
    
    # 預測並評估
    y_pred = stacking.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"分類任務準確率: {accuracy:.4f}")
    
    # 保存模型
    stacking.save_model('stacking_classification_model.pkl')
    
    return stacking

def example_regression():
    """回歸任務範例"""
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error
    
    # 載入數據 (注意: load_boston 已經被棄用)
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
    except:
        data = load_boston()
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42)
    
    # 創建基礎模型
    base_models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        ElasticNet(random_state=42)
    ]
    
    # 創建並訓練 stacking 模型
    stacking = StackingModel(
        base_models=base_models,
        task_type='regression'
    )
    stacking.fit(X_train, y_train)
    
    # 預測並評估
    y_pred = stacking.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"回歸任務均方誤差: {mse:.4f}")
    
    # 保存模型
    stacking.save_model('stacking_regression_model.pkl')
    
    return stacking

def example_prob_classification():
    """概率分類任務範例"""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import f1_score, accuracy_score
    
    # 載入數據
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42)
    
    # 創建基礎模型（全部使用回歸模型）
    base_models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        Ridge(random_state=42)
    ]
    
    # 創建並訓練 stacking 模型
    stacking = StackingModel(
        base_models=base_models,
        task_type='prob_classification',
        threshold=0.5  # 初始閾值
    )
    stacking.fit(X_train, y_train)
    
    # 優化閾值
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)
    
    best_threshold, best_score = stacking.optimize_threshold(X_val, y_val)
    print(f"最佳閾值: {best_threshold:.2f}, F1分數: {best_score:.4f}")
    
    # 使用優化的閾值進行預測
    y_pred = stacking.predict(X_test_final)
    accuracy = accuracy_score(y_test_final, y_pred)
    f1 = f1_score(y_test_final, y_pred)
    print(f"概率分類任務準確率: {accuracy:.4f}, F1分數: {f1:.4f}")
    
    # 保存模型
    stacking.save_model('stacking_prob_classification_model.pkl')
    
    return stacking

if __name__ == "__main__":
    # 運行範例
    print("分類任務示例:")
    example_classification()
    
    print("\n回歸任務示例:")
    example_regression()
    
    print("\n概率分類任務示例:")
    example_prob_classification()