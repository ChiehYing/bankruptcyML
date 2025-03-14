from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 驗證模型
def model_validation(model, X_train, y_train, X_valid=None, y_valid=None):
       if X_valid is not None or y_valid is not None:
              y_pred_train = model.predict(X_train)
              r2_train = r2_score(y_train, y_pred_train)
              print("訓練組 R2:", r2_train)
              
              y_pred_valid = model.predict(X_valid)
              r2_valid = r2_score(y_valid, y_pred_valid)
              print("驗證組 R2:", r2_valid)
              
              return y_pred_train, y_pred_valid
              
       else:
              y_pred_train = model.predict(X_train)
              r2_train = r2_score(y_train, y_pred_train)
              print("訓練組 R2:", r2_train)

              y_pred_valid = None
              
              return y_pred_train, y_pred_valid


# 繪製散布圖
def scatter_plot(y1_true, y1_pred, y2_true=None, y2_pred=None):
       if y2_true is not None or y2_pred is not None:
              fig, axs = plt.subplots(1, 2, figsize=(12, 6))
              axs[0].scatter(y1_true, y1_pred, alpha=0.6, color="b")
              axs[1].scatter(y2_true, y2_pred, alpha=0.6, color="r")       
       
       else:
              plt.scatter(y1_true, y1_pred, alpha=0.6, color="b")
              
       plt.tight_layout()
       plt.show()
       

# 驗證模型並繪製散布圖
def regression_validation(model, X_train, y_train, X_valid=None, y_valid=None):
       y_pred_train, y_pred_valid = model_validation(model, X_train, y_train, X_valid, y_valid)
       scatter_plot(y_train, y_pred_train, y_valid, y_pred_valid)