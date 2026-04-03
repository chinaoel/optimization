import numpy as np
from sklearn.datasets import make_blobs, make_circles

def get_linear_separable_data(n_samples=100):
    """產生線性可分的點，標籤轉換為 SVM 專用的 +1 與 -1"""
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42, cluster_std=1.2)
    # ⚠️ 關鍵：將 Scikit-learn 預設的 0 標籤轉換為 -1
    y = np.where(y == 0, -1, 1)
    return X, y

def get_circle_data(n_samples=100, noise=0.1, factor=0.5):
    """產生同心圓資料，標籤轉換為 SVM 專用的 +1 與 -1"""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    # ⚠️ 關鍵：將 Scikit-learn 預設的 0 標籤轉換為 -1
    y = np.where(y == 0, -1, 1)
    return X, y