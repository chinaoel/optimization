import math
import random
import numpy as np
from sklearn.datasets import load_digits
from core import AutoDiffEngine, NNOps

random.seed(67)

def simple_logistic_loss(params, img, label):
    """Logistic Regression 的 Loss Function"""
    W = params[:-1]
    b = params[-1]
    
    z = NNOps.dot_product(W, img) + b
    y_pred = NNOps.sigmoid(z)
    
    # MSE Loss
    diff = y_pred - label
    return diff ** 2

def train_image_recognition():
    # 1. 準備資料
    digits = load_digits()
    X_data = digits.data / 16.0  # 歸一化
    y_binary = np.where(digits.target == 0, 1.0, 0.0)
    
    # 2. 初始化參數 (64 Weights + 1 Bias)
    params = [random.uniform(-0.1, 0.1) for _ in range(65)]
    learning_rate = 0.5
    epochs = 10
    
    print(f"開始影像辨識訓練! 總參數數量: {len(params)}")
    print("-" * 30)
    
    # 3. 訓練迴圈 (SGD)
    for step in range(epochs):
        curr_img = X_data[step]
        curr_label = y_binary[step]
        
        # 使用 lambda 將當前資料包進目標函數中
        loss_func = lambda p: simple_logistic_loss(p, curr_img, curr_label)
        
        # 呼叫核心運算計算梯度
        grads = AutoDiffEngine.gradient(loss_func, params)
        
        # 參數更新 (Gradient Descent)
        for i in range(len(params)):
            params[i] -= learning_rate * grads[i]
            
        # 監控 Loss (用浮點數算一次純量)
        W_curr, b_curr = params[:-1], params[-1]
        z = sum([w * x for w, x in zip(W_curr, curr_img)]) + b_curr
        pred = 1 / (1 + math.exp(-z))
        loss_val = (pred - curr_label) ** 2
        print(f"Step {step+1:2d} | Label: {int(curr_label)} | Pred: {pred:.4f} | Loss: {loss_val:.4f}")

if __name__ == "__main__":
    train_image_recognition()