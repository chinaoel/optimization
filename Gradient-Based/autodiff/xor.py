import random
import math
from core import AutoDiffEngine, NNOps

random.seed(67)


def xor_loss(params, x, y):
    """
    2層神經網路的 Loss Function
    Layer 1: 2個神經元 (W11, W12)
    Layer 2: 1個神經元 (W22)
    總共 9 個參數: 每個神經元 2個權重 + 1個偏差值
    """
    W11 = params[:3]
    W12 = params[3:6]
    W22 = params[6:]

    # Layer 1
    N11 = NNOps.sigmoid(NNOps.dot_product(W11[:2], x) + W11[-1])
    N12 = NNOps.sigmoid(NNOps.dot_product(W12[:2], x) + W12[-1])
    
    # Layer 2 
    N22 = NNOps.dot_product(W22[:2], [N11, N12]) + W22[-1]
    pred = NNOps.sigmoid(N22)
    
    # MSE Loss
    diff = y - pred
    return diff ** 2

def train_xor():
    # XOR 真值表
    X_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_data = [0, 1, 1, 0]
    
    # 初始 9 個參數
    params = [random.uniform(-1, 1) for _ in range(9)]
    learning_rate = 1.0
    epochs = 5000  # XOR 通常需要較多次迭代
    
    print(f"開始 XOR 模型訓練! 總參數數量: {len(params)}")
    print("-" * 30)
    
    for step in range(epochs):
        total_loss = 0
        grads_accum = [0.0] * len(params)
        
        # Batch Gradient Descent (將 4 筆資料梯度加總)
        for i in range(4):
            curr_X = X_data[i]
            curr_y = y_data[i]
            
            loss_func = lambda p: xor_loss(p, curr_X, curr_y)
            grads = AutoDiffEngine.gradient(loss_func, params)
            
            for j in range(len(params)):
                grads_accum[j] += grads[j]
        
        # 參數更新
        for j in range(len(params)):
            params[j] -= learning_rate * (grads_accum[j] / 4.0)

        # 每 100 step 印出目前的預測狀態
        if (step + 1) % 100 == 0:
            print(f"\n--- Epoch {step+1} ---")
            for i in range(4):
                W11, W12, W22 = params[:3], params[3:6], params[6:]
                N11 = 1 / (1 + math.exp(-(W11[0]*X_data[i][0] + W11[1]*X_data[i][1] + W11[2])))
                N12 = 1 / (1 + math.exp(-(W12[0]*X_data[i][0] + W12[1]*X_data[i][1] + W12[2])))
                z_out = W22[0]*N11 + W22[1]*N12 + W22[2]
                pred = 1 / (1 + math.exp(-z_out))
                print(f"Input: {X_data[i]} | True: {y_data[i]} | Pred: {pred:.4f}")

if __name__ == "__main__":
    train_xor()