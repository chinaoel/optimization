import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 建立金融問題參數 (The Setup)
# ==========================================
# 預期報酬 (Tech, Crypto, Bonds)
expected_returns = np.array([0.08, 0.15, 0.03])
c = -expected_returns  # 因為我們是 minimize，所以加負號

# 共變異數矩陣 Q (Risk)
Q = np.array([
    [0.04, 0.01, 0.00],  # Tech 風險適中
    [0.01, 0.16, -0.01], # Crypto 風險極高
    [0.00, -0.01, 0.005] # Bonds 風險極低，且與 Crypto 負相關
])

# 不等式限制 Gx <= h
# 包含 -x <= 0 (三條) 和 sum(x) <= 1 (一條)
G = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [ 1.0,  1.0,  1.0]
])
h = np.array([0.0, 0.0, 0.0, 1.0])

# ==========================================
# 2. 核心數學函數 (Oracle)
# ==========================================
def objective(x):
    """計算原始目標函數 f(x) = 0.5 * x^T Q x + c^T x"""
    return 0.5 * x.T @ Q @ x + c.T @ x

def obj_grad_hess(x):
    """回傳 f(x) 的 Gradient 和 Hessian"""
    grad = Q @ x + c
    hess = Q
    return grad, hess

def barrier(x):
    """計算 Log Barrier 的值: -sum(log(h - Gx))"""
    slack = h - G @ x
    if np.any(slack <= 0):
        return np.inf # 撞牆了
    return -np.sum(np.log(slack))

def barrier_grad_hess(x):
    """回傳 Barrier 函數的 Gradient 和 Hessian (這部分矩陣微積分已幫你推導好)"""
    slack = h - G @ x
    if np.any(slack <= 0):
        raise ValueError("x is outside the strictly feasible region!")
    
    d = 1.0 / slack
    grad = G.T @ d                  # 第一階導數
    hess = G.T @ np.diag(d**2) @ G  # 第二階導數
    return grad, hess

# ==========================================
# 3. 演算法核心 (Your Playground)
# ==========================================
'''
def backtracking_line_search(x, delta_x, gradient, t, alpha=0.1, beta=0.7):
    """
    TODO 1: 實作回溯線搜索 (Backtracking Line Search)
    確保 x_new = x + step_size * delta_x 不會超出邊界 (G @ x_new < h)
    並且滿足 Armijo condition (或簡單確認 Objective 有下降)
    Alpha : 預期進步比率(根據梯度)
    """
    step_size = 1.0
    fn_old = objective(x) + 1/t * barrier(x)
        
    while True:
        x_new = x + step_size * delta_x
        slack_new = h - G @ x_new
        # the distance of this x setting between each wall



        

        if all(slack_new > 0):
            # if all slack is positive, log can be calculated
            fn_new = objective(x_new) + 1/t * barrier(x_new)
            # check for Armijo condition 
            # Armijo 公式：$F_{new} \le F_{old} + \alpha \cdot \text{step\_size} \cdot (\nabla F^T \Delta x)$。
            # if pass then break, 
            if fn_new <= (fn_old + alpha * step_size * (gradient.T @ delta_x)):
                break
            
 
        step_size = beta * step_size
        
        
        # 條件 1: 必須留在房間內 (Strictly feasible)
        # 條件 2: 函數值必須下降 (為了簡化，你可以在這裡只檢查有沒有撞牆，
        #         進階一點可以加入 f(x_new) < f(x) + alpha * step_size * grad^T delta_x)
        
        # --- 你的邏輯寫這裡 ---
        # if ... 
        #     break
        # else:
        #     step_size *= beta
        # ----------------------
        
        break # 記得刪掉這行
        
    return step_size
    '''

def backtracking_line_search(x, delta_x, gradient, t, alpha=0.1, beta=0.7):
    # ---------------------------------------------------------
    # 1. 計算撞牆極限 (Fraction to the Boundary)
    # ---------------------------------------------------------
    s = h - G @ x              # 目前離各個牆壁的距離 (保證 > 0)
    delta_s = G @ delta_x      # 沿著 delta_x 走，距離的變化率
    
    # 找出所有「正在靠近牆壁」的限制式 (也就是 delta_s > 0)
    # 加上 1e-12 是為了避免除以 0 的浮點數警告
    mask = delta_s > 1e-12 
    
    if np.any(mask):
        # s / delta_s 就是「還要走多遠會剛好撞牆」
        alphas_to_walls = s[mask] / delta_s[mask]
        
        # 找出最快撞上的那面牆，並乘上 0.99 留出絕對安全的緩衝區
        alpha_max = min(1.0, 0.99 * np.min(alphas_to_walls))
    else:
        # 如果前方一片平坦 (甚至在遠離所有牆壁)
        alpha_max = 1.0 
        
    # ---------------------------------------------------------
    # 2. 從安全極限開始進行 Armijo Backtracking
    # ---------------------------------------------------------
    step_size = alpha_max  # 不再是死板的 1.0，而是安全的 alpha_max
    fn_old = objective(x) + (1.0 / t) * barrier(x)
    expected_descent = gradient.T @ delta_x 
        
    while True:
        x_new = x + step_size * delta_x
        slack_new = h - G @ x_new
        
        if np.all(slack_new > 0):
            fn_new = objective(x_new) + (1.0 / t) * barrier(x_new)
            if fn_new <= fn_old + alpha * step_size * expected_descent:
                break 
 
        step_size = beta * step_size
        
        # 終極防呆：如果步長被打折到比奈米還小，代表牛頓方向有問題或已到極限
        if step_size < 1e-15:
            break
            
    return step_size

def newton_method_for_fixed_t(x_init, t, max_inner_iters=100, tol=1e-5):
    """
    TODO 2: 實作給定 t 下的牛頓法 (Inner Loop)
    目標函數是 F(x) = f(x) + (1/t) * barrier(x)
    """
    x = x_init.copy()
    
    for i in range(max_inner_iters):
        # 1. 取得 f(x) 和 barrier(x) 的 grad 與 hess
        f_grad, f_hess = obj_grad_hess(x)
        phi_grad, phi_hess = barrier_grad_hess(x)
        
        # 2. 組合出當前 F(x) 的 Gradient 和 Hessian
        # --- 你的邏輯寫這裡 ---
        # F_grad = ...
        # F_hess = ...
        # ----------------------
        F_grad = f_grad + 1/t * phi_grad 
        F_hess = f_hess + 1/t * phi_hess 
        
        # 3. 解牛頓方向 delta_x = - Hessian^-1 * Gradient
        # 提示: 使用 np.linalg.solve(A, b) 來解 A*x = b，不要用反矩陣
        # --- 你的邏輯寫這裡 ---
        # delta_x = ...
        # ----------------------
        delta_x = np.linalg.solve(F_hess, -F_grad)

        
        # 4. 檢查停止條件 (Newton Decrement)
        # 簡單版: if np.linalg.norm(delta_x) < tol: break
        
        # 5. Line Search 決定步長
        # step_size = backtracking_line_search(x, delta_x, F_grad, t)
        step_size = backtracking_line_search(x, delta_x, F_grad, t)

        # 6. 更新 x
        # x = x + step_size * delta_x
        x = x + step_size * delta_x
                
    return x

def barrier_method():
    """
    TODO 3: 實作外層迴圈 Sequential Unconstrained Minimization
    """
    # 找一個絕對安全的起始點 (資金平分，且總和 < 1)
    x = np.array([0.1, 0.1, 0.1]) 
    
    m = len(h) # 不等式限制的數量
    t = 1.0    # 初始 t
    mu = 15.0  # t 每次放大的倍數 (Lecture 3 建議值介於 10~20)
    epsilon = 1e-6 # 最終容忍誤差
    
    history = [x]
    
    print("Starting Barrier Method...")
    # --- 你的邏輯寫這裡 ---
    # while ... (利用 m / t 來判斷是否已經夠接近最佳解):
    #     x = newton_method_for_fixed_t(x, t)
    #     history.append(x)
    #     t = t * mu
    #     print(f"t = {t:.1f}, Gap = {m/t:.6f}, x = {x}")
    # ----------------------
    while m/t > 10e-14:
        x = newton_method_for_fixed_t(x, t)
        history.append(x)
        t = t * mu
        print(f"t = {t:.1f}, Gap = {m/t:.6f}, x = {x}")

    return np.array(history)