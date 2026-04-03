import numpy as np
from abc import ABC, abstractmethod
import random
from IPM.core.qp_problem import QPProblem
from IPM.core.primal_dual import PrimalDualSolver


class SVMSolver(ABC):
    """求解器抽象介面：只要吃進資料，就必須吐出 alpha 和 b"""
    @abstractmethod
    def solve(self, X, y, C, kernel_func) -> tuple[np.ndarray, float]:
        pass

class SMOSolver(SVMSolver):
    def __init__(self, tol=1e-3, max_passes=5):
        self.tol = tol
        self.max_passes = max_passes
        
    def solve(self, X, y, C, kernel_func):
        m = X.shape[0]
        alphas = np.zeros(m)
        b = 0.0
        passes = 0
        
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                # 計算 E_i (預測誤差)
                # 這裡為了維持 SMO 原意，保留簡單的迴圈計算
                K_i = kernel_func(X, X[i])
                f_xi = np.sum(alphas * y * K_i) + b
                E_i = f_xi - y[i]
                
                # 檢查 KKT 違規條件
                if (y[i] * E_i < -self.tol and alphas[i] < C) or (y[i] * E_i > self.tol and alphas[i] > 0):
                    # 隨機挑選 j (簡化版)
                    j = np.random.choice([x for x in range(m) if x != i])
                    
                    K_j = kernel_func(X, X[j])
                    f_xj = np.sum(alphas * y * K_j) + b
                    E_j = f_xj - y[j]
                    
                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                    
                    # 計算邊界 L, H
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - C)
                        H = min(C, alphas[i] + alphas[j])
                        
                    if L == H:
                        continue
                        
                    # 計算 eta
                    eta = 2 * kernel_func(X[i], X[j]) - kernel_func(X[i], X[i]) - kernel_func(X[j], X[j])
                    if eta >= 0:
                        continue
                        
                    # 更新 alpha_j 並裁剪
                    alphas[j] -= y[j] * (E_i - E_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue
                        
                    # 更新 alpha_i
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # 更新 b
                    b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * kernel_func(X[i], X[i]) - y[j] * (alphas[j] - alpha_j_old) * kernel_func(X[i], X[j])
                    b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * kernel_func(X[i], X[j]) - y[j] * (alphas[j] - alpha_j_old) * kernel_func(X[j], X[j])
                    
                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                        
                    num_changed_alphas += 1
                    
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
        return alphas, b

class IPMSolver(SVMSolver):
    def __init__(self, ipm_solve_func):
        """傳入你外部寫好的 ipm_solve 函數"""
        self.ipm_solve_func = ipm_solve_func
        
    def _build_qp(self, X, y, C, kernel_func) -> QPProblem:
        n = X.shape[0]
        K_matrix = kernel_func(X, X)
        y_col = y.reshape(-1, 1)
        
        Q = (y_col @ y_col.T) * K_matrix
        Q = Q + 1e-5 * np.eye(n)
        c = -np.ones(n)
        
        G = np.vstack((-np.eye(n), np.eye(n)))
        h = np.concatenate((np.zeros(n), np.full(n, C)))
        
        A = y.reshape(1, n)
        b = np.array([0.0])
        
        return QPProblem(Q=Q, c=c, G=G, h=h, A=A, b=b)

    def _generate_start_point(self, y, C):
        n = len(y)
        alphas = np.zeros(n)
        is_pos = (y == 1)
        is_neg = (y == -1)
        S = (C * min(np.sum(is_pos), np.sum(is_neg))) / 2.0
        alphas[is_pos] = S / np.sum(is_pos)
        alphas[is_neg] = S / np.sum(is_neg)
        return alphas

    def solve(self, X, y, C, kernel_func):
        # 1. 翻譯成 QP 問題
        qp_prob = self._build_qp(X, y, C, kernel_func)
        x0 = self._generate_start_point(y, C)
        
        # 2. 呼叫外部 IPM 引擎 (假設你的求解器吃 qp_prob 和 x0)
        solver = self.ipm_solve_func(problem=qp_prob, tol=1e-8)
        alphas, info = solver.solve(x_init = x0)

        # 3. 逆向推導截距 b
        sv_margin_mask = (alphas > 1e-5) & (alphas < C - 1e-5)
        if np.any(sv_margin_mask):
            idx = np.where(sv_margin_mask)[0][0]
            k_vec = kernel_func(X, X[idx])
            b = y[idx] - np.sum(alphas * y * k_vec)
        else:
            sv_mask = alphas > 1e-5
            sv_indices = np.where(sv_mask)[0]
            b_sum = 0
            for idx in sv_indices:
                k_vec = kernel_func(X, X[idx])
                b_sum += y[idx] - np.sum(alphas * y * k_vec)
            b = b_sum / len(sv_indices) if len(sv_indices) > 0 else 0.0
            
        return alphas, b