import numpy as np
from typing import Tuple, Dict

from .base_solver import BaseQPSolver
from .qp_problem import QPProblem

class PrimalDualSolver(BaseQPSolver):
    """
    Primal-Dual Interior Point Method 求解 Convex QP。
    直接對 KKT 條件進行牛頓法迭代，收斂速度極快，且原生支援等式限制 (Ax = b)。
    """
    
    def __init__(self, problem: QPProblem, mu_param: float = 10.0, tol: float = 1e-8):
        super().__init__(problem)
        self.mu_param = mu_param
        self.tol = tol
        
        # Backtracking Line Search 參數
        self.alpha_bt = 0.1
        self.beta_bt = 0.5

    def solve(self, x_init: np.ndarray, max_iters: int = 100) -> Tuple[np.ndarray, Dict]:
        x = x_init.copy()
        
        n = self.problem.num_vars
        m = self.problem.num_ineq_constraints
        p = self.problem.num_eq_constraints
        
        # 初始化對偶變數 (Dual variables)
        # lambda 對應不等式 (必須 > 0)，nu 對應等式 (無正負限制)
        lmbda = np.ones(m)
        nu = np.zeros(p) if p > 0 else np.array([])
        
        history_x = [x.copy()]
        history_gap = []
        
        for i in range(max_iters):
            # 1. 計算目前的 Slack (離牆壁的距離) 與 Surrogate Duality Gap
            s = self.problem.h - self.problem.G @ x
            if np.any(s <= 0):
                raise ValueError("Current x is not strictly feasible (s <= 0).")
                
            gap = lmbda.T @ s
            history_gap.append(gap)
            
            # 自適應計算 Barrier Parameter t
            t = (self.mu_param * m) / gap
            
            # 2. 計算 KKT 系統的三大殘差 (Residuals)
            eq_dual = self.problem.A.T @ nu if p > 0 else np.zeros(n)
            r_dual = self.problem.Q @ x + self.problem.c + self.problem.G.T @ lmbda + eq_dual
            r_cent = lmbda * s - (1.0 / t) * np.ones(m)
            r_pri = self.problem.A @ x - self.problem.b if p > 0 else np.array([])
            
            # 將殘差組裝並計算 Norm
            res_concat = np.concatenate([r_dual, r_cent, r_pri]) if p > 0 else np.concatenate([r_dual, r_cent])
            res_norm = np.linalg.norm(res_concat)
            
            # 檢查停止條件
            if gap < self.tol and res_norm < self.tol:
                break
                
            # 3. 組裝 KKT 巨型矩陣 (The Block Matrix) 與 RHS
            if p > 0:
                KKT_matrix = np.block([
                    [self.problem.Q, self.problem.G.T, self.problem.A.T],
                    [-np.diag(lmbda) @ self.problem.G, np.diag(s), np.zeros((m, p))],
                    [self.problem.A, np.zeros((p, m)), np.zeros((p, p))]
                ])
            else:
                KKT_matrix = np.block([
                    [self.problem.Q, self.problem.G.T],
                    [-np.diag(lmbda) @ self.problem.G, np.diag(s)]
                ])
                
            # 解牛頓方向
            step_dir = np.linalg.solve(KKT_matrix, -res_concat)
            
            # 切割方向向量
            dx = step_dir[0:n]
            dlmbda = step_dir[n:n+m]
            dnu = step_dir[n+m:] if p > 0 else np.array([])
            
            # 4. Fraction to the boundary (防撞牆極限步長)
            ds = -self.problem.G @ dx
            mask_s = ds < 0
            step_s = min(1.0, 0.99 * np.min(-s[mask_s] / ds[mask_s])) if np.any(mask_s) else 1.0
            
            mask_l = dlmbda < 0
            step_l = min(1.0, 0.99 * np.min(-lmbda[mask_l] / dlmbda[mask_l])) if np.any(mask_l) else 1.0
            
            step_max = min(step_s, step_l)
            step = step_max
            
            # 5. Backtracking Line Search (確保殘差穩定下降)
            while True:
                x_new = x + step * dx
                lmbda_new = lmbda + step * dlmbda
                nu_new = nu + step * dnu if p > 0 else np.array([])
                
                s_new = self.problem.h - self.problem.G @ x_new
                
                # 計算新點的殘差
                eq_dual_new = self.problem.A.T @ nu_new if p > 0 else np.zeros(n)
                r_dual_new = self.problem.Q @ x_new + self.problem.c + self.problem.G.T @ lmbda_new + eq_dual_new
                r_cent_new = lmbda_new * s_new - (1.0 / t) * np.ones(m)
                r_pri_new = self.problem.A @ x_new - self.problem.b if p > 0 else np.array([])
                
                res_concat_new = np.concatenate([r_dual_new, r_cent_new, r_pri_new]) if p > 0 else np.concatenate([r_dual_new, r_cent_new])
                res_new_norm = np.linalg.norm(res_concat_new)
                
                # Armijo 條件：殘差必須有足夠的下降
                if res_new_norm <= (1 - self.alpha_bt * step) * res_norm:
                    break
                    
                step *= self.beta_bt
                if step < 1e-15:
                    break
                    
            # 6. 正式更新變數
            x = x_new
            lmbda = lmbda_new
            nu = nu_new
            
            history_x.append(x.copy())
            
        info = {
            "history_x": np.array(history_x),
            "history_gap": history_gap,
            "final_res_norm": res_norm,
            "iterations": i + 1
        }
        return x, info