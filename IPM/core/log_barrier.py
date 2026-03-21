import numpy as np
from typing import Tuple, Dict

# 引入剛剛寫好的父類別與資料結構
from .base_solver import BaseQPSolver
from .qp_problem import QPProblem

class LogBarrierSolver(BaseQPSolver):
    """
    Log-Barrier Method (障礙函數法) 求解 Convex QP。
    利用對數障礙函數將不等式限制轉換為目標函數的懲罰項。
    """
    
    def __init__(self, problem: QPProblem, mu: float = 15.0, inner_tol: float = 1e-5, outer_tol: float = 1e-6):
        super().__init__(problem)
        self.mu = mu
        self.inner_tol = inner_tol
        self.outer_tol = outer_tol
        
        # 工程防呆：基本的 Log-Barrier 不處理等式限制 (Ax=b)
        # 如果使用者誤傳了 A 和 b，提早報錯，並引導他們去用 Primal-Dual
        if self.problem.A is not None:
            raise NotImplementedError(
                "This basic Log-Barrier implementation does not support equality constraints. "
                "Please use PrimalDualSolver for problems with Ax = b constraints."
            )

    def _barrier_val(self, x: np.ndarray) -> float:
        """[Private] 計算 Barrier 函數的值: -sum(log(h - Gx))"""
        slack = self.problem.h - self.problem.G @ x
        if np.any(slack <= 0):
            return np.inf  # 撞牆了
        return -np.sum(np.log(slack))

    def _barrier_grad_hess(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """[Private] 計算 Barrier 函數的 Gradient 和 Hessian"""
        slack = self.problem.h - self.problem.G @ x
        if np.any(slack <= 0):
            raise ValueError("x is outside the strictly feasible region!")
        
        d = 1.0 / slack
        grad = self.problem.G.T @ d
        hess = self.problem.G.T @ np.diag(d**2) @ self.problem.G
        return grad, hess

    def _line_search(self, x: np.ndarray, delta_x: np.ndarray, F_grad: np.ndarray, t: float, alpha: float = 0.1, beta: float = 0.7) -> float:
        """[Private] Backtracking Line Search (結合了防撞牆 Fraction-to-the-boundary)"""
        s = self.problem.h - self.problem.G @ x
        delta_s = self.problem.G @ delta_x
        
        mask = delta_s > 1e-12
        # 計算不撞牆的最大安全步長
        alpha_max = min(1.0, 0.99 * np.min(s[mask] / delta_s[mask])) if np.any(mask) else 1.0
        
        step_size = alpha_max
        fn_old = self.compute_objective(x) + (1.0 / t) * self._barrier_val(x)
        expected_descent = F_grad.T @ delta_x
        
        while step_size > 1e-15:
            x_new = x + step_size * delta_x
            slack_new = self.problem.h - self.problem.G @ x_new
            
            # 確保新點在可行域內 (Strictly feasible)
            if np.all(slack_new > 0):
                fn_new = self.compute_objective(x_new) + (1.0 / t) * self._barrier_val(x_new)
                # Armijo Condition
                if fn_new <= fn_old + alpha * step_size * expected_descent:
                    break
            
            step_size *= beta
            
        return step_size

    def solve(self, x_init: np.ndarray, max_outer_iters: int = 50, max_inner_iters: int = 100) -> Tuple[np.ndarray, Dict]:
        """
        [Public] 執行 Sequential Unconstrained Minimization
        這是實作 BaseQPSolver 所規範的強制方法。
        """
        x = x_init.copy()
        m = self.problem.num_ineq_constraints
        t = 1.0
        
        history_x = [x.copy()]
        history_steps = []
        cumulative_steps = 0
        
        # Outer Loop: 逐步放大 t
        for _ in range(max_outer_iters):
            if (m / t) < self.outer_tol:
                break
                
            # Inner Loop: Newton's Method for fixed t
            for _ in range(max_inner_iters):
                cumulative_steps += 1
                
                # 這裡直接呼叫父類別寫好的 compute_obj_grad_hess！
                f_grad, f_hess = self.compute_obj_grad_hess(x)
                phi_grad, phi_hess = self._barrier_grad_hess(x)
                
                F_grad = f_grad + (1.0 / t) * phi_grad
                F_hess = f_hess + (1.0 / t) * phi_hess
                
                # 解牛頓方向
                delta_x = np.linalg.solve(F_hess, -F_grad)
                
                # 使用 Newton Decrement 作為停止條件
                lambda_sq = F_grad.T @ np.linalg.solve(F_hess, F_grad)
                if lambda_sq / 2.0 < self.inner_tol:
                    break
                    
                # 決定步長並更新 x
                step_size = self._line_search(x, delta_x, F_grad, t)
                x = x + step_size * delta_x
                
            history_x.append(x.copy())
            history_steps.append(cumulative_steps)
            t *= self.mu
            
        # 回傳最佳解與軌跡資訊 (為了畫對比圖)
        info = {
            "history_x": np.array(history_x),
            "history_steps": history_steps,
            "final_gap": m / t
        }
        return x, info