import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any

# 假設 qp_problem.py 和這個檔案在同一個 core/ 目錄下
from .qp_problem import QPProblem 

class BaseQPSolver(ABC):
    """
    所有 Convex QP Solver 的抽象父類別 (Abstract Base Class)。
    提供共用的數學運算 (例如計算目標函數、梯度與海森矩陣)，
    並強制子類別必須實作 `solve` 方法。
    """
    
    def __init__(self, problem: QPProblem):
        """
        透過依賴注入 (Dependency Injection) 的方式傳入問題定義。
        """
        self.problem = problem

    def compute_objective(self, x: np.ndarray) -> float:
        """
        計算原始 QP 的目標函數值: f(x) = 1/2 x^T Q x + c^T x
        """
        return 0.5 * x.T @ self.problem.Q @ x + self.problem.c.T @ x

    def compute_obj_grad_hess(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算原始 QP 目標函數的梯度 (Gradient) 與海森矩陣 (Hessian)。
        因為是二次規劃，Hessian 永遠是常數矩陣 Q。
        """
        grad = self.problem.Q @ x + self.problem.c
        hess = self.problem.Q
        return grad, hess

    @abstractmethod
    def solve(self, x_init: np.ndarray, **kwargs) -> Tuple[np.ndarray, Any]:
        """
        強制所有繼承此類別的子類別都必須實作這個方法。
        
        Args:
            x_init: 初始猜測的權重向量
            **kwargs: 其他演算法專屬的超參數 (如 tol, max_iters 等)
            
        Returns:
            x_star: 最佳化後的權重向量
            history/info: 演算法收斂過程的歷史紀錄或狀態
        """
        pass