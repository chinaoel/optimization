import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class QPProblem:
    """
    Standard form Convex Quadratic Programming (QP) problem data container.
    
    Objective: 
        min 1/2 x^T Q x + c^T x
    
    Subject to: 
        Gx <= h
        Ax = b
    """
    Q: np.ndarray
    c: np.ndarray
    G: np.ndarray
    h: np.ndarray
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        初始化後自動執行的維度檢查 (Dimensionality Checks)。
        這能確保傳入的矩陣形狀完全匹配，提早發現錯誤 (Fail Fast)。
        """
        n = len(self.c)
        
        # 檢查目標函數矩陣
        if self.Q.shape != (n, n):
            raise ValueError(f"Q must be a square matrix of shape ({n}, {n}), got {self.Q.shape}")
            
        # 檢查不等式限制矩陣
        if self.G.shape[1] != n:
            raise ValueError(f"G must have {n} columns to match variables, got {self.G.shape[1]}")
        if len(self.h) != self.G.shape[0]:
            raise ValueError(f"h length ({len(self.h)}) must match G rows ({self.G.shape[0]})")
            
        # 檢查等式限制矩陣 (如果有的話)
        if self.A is not None and self.b is not None:
            if self.A.shape[1] != n:
                raise ValueError(f"A must have {n} columns to match variables, got {self.A.shape[1]}")
            if len(self.b) != self.A.shape[0]:
                raise ValueError(f"b length ({len(self.b)}) must match A rows ({self.A.shape[0]})")

    @property
    def num_vars(self) -> int:
        """回傳變數的數量 (n)"""
        return len(self.c)
        
    @property
    def num_ineq_constraints(self) -> int:
        """回傳不等式限制的數量 (m)"""
        return len(self.h)
        
    @property
    def num_eq_constraints(self) -> int:
        """回傳等式限制的數量 (p)"""
        return len(self.b) if self.b is not None else 0