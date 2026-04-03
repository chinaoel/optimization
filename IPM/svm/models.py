import numpy as np
from .kernel import BaseKernel, RBFKernel
from .solver import SMOSolver,SVMSolver

class SVM:
    def __init__(self, C=1.0, kernel: BaseKernel = None, solver: SVMSolver = None):
        """
        依賴注入 (Dependency Injection)：
        SVM 是一個空殼，它只負責管理狀態與預測，把數學運算交給 Kernel 和 Solver
        """
        self.C = C
        self.kernel = kernel if kernel else RBFKernel()
        self.solver = solver if solver else SMOSolver()
        
        # 訓練後生成的參數
        self.alphas = None
        self.b = None
        self.support_vectors = None
        self.sv_y = None
        self.sv_alphas = None

    def fit(self, X, y):
        # 把資料丟給外包求解器
        self.alphas, self.b = self.solver.solve(X, y, self.C, self.kernel)
        
        # 訓練完畢，立刻把「無用資料 (alpha=0)」丟掉，只保留菁英
        sv_mask = self.alphas > 1e-5
        self.support_vectors = X[sv_mask]
        self.sv_y = y[sv_mask]
        self.sv_alphas = self.alphas[sv_mask]

    def predict(self, X_new):
        if self.alphas is None:
            raise ValueError("Model is not fitted yet!")
            
        # 終極向量化預測：只算新資料跟 Support Vectors 的距離
        K_matrix = self.kernel(self.support_vectors, X_new)
        decision_scores = np.dot(self.sv_alphas * self.sv_y, K_matrix) + self.b
        return np.where(decision_scores < 0, -1, 1)