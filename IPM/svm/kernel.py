import numpy as np
from abc import ABC, abstractmethod


class BaseKernel(ABC):
    @abstractmethod
    def __call__(self, x1, x2):
        pass

class LinearKernel(BaseKernel):
    def __call__(self, x1, x2):
        return np.dot(x1, x2.T)

class RBFKernel(BaseKernel):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        
    def __call__(self, x1, x2):
        # 終極維度防禦版：支援 Vector vs Vector, Matrix vs Vector, Matrix vs Matrix
        if x1.ndim == 2 and x2.ndim == 2:
            diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        else:
            diff = x1 - x2
        sq_dist = np.sum(diff**2, axis=-1)
        return np.exp(-sq_dist / (2 * self.sigma ** 2))