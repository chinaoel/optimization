import math

class Dual:
    def __init__(self, primal, deriv):
        self.primal = primal
        self.deriv = deriv
    
    def __repr__(self):
        return f"primal : {self.primal:.4f}, deriv : {self.deriv:.4f}"

    def __add__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other, 0.0)
        return Dual(self.primal + other.primal, self.deriv + other.deriv)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other, 0.0)
        return Dual(self.primal - other.primal, self.deriv - other.deriv)

    def __rsub__(self, other):
        return Dual(other, 0.0) - self

    def __mul__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other, 0.0)
        return Dual(self.primal * other.primal, self.deriv * other.primal + self.primal * other.deriv)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        assert other.primal != 0, "Division by zero (primal value)"
        deriv = (self.deriv * other.primal - self.primal * other.deriv) / other.primal ** 2 
        return Dual(self.primal / other.primal, deriv)
    
    def __rtruediv__(self, other):
        return Dual(other, 0.0) / self 

    def __pow__(self, const):
        return Dual(self.primal ** const, const * self.primal ** (const - 1) * self.deriv)
    
    def __neg__(self):
        return Dual(-1 * self.primal, -1 * self.deriv)

    def exp(self):
        return Dual(math.exp(self.primal), math.exp(self.primal)*self.deriv)
    
    def log(self):
        return Dual(math.log(self.primal), self.deriv * (1 / self.primal))

class AutoDiffEngine:
    @staticmethod
    def gradient(f, params: list[float]) -> list[float]:
        """計算函數 f 對參數 params 的梯度"""
        grads = []
        n = len(params)
        for i in range(n):
            dual_inputs = []
            for j in range(n):
                val = params[j]
                if i == j:
                    dual_inputs.append(Dual(val, 1.0)) # 打開針對這個參數的導數開關
                else:
                    dual_inputs.append(Dual(val, 0.0)) # 其他視為常數
            
            loss = f(dual_inputs)
            grads.append(loss.deriv)
            
        return grads

class NNOps:
    """神經網路共用運算"""
    @staticmethod
    def sigmoid(x):
        if isinstance(x, Dual):
            return 1 / (1 + (-x).exp())
        else:
            return 1 / (1 + math.exp(-x))

    @staticmethod
    def dot_product(w, x):
        return sum([wi * xi for wi, xi in zip(w, x)])