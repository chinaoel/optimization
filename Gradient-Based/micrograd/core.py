from typing import Self
import math

class Value():
    def __init__(self, data:float, _children:tuple=(), _op:str="", label:str=""):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._op = _op
        self._backward = lambda : None # base case leaf node
        # for visualization purpose
        self.label = label

    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other:Self):
        if not isinstance(other, Value):
            other = Value(other) # assume it is float or int
        
        out = Value(self.data + other.data, _children=(self, other),_op="+")
        def _backward():
            # set the gradient of the child nodes
            # node = self + other , we need to flow these gradient to its child nodes
            # local deriv = 1 global deriv is self.grad
            # out.grad will be available when this function is called
            self.grad += out.grad # 1.0 * out.grad
            other.grad += out.grad # 1.0 * out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, other:Self):
        return self + other
    
    def __neg__(self):
        return self * (-1)
    def __sub__(self, other:Self):
        return self + (-other)

    def __rmul__(self, other:Self):
        # if self is not a Value
        return self * other
    
    def __mul__(self, other:Self):
        if not isinstance(other, Value):
            other = Value(other) # assume it is float or int
        out = Value(self.data * other.data, _children=(self, other),_op="*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op="tanh")
        def _backward():
            self.grad = (1 - t ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        # -1 * (6.8284**-2) * 4.8284
        x = self.data
        out = Value(math.exp(x), (self, ), _op="exp")
        
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other:Self):
        '''x ** k'''
        return self * (other ** -1)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float powers for now"
        out = Value(self.data ** other, (self,),f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other -1)) * out.grad

        out._backward = _backward        
        return out

    def backward(self):
        # encapsulate topo list only called by the root
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()