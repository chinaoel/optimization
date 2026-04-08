import random
from core import Value

class Nueron():
    def __init__(self,nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin

    def __call__(self,x):
        preact = sum([wi * xi for wi, xi in zip(self.w, x)]) + self.b # takes two iterator and return a iterator of tuples
        postact = preact.tanh()
        return postact
    
    def parameters(self):
        return self.w + [self.b]


    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer():
    def __init__(self, nin, nout):
        self.neurons = [Nueron(nin) for _ in range(nout)] # this neuron has how many weight and how many of this kind of neuron

    def __call__(self,x):
        
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
            return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP():
    # fully connected mlp # of layers, in out for each layer (Layer(in, out))
    # out -> next layer's in
    # 10, 4, 2, 1
    # (10, 4), (4,2), (2,1)
    def __init__(self, layers):
    
        self.layers = [Layer(layers[i],layers[i+1]) for i in range(len(layers)-1)]

    def __call__(self, x):
        # Sequential
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
