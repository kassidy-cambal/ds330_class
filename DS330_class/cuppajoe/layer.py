"""This is layer of neurons that contains a set of tensors.
    We will have to keep track of the ability to run and train."""

import numpy as np

from DS330_class.cuppajoe import tensor


class Layer():
    def __init__(self):
         # y = wx + b
        self.w = tensor.Tensor # w represents the weight
        self.b = tensor.Tensor 
        self.x = None # these are the inputs to the layers
        self.grad_w = 0
        self.grad_b = 0

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """A forward pass through the layer"""
        raise NotImplementedError
    
    def backward(self, x: tensor.Tensor) -> tensor.Tensor:
        """A training/ backpropagation pass through the layer"""
        raise NotImplementedError
    
# subclass of Layer (takes all of the fns)
class Linear(Layer): 
    def __init__(self, input_size: int, output_size: int):
        """Create a new linear layer"""
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)

    # does a computation on a single layer 
    # takes a vector, multiplies it, outputs vector
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """The forward computation is y = w x + b"""
        self.x = x
        return self.w @ self.x + self.b # @ is multiplication of arrays, built in from numpy
    
    def backward(self, x: tensor.Tensor) -> tensor.Tensor:
        """We will compute the derivative on data passing backwards
        through the network to figure out the step we should take 
        to train our network.
        
        We will compute the gradient for 
        X = w*x + b
        y = f(x)
        dy/dw = f'(X) * x
        dy/dx = f'(X) * w
        dy/db = f'(x)
        
        The new component being added to our variables in tensor form:
        if y = ff(x) and X = x @ w + b and f'(x) is the gradient then
        dy/dx = f'(X) @ w.T
        dy/dw = x.T @ f'(X)
        dy/db = f'(X)"""

class Tanh(Linear):
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        self.x = x 
        return np.tanh(super().forward(x))
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        grad = super().backward(grad)
        y = np.tanh(grad)
        return 1 - y**2
