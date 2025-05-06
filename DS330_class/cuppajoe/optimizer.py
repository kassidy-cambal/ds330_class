""""An optimizer update sthe paramaters of the layers based on the output
of the gradient."""

from DS330_class.cuppajoe import mlp

class Optimizer():
    def __init__(self, neural_network: mlp.MLP, learning_rate: float = 0.01):
        self.net = neural_network
        self.lr

    def step(self):
        """Take a step forward, backpropagate the error."""
        raise NotImplemented
    
class SGD(Optimizer):
    """Stochastic gradient descent optimizer"""
    def step(self):
        for param, grad in self.net.params_and_grads():
            param -= grad*self.lr
