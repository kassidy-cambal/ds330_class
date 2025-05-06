"""Loss fns are used to train a neural network. These measure the
difference between the predictions and the labels."""

import numpy as np

from DS330_class.cuppajoe import tensor

class Loss():
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        """From predictions and labels, figure out how wrong we are
        
        Returns: 
            float: wrongness
        """
        raise NotImplementedError
    
    def grad(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> tensor:
        """The gradient of the loss fn w/ respect to the predictions"""
        raise NotImplementedError
    
class MSE(Loss):
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor):
        return np.mean((predictions))