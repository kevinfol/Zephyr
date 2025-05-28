import numpy as np
from . import NeuralNetworkRegression
import torch


class QuantileNeuralNetworkRegression(NeuralNetworkRegression):

    def __init__(self, *args, **kwargs):

        NeuralNetworkRegression.__init__(self, *args, **kwargs)
        self.loss_function = self.quantile_loss

    def quantile_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the 50% quantile loss

        Args:
            output (torch.Tensor): predicted values from the network
            target (torch.Tensor): observed values

        Returns:
            torch.Tensor: loss tensor
        """

        loss = 0.5 * (target - output)
        return 2 * loss
