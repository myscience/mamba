import torch
import torch.nn as nn
from torch import Tensor
from typing import TypeVar

T = TypeVar('T')
D = TypeVar('D')

def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

# This implementation of RMSNorm is taken directly from:
# https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    '''
    A module that performs RMS normalization on the input tensor.

    Args:
        d_model (int): The size of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-8.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (torch.Tensor): A learnable parameter used to scale the normalized input tensor.

    Methods:
        forward(x): Performs RMS normalization on the input tensor.

    Example:
        >>> rms_norm = RMSNorm(d_model=512)
        >>> input_tensor = torch.randn(10, 512)
        >>> output_tensor = rms_norm(input_tensor)
    '''
    
    def __init__(
        self,
        d_model : int,
        eps : float = 1e-8,
    ) -> None:
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x : Tensor) -> Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output