'''
    Implementation of the parallel scan algorithm in PyTorch.
    This version is basically ported as-is from the codes in:
    - https://github.com/alxndrTL/mamba.py/blob/main/pscan.py
    - https://github.com/kyegomez/zeta/blob/be1c7e14d6c5a78f7d558ad919ec774a5f018042/zeta/nn/modules/p_scan.py
    to which all the credit goes.
'''

import math
import torch

from torch import Tensor
from einops import rearrange

from typing import Callable, Tuple

from torch.autograd import Function

class PScan(Function):
    '''
    Implementation of the parallel scan algorithm in PyTorch for
    the particular case of the cumulative filtering needed by the
    mamba architecture in its SSM stage.
    '''
    
    @staticmethod
    def forward(
        ctx,
        A_inp: Tensor,
        X_inp: Tensor,
    ) -> Tensor:
        '''Forward pass of the pscan module.

        This method performs the forward pass of the pscan module.
        It takes in two input tensors, A and X, and returns a tensor
        as output containing the result of the following operation:
        
        Y[t] = A[t] * Y[t - 1] + X[t]

        Args:
            ctx (_type_): The context object.
            A (Tensor): The input tensor A of expected shape:
                (seq_len, batch_size, d_model, d_state).
            X (Tensor): The input tensor X of expected shape:
                (seq_len, batch_size, d_model, d_state).

        Returns:
            Tensor: The result of the parallel scan.
        '''
        
        # Clone the tensors because we will modify them in-place
        A = A_inp.clone()
        X = X_inp.clone()
        
        A = rearrange(A, 'l b d s -> b d l s')
        X = rearrange(X, 'l b d s -> b d l s')
        
        # Perform the parallel scan, which modifies the input tensors in-place
        PScan._forward(A, X)
        
        ctx.save_for_backward(A.clone(), X)
        
        return rearrange(X, 'b d l s -> b l d s')

    @staticmethod
    # TODO: Understand the implementation of the backward pass
    def backward(ctx, grad_inp: Tensor) -> Tuple[Tensor, Tensor]:
        '''Implements the backward pass for the pscan module.
        Tells the gradient how to propagate through the pscan module.

        Args:
            ctx (A, X): Saved tensors from the forward pass.
                A_in: The input tensor A of expected shape:
                    (seq_len, batch_size, d_model, d_state).
                X: The input tensor X of expected shape:
                    (seq_len, batch_size, d_model, d_state).
            grad_outputs (Tensor): The incoming gradients

        Returns:
            Tuple of Tensor: Gradients with respect to the A and X tensors.
                grad_A: The gradient with respect to A.
                grad_X: The gradient with respect to X.
                both tensor have the same shape as the input tensors.
        '''
        
        A, X = ctx.saved_tensors
        
        # Reverse both the A and grad tensor along the sequence dim
        # NOTE: Apparently A needs to be "shifted by one" to the right
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        
        grad_out = rearrange(grad_inp, 'b l d s -> b d l s')
        
        # Perform the reverse parallel scan
        grad_out = grad_out.flip(2)
        PScan._forward(A, grad_out)
        grad_out = grad_out.flip(2)
        
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_out[:, :, 1:])
        
        Q = rearrange(Q, 'b d l s -> b l d s')
        grad_out = rearrange(grad_out, 'b d l s -> b l d s')

        return Q, grad_out
        
    
    @staticmethod
    def _forward(A: Tensor, X: Tensor) -> None:
        '''Perform the forward pass of the parallel scan algorithm.
        Modify the input tensors in-place.

        Args:
            A (Tensor): Tensor of expected shape (batch_size, d_model, seq_len, d_state).
            X (Tensor): Tensor of expected shape (batch_size, d_model, seq_len, d_state).
        '''
        
        # Get the dimensions of the input tensors
        b, d, l, s = A.shape
        
        num_steps = int(math.log2(l))
        
        # * Upsweep phase of the scan (going up the three)
        Av = A
        Xv = X
        for _ in range(num_steps):
            T = Xv.size(2)
            
            Av = Av[:, :, :T].reshape(b, d, T // 2, 2, -1)
            Xv = Xv[:, :, :T].reshape(b, d, T // 2, 2, -1)
            
            Xv[:, :, :, 1].add_(Av[:, :, :, 1].mul(Xv[:, :, :, 0]))
            Av[:, :, :, 1].mul_(Av[:, :, :, 0])
            
            Av = Av[:, :, :, 1]
            Xv = Xv[:, :, :, 1]
            
        # * Downsweep phase of the scan (going down the three)
        for k in range(num_steps - 1, -1, -1):
            Av = A[:, :, 2**k - 1 : l : 2**k]
            Xv = X[:, :, 2**k - 1 : l : 2**k]
            
            T = 2 * (Xv.size(2) // 2)

            if T < Xv.size(2):
                Xv[:, :, -1].add_(Av[:, :, -1].mul(Xv[:, :, -2]))
                Av[:, :, -1].mul_(Av[:, :, -2])

            Av = Av[:, :, :T].reshape(b, d, T // 2, 2, -1)
            Xv = Xv[:, :, :T].reshape(b, d, T // 2, 2, -1)

            Xv[:, :, 1:, 0].add_(Av[:, :, 1:, 0].mul(Xv[:, :, :-1, 1]))
            Av[:, :, 1:, 0].mul_(Av[:, :, :-1, 1])
    
pscan : Callable[[Tensor, Tensor], Tensor] = PScan.apply # type: ignore