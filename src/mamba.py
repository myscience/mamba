import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from einops import rearrange

from torch.nn.functional import silu
from torch.nn.functional import softplus

from utils import default

class Mamba(nn.Module):
    '''
    Class representing the Mamba model as introduced in Gu & Dao (2023)
    (see paper: https://arxiv.org/abs/2312.00752). It is a State Space
    Model with context-dependent capability that matches the performances
    of the strongest Transformer competitor (albeit only tested for small
    scales) while being much more compute efficient.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
class MambaBlock(nn.Module):
    '''
    Class representing the MambaBlock as introduced in Gu & Dao (2023).
    '''
    
    def __init__(
        self, 
        d_input : int,
        d_model : int,
        d_state : int = 16,
        d_discr : int | None = None,
        ker_size : int = 4,
    ) -> None:
        super().__init__()
        
        d_discr = default(d_discr, d_model // 16)
        
        # Projection matrices from the input sequence space to the
        # model state space (of dimension d_model) and back.
        # NOTE: The in_proj matrix has a factor of 2 because it is
        #       use to split the input sequence into two branches
        self.in_proj  = nn.Linear(d_input, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_input, bias=False)
        
        # Projection matrices for  endowing the SSM stage with
        # context-dependent capability (i.e. input dependence)
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_D = nn.Sequential(
            nn.Linear(d_model, d_discr, bias=False), # Fixing matrix rank to d_disc
            nn.Linear(d_discr, d_model, bias=False),
        )
        
        
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=ker_size,
            padding=ker_size - 1,
            groups=d_model,
            bias=True,
        )
        
        # Parameters for the SSM. Follows the S4 initialization
        self.A = nn.Parameter(torch.arange(1, d_state + 1).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, seq : Tensor) -> Tensor:
        '''
        Forward pass of the MambaBlock.
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, d_seq).
            
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, d_seq).
        '''
        
        # Project the input sequence from d_seq to d_model and into two
        # distinct branches, one for the SSM and the residual branch
        # (see Fig. 3 of the Mamba paper). The resulting shapes are:
        # a: (batch_size, seq_len, d_model), b: (batch_size, seq_len, d_model)
        a, b = self.in_proj(seq).chunk(2, dim=-1)
        
        # * The SSM branch
        # Apply the convolutional layer to the SSM branch
        # NOTE: We need to move the channel dimension to the second dimension
        #       for the convolution to work properly, hence the rearrange
        a = rearrange(a, 'b l d -> b d l')
        a = self.conv(a)
        a = rearrange(a, 'b d l -> b l d')
        
        # Apply the SSM
        a = silu(a)
        a = self.ssm(a) 
        
        # * The residual branch
        b = silu(b)
        
        # Combine the two branches
        out = a * b
        
        return self.out_proj(out)
    
    def ssm(self, seq : Tensor) -> Tensor:
        '''
        State Space Model (SSM) of the MambaBlock.
        
        Args:
            seq (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        '''
        
        # Compute the context-dependent projections
        A = -self.A # shape: (d_model, d_state)
        D = +self.D # shape: (d_model, )
        
        B = self.s_B(seq)               # shape: (batch_size, seq_len, d_state)
        C = self.s_C(seq)               # shape: (batch_size, seq_len, d_state)
        Δ = softplus(D + self.s_D(seq)) # shape: (batch_size, seq_len, d_model)
        
        # Discretize the A and B parameters using Δ
        A_bar = einsum(torch.exp(A), Δ, 'd s,   b l d -> b l d s')
        B_bar = einsum(          B,  Δ, 'b l s, b l d -> b l d s')
        
        X = einsum(B_bar, seq, 'b l d s, b l d -> b l d s')
        
        # Compute the state sequence
        raise NotImplementedError('SSM computation not implemented yet')
    
        out = out + D * seq
        
        return out