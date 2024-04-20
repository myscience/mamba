import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from einops import rearrange

from torch.nn.functional import silu
from torch.nn.functional import softplus

from utils import default
from utils import RMSNorm

class Mamba(nn.Module):
    '''
    Class representing the Mamba model as introduced in Gu & Dao (2023)
    (see paper: https://arxiv.org/abs/2312.00752). It is a State Space
    Model with context-dependent capability that matches the performances
    of the strongest Transformer competitor (albeit only tested for small
    scales) while being much more compute efficient.
    '''
    
    def __init__(
        self,
        num_layers : int,
        d_input : int,
        d_model : int,
        d_state : int = 16,
        d_discr : int | None = None,
        ker_size : int = 4,
        parallel : bool = False,
    ) -> None:
        super().__init__()
        
        mamba_par = {
            'd_input' : d_input,
            'd_model' : d_model,
            'd_state' : d_state,
            'd_discr' : d_discr,
            'ker_size': ker_size,
            'parallel': parallel,
        }
        
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
        self.layers = nn.ModuleList([
            (
                MambaBlock(**mamba_par),
                RMSNorm(d_model)
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, seq : Tensor) -> Tensor:
        '''
        Forward pass of the Mamba model.
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, d_seq).
            
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, d_seq).
        '''
        
        for mamba, norm in self.layers:
            # Apply the MambaBlock and normalize the
            # output plus the residual connection
            seq = mamba(norm(seq)) + seq
            
        return seq
        
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
        parallel : bool = False,
    ) -> None:
        '''Initialize the Mamba model.

        Args:
            d_input (int): The dimension of the input sequence.
            d_model (int): The dimension of the model state space.
            d_state (int, optional): The dimension of the state space in the SSM stage. Defaults to 16.
            d_discr (int | None, optional): The dimension of the discrete space in the SSM stage. Defaults to None.
            ker_size (int, optional): The kernel size for the convolutional layer. Defaults to 4.
            parallel (bool, optional): Whether to use parallel scan for the SSM stage. Defaults to False.
        '''
        super().__init__()
        
        d_discr = default(d_discr, d_model // 16)
        
        # Projection matrices from the input sequence space to the
        # model state space (of dimension d_model) and back.
        # NOTE: The in_proj matrix has a factor of 2 because it is
        #       used to split the input sequence into two branches
        self.in_proj  = nn.Linear(d_input, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_input, bias=False)
        
        # Projection matrices for endowing the SSM stage with
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
        
        # Whether to use or not the parallel scan for the SSM
        self.parallel = parallel
        
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
        
        # Compute the state space hidden states
        # NOTE: This can be done either sequentially (slow) or with
        # a parallel scan (fast)
        hid = self._hid_states(A_bar, X, parallel=self.parallel)
    
        # Compute the output based on the hidden states
        out = einsum(hid, C, 'b l d s, b l s -> b l d 1')
    
        out = out + D * seq
        
        return out
    
    def _hid_states(
        self,
        A_bar   : Tensor,
        X       : Tensor,
        parallel: bool = False,
    ) -> Tensor:
        '''
        Calculate the hidden states of the SSM.

        Args:
            A_bar (Tensor): The tensor representing A_bar.
            X (Tensor): The tensor representing X.
            parallel (bool): Whether to use parallel scan or not.

        Returns:
            Tensor: The tensor representing the hidden states.
        '''
        if parallel:
            return self._hid_states_par(A_bar, X)
        else:
            return self._hid_states_seq(A_bar, X)
    
    def _hid_states_par(
            self,
            A_bar : Tensor,
            X     : Tensor,
        ) -> Tensor:
        '''Calculate the hidden states in parallel.

        This method calculates the hidden states in parallel
        (via pscan algorithm) a using the given input tensors.

        Args:
            A_bar (Tensor): The tensor representing A_bar.
            X (Tensor): The tensor representing X.

        Returns:
            Tensor: The tensor representing the calculated hidden states.
        '''
        raise NotImplementedError('Parallel scan not implemented yet.')
    
    def _hid_states_seq(
            self,
            A_bar : Tensor,
            X     : Tensor,
        ) -> Tensor:
        '''
        Calculate the hidden states.

        Args:
            A_bar (Tensor): The A_bar tensor.
            X (Tensor): The X tensor.

        Returns:
            Tensor: The hidden states tensor.
        '''
        b, l, d, s = A_bar.shape
        
        hs = [
            torch.zeros(b, l, d, s, device=self.device),
        ]
        
        for t in range(1, l):
            hs.append(
                A_bar[:, t] * hs[-1] + X[:, t]
            )
            
        # Stuck the hidden states along the sequence length dimension
        return torch.stack(hs, dim=1)

    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        '''
        return next(self.parameters()).device