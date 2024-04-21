import unittest
import torch
from torch import Tensor
from mamba import Mamba

class TestMamba(unittest.TestCase):
    def test_mamba_seq(self):

        d_input = 16
        seq_len = 32
        batch_size = 4

        model = Mamba(
        num_layers = 6,    # Number of layers of the full model
        d_input = d_input, # Dimension of each vector in the input sequence (i.e. token size)
        d_model = 64,      # Dimension of the visible state space
        d_state = 64,      # Dimension of the latent hidden states
        d_discr = 16,      # Rank of the discretization matrix Δ
        ker_size = 4,      # Kernel size of the convolution in the MambaBlock
        parallel = False,  # Whether to use the sequential or the parallel implementation
        )

        # Mockup input for example purposes
        inp = torch.randn(batch_size, seq_len, d_input)

        # Compute the output using the Mamba architecture
        out : Tensor = model(inp) # (batch_size, seq_len, d_input)
        
        self.assertEqual(inp.shape, out.shape)
        
    def test_mamba_par(self):

        d_input = 16
        seq_len = 32
        batch_size = 4

        model = Mamba(
        num_layers = 6,    # Number of layers of the full model
        d_input = d_input, # Dimension of each vector in the input sequence (i.e. token size)
        d_model = 64,      # Dimension of the visible state space
        d_state = 64,      # Dimension of the latent hidden states
        d_discr = 16,      # Rank of the discretization matrix Δ
        ker_size = 4,      # Kernel size of the convolution in the MambaBlock
        parallel = True,   # Whether to use the sequential or the parallel implementation
        )

        # Mockup input for example purposes
        inp = torch.randn(batch_size, seq_len, d_input)

        # Compute the output using the Mamba architecture
        out : Tensor = model(inp) # (batch_size, seq_len, d_input)
        
        self.assertEqual(inp.shape, out.shape)