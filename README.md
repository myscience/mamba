# Mamba Model in Easy Pytorch 🐍

This repo contains the _unofficial_ PyTorch implementation of the `Mamba` model as introduced in [Gu & Dao (2023)](https://arxiv.org/abs/2312.00752). One can find the official (CUDA) implementation [here](https://github.com/state-spaces/mamba) or the nice [alxndrTL](https://github.com/alxndrTL/mamba.py) alternative Python implementation. This repo is developed mainly for didactic purposes to spell out the details of a `State Space Models`.

# Usage

The basic usage is to instantiate a `Mamba` model and run it on the input of choice. The model expects the input tensor to have shape `(batch_size, seq_len, d_input)` and outputs a similarly shaped tensor.

```python
from mamba import Mamba

d_input = 16
seq_len = 32
batch_size = 4

model = Mamba(
  num_layers = 6,   # Number of layers of the full model
  d_input = d_input # Dimension of each vector in the input sequence (i.e. token size)
  d_model = 64,     # Dimension of the visible state space
  d_state = 64,     # Dimension of the latent hidden states
  d_discr = 16,     # Rank of the discretization matrix Δ
  ker_size = 4,     # Kernel size of the convolution in the MambaBlock
  parallel = False, # Whether to use the sequential or the parallel implementation
)

# Mockup input for example purposes
inp = torch.randn(batch_size, seq_len, d_input)

# Compute the output using the Mamba architecture
out = model(inp) # (batch_size, seq_len, d_input)
```

# Roadmap

- [x] Put all the essential pieces together
- [x] Add functioning parallel implementation (p-scan) (🙏🏻 @ [Zeta Project](https://github.com/kyegomez/zeta/blob/be1c7e14d6c5a78f7d558ad919ec774a5f018042/zeta/nn/modules/p_scan.py) & [alxndrTL](https://github.com/alxndrTL/mamba.py/tree/main))
- [ ] Add functioning training script (Lightning)
- [ ] Show some results

# Citations

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
