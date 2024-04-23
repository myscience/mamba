# Mamba Model in Easy Pytorch 🐍

This repo contains the _unofficial_ PyTorch implementation of the `Mamba` model as introduced in [Gu & Dao (2023)](https://arxiv.org/abs/2312.00752). One can find the official (CUDA) implementation [here](https://github.com/state-spaces/mamba) or the nice [alxndrTL](https://github.com/alxndrTL/mamba.py) alternative Python implementation. This repo is developed mainly for didactic purposes to spell out the details of a `State Space Models`.

The main contribution of this repo is the implementation of the `Mamba` architecture (upon which a basic LLM is built, see `📂 mamba.llm`) using the [Lightning Framework](https://lightning.ai/docs/pytorch/stable/) which abstract away the difficulty of training on distributed settings.

# Usage

The basic usage is to instantiate a `Mamba` model and run it on the input of choice. The model expects the input tensor to have shape `(batch_size, seq_len, d_input)` and outputs a similarly shaped tensor.

```python
from mamba import Mamba

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
out = model(inp) # (batch_size, seq_len, d_input)
```

This repo also offers a didactic implementation of an LLM built using Lightning which unlocks easy training on multi-gpus. To use it one can simply run the following example:

```python
from lightning import Trainer
from transformers import AutoTokenizer

from mamba import MambaLLM
from mamba.tiny import TinyStoriesLightning

config = ... # path to YAML configuration file

# Load an off-the-shelf tokenizer from HF
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

# Load the Mamba model from a config file
model = MambaLLM.from_config(config, key='mamba')

# Load the dataset
dataset = TinyStoriesLightning.from_config(
    config,
    tokenizer,
    key='dataset'
)

trainer = Trainer(
  max_epochs  = 500,
  accelerator = 'gpu',
  devices     = 4, # Piece of cake multi-gpu support!
  strategy    = 'ddp_find_unused_parameters_false',
)

# Train the model
trainer.fit(model, dataset)
```

Alternatively, one can also run the training script `train.py` directly which expects the configuration file path and accepts all the Trainer arguments.

```bash
python train.py --config <path_to_config_file>\
  --max_epochs 500\
  --accelerator gpu\
  --devices 4
```

A cool feature of `MambaLLM` implementation is the lazy (batched-) inference implemented via a generator. One can thus print tokens on screen as they are streamed by the model, no need to wait for the whole inference to finish! A mock-up script would look like the following.

```python
from mamba import MambaLLM
from transformers import AutoTokenizer

# Get an off-the-shelf tokenizer
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

# Parameters for the LLM
vocab_size = tokenizer.vocab_size + 1
num_layers = 6
d_input = 16
d_model = 64
d_state = 64
d_discr = 16
ker_size = 4
parallel = True

model = MambaLLM(
    vocab_size = vocab_size,
    num_layers = num_layers,
    d_input = d_input,
    d_model = d_model,
    d_state = d_state,
    d_discr = d_discr,
    ker_size = ker_size,
    parallel = parallel
)

# Parameters for the inference
token_lim = 16
use_top_k = 50
temperature = 0.7

# Generate text
stream = model.generate(
  # We can provide more than one prompt!
  prompt=[
      'Once upon a time',
      'In a galaxy far far away',
  ],
  tokenizer=tokenizer,
  token_lim=token_lim,
  use_top_k=use_top_k,
  temperature=temperature,
)

for token in stream:
    # Each token is a dictionary indexed by the
    # batch-id and contains the produced string
    # as value, so we can print the first batch as:
    print(token[0], end='')
```

# Roadmap

- [x] Put all the essential pieces together
- [x] Add functioning parallel implementation (p-scan) (🙏🏻 @ [Zeta Project](https://github.com/kyegomez/zeta/blob/be1c7e14d6c5a78f7d558ad919ec774a5f018042/zeta/nn/modules/p_scan.py) & [alxndrTL](https://github.com/alxndrTL/mamba.py/tree/main))
- [x] Add functioning training script (Lightning)
- [ ] Show some results

# Requirements

This repo build on top of some cool libraries, one can easily install the dependencies via `pip install -r requirements.txt`

```
einops==0.7.0
lightning==2.2.2
PyYAML==6.0
PyYAML==6.0.1
torch==2.0.1
transformers==4.40.0
```

# Citations

Some of the code is inspired by the great repo [LLM From Scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main) that was used as a guide to elucidate several doubts along the way.

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
