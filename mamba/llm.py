import yaml
import torch.nn as nn
from lightning import LightningModule

from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.nn.functional import cross_entropy

from .utils import RMSNorm
from .mamba import MambaBlock

from typing import Tuple

class MambaLLM(LightningModule):
    '''
    Class representing a (Pytorch Lightning) Large Language Model based
    on the Mamba architecture as introduced in Gu & Dao (2023)
    (see paper: https://arxiv.org/abs/2312.00752). Mamba is a State Space
    Model with context-dependent capability that matches the performances
    of the strongest Transformer competitor (albeit only tested for small
    scales) while being much more compute efficient.
    '''
    
    @classmethod
    def from_config(cls, conf_path : str, key : str | None = 'LLM') -> 'MambaLLM':
        '''
        Construct a MambaLLM from a configuration file.
        '''

        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)

        conf = conf if key is None else conf[key]

        return cls(
            **conf,
        )
    
    def __init__(
        self,
        vocab_size : int,
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
        
        # Needed embedding layer for mapping input tokens to the network
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
        
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
        self.llm = nn.ModuleList([
            nn.ModuleList(
                [
                    MambaBlock(**mamba_par),
                    RMSNorm(d_input)
                ]
            )
            for _ in range(num_layers)
        ])
        
        # Prediction head to map the output of the Mamba model to the vocabulary
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.save_hyperparameters()
        
    def forward(self, tok : Tensor, cache : Tensor | None = None) -> Tuple[Tensor, Tensor | None]:
        '''
        Forward pass of the Mamba model.
        
        Args:
            tok (Tensor): Input sequence of word tokens, has expected
                shape: (batch_size, seq_len, vocab_size).
            cache (Tensor, optional): Cache tensor to store the hidden states
                of the model. Default is None.
            
        Returns:
            Tensor: Predicted logits. If cache was provided return tensor has
                shape: (batch_size, vocab_size), while if no cache was provided
                output shape is: (batch_size, seq_len, vocab_size).
        '''
        
        seq = self.embedding(tok)
        
        for mamba, norm in self.llm: # type: ignore
            # Apply the MambaBlock and normalize the
            # output plus the residual connection
            out, cache = mamba(norm(seq), cache)
            seq = out + seq
            
        logits = self.head(seq)
            
        return logits, cache
    
    def compute_loss(self, prev : Tensor, next : Tensor) -> Tensor:
        # Compute model predictions for the previous tokens
        pred, _ = self(prev)
        
        # Compute the loss using the cross entropy loss
        loss = cross_entropy(pred, next)
        
        return loss
    
    def training_step(self, batch : Tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        prev_tok, next_tok = batch
        
        loss = self.compute_loss(prev_tok, next_tok)

        self.log_dict(
            {'train_loss' : loss},
            logger=True,
            on_step=True,
            sync_dist=True
        )
        
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        prev_tok, next_tok = batch
        
        loss = self.compute_loss(prev_tok, next_tok)

        self.log_dict(
            {'val_loss' : loss},
            logger=True,
            on_step=True,
            sync_dist=True
        )
        
        return loss
    
    def configure_optimizers(self) -> Optimizer:
        optim = AdamW(
            self.parameters(),
            lr=1e-3
        )
        
        return optim