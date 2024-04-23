import yaml
import torch
import torch.nn as nn
from lightning import LightningModule

from torch import Tensor, NumberType
from torch.optim import AdamW, Optimizer
from einops import rearrange
from torch.nn.functional import pad
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from transformers import PreTrainedTokenizerBase

from .utils import Cache
from .utils import RMSNorm
from .mamba import MambaBlock

from typing import Dict, List, Tuple, Generator

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
    def from_config(cls, conf_path : str, key : str | None = 'llm') -> 'MambaLLM':
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
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.inference_kw = kwargs
        
        self.mamba_par = {
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
            embedding_dim=d_input
        )
        
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
        self.llm = nn.ModuleList([
            nn.ModuleList(
                [
                    MambaBlock(**self.mamba_par),
                    RMSNorm(d_input)
                ]
            )
            for _ in range(num_layers)
        ])
        
        # Prediction head to map the output of the Mamba model to the vocabulary
        self.head = nn.Linear(d_input, vocab_size, bias=False)
        
        self.save_hyperparameters()
        
    def forward(self, tok : Tensor, cache : Cache = None) -> Tuple[Tensor, Cache]:
        '''
        Forward pass of the Mamba model.
        
        Args:
            tok (Tensor): Input sequence of word tokens, has expected
                shape: (batch_size, seq_len).
            cache (Tensor, optional): Cache tensor to store the hidden states
                of the model. Default is None.
            
        Returns:
            Tensor: Predicted logits. If cache was provided return tensor has
                shape: (batch_size, vocab_size), while if no cache was provided
                output shape is: (batch_size, seq_len, vocab_size).
        '''
        
        tok = torch.atleast_2d(tok)
        seq = self.embedding(tok)
        
        for mamba, norm in self.llm: # type: ignore
            # Apply the MambaBlock and normalize the
            # output plus the residual connection
            out, cache = mamba(norm(seq), cache)
            seq = out + seq
            
        logits = self.head(seq)
            
        return logits, cache
    
    @torch.no_grad()
    def generate(
        self,
        prompt : str | List[str],
        tokenizer : PreTrainedTokenizerBase, 
        token_lim : int = 300,
        use_top_k : int = 50,
        temperature : float = 1.0,
    ) -> Generator[Dict[int, str], None, None]:
        # Set model in evaluation model for inference
        self.eval()
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Encode the prompt using the tokenizer
        inp = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).input_ids
        
        batch_size, inp_len = inp.shape
        vocab_size = tokenizer.vocab_size # type: ignore
        
        d_model, ker_size = self.mamba_par['d_model'], self.mamba_par['ker_size']
        cache = (None, torch.zeros(batch_size, d_model, ker_size - 1, device=self.device))
        
        # Consume the prompt to get the hidden states
        for tok in rearrange(inp, 'b s -> s b 1'):
            logits, cache = self(tok, cache)
        
        # Start generating the output sequence until either the maximum
        # token limit is reach or the model generates the<|endoftext|> token
        num_tokes = 0
        out, pred = [inp], tok
        pidx = torch.arange(batch_size)
        
        yield {int(pid) : tokenizer.decode(raw, skip_special_tokens=True) for pid, raw in zip(pidx, inp)}
        
        while num_tokes < token_lim and len(pred):
            logits, cache = self(pred, cache)
            
            # Get the token with the highest probability by zeroing out
            # the probability of the lowest probability tokens
            prob = softmax(logits[:, -1] / temperature, dim=-1)
            idxs = prob.topk(k=vocab_size - use_top_k, largest=False, sorted=False).indices
            prob.scatter_(dim=-1, index=idxs, src=torch.zeros_like(prob))
            prob /= prob.sum(dim=-1, keepdim=True)
            
            # Sample the next token from the distribution modelled by the llm
            pred = torch.multinomial(prob, num_samples=1, replacement=True)
            
            # Append the token to the input sequence
            out.append(pred)
            
            num_tokes += 1
            
            # Drop from the batch every prediction that reached the <|endoftext|> token
            mask = pred.squeeze() != tokenizer.eos_token_id
            
            pred  = pred[mask]
            pidx  = pidx[mask]
            cache = (cache[0][mask], cache[1][mask])
            
            # Yield the decoded tokens
            yield {int(pid) : tokenizer.decode(raw, skip_special_tokens=True) for pid, raw in zip(pidx, pred)}
        
        self.train()
    
    def compute_loss(self, prev : Tensor, post : Tensor) -> Tensor:
        # Compute model predictions for the previous tokens
        pred, _ = self(prev)

        pred = rearrange(pred, 'b s v -> (b s) v')
        post = rearrange(post, 'b s -> (b s)')
        
        # Compute the loss using the cross entropy loss
        loss = cross_entropy(pred, post)
        
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
    
    def on_validation_end(self) -> None:
        
        inference_kw = {
            'prompt' : 'Once upon a time',
            'tokenizer' : self.tokenizer,
            **self.inference_kw
        }
        
        # Generate the model output on the given prompt
        output = list( # List needed to consume the generator
            self.generate(
                **inference_kw
            )
        )
        
        # Assemble the outputs based on the batch id
        pids = list(output[0].keys())
        output = {pid : ''.join([out[pid] for out in output]) for pid in pids}
        
        for pid, text in output.items():
            self.logger.experiment.add_text({ # type: ignore
                    f'Prompt {pid}' : text
                },
                global_step=self.global_step,
            )
    
    def configure_optimizers(self) -> Optimizer:
        optim = AdamW(
            self.parameters(),
            lr=1e-3
        )
        
        return optim