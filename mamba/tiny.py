import torch
from os import path
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from typing import Literal, Tuple
from transformers import PreTrainedTokenizerBase

from .data import LightningDataset

class TinyStories(Dataset):
    
    def __init__(
        self,
        root : str,
        tokenizer : PreTrainedTokenizerBase,
        max_length : int = 256,
        data_split : Literal['train', 'valid', 'test'] = 'train',
    ) -> None:
        super().__init__()
        
        # Tokenize the stories
        text_path = path.join(root, f'{data_split}.txt')
        with open(text_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        
        tokens = tokenizer.encode(raw, allowed_special={"<|endoftext|>"}) # type: ignore
        
        self.prev = torch.tensor(tokens    ).chunk(max_length)
        self.next = torch.tensor(tokens[1:]).chunk(max_length)

    def __len__(self) -> int:
        return len(self.prev)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.prev[index], self.next[index]

class TinyStoriesLightning(LightningDataset):
    '''Lightning Dataset class for the Tiny Stories dataset. The Tiny
    Stories dataset is a small dataset of short stories, each consisting
    of a few sentences. The dataset is used for training a language model.
    '''
    
    def __init__(
        self,
        root : str,
        tokenizer : PreTrainedTokenizerBase,
        max_length : int = 256,
        **kwargs,    
    ) -> None:
        super().__init__(**kwargs)
        
        self.root = root
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def setup(self, stage: str) -> None:

        match stage:
            case 'fit':
                self.train_dataset = TinyStories(
                    root=self.root,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    data_split='train',
                )
                self.valid_dataset = TinyStories(
                    root=self.root,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    data_split='valid',
                )
            case 'test':
                self.test__dataset = TinyStories(
                    root=self.root,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    data_split='test',
                )
            case _:
                raise ValueError(f'Invalid stage: {stage}')