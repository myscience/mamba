import unittest

import yaml
import torch

from transformers import AutoTokenizer

from mamba import MambaLLM
from mamba.tiny import TinyStoriesLightning

class TestMambaLLM(unittest.TestCase):
    
    def test_llm_forward(self):
        
        vocab_size = 32
        num_layers = 6
        d_input = 16
        d_model = 64
        d_state = 64
        d_discr = 16
        seq_len = 32
        ker_size = 4
        parallel = True
        batch_size = 4
        
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
        
        # Mockup input for example purposes
        tok = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Compute the output using the Mamba architecture
        out, _ = model(tok)
        
        self.assertEqual(out.shape, (batch_size, seq_len, vocab_size))
            
    def test_llm_dataloader(self):
        
        vocab_size = 32
        num_layers = 6
        d_input = 16
        d_model = 64
        d_state = 64
        d_discr = 16
        seq_len = 32
        ker_size = 4
        parallel = True
        batch_size = 4
        
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
        
        # Get the local path to tiny stories
        with open('.local.yaml') as f:
            root = yaml.safe_load(f)['tiny_stories_path']
            
        # Get an off-the-shelf tokenizer
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        
        loader = TinyStoriesLightning(
            root,
            tokenizer,
            max_length=seq_len,
            batch_size=batch_size,
        )
        
        loader.prepare_data()
        batch = next(iter(loader.train_dataloader()))
        
        loss, _ = model(batch)
        
        self.assertTrue((loss >= 0).all())
        self.assertEqual(loss.shape, (batch_size, ))