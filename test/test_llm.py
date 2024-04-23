import unittest


import yaml
import torch
from os import path

from transformers import AutoTokenizer

from mamba import MambaLLM
from mamba.tiny import TinyStoriesLightning
from mamba.utils import default_iterdata_worker_init

# Loading `local_settings.json` for custom local settings
test_folder = path.dirname(path.abspath(__file__))
local_settings = path.join(test_folder, '.local.yaml')

class TestMambaLLM(unittest.TestCase):
    
    def test_llm_forward(self):
        
        vocab_size = 24
        num_layers = 6
        d_input = 16
        d_model = 64
        d_state = 42
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
        out, _ = model.forward(tok)
        
        self.assertEqual(out.shape, (batch_size, seq_len, vocab_size))
            
    def test_llm_dataloader(self):
        
        # Get the local path to tiny stories
        with open(local_settings, 'r') as f:
            root = yaml.safe_load(f)['tiny_stories_path']
            
        # Get an off-the-shelf tokenizer
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        
        vocab_size = tokenizer.vocab_size
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
        
        loader = TinyStoriesLightning(
            root,
            tokenizer,
            max_length=seq_len,
            batch_size=batch_size,
            worker_init_fn=default_iterdata_worker_init,
        )
        
        loader.setup(stage='fit')
        batch = next(iter(loader.train_dataloader()))
        
        prev, post = batch
        
        logits, _ = model(prev)
        
        loss = model.compute_loss(prev, post)
        
        self.assertTrue((loss >= 0).all())
        self.assertEqual(logits.shape, (*prev.shape, vocab_size))
        
    def test_llm_generate(self):
        # Get an off-the-shelf tokenizer
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        
        vocab_size = tokenizer.vocab_size + 1
        num_layers = 6
        d_input = 16
        d_model = 64
        d_state = 64
        d_discr = 16
        ker_size = 4
        parallel = True
        token_lim = 16
        
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
        
        # Generate text
        gen = model.generate(
            prompt=[
                'Once upon a time',
                'In a galaxy far far away',
            ],
            tokenizer=tokenizer,
            token_lim=token_lim,
        )
        
        for tok in gen:
            print(tok[0], end='')
        
        self.assertTrue(True)