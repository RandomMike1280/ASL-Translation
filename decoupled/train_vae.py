from vae import Encoder, Decoder, DeepSeekV3Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
import pickle
from tokenizer import Tokenizer

### PARAMETERS ###
num_epochs = 1
batch_size = 1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderConfig(DeepSeekV3Config):
    """Configuration for the VAE Encoder."""
    def __init__(self):
        super().__init__()
        # Small encoder config (~million params)
        self.hidden_size = 128
        self.intermediate_size = 512
        self.num_attention_heads = 4
        self.head_dim = 32
        self.kv_compress_dim = 64
        self.q_compress_dim = 128
        self.rope_dim = 16
        self.num_layers = 2
        self.vocab_size = 100000
        self.latent_dim = 512
        self.num_experts = 2
        self.moe_top_k = 1
        self.max_position_embeddings = 512
        self.token_embedding_type = 'tokens'

class DecoderConfig(DeepSeekV3Config):
    """Configuration for the VAE Decoder."""
    def __init__(self):
        super().__init__()
        # Small decoder config (~million params)
        self.hidden_size = 128
        self.intermediate_size = 512
        self.num_attention_heads = 4
        self.head_dim = 32
        self.kv_compress_dim = 64
        self.q_compress_dim = 128
        self.rope_dim = 16
        self.num_layers = 2
        self.vocab_size = 100000
        self.num_experts = 2
        self.moe_top_k = 1
        self.max_position_embeddings = 512
        self.token_embedding_type = 'tokens'
        self.latent_dim = 512

ds = load_dataset("agentlans/high-quality-english-sentences")
tokenizer = Tokenizer(level='sub_word', max_vocab_size=100000, bpe_iterations=10)
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

class Dataset:
    def __init__(self, ds, tokenizer, max_length=512):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[int(idx)]['text']
        tokens = tokenizer.encode(text)
        return torch.tensor(tokens, dtype=torch.long)

dataloader = DataLoader(Dataset(ds['train'], tokenizer), batch_size=batch_size, shuffle=True)

encoder = Encoder(EncoderConfig()).to(device)
decoder = Decoder(DecoderConfig()).to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# for key, value in tokenizer.vocab.items():
#     print(f"  {key}: {value}")
for num_e in range(num_epochs):
    print(f"Epoch {num_e+1}/{num_epochs}")
    for i, batch in enumerate(dataloader):
        print(batch)
        print(tokenizer.decode(list(batch[0].detach().cpu().numpy())))
        break
        
    