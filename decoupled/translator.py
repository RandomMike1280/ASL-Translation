import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module): # basically self-attention
	"""attention, but only one at a time, one head of self-attention"""
	# CHANGE: Renamed 'size' to 'head_size' for clarity and added 'block_size'
	def __init__(self, num_embed, head_size, block_size, dropout):
		super().__init__()
		self.Wk = nn.Linear(num_embed, head_size, bias=False)
		self.Wq = nn.Linear(num_embed, head_size, bias=False)
		self.Wv = nn.Linear(num_embed, head_size, bias=False)
		# CHANGE: 'tril' is now correctly sized based on the maximum sequence length (block_size)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape # batch, time, channels
		k = self.Wk(x) # (B, T, head_size)
		q = self.Wq(x) # (B, T, head_size)
		# compute attention scores
		# The 'C' in the original formula refers to the head dimension, not the embedding dimension
		head_size = q.shape[-1]
		score = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
		# score = (q @ k^T) / sqrt(d_k)
		score = score.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		score = F.softmax(score, dim=-1) # Softmax over the last dimension
		score = self.dropout(score)
		# perform weighted aggregation of the values
		v = self.Wv(x)
		output = score @ v
		return output
	
class MultiHeadAttention(nn.Module):
	"""multiple heads of self-attention, and performs in parallel"""
	# CHANGE: Added 'block_size' to pass to AttentionHead
	def __init__(self, num_embed, num_heads, head_size, block_size, dropout):
		super().__init__()
		# CHANGE: Pass 'block_size' to each head
		self.heads = nn.ModuleList([AttentionHead(num_embed, head_size, block_size, dropout) for _ in range(num_heads)])
		# The output of concatenation is num_heads * head_size, which should equal num_embed
		self.projection = nn.Linear(num_heads * head_size, num_embed)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		output = torch.cat([h(x) for h in self.heads], dim=-1)
		output = self.dropout(self.projection(output))
		return output

class FeedForward(nn.Module):
	"""decompress and compress information"""
	def __init__(self, num_embed, dropout):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(num_embed, 4*num_embed),
			nn.ReLU(),
			nn.Linear(4 * num_embed, num_embed),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.net(x)
	
class Block(nn.Module):
	"""Transformer block, 'Communication followed by computation'"""
	# CHANGE: Added 'block_size' to pass to MultiHeadAttention
	def __init__(self, num_embed, num_head, block_size, dropout):
		# num_embed: is embedding dimension, num_head: is the amount of heads we'd like
		super().__init__()
		head_size = num_embed // num_head
		self.ln1 = nn.LayerNorm(num_embed)
		# CHANGE: Pass 'block_size' to MultiHeadAttention
		self.Att = MultiHeadAttention(num_embed, num_head, head_size, block_size, dropout)
		self.ln2 = nn.LayerNorm(num_embed)
		self.ffwd = FeedForward(num_embed, dropout)

	def forward(self, x):
		# Pre-LayerNorm variant is more common and stable
		x = x + self.Att(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x

class Sinusoidal(nn.Module): # NOTE: This class is defined but not used in your Encoder
    """Sinusoidal positional embedding"""
    def __init__(self, block_size, num_embed):
        super().__init__()
        # ... implementation ...
        
class Encoder(nn.Module):
    """Transformer text encoder that compresses a token sequence into a fixed-size latent vector."""
    def __init__(self, config):
        super().__init__()
        self.name = "ASL-V1"
        self.config = config
        self.latent_dim = config.latent_dim
        self.hidden_size = config.hidden_size
        self.input_dim = config.input_dim

        self.input_proj = nn.Linear(self.input_dim, self.hidden_size)

        # Transformer encoder blocks
        # CHANGE: Pass the max sequence length from config to the Block
        self.layers = nn.ModuleList([Block(
            num_embed=config.hidden_size, 
            num_head=config.num_attention_heads,
            block_size=config.max_position_embeddings,
            dropout=0.1
        ) for _ in range(config.num_layers)])

        self.norm = nn.LayerNorm(self.hidden_size)
        self.latent_proj = nn.Linear(self.hidden_size, self.latent_dim)

    def forward(self, input_vectors: torch.FloatTensor, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None):
        hidden_states = self.input_proj(input_vectors)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        pooled = hidden_states.mean(dim=1)
        latent = self.latent_proj(pooled)
        return latent

if __name__ == "__main__":
    class ModelConfig:
        hidden_size = 256
        num_attention_heads = 8
        # head_dim = 64 # This is redundant as hidden_size / num_attention_heads = 32
        max_position_embeddings = 1024 # This is our block_size
        num_layers = 2
        latent_dim = 1024
        input_dim = 256

    config = ModelConfig()
    model = Encoder(config)

    batch_size = 2
    seq_len = 384
    input_vectors = torch.randn(batch_size, seq_len, config.input_dim)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    latent_vector = model(input_vectors=input_vectors, position_ids=position_ids)
    print(f"Output latent vector shape: {latent_vector.shape}")
    print("Model ran successfully!")