import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dims:list, output_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x / (x.norm(keepdim=True, dim=-1) + self.eps))

class RoPEEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        self.dim = dim
        # The 'dim' passed here is expected to be rope_dim per head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Create position indices
        position = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum('i,j->ij', position, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()) # [max_pos, rope_dim]
        self.register_buffer('sin_cached', emb.sin()) # [max_pos, rope_dim]

    def forward(self, x, position_ids, num_heads):
        # x: [batch_size, num_heads, seq_len, head_dim] (unused but kept for potential future use)
        # position_ids: [batch_size, seq_len]
        # num_heads: int
        # Output shape needs to be [bs, num_heads, seq_len, rope_dim]
        cos = self.cos_cached[position_ids]  # [bs, seq_len, rope_dim]
        sin = self.sin_cached[position_ids]  # [bs, seq_len, rope_dim]
        # Expand num_heads dimension
        cos = cos.unsqueeze(1).expand(-1, num_heads, -1, -1) # [bs, num_heads, seq_len, rope_dim]
        sin = sin.unsqueeze(1).expand(-1, num_heads, -1, -1) # [bs, num_heads, seq_len, rope_dim]
        return cos, sin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Crop rotary embeddings if cos/sin sequence length > q's seq_len
    seq_len = q.size(2)
    if cos.size(2) != seq_len:
        cos = cos[:, :, :seq_len, :]
        sin = sin[:, :, :seq_len, :]

    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [batch_size, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim  # 128
        self.kv_compress_dim = config.kv_compress_dim  # 512
        self.q_compress_dim = config.q_compress_dim  # 1536
        self.rope_dim = config.rope_dim  # 64 per head
        
        # Compression matrices
        self.W_DKV = nn.Linear(self.hidden_size, self.kv_compress_dim)
        self.W_DQ = nn.Linear(self.hidden_size, self.q_compress_dim)
        
        # Up-projection matrices
        self.W_UK = nn.Linear(self.kv_compress_dim, self.num_heads * self.head_dim)
        self.W_UV = nn.Linear(self.kv_compress_dim, self.num_heads * self.head_dim)
        self.W_UQ = nn.Linear(self.q_compress_dim, self.num_heads * self.head_dim)
        
        # RoPE projection matrices
        self.W_KR = nn.Linear(self.hidden_size, self.num_heads * self.rope_dim)
        self.W_QR = nn.Linear(self.q_compress_dim, self.num_heads * self.rope_dim)
        
        # Output projection
        self.W_O = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        
        # RoPE embedding - Initialize with rope_dim per head
        self.rope = RoPEEmbedding(self.rope_dim, config.max_position_embeddings)
        # Initialize cache for compressed KV and RoPE
        self.register_buffer('cache_c_kv', None)
        self.register_buffer('cache_k_rope', None)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # KV Path
        c_kv = self.W_DKV(hidden_states)  # [B, T_new, dc_KV]
        # Use cached c_kv if available
        if self.cache_c_kv is not None:
            # Concatenate along the sequence dimension
            c_kv = torch.cat([self.cache_c_kv, c_kv], dim=1) # [B, T_cache + T_new, dc_KV]
        # Update cache for next iteration
        self.cache_c_kv = c_kv.detach()
        
        # Get the actual sequence length after potential caching
        current_seq_len = c_kv.shape[1]
        
        k_latent = self.W_UK(c_kv)  # [B, T_current, nh*dh]
        v_latent = self.W_UV(c_kv)  # [B, T_current, nh*dh]
        
        # k_rope_proj needs to be calculated based on the *new* hidden_states, not cached ones
        k_rope_proj = self.W_KR(hidden_states)  # [B, T_new, nh*d_R']
        
        # Reshape for RoPE (using T_new)
        k_rope = k_rope_proj.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        k_rope = k_rope.transpose(1, 2)  # [B, nh, T_new, d_R']
        # Use cached k_rope if available
        if self.cache_k_rope is not None:
            # Concatenate along the sequence dimension (dim=2 for [B, nh, T, d_R'])
            k_rope = torch.cat([self.cache_k_rope, k_rope], dim=2) # [B, nh, T_cache + T_new, d_R']
        # Update cache for next iteration
        self.cache_k_rope = k_rope.detach()
        
        # Query Path (only uses current hidden_states)
        c_q = self.W_DQ(hidden_states)  # [B, T_new, dc_Q]
        q_latent = self.W_UQ(c_q)  # [B, T_new, nh*dh]
        q_rope_proj = self.W_QR(c_q)  # [B, T_new, nh*d_R']
        
        # Reshape for RoPE (using T_new)
        q_rope = q_rope_proj.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        q_rope = q_rope.transpose(1, 2)  # [B, nh, T_new, d_R']
        
        # Reshape latent vectors using the correct sequence lengths
        # Q uses T_new (seq_len), K/V use T_current (current_seq_len)
        q_latent = q_latent.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T_new, dh]
        k_latent = k_latent.view(batch_size, current_seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T_current, dh]
        v_latent = v_latent.view(batch_size, current_seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T_current, dh]
        
        # Apply RoPE only to the first rope_dim dimensions of q_latent and k_latent
        # Position IDs should correspond to the *current* sequence length for K/V
        # For Q, it should correspond to the *new* sequence length
        # We need position_ids for the full sequence [0, ..., T_current-1]
        # The provided position_ids might only be for the new tokens [T_cache, ..., T_cache + T_new - 1]
        # Assuming position_ids passed are for the *new* tokens only
        # We need to generate position IDs for the cached part if caching is used
        if self.cache_c_kv is not None:
             # If cache exists, position_ids are for the new part [T_cache, ..., T_cache + T_new - 1]
             # We need full position IDs [0, ..., T_current - 1] for K
             cached_seq_len = self.cache_c_kv.shape[1]
             full_position_ids = torch.arange(current_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        else:
             # No cache, position_ids are [0, ..., T_new - 1]
             full_position_ids = position_ids

        # Pass num_heads to rope forward
        # Use full_position_ids for K, and original position_ids for Q
        cos_k, sin_k = self.rope(hidden_states, full_position_ids, self.num_heads)
        cos_q, sin_q = self.rope(hidden_states, position_ids, self.num_heads)

        q_rot, q_pass = q_latent[..., :self.rope_dim], q_latent[..., self.rope_dim:]
        k_rot, k_pass = k_latent[..., :self.rope_dim], k_latent[..., self.rope_dim:]
        
        # Apply RoPE
        # cos/sin shape: [bs, num_heads, seq_len, rope_dim]
        # q_rot/k_rot shape: [bs, num_heads, seq_len, rope_dim]
        q_rot, _ = apply_rotary_pos_emb(q_rot, q_rot, cos_q, sin_q) # Apply Q RoPE
        _, k_rot = apply_rotary_pos_emb(k_rot, k_rot, cos_k, sin_k) # Apply K RoPE
        
        # Concatenate rotated and unrotated parts
        q_latent = torch.cat([q_rot, q_pass], dim=-1)
        k_latent = torch.cat([k_rot, k_pass], dim=-1)
        
        # Ensure RoPE projections have the correct shape [B, nh, T, d_R']
        # q_rope is already [B, nh, T_new, d_R']
        # k_rope is already [B, nh, T_current, d_R']

        # Concat latent and RoPE components
        # q_latent shape: [B, nh, T_new, dh]
        # q_rope shape: [B, nh, T_new, d_R']
        q = torch.cat([q_latent, q_rope], dim=-1)  # [B, nh, T_new, dh + d_R']
        # k_latent shape: [B, nh, T_current, dh]
        # k_rope shape: [B, nh, T_current, d_R']
        k = torch.cat([k_latent, k_rope], dim=-1)  # [B, nh, T_current, dh + d_R']
        v = v_latent  # [B, nh, T_current, dh]
        
        # Attention computation
        # q: [B, nh, T_new, D], k: [B, nh, T_current, D], v: [B, nh, T_current, dh]
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / (q.size(-1) ** 0.5)
        
        # Apply attention mask if provided (needs careful handling with caching)
        # The mask should cover the full K sequence length (T_current)
        # And only allow Q (T_new) to attend to relevant K positions
        if attention_mask is not None:
            # Assuming attention_mask is [B, 1, T_new, T_current]
            attn_weights = attn_weights + attention_mask # Use additive mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        # attn_weights: [B, nh, T_new, T_current], v: [B, nh, T_current, dh]
        attn_output = torch.einsum('bhqk,bhvd->bhqd', attn_weights, v) # [B, nh, T_new, dh]
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [B, T_new, nh*dh]
        attn_output = self.W_O(attn_output)  # [B, T_new, d]
        
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.fc1 = nn.Linear(self.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, self.hidden_size)
        
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # RMS Normalization layers
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        
        # Attention
        self.mla = MLA(config)
        
        # Feed-forward network
        self.ffn = FeedForward(config)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # First residual connection with MLA
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mla(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # Second residual connection with FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class Encoder(nn.Module):
    """Transformer text encoder that compresses a token sequence into a fixed-size latent vector."""
    def __init__(self, config):
        super().__init__()
        self.name = "DeepSeekASL-V1"
        self.config = config
        self.latent_dim = config.latent_dim
        self.hidden_size = config.hidden_size
        self.input_dim = config.input_dim

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_size)

        # Transformer encoder blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # Final layer-norm
        self.norm = RMSNorm(self.hidden_size)

        # Projection to latent space
        self.latent_proj = nn.Linear(self.hidden_size, self.latent_dim)

    def forward(self, input_vectors: torch.FloatTensor, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None):
        """Args:
            input_vectors: [B, T, input_dim] sequence of input vectors.
            attention_mask: optional [B, T] mask.
            position_ids: optional [B, T] ids.
        Returns:
            latent: [B, latent_dim] vector representing the whole sequence.
        """
        # Project input vectors to hidden size
        hidden_states = self.input_proj(input_vectors)  # [B, T, d]

        # Encoder stack
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        # Norm
        hidden_states = self.norm(hidden_states)  # [B, T, d]

        # Mean-pool over sequence length
        pooled = hidden_states.mean(dim=1)  # [B, d]

        # Project to latent space
        latent = self.latent_proj(pooled)  # [B, latent_dim]
        return latent

if __name__ == "__main__":
    class ModelConfig:
        hidden_size = 256
        num_attention_heads = 4
        head_dim = 64
        kv_compress_dim = 128
        q_compress_dim = 256
        rope_dim = 32
        max_position_embeddings = 1024
        intermediate_size = 512
        num_layers = 2
        latent_dim = 512
        input_dim = 128  # Dimension of input vectors

    config = ModelConfig()
    model = Encoder(config)

    batch_size = 2
    seq_len = 10
    input_vectors = torch.randn(batch_size, seq_len, config.input_dim)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    latent_vector = model(input_vectors=input_vectors, position_ids=position_ids)
    print(f"Output latent vector shape: {latent_vector.shape}")
