import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Create position indices
        position = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum('i,j->ij', position, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, position_ids):
        # x: [batch_size, num_heads, seq_len, head_dim]
        # position_ids: [batch_size, seq_len]
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos, sin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
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
        
        # RoPE embedding
        self.rope = RoPEEmbedding(self.rope_dim * self.num_heads)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # KV Path
        c_kv = self.W_DKV(hidden_states)  # [B, T, dc_KV]
        k_latent = self.W_UK(c_kv)  # [B, T, nh*dh]
        v_latent = self.W_UV(c_kv)  # [B, T, nh*dh]
        k_rope_proj = self.W_KR(hidden_states)  # [B, T, nh*d_R']
        
        # Reshape for RoPE
        k_rope = k_rope_proj.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        k_rope = k_rope.transpose(1, 2)  # [B, nh, T, d_R']
        
        # Query Path
        c_q = self.W_DQ(hidden_states)  # [B, T, dc_Q]
        q_latent = self.W_UQ(c_q)  # [B, T, nh*dh]
        q_rope_proj = self.W_QR(c_q)  # [B, T, nh*d_R']
        
        # Reshape for RoPE
        q_rope = q_rope_proj.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        q_rope = q_rope.transpose(1, 2)  # [B, nh, T, d_R']
        
        # Apply RoPE
        cos, sin = self.rope(hidden_states, position_ids)
        q_latent = q_latent.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_latent = k_latent.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_latent, k_latent = apply_rotary_pos_emb(q_latent, k_latent, cos, sin)
        
        # Concat latent and RoPE components
        q = torch.cat([q_latent, q_rope], dim=-1)  # [B, nh, T, dh + d_R']
        k = torch.cat([k_latent, k_rope], dim=-1)  # [B, nh, T, dh + d_R']
        v = v_latent.view(batch_size, seq_len, self.num_heads, self.head_dim)  # [B, nh, T, dh]
        
        # Attention computation
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / (q.size(-1) ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.einsum('bhqk,bhvd->bhqd', attn_weights, v)
        
        # Reshape and project back
        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)  # [B, T, nh*(dh + d_R')]
        attn_output = self.W_O(attn_output)  # [B, T, d]
        
        return attn_output

class DeepSeekMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        
        # Gate
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, self.hidden_size)
            ) for _ in range(self.num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Flatten sequence and batch dimensions
        flat_input = hidden_states.view(-1, hidden_dim)  # [B*T, d]
        
        # Get gate logits and probabilities
        gate_logits = self.gate(flat_input)  # [B*T, E]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B*T, E]
        
        # Select top-k experts
        gate_topk_probs, gate_topk_indices = torch.topk(
            gate_probs, self.top_k, dim=-1
        )  # [B*T, K], [B*T, K]
        
        # Normalize top-k probabilities
        gate_topk_probs = gate_topk_probs / gate_topk_probs.sum(dim=-1, keepdim=True)
        
        # Initialize combined output
        combined_output = torch.zeros_like(flat_input)  # [B*T, d]
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Create mask for tokens routed to this expert
            expert_mask = (gate_topk_indices == expert_idx)  # [B*T, K]
            num_tokens = expert_mask.sum().item()
            
            if num_tokens == 0:
                continue
            
            # Get input for this expert
            expert_input = flat_input.masked_select(expert_mask.any(dim=-1).unsqueeze(-1))
            expert_input = expert_input.view(-1, hidden_dim)  # [N, d]
            
            # Process through expert
            expert_output = expert(expert_input)  # [N, d]
            
            # Get gate weights for this expert
            expert_weights = gate_topk_probs.masked_select(expert_mask)
            expert_weights = expert_weights.view(-1, 1)  # [N, 1]
            
            # Apply weights and add to combined output
            combined_output[expert_mask.any(dim=-1)] += (expert_output * expert_weights)
        
        # Reshape back to original dimensions
        output = combined_output.view(batch_size, seq_len, hidden_dim)  # [B, T, d]
        return output

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
        self.moe = DeepSeekMoE(config)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # First residual connection with MLA
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mla(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # Second residual connection with MoE
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class DeepSeekV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final RMS Normalization
        self.norm = RMSNorm(self.hidden_size)
        
        # Output head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Weight tying
        self.embed_tokens.weight = self.lm_head.weight
        
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)  # [B, T, d]
        
        # Apply transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Output logits
        logits = self.lm_head(hidden_states)  # [B, T, V]
        
        return logits

# Configuration class for DeepSeek-V3
class DeepSeekV3Config:
    def __init__(self):
        self.hidden_size = 7168
        self.intermediate_size = 28672  # 4x hidden_size
        self.num_attention_heads = 128
        self.head_dim = 128
        self.kv_compress_dim = 512
        self.q_compress_dim = 1536
        self.rope_dim = 64
        self.num_layers = 61
        self.vocab_size = 102400  # random ahh size
        self.num_experts = 64
        self.moe_top_k = 2
        self.max_position_embeddings = 8192