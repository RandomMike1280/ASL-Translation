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
        if config.token_embedding_type == 'tokens':
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        elif config.token_embedding_type == 'vector':
            self.embed_tokens = MLP(input_dim=120, hidden_dims=[256, 128], output_dim=self.hidden_size)
        
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
        self.token_embedding_type = 'tokens'


if __name__ == "__main__":
    import torch
    
    def test_model_prediction(n: int = 5):
        # Create tiny model configuration
        class TinyConfig:
            # Model architecture dimensions - Increased to roughly double size
            hidden_size = 768
            intermediate_size = 3072
            num_attention_heads = 12
            head_dim = 64
            token_embedding_type = 'tokens' # or 'vector'
            
            # Compression dimensions for MLA (Scaled proportionally)
            kv_compress_dim = 192 
            q_compress_dim = 384
            rope_dim = 32
            
            # Model structure
            num_layers = 12
            vocab_size = 16
            
            # Mixture of Experts settings
            num_experts = 32
            moe_top_k = 2
            
            # Sequence length settings
            max_position_embeddings = 1024

        # Initialize model and move to GPU if available
        config = TinyConfig()
        model = DeepSeekV3(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Calculate and print parameter/expert counts
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

        if hasattr(config, 'num_experts') and config.num_experts > 0:
            total_experts = config.num_experts * config.num_layers
            active_experts = config.moe_top_k * config.num_layers
            print(f"Total Experts: {total_experts}")
            print(f"Active Experts per Token: {active_experts}")

            # Calculate active parameters (approximate)
            non_expert_params = 0
            expert_params_per_layer = 0

            # Embeddings and final norm/head
            non_expert_params += sum(p.numel() for p in model.embed_tokens.parameters() if p.requires_grad)
            non_expert_params += sum(p.numel() for p in model.norm.parameters() if p.requires_grad)
            # lm_head shares weights with embed_tokens, so no need to add again

            # Per-layer non-expert params
            if config.num_layers > 0:
                layer = model.layers[0]
                non_expert_params_per_layer = 0
                non_expert_params_per_layer += sum(p.numel() for p in layer.norm1.parameters() if p.requires_grad)
                non_expert_params_per_layer += sum(p.numel() for p in layer.norm2.parameters() if p.requires_grad)
                non_expert_params_per_layer += sum(p.numel() for p in layer.mla.parameters() if p.requires_grad)
                non_expert_params_per_layer += sum(p.numel() for p in layer.moe.gate.parameters() if p.requires_grad)
                non_expert_params += non_expert_params_per_layer * config.num_layers

                # Expert params (for one expert)
                expert_params_per_layer = sum(p.numel() for p in layer.moe.experts[0].parameters() if p.requires_grad)

            active_params = non_expert_params + (expert_params_per_layer * active_experts)
            print(f"Active Parameters per Token (Approx): {active_params / 1e6:.2f}M")
        else:
            print("Model does not use MoE.")
            print(f"Active Parameters per Token: {total_params / 1e6:.2f}M") # All params are active

        model.eval()  # Set to evaluation mode

        # Generate random input tokens
        batch_size = 1
        seq_len = n
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        print(f"Input shape: {input_ids.shape}")  # Should be [1, n]
        print(f"Input tokens: {input_ids[0]}")  # Print the first sequence of tokens
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).expand(batch_size, -1)
        
        # First forward pass with all tokens
        with torch.no_grad():
            output = model(input_ids, position_ids=position_ids)
        
        print(f"Output shape: {output.shape}")  # Should be [1, n, 1000]
        for i in range(n):
            if config.vocab_size < 100:
                print(output[0, i])
            print(f"Token {i+1} prediction: {output[0, i].argmax().item()}")
        
        # Verify KV cache updates by processing tokens sequentially
        print("\nTesting KV cache updates...")
        # Reset cache
        for layer in model.layers:
            layer.mla.cache_c_kv = None
            layer.mla.cache_k_rope = None
        
        # Process tokens one by one
        for i in range(n):
            with torch.no_grad():
                _ = model(input_ids[:, i:i+1], position_ids=position_ids[:, i:i+1])
            
            # Check cache shapes after each token
            c_kv_cache = model.layers[0].mla.cache_c_kv
            k_rope_cache = model.layers[0].mla.cache_k_rope
            print(f"\nAfter token {i+1}:")
            print(f"c_kv cache shape: {c_kv_cache.shape if c_kv_cache is not None else None}")
            print(f"k_rope cache shape: {k_rope_cache.shape if k_rope_cache is not None else None}")

    test_model_prediction()