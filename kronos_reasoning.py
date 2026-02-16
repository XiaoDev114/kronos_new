import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import ModelConfig


class StandardAttention(nn.Module):
    """标准注意力模块，用于可视化和调试。"""
    
    def __init__(self, dim=ModelConfig.dim, num_heads=ModelConfig.heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        
        if return_attention:
            return self.to_out(out), attn
        return self.to_out(out)


class LinearAttention(nn.Module):
    """线性注意力模块，支持O(N)复杂度的长序列处理。"""
    
    def __init__(self, dim=ModelConfig.dim, num_heads=ModelConfig.heads, chunk_size=ModelConfig.chunk_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if return_attention:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, N, -1)
            return self.to_out(out), attn
        
        if N <= self.chunk_size:
            return self._linear_attention_chunk(q, k, v)
        return self._linear_attention_chunked_long(q, k, v)

    def _linear_attention_chunk(self, q, k, v):
        """单块线性注意力计算 (Causal)。"""
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Causal Masking using cumsum
        # k, v: [B, H, N, D]
        # kv_cum: [B, H, N, D, D] - accumulating outer products
        # This is memory intensive for large N! 
        # Optimized: S_t = S_{t-1} + k_t^T v_t
        # q_t S_t
        
        # PyTorch efficient implementation using cumsum:
        # KV = einsum(k, v) -> [B, H, N, D, D] -> cumsum -> [B, H, N, D, D]
        # This is O(N * D^2) memory. D=dim/heads = 256/8 = 32. D^2=1024.
        # N=10000. 10000 * 1024 * 4 bytes approx 40MB. Feasible.
        
        kv = torch.einsum('bhnd,bhne->bhn de', k, v)
        kv_cum = torch.cumsum(kv, dim=2)
        
        k_cum = torch.cumsum(k, dim=2) # [B, H, N, D]
        
        # q: [B, H, N, D]
        # kv_cum: [B, H, N, D, D]
        # out = q * kv_cum
        
        out = torch.einsum('bhnd,bhnde->bhne', q, kv_cum)
        
        k_sum = torch.einsum('bhnd,bhnd->bhn', q, k_cum).unsqueeze(-1) + 1e-6
        
        out = out / k_sum
        return self.to_out(out.reshape(out.shape[0], out.shape[2], -1))

    def _linear_attention_chunked_long(self, q, k, v):
        """分块线性注意力计算，用于超长序列 (Causal)。"""
        # For very long sequences, we cannot materialize [N, D, D].
        # We need to process in chunks and carry over state.
        B, H, N, D = q.shape
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        out = torch.zeros_like(v)
        kv_state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        k_state = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
        
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            q_chunk = q[:, :, i:end]
            k_chunk = k[:, :, i:end]
            v_chunk = v[:, :, i:end]
            
            # Intra-chunk causal attention
            kv_chunk = torch.einsum('bhnd,bhne->bhnde', k_chunk, v_chunk)
            kv_cum = torch.cumsum(kv_chunk, dim=2) + kv_state.unsqueeze(2)
            
            k_cum = torch.cumsum(k_chunk, dim=2) + k_state.unsqueeze(2)
            
            out_chunk = torch.einsum('bhnd,bhnde->bhne', q_chunk, kv_cum)
            norm_chunk = torch.einsum('bhnd,bhnd->bhn', q_chunk, k_cum).unsqueeze(-1) + 1e-6
            
            out[:, :, i:end] = out_chunk / norm_chunk
            
            # Update state
            kv_state = kv_cum[:, :, -1]
            k_state = k_cum[:, :, -1]
            
        return self.to_out(out.reshape(B, N, -1))


class RingAttentionBlock(nn.Module):
    """Ring Attention块，包含注意力和前馈网络。"""
    
    def __init__(self, dim=ModelConfig.dim, num_heads=ModelConfig.heads, dropout=ModelConfig.dropout):
        super().__init__()
        self.attn = LinearAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            return x + self.ffn(self.norm2(x)), attn_weights
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ThinkingLayer(nn.Module):
    """思维层，使用GRU实现潜在思维链推理。"""
    
    def __init__(self, dim=ModelConfig.dim, num_thoughts=ModelConfig.num_thoughts):
        super().__init__()
        self.num_thoughts = num_thoughts
        self.thought_gru = nn.GRUCell(dim, dim)
        self.thought_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        h = torch.zeros(B * N, D, device=x.device, dtype=x.dtype)
        thought_states = []
        
        for _ in range(self.num_thoughts):
            h = self.thought_gru(x.reshape(-1, D), h)
            thought_states.append(h.reshape(B, N, D))
        
        return self.thought_proj(thought_states[-1]), torch.stack(thought_states)


class KronosReasoningGPT(nn.Module):
    """Kronos推理GPT模型，支持分层token和潜在思维链。"""
    
    def __init__(self, dim=ModelConfig.dim, depth=ModelConfig.depth, heads=ModelConfig.heads, 
                 num_thoughts=ModelConfig.num_thoughts, max_len=ModelConfig.max_len):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        
        self.token_emb_coarse = nn.Embedding(ModelConfig.vocab_size_coarse, dim)
        self.token_emb_fine = nn.Embedding(ModelConfig.vocab_size_fine, dim)
        self.sector_emb = nn.Embedding(ModelConfig.sector_vocab_size, dim)
        
        self.time_emb_min = nn.Embedding(240, dim)
        self.time_emb_day = nn.Embedding(31, dim)
        self.time_emb_month = nn.Embedding(12, dim)
        self.time_emb_year = nn.Embedding(100, dim)
        
        self.blocks = nn.ModuleList([RingAttentionBlock(dim, heads) for _ in range(depth)])
        self.thinking_layers = nn.ModuleList([ThinkingLayer(dim, num_thoughts) for _ in range(2)])
        
        self.norm = nn.LayerNorm(dim)
        self.head_coarse = nn.Linear(dim, ModelConfig.vocab_size_coarse)
        self.head_fine = nn.Linear(dim, ModelConfig.vocab_size_fine)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, idx_coarse, idx_fine, sector_id, t_min, t_day, t_month, t_year, return_attention=False):
        B, N = idx_coarse.shape
        
        x = self.token_emb_coarse(idx_coarse) + self.token_emb_fine(idx_fine)
        x = x + self.sector_emb(sector_id).unsqueeze(1)
        x = x + self.time_emb_min(t_min) + self.time_emb_day(t_day) + self.time_emb_month(t_month) + self.time_emb_year(t_year)
        x = x + self.pos_emb[:, :N, :]
        
        attention_weights = []
        
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x)
        
        thought_states_all = None
        for thinking in self.thinking_layers:
            x, thoughts = thinking(x)
            if thought_states_all is None:
                thought_states_all = thoughts
            else:
                thought_states_all = torch.cat([thought_states_all, thoughts], dim=0)
        
        x = self.norm(x)
        
        if return_attention:
            return self.head_coarse(x), self.head_fine(x), thought_states_all, attention_weights
        
        return self.head_coarse(x), self.head_fine(x), thought_states_all
