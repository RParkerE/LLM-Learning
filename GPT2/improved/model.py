import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

device = torch.device("cuda")


# RoPE Implementation
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        assert dim % 2 == 0

        self.dim = dim
        self.max_seq_len = max_seq_len

        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)

        # Compute Frequency Terms
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # Compute Rotation Angles
        freqs = position * inv_freq
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x):
        """
        x: shape (batch, heads, seq_len, head_dim)
        """
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Apply Rotation
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        return x_rotated


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads):
        super().__init__()
        assert dim_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.rate = dropout

        # Q, K, V Projections
        self.W_q = nn.Linear(dim_in, dim_out)
        self.W_k = nn.Linear(dim_in, dim_out)
        self.W_v = nn.Linear(dim_in, dim_out)

        # Final Output Projection
        self.out_proj = nn.Linear(dim_out, dim_out)

        # RoPE
        self.rope = RotaryPositionalEncoding(self.head_dim, context_length)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Casual Masking (Upper Triangular Matrix)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # Exclude This Mask From Being Updated During Backward
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch_size, embed_size, _ = x.size()

        # Linear Projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split The Features Equally Into Every Head
        Q = Q.view(batch_size, embed_size, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, embed_size, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, embed_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # Attention Scores
        mask = self.mask[:embed_size, :embed_size] == 1
        context = F.scaled_dot_product_attention(query=Q, key=K, value=V, attn_mask=mask, dropout_p=self.rate)
        context = context.transpose(1, 2).contiguous().view(batch_size, embed_size, -1)

        return self.out_proj(context)


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x1, x2):
        # Gate, Value, Bias
        return self.silu(x1) * x2


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.embedding_dim, 2 * config.embedding_dim)
        self.value_proj = nn.Linear(config.embedding_dim, 2 * config.embedding_dim)
        self.out_proj = nn.Linear(2 * config.embedding_dim, config.embedding_dim)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_value = self.value_proj(x)
        
        gated_x = self.act(x_gate, x_value)

        return self.dropout(self.out_proj(gated_x))


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            dim_in=config.embedding_dim,
            dim_out=config.embedding_dim,
            context_length=config.context_length,
            num_heads=config.attention_heads,
            dropout=config.drop_rate
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.embedding_dim)
        self.norm2 = LayerNorm(config.embedding_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        self.transformer_layers = nn.Sequential(
            *[TransformerLayer(config) for _ in range(config.transformer_layer)]
        )

        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.final_output = nn.Linear(config.embedding_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        token_embedding = self.token_embedding(in_idx)
        
        x = self.dropout(token_embedding)
        x = self.transformer_layers(x)
        x = self.final_norm(x)
        
        logits = self.final_output(x)
        
        return logits


class Config:
    def __init__(self):
        self.vocab_size = 50257
        self.embedding_dim = 768
        self.drop_rate = 0.1
        self.transformer_layer = 12
        self.attention_heads = 12
        self.context_length = 1024