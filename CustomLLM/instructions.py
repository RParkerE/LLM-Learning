import torch
import torch.nn as nn

## GPT2-XL
# embedding_dim = 1600
# transformer_layer = 48
# attention_heads = 25
## GPT2-Large
# embedded_dim = 1280
# transformer_layer = 36
# attention_heads = 20
## GPT2-Medium
# embedded_dim = 1024
# transformer_layer = 24
# attention_heads = 12
## GPT2-Small
# embedded_dim = 768
# transformer_layer = 12
# attention_heads = 12
class Config:
    def __init__(self):
        self.vocab_size = 50257
        self.embedding_dim = 768
        self.drop_rate = 0.1
        self.transformer_layer = 12
        self.attention_heads = 12
        self.context_length = 256
        self.qkv_bias = False


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)

        return context_vec


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ## TODO: Look into implementing own Embedding Layer (one-hot encoding + matrix multiplication)
        ## https://mng.bz/ZEB5
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.transformer_layer)]
        )
        self.final_norm = LayerNorm(cfg.embedding_dim)
        self.out_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.embedding_dim,
            d_out=cfg.embedding_dim,
            context_length=cfg.context_length,
            num_heads=cfg.attention_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.embedding_dim)
        self.norm2 = LayerNorm(cfg.embedding_dim)
        self.dropout_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        ## TODO: Implement my own SwiGLU
        self.silu = nn.SiLU()

    def forward(self, x):
        gate = self.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden_states = gate * up
        hidden_states = self.down_proj(hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.swiglu_block = SwiGLU(
            hidden_size=cfg.embedding_dim,
            intermediate_size=4 * cfg.embedding_dim
        )

    def forward(self, x):
        return self.swiglu_block(x)
