import tiktoken
import torch.nn.functional as F
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads):
        super().__init__()
        assert dim_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V Projections
        self.W_q = nn.Linear(dim_in, dim_out)
        self.W_k = nn.Linear(dim_in, dim_out)
        self.W_v = nn.Linear(dim_in, dim_out)

        # Final Output Projection
        self.out_proj = nn.Linear(dim_out, dim_out)

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

        # Attention Scores
        attention_scores = (Q @ K.transpose(-2, -1)) * self.scale

        # Apply Casual Mask
        mask = self.mask[:embed_size, :embed_size] == 1
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # Softmax And Dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted Sum Of Values
        context = attention_weights @ V
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

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3)) ))

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.layers(x)

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
        self.ff = FeedForward(config.embedding_dim)
        self.norm1 = LayerNorm(config.embedding_dim)
        self.norm2 = LayerNorm(config.embedding_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x):
        residual1 = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + residual1
        residual2 = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + residual2
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        self.transformer_layers = nn.Sequential(
            * [TransformerLayer(config) for _ in range(config.transformer_layer)]
        )

        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.final_output = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embedding = self.token_embedding(in_idx)
        positional_embedding = self.positional_embedding(torch.arange(seq_len, device=in_idx.device))
        x = token_embedding + positional_embedding
        x = self.dropout(x)

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

config = Config()
model = GPT2(config)

start_context = "The bluefooted booby is a bird found primarily in the"

tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)


########## CODE TO TRAIN GPT2 MODEL ##########
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, block_size=32):
        self.tokenizer = tokenizer
        self.data = tokenizer.encode(data)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

with open("custom.txt", "r") as f:
    raw_text = f.read()

# Prepare dataset
dataset = CustomDataset(raw_text, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train(model, dataloader, optimizer, epochs=10)

# Save model
torch.save(model.state_dict(), "gpt_custom.pt")


########## CODE TO TEST GPT2 MODEL ##########
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

model.load_state_dict(torch.load("gpt_custom.pt"))
model.eval() 

print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
print("\nInput text:", start_context)
print("Encoded input text:", encoded)
print("encoded_tensor.shape:", encoded_tensor.shape)

out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=config.context_length
    )
decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
print("\nOutput:", out)
print("Output length:", len(out[0]))
print("Output text:", decoded_text)
