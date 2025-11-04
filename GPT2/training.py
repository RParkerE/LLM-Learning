import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken

from model import GPT2, Config


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


if __name__ == "__main__":
    # Initialize model and tokenizer
    config = Config()
    model = GPT2(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load training data
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
    print("\nModel saved to gpt_custom.pt")