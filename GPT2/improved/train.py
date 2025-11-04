import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tiktoken
from sklearn.model_selection import train_test_split
import random
import numpy as np

from model import GPT2, Config


SEED = 2025
BATCH_SIZE = 2
EPOCHS = 10
PATIENCE = 5
RATE = 1e-3
SCHEDULE = 0.1
CLIP = 1.0
FILE = "best_custom.pt"

GREEN = '\033[32m'
YELLOW = '\033[33m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
RESET = '\033[0m' 

# Temporarily force CPU to debug
device = torch.device("cpu")
print(f"Using device: {device}")

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


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


class EarlyStopper:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model) 
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(GREEN + f'Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}). Saving model ...' + RESET)
        torch.save(model.state_dict(), self.path)


def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='sum')
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    model.train()
    return avg_loss


def train(model, train_loader, val_loader, optimizer, epochs, patience, clip_value, model_path="best_custom.pt"):
    early_stopper = EarlyStopper(patience=patience, verbose=True, path=model_path)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            
            # Gradient clipping BEFORE optimizer step
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Loop 
        avg_val_loss = validate(model, val_loader)
        
        # Print metrics
        print(YELLOW + f"\n--- Epoch {epoch+1}/{epochs} ---" + RESET)
        print(MAGENTA + f"Train Loss: {avg_train_loss:.4f}" + RESET)
        print(CYAN + f"Validation Loss: {avg_val_loss:.4f}" + RESET)

        # Update Scheduler
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

        # Early Stopping Check
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(GREEN + f"Early stopping triggered after {epoch+1} epochs! Loading best model weights..." + RESET)
            break
            
    model.load_state_dict(torch.load(model_path))
    print(GREEN + "Training complete and best model weights loaded." + RESET)


if __name__ == "__main__":
    # Initialize model and tokenizer
    config = Config()
    model = GPT2(config).to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load training data
    with open("custom.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Total characters in dataset: {len(raw_text)}")

    # Split the text manually (simple split on sentences or paragraphs)
    # Using 80/20 split
    split_idx = int(len(raw_text) * 0.8)
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    
    print(f"Train data length: {len(train_data)}")
    print(f"Val data length: {len(val_data)}")
    
    train_dataset = CustomDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = CustomDataset(val_data, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    print(f"\nStarting training...\n")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=RATE, weight_decay=SCHEDULE)
    train(model, train_dataloader, val_dataloader, optimizer, EPOCHS, PATIENCE, CLIP, FILE)