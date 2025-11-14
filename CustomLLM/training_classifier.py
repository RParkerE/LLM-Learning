import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import pandas as pd

from classifier import GPTModel, Config

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
 #1
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
 #2
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

 #3
        self.encoded_texts = [
            encoded_text + [pad_token_id] * 
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]


    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def create_data_loader_v1(csv, batch_size=4, max_length=256, 
                          stride=128, shuffle=True, drop_last=True,
                          num_workers=0):
    ## TODO: Look into creating custom BPE tokenizer not just use tiktoken - gpt2
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SpamDataset(csv, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def random_split(df, train_frac, validation_frac):

    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)               
    train_end = int(len(df) * train_frac)          
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]     
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )                                         
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])                               
    return balanced_df

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
        
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predict_labels = torch.argmax(logits, dim=-1)

            num_examples += predict_labels.shape[0]
            correct_predictions += (
                (predict_labels == target_batch).sum().item()
            )
        else:
            break

    return correct_predictions / num_examples

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = nn.functional.cross_entropy(logits, target_batch)

    return loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, 
                        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoc in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(data_loader=train_loader, model=model, device=device, num_batches=eval_iter)
        val_loss = calc_loss_loader(data_loader=val_loader, model=model, device=device, num_batches=eval_iter)
    model.train()
    
    return train_loss, val_loss

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

torch.manual_seed(123)

device = torch.device("mps")
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

num_workers = 0
batch_size = 8

config = Config()

data_file_path = "train.tsv"

df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
balanced_df = create_balanced_dataset(df)
train_df, validation_df = random_split(balanced_df, 0.8, 0.2)
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)

train_data = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
val_data = SpamDataset(
    csv_file="validation.csv",
    max_length=None,
    tokenizer=tokenizer
)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

model = GPTModel(config)
for param in model.parameters():
    param.requires_grad = False
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
model.to(device)
optimizer = torch.optim.AdamW(
     model.parameters(),  
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

with torch.no_grad():
    train_loss = calc_loss_loader(data_loader=train_loader, model=model, device=device)
    val_loss = calc_loss_loader(data_loader=val_loader, model=model, device=device)

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
torch.save(model.state_dict(), "review_classifier.pth")
print(train_loss)
print(val_loss)
print("=============================")
print(f"{train_accuracy}%")
print(f"{val_accuracy}%")