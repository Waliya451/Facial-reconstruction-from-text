import torch
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os

# Load dataset
print("Loading dataset...")
df = pd.read_csv("descriptions.csv")
df.columns = df.columns.str.strip().str.lower()

# Check if dataset is loaded correctly
if "description" not in df.columns:
    raise ValueError("Column 'description' not found in dataset!")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset for MLM Training
class DescriptionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=310):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Mask 15% of tokens (MLM training)
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Split data
print("Splitting data into train and validation sets...")
train_texts, val_texts = train_test_split(df['description'].tolist(), test_size=0.2)

# Create dataset and dataloader
train_dataset = DescriptionDataset(train_texts, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Verify tokenization
print("\nSample tokenized description:")
print(tokenizer.decode(train_dataset[0]['input_ids']))

# Model Setup: Using BERT for Masked Language Modeling (MLM)
print("Initializing BERT model...")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer & Loss Function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Checkpoint path
checkpoint_path = "bert_checkpoint.pth"

# Load checkpoint if exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
    print(f"Resuming training from epoch {start_epoch}...\n")

# Define total number of epochs
num_epochs = 5

# Train Model
print("Starting training...\n")
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch}/{num_epochs}")
    total_loss = 0
    model.train()
    
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print loss every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")

    # Save checkpoint after every epoch
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}!\n")

# Save the final trained model
torch.save(model.state_dict(), "bert_mlm_model.pth")
print("BERT Model training completed and saved!")

def get_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # Extract hidden states
    return hidden_states[-1].mean(dim=1)  # Mean pooling over the last hidden state

# # Function to Extract Embeddings from Trained BERT
# def get_embedding(text, model, tokenizer, device):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     inputs = {key: val.to(device) for key, val in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     return outputs.last_hidden_state.mean(dim=1)  # Mean-pooling over tokens

# Test model on a sample input
print("\nTesting model on a sample description:")
sample_text = "A man with short black hair and a beard."
embedding = get_embedding(sample_text, model, tokenizer, device)
print(f"Extracted embedding shape: {embedding.shape}")  # Should be [1, 768]
