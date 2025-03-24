import torch
import os
import clip
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# ------------------------- Step 1: Load Models -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()

bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

# Load BERT checkpoint safely
bert_checkpoint_path = "bert_checkpoint.pth"
if os.path.exists(bert_checkpoint_path):
    bert_checkpoint = torch.load(bert_checkpoint_path, map_location=device)
    bert_model.load_state_dict(bert_checkpoint["model_state_dict"], strict=False)
    print("✅ Loaded BERT checkpoint.")

# ------------------------- Step 2: Modify CLIP to Use Trained BERT -------------------------
class CustomCLIP(nn.Module):
    def __init__(self, clip_model, bert_model):
        super(CustomCLIP, self).__init__()
        self.clip_vision = clip_model.visual  
        self.bert_text_encoder = bert_model  
        self.text_projection = nn.Linear(768, 512)  
        self.image_projection = nn.Linear(clip_model.visual.output_dim, 512)  

    def encode_image(self, image):
        image_features = self.clip_vision(image)
        return self.image_projection(image_features)  

    def encode_text(self, input_ids, attention_mask):
        text_embeddings = self.bert_text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        return self.text_projection(text_embeddings)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)
        return image_features, text_features

model = CustomCLIP(clip_model, bert_model).to(device)

# ------------------------- Step 3: Load Compiled Dataset in Batches -------------------------
compiled_dataset_path = "compiled_dataset.pth"
if not os.path.exists(compiled_dataset_path):
    raise FileNotFoundError("❌ Compiled dataset not found! Run data_loader.py first.")

compiled_data = torch.load(compiled_dataset_path)

batch_size = 16  # Adjust based on RAM availability

# Resize images to save memory
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
])

# Process dataset in chunks to avoid OOM issues
dataset = []
for i in range(0, len(compiled_data), batch_size):
    batch = compiled_data[i:i + batch_size]
    # images = torch.stack([transform(sample["image"]) for sample in batch])
    images = torch.stack([sample["image"].float() / 255.0 for sample in batch])  # Normalize manually
    input_ids = torch.stack([sample["input_ids"] for sample in batch])
    attention_mask = torch.stack([sample["attention_mask"] for sample in batch])
    dataset.append(TensorDataset(images, input_ids, attention_mask))

print(f"✅ Loaded compiled dataset with {len(dataset) * batch_size} samples.")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

# ------------------------- Step 4: Load Checkpoint if Available -------------------------
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CosineEmbeddingLoss()
num_epochs = 10
start_epoch = 0

checkpoint_path = "clip_checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1  
    print(f"✅ Resuming training from epoch {start_epoch}.")

# ------------------------- Step 5: Train Model & Save Every 50 Images -------------------------
progress_count = 0

for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    model.train()

    for i, batch in enumerate(dataloader):
        images, input_ids, attention_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        # Convert to float16 for memory efficiency
        images = images.half()
        input_ids = input_ids.half()
        attention_mask = attention_mask.half()
        
        optimizer.zero_grad()
        image_embeddings, text_embeddings = model(images, input_ids, attention_mask)
        
        target = torch.ones(image_embeddings.shape[0]).to(device)  
        loss = criterion(image_embeddings, text_embeddings, target)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Save progress every 50 images
        progress_count += len(images)
        if progress_count % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, checkpoint_path)
            print(f"✅ Progress saved after {progress_count} images.")

    print(f"✅ Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Save model after every epoch
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, checkpoint_path)

    print(f"✅ Epoch {epoch+1} checkpoint saved!")

print("✅ Training complete! Model saved!")
