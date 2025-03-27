import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and DistilBERT model (for text embeddings)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()

# Define image transformation: resize to 64x64, convert to grayscale, and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Paths (adjust as needed)
image_folder = r"C:\Users\Waqas Ahmed\Desktop\py\images"       # Folder containing images
description_csv = r"C:\Users\Waqas Ahmed\Desktop\py\descriptions.csv"  # CSV with "image, description" rows

# Load CSV into DataFrame
df = pd.read_csv(description_csv, header=None, names=["image", "description"])

# Define Dataset that returns both the image and its text embedding.
# This version skips samples with missing images.
class ImageTextDataset(Dataset):
    def __init__(self, dataframe, image_folder, tokenizer, bert_model, transform, max_length=128):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.transform = transform
        self.max_length = max_length

        self.samples = []  # List to store valid (image_filename, text_embedding) tuples
        self.bert_model.eval()
        with torch.no_grad():
            for i in range(len(dataframe)):
                img_filename = str(dataframe.iloc[i, 0])
                img_path = os.path.join(image_folder, img_filename)
                # Skip this sample if image file is not found
                if not os.path.exists(img_path):
                    continue
                # Process description and compute embedding
                text = str(dataframe.iloc[i, 1])
                encoding = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                outputs = self.bert_model(input_ids, attention_mask=attention_mask)
                # Use the CLS token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
                self.samples.append((img_filename, cls_embedding))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_filename, text_embedding = self.samples[idx]
        img_path = os.path.join(self.image_folder, img_filename)
        # Load image and apply transformation
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # 64x64 grayscale tensor
        # Return image and text embedding (unsqueeze embedding to have shape [1, embedding_dim])
        return {"image": image, "text_embedding": text_embedding.unsqueeze(0)}

# Create dataset and DataLoader
dataset = ImageTextDataset(df, image_folder, tokenizer, bert_model, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the GAN models
class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, image_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )
    
    def forward(self, noise, embedding):
        # Concatenate noise and text embedding along dimension=1
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)  # Normalize embeddings
        x = torch.cat((noise, embedding), dim=1)
        return self.model(x).view(-1, 1, 64, 64)  # Output: 64x64 grayscale image

# class Generator(nn.Module):
#     def __init__(self, noise_dim, embedding_dim):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(noise_dim + embedding_dim, 128, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
#             nn.Tanh()
#         )

#     # def forward(self, noise, embedding):
#     #     x = torch.cat((noise, embedding), dim=1).unsqueeze(2).unsqueeze(3)  # Reshape for ConvTranspose2d
#     #     return self.model(x)  # Output: 1x64x64

#     def forward(self, noise, embedding):
#         embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)  # Normalize embeddings
#         x = torch.cat((noise, embedding), dim=1).unsqueeze(2).unsqueeze(3)
#         return self.model(x)



# class Discriminator(nn.Module):
#     def __init__(self, image_dim):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(image_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, image):
#         return self.model(image.view(image.size(0), -1))

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, image):
        return self.model(image.view(image.size(0), -1))  # Remove sigmoid here


# Hyperparameters
noise_dim = 100
# Get embedding dimension from precomputed embedding (assumes all embeddings have same dim)
embedding_dim = dataset.samples[0][1].shape[0]
image_dim = 64 * 64  # For 64x64 grayscale images
num_epochs = 220  # Adjust epochs as needed
# lr = 0.0002
checkpoint_path = "checkpoint.pth"

# Initialize models
generator = Generator(noise_dim, embedding_dim, image_dim).to(device)
discriminator = Discriminator(image_dim).to(device)

# Optimizers and Loss
# optimizer_G = optim.Adam(generator.parameters(), lr=lr)
# optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
# criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Lower LR for generator
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))  # Even lower for discriminator
criterion = nn.BCEWithLogitsLoss()  # More stable

# Check if checkpoint exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state"])
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    optimizer_G.load_state_dict(checkpoint["optimizer_g_state"])
    optimizer_D.load_state_dict(checkpoint["optimizer_d_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")

# Training loop for the conditional GAN

for epoch in range(start_epoch,num_epochs):
    for batch in dataloader:
        # Get real images and text embeddings
        real_images = batch["image"].to(device)  # shape: [B, 1, 64, 64]
        # The text embeddings come with shape [B, 1, embedding_dim]. Squeeze to [B, embedding_dim]
        text_embeddings = batch["text_embedding"].squeeze(1).to(device)
        batch_size_current = real_images.size(0)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size_current, 1).to(device)
        fake_labels = torch.zeros(batch_size_current, 1).to(device)
        
        # ------------------- Train Discriminator -------------------
        noise = torch.randn(batch_size_current, noise_dim).to(device)
        fake_images = generator(noise, text_embeddings)
        
        real_loss = criterion(discriminator(real_images), real_labels)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # ------------------- Train Generator -------------------
        noise = torch.randn(batch_size_current, noise_dim).to(device)
        fake_images = generator(noise, text_embeddings)
        g_loss = criterion(discriminator(fake_images), real_labels)
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    
    # Save model checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        checkpoint = {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "optimizer_g_state": optimizer_G.state_dict(),
            "optimizer_d_state": optimizer_D.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")


# Save the trained generator model
torch.save(generator.state_dict(), "generator_final.pth")
torch.save(discriminator.state_dict(), "discriminator_final.pth")
print("Generator model saved successfully")
print("GAN Training Complete")