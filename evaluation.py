import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# Reuse your Dataset, Generator, Discriminator classes
# Assuming they are available in the same file or imported from a module

# ---------------- Config ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_folder = r"C:\Users\Waqas Ahmed\Desktop\py\images"
description_csv = r"C:\Users\Waqas Ahmed\Desktop\py\desc.csv"
generator_path = "generator_final.pth"
discriminator_path = "discriminator_final.pth"

# ---------------- Load Data ----------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

df = pd.read_csv(description_csv, header=None, names=["image", "description"])
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

from torch.utils.data import Dataset

class ImageTextDataset(Dataset):
    def __init__(self, dataframe, image_folder, tokenizer, bert_model, transform, max_length=128):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.transform = transform
        self.max_length = max_length
        self.samples = []

        self.bert_model.eval()
        with torch.no_grad():
            for i in range(len(dataframe)):
                img_filename = str(dataframe.iloc[i, 0])
                img_path = os.path.join(image_folder, img_filename)
                if not os.path.exists(img_path):
                    continue
                text = str(dataframe.iloc[i, 1])
                encoding = self.tokenizer(
                    text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt'
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                outputs = self.bert_model(input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
                self.samples.append((img_filename, cls_embedding))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_filename, text_embedding = self.samples[idx]
        img_path = os.path.join(self.image_folder, img_filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "text_embedding": text_embedding.unsqueeze(0)}

dataset = ImageTextDataset(df, image_folder, tokenizer, bert_model, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ---------------- Load Models ----------------
embedding_dim = dataset.samples[0][1].shape[0]
image_dim = 64 * 64
noise_dim = 100

# Define Generator and Discriminator again or import
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
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        x = torch.cat((noise, embedding), dim=1)
        return self.model(x).view(-1, 1, 64, 64)

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
        return self.model(image.view(image.size(0), -1))

generator = Generator(noise_dim, embedding_dim, image_dim).to(device)
discriminator = Discriminator(image_dim).to(device)

generator.load_state_dict(torch.load(generator_path, map_location=device))
discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
generator.eval()
discriminator.eval()

# ---------------- Evaluation ----------------
fooling_confidences = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating Generator Fooling Power"):
        batch_size = batch["image"].size(0)
        text_embeddings = batch["text_embedding"].squeeze(1).to(device)
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(noise, text_embeddings)
        d_outputs = discriminator(fake_images)
        d_probs = torch.sigmoid(d_outputs).squeeze()

        fooling_confidences.extend(d_probs.cpu().numpy())

# ---------------- Results ----------------
fooling_confidences = torch.tensor(fooling_confidences)
fooling_rate = (fooling_confidences > 0.5).float().mean().item() * 100
avg_confidence = fooling_confidences.mean().item()

print(f"\nGenerator Evaluation:")
print(f"Fooling Rate (D thinks generated is real): {fooling_rate:.2f}%")
print(f"Average Confidence from Discriminator: {avg_confidence:.4f}")
