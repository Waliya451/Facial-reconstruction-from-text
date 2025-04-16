import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import pathlib

st.set_page_config(page_title="Face Generator", layout="wide")
def load_css(file_path):
    with open(file_path) as f:
        st.html(
            '''
        <style>
            st-key-container1{
            display:flex;
            gap:2em;
            background-color: yellow;
            padding: 3em;

        }
        </style>
        ''')
file_path = pathlib.Path("./styles/styles.css")
load_css(file_path)




# Define the Generator (must match the training architecture)
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
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        x = torch.cat((noise, embedding), dim=1)
        return self.model(x).view(-1, 1, 64, 64)  # Output: 64x64 grayscale image


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
noise_dim = 100
embedding_dim = 768  # DistilBERT hidden size for distilbert-base-uncased
image_dim = 64 * 64  # 64x64 grayscale image

# Initialize and load the trained generator model
generator = Generator(noise_dim, embedding_dim, image_dim).to(device)
generator.load_state_dict(torch.load("E:\FYDP - Train\Tests\Generator_no_gender.pth", map_location=device))
generator.eval()

# Load DistilBERT tokenizer and model for text embedding
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()

# Title
st.title("🎨 Face Generation from Description")
container1 = st.container(key="container1", border=1)
col1, col2, col3 = container1.columns([2,1,3])
with col1:
    container2 = st.container()
with col3:
    container3 = st.container()
container2.write("""
Describe a face in natural language, and watch the AI generate a grayscale face image based on your description!
""")

# User input
description = container2.text_area("✏️ Enter Face Description:", height=200, placeholder="e.g. A young person with blond hair, round eyes...")

if st.button("Generate Image"):
    if description.strip() == "":
        st.warning("Please enter a valid description.")
    else:
        with st.spinner("Generating image..."):
            # Tokenize description
            encoding = tokenizer(description, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Get embedding
            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
            text_embedding = outputs.last_hidden_state[:, 0, :]

            # Generate noise
            noise = torch.randn(1, noise_dim).to(device)

            # Generate image
            with torch.no_grad():
                fake_image = generator(noise, text_embedding)
            fake_image = fake_image.squeeze(0).squeeze(0).cpu().detach().numpy()
            fake_image = (fake_image + 1) / 2  # Denormalize

            # Convert to image
            image_pil = Image.fromarray((fake_image * 255).astype(np.uint8))

            container3.image(image_pil, caption="Generated Face", width=500)

# st.markdown("---")
st.caption("Powered by CIS Undergrads")
