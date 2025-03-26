import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel

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
generator.load_state_dict(torch.load("generator_final.pth", map_location=device))
generator.eval()

# Load DistilBERT tokenizer and model for text embedding
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()

def generate_image_from_description(description, noise_dim=noise_dim):
    # Tokenize and encode the description
    encoding = tokenizer(
        description, 
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get text embedding from DistilBERT (CLS token)
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
    text_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [1, embedding_dim]
    
    # Generate random noise
    noise = torch.randn(1, noise_dim).to(device)
    
    # Generate image using the generator
    with torch.no_grad():
        fake_image = generator(noise, text_embedding)
    
    return fake_image

# Example usage
description = "This person has a youthful appearance, with a medium skin tone and an oval-shaped face that is slightly asymmetric. Their hair is blond, straight, and frames their face nicely. They have a pointed nose and heart-shaped lips that curve upwards into a bright, smiling expression. Their eyes are round and wide-eyed, giving them a lively and alert look. Although they don't have a beard, they do have a hint of a 5 o'clock shadow, which adds a touch of maturity to their overall appearance. With an age of around 25, this individual has a fresh and attractive face, and their overall features suggest a person of white ethnicity. Overall, their face is pleasant to look at, with a mix of soft and striking features that make them stand out."  # Change this to your desired text
fake_image = generate_image_from_description(description)

# Convert generated image tensor to a displayable format
# fake_image shape: [1, 1, 64, 64] -> Remove batch and channel dims
fake_image = fake_image.squeeze(0).squeeze(0).cpu().detach()

# Denormalize the image: Original normalization was (-1,1); bring it back to (0,1)
fake_image = (fake_image + 1) / 2

# Display the generated image using matplotlib
plt.imshow(fake_image, cmap="gray")
# plt.title("Generated image for: " + description)
plt.axis("off")
plt.show()
