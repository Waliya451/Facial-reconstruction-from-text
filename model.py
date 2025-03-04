# Required Libraries
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

# Get the absolute path to the stylegan2_pytorch package
current_dir = os.path.dirname(os.path.abspath(__file__))
stylegan2_path = os.path.join(current_dir, "stylegan2-pytorch")  # Adjust to your structure

# Add the package directory to sys.path
sys.path.insert(0, stylegan2_path)

# Now, you can import it
import stylegan2_pytorch
print(f"Using stylegan2_pytorch from: {stylegan2_pytorch.__file__}")

from stylegan2_pytorch.stylegan2_pytorch import Generator
import matplotlib.pyplot as plt

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

# Load Summarization Model (BART)
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(device)

# Summarization Function
def summarize_text(description, max_length=77):
    """Summarizes text to ensure it fits within 77 tokens for CLIP."""
    inputs = summarizer_tokenizer(
        description, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    
    with torch.no_grad():
        summary_ids = summarizer_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Dataset Class
class CSVDataset(Dataset):
    def __init__(self, csv_file, processor, image_root=""):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.image_root = image_root
        self.filtered_data = []

        # Filter out rows with missing images
        for idx in range(len(self.data)):
            img_name = self.data.iloc[idx]['Image Name']
            img_path = os.path.join(self.image_root, img_name)
            if os.path.exists(img_path):  # Check if the image file exists
                self.filtered_data.append(self.data.iloc[idx])
            else:
                print(f"Image not found, skipping: {img_path}")

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        row = self.filtered_data[idx]
        img_name = row['Image Name']
        description = row['Description']
        img_path = os.path.join(self.image_root, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to match CLIP input

        # Summarize description
        summarized_text = summarize_text(description, max_length=77)

        # Process image and text
        processed = self.processor(text=[summarized_text], images=[image], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        
        return {
            "image": processed['pixel_values'].squeeze(0),  # Image tensor
            "text": summarized_text,  # Raw summarized text
            "text_inputs": processed['input_ids'].squeeze(0)  # Tokenized text with padding
        }


# Conditional StyleGAN

class ConditionalStyleGAN(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim, generator):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.generator = generator
        # self.text_to_latent = nn.Linear(text_embedding_dim, latent_dim)
        self.text_to_latent = nn.Linear(text_embedding_dim, 512)


    # def forward(self, z, text_embedding):
    #     print(f"text_embedding shape before reshape: {text_embedding.shape}")
    #     text_latent = self.text_to_latent(text_embedding)
    #     print(f"text_latent shape after passing through text_to_latent: {text_latent.shape}")
    #     conditioned_latent = z + text_latent
    #     conditioned_latent = conditioned_latent.view(z.shape[0], 512, 1, 1)

    #     print(f"conditioned_latent shape: {conditioned_latent.shape}")
        
    #     # Convert noise dictionary to a list
    #     noise_resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #     input_noise = {res: torch.randn(z.shape[0], 1, res, res, device=z.device) for res in [4, 8, 16, 32, 64, 128, 256, 512, 1024]}
    #     max_res = max(input_noise.keys())  # Find the highest resolution
    #     inoise = torch.cat(
    #         [F.interpolate(input_noise[res], size=(max_res, max_res), mode='bilinear', align_corners=False)
    #         for res in sorted(input_noise.keys())],
    #         dim=1
    #     )

    #     #input_noise = [torch.randn(1, 512, 1, 1).to(device) for _ in range(len(noise_resolutions))]
    #     # input_noise = [torch.randn(z.shape[0], 1, res, res, device=z.device) for res in noise_resolutions]

    #     return self.generator(conditioned_latent, inoise)

    def forward(self, z, text_embedding):
        print(f"text_embedding shape before reshape: {text_embedding.shape}")

        # Convert text embedding to latent space
        text_latent = self.text_to_latent(text_embedding)
        print(f"text_latent shape after passing through text_to_latent: {text_latent.shape}")

        # Ensure text_latent and z have the same shape
        if text_latent.shape[-1] != z.shape[-1]:
            text_latent = text_latent[:, :z.shape[-1]]  # Adjust if needed

        # Combine latent vectors
        conditioned_latent = z + text_latent
        conditioned_latent = conditioned_latent.view(z.shape[0], 512, 1, 1)
        print(f"conditioned_latent shape: {conditioned_latent.shape}")

        # Generate per-resolution noise maps
        noise_resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        # input_noise = {res: torch.randn(z.shape[0], 1, res, res, device=z.device) for res in noise_resolutions}
        input_noise = [torch.randn(z.shape[0], 1, res, res, device=z.device) for res in noise_resolutions]

        # Pass the noise dictionary as expected by StyleGAN2
        return self.generator(conditioned_latent, input_noise)


    # import torch.nn.functional as F

    # def forward(self, z, text_embedding):
    #     print(f"text_embedding shape before reshape: {text_embedding.shape}")

    #     # Ensure text_embedding is correctly shaped for processing
    #     text_latent = self.text_to_latent(text_embedding)
    #     print(f"text_latent shape after passing through text_to_latent: {text_latent.shape}")

    #     # Fix dimension mismatch if necessary
    #     if text_latent.shape[-1] != z.shape[-1]:
    #         text_latent = text_latent[:, :z.shape[-1]]  # Adjust size if necessary

    #     # Combine text and noise latent vectors
    #     conditioned_latent = z + text_latent
    #     conditioned_latent = conditioned_latent.view(z.shape[0], 512, 1, 1)
    #     print(f"conditioned_latent shape: {conditioned_latent.shape}")

    #     # Generate noise for all resolutions
    #     noise_resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #     input_noise = {res: torch.randn(z.shape[0], 1, res, res, device=z.device) for res in noise_resolutions}

    #     # Ensure noise matches expected generator input
    #     max_res = max(input_noise.keys())  
    #     inoise = torch.cat(
    #         [F.interpolate(input_noise[res], size=(max_res, max_res), mode='bilinear', align_corners=False)
    #         for res in sorted(input_noise.keys())],
    #         dim=1
    #     )

    #     return self.generator(conditioned_latent, inoise)


    # def forward(self, z, text_embedding):
    #     print(f"text_embedding shape before reshape: {text_embedding.shape}")
        
    #     # Ensure correct dimensions in text_to_latent
    #     text_latent = self.text_to_latent(text_embedding)
    #     print(f"text_latent shape after passing through text_to_latent: {text_latent.shape}")

    #     # Fix dimension mismatch between z and text_latent
    #     if text_latent.shape[-1] != z.shape[-1]:
    #         text_latent = text_latent[:, :z.shape[-1]]  # Adjust size

    #     # conditioned_latent = z + text_latent
    #     conditioned_latent = z + text_latent[:, :z.shape[-1]]  # Adjusts size if mismatch
    #     conditioned_latent = conditioned_latent.view(z.shape[0], 512, 1, 1)
    #     print(f"conditioned_latent shape: {conditioned_latent.shape}")
        
    #     # Convert noise dictionary to a tensor
    #     noise_resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #     input_noise = {res: torch.randn(z.shape[0], 1, res, res, device=z.device) for res in [4, 8, 16, 32, 64, 128, 256, 512, 1024]}
        
    #     max_res = max(input_noise.keys())  
    #     inoise = torch.cat(
    #         [F.interpolate(input_noise[res], size=(max_res, max_res), mode='bilinear', align_corners=False)
    #         for res in sorted(input_noise.keys())],
    #         dim=1
    #     )

    #     return self.generator(conditioned_latent, inoise)


    # def forward(self, z, text_embedding):
    #     # Ensure text_embedding has the correct shape
    #     text_embedding = text_embedding.view(text_embedding.shape[0], -1)

    #     # Convert text embedding to latent space
    #     text_latent = self.text_to_latent(text_embedding)  # Ensure it outputs (batch, 512)
    #     text_latent = text_latent.view(z.shape[0], 512)   # Reshape if needed

    #     # Combine text and noise latent vectors
    #     conditioned_latent = z + text_latent

    #     # Convert noise dictionary to a list
    #     noise_resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #     input_noise = [torch.randn(z.shape[0], 1, res, res, device=z.device) for res in noise_resolutions]

    #     # Reshape latent vector to match the generator's expected input
    #     conditioned_latent = conditioned_latent.view(z.shape[0], 512, 1, 1)

    #     return self.generator(conditioned_latent, input_noise)
    
    # def forward(self, z, text_embedding):
    #     print(f"text_embedding shape before reshape: {text_embedding.shape}")

    #     # Ensure text_embedding has correct shape for text_to_latent
    #     text_embedding = text_embedding.view(text_embedding.shape[0], -1)
    #     print(f"text_embedding shape after reshape: {text_embedding.shape}")

    #     # Convert text embedding to latent space
    #     text_latent = self.text_to_latent(text_embedding)
    #     print(f"text_latent shape after passing through text_to_latent: {text_latent.shape}")

    #     text_latent = text_latent.view(z.shape[0], 512)
    #     print(f"text_latent reshaped: {text_latent.shape}")

    #     # Combine text and noise latent vectors
    #     conditioned_latent = z + text_latent
    #     print(f"conditioned_latent shape: {conditioned_latent.shape}")

    #     # Convert noise dictionary to a list
    #     noise_resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #     input_noise = [torch.randn(z.shape[0], 1, res, res, device=z.device) for res in noise_resolutions]

    #     # Reshape latent vector to match generator input
    #     conditioned_latent = conditioned_latent.view(z.shape[0], 512, 1, 1)
    #     print(f"conditioned_latent reshaped for generator: {conditioned_latent.shape}")

    #     return self.generator(conditioned_latent, input_noise)






# Initialize StyleGAN
stylegan_generator = Generator(image_size=1024, latent_dim=512, network_capacity=16).to(device)
conditional_gan = ConditionalStyleGAN(latent_dim=512, text_embedding_dim=512, generator=stylegan_generator).to(device)

# Dataset and DataLoader
csv_file = "C:/Users/Waqas Ahmed/Desktop/py/descriptions.csv"
image_dir = "C:/Users/Waqas Ahmed/Desktop/py/images"
dataset = CSVDataset(csv_file, clip_processor, image_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training the Conditional GAN
optimizer = optim.Adam(conditional_gan.parameters(), lr=1e-4)
num_epochs = 10

scaler = torch.cuda.amp.GradScaler("cuda")

scaler = torch.amp.GradScaler()  # Fix GradScaler deprecation

torch.cuda.empty_cache()
import torch
from torch.utils.checkpoint import checkpoint

torch.backends.cudnn.benchmark = True  # Speed up training

def memory_efficient_forward(model, *inputs):
    return checkpoint(model, *inputs)

conditional_gan.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['image'].to(device)
        text_inputs = batch['text_inputs'].to(device)

        with torch.no_grad():
            text_embeddings = clip_model.get_text_features(input_ids=text_inputs)

        latent_vectors = torch.randn(len(images), 512, device=device, requires_grad=True)

        # Free memory before forward pass
        torch.cuda.empty_cache()

        with torch.set_grad_enabled(True):
            with torch.autocast(device_type="cuda"):
                generated_images = memory_efficient_forward(conditional_gan, latent_vectors, text_embeddings)
                generated_images = F.interpolate(generated_images, size=(224, 224), mode='bilinear', align_corners=False)

                loss = ((generated_images - images) ** 2).mean()

        # Free memory before backward pass
        del latent_vectors, text_embeddings
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        del images, generated_images, loss
        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



# Generate Images from Text
def generate_image_from_text_with_summary(description, generator, clip_model, processor):
    """Summarizes text, generates embedding, and creates an image."""
    summarized_text = summarize_text(description, max_length=77)
    print(f"Original: {description}\nSummarized: {summarized_text}")

    inputs = processor(text=[summarized_text], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    
    latent_vector = torch.randn(1, 512).to(device)
    generated_image = generator(latent_vector, text_embedding).squeeze(0).detach().cpu()
    
    return generated_image

# Example
user_input = "A man with short brown hair, wearing glasses, smiling warmly at the camera in a sunny park."
generated_image = generate_image_from_text_with_summary(user_input, conditional_gan, clip_model, clip_processor)

# Display the Generated Image
plt.imshow(generated_image.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
plt.axis("off")
plt.show()
