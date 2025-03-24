import torch
import pandas as pd
import os
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ------------------------- Step 1: Dataset Class -------------------------
class ImageTextDataset(Dataset):
    def __init__(self, csv_file, image_folder, tokenizer, transform=None):
        print(f" Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform

        # Normalize column names
        self.data.columns = self.data.columns.str.strip().str.lower().str.replace(" ", "_")
        if "description" not in self.data.columns or "image_name" not in self.data.columns:
            raise ValueError(" CSV file must have 'description' and 'image_name' columns!")

        print(f"✅ Dataset loaded successfully! Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image_name"]
        img_path = os.path.join(self.image_folder, img_name)

        # Ensure the image file exists
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Image {img_path} not found! Skipping...")
            return None  

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text = self.data.iloc[idx]["description"]

        # Print text before tokenization
        print(f"🔤 Processing text for image {img_name}: {text[:50]}...")

        tokens = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=310, return_tensors="pt"
        )

        print(f"📌 Tokenized input_ids shape: {tokens['input_ids'].shape}")
        
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "description": text  # Keep text for saving
        }

# ------------------------- Step 2: Define Transform & Tokenizer -------------------------
print("🛠 Initializing tokenizer and image transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("✅ Tokenizer and transformations initialized!")

# ------------------------- Step 3: Load Dataset -------------------------
if __name__ == '__main__':
    dataset = ImageTextDataset("descriptions.csv", "images", tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    print(f"📊 Total dataset size: {len(dataset)} images")

    # ------------------------- Step 4: Save Compiled Dataset -------------------------
    compiled_dataset_path = "compiled_dataset.pth"

    if not os.path.exists(compiled_dataset_path):
        print(f"💾 Saving compiled dataset to {compiled_dataset_path}...")
        compiled_data = []

        for i, sample in enumerate(dataset):
            if sample is None:
                continue  # Skip missing images

            compiled_data.append({
                "image": sample["image"],
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
                "description": sample["description"]
            })

            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"📝 Processed {i+1}/{len(dataset)} samples...")

        torch.save(compiled_data, compiled_dataset_path)
        print(" Compiled dataset saved successfully!")
    else:
        print(" Compiled dataset already exists. Skipping save step.")

    # ------------------------- Step 5: Save Progress After Every 20 Images -------------------------
    save_progress_path = "dataloader_checkpoint.pth"
    progress_count = 0

    for i, batch in enumerate(dataloader):
        torch.save(batch, save_progress_path)  
        progress_count += len(batch["image"])
        
        print(f" Processed {progress_count} images so far...")

        if progress_count % 20 == 0:
            print(f"💾 Saving progress after {progress_count} images!")

    print(" Dataloader processing complete!")
