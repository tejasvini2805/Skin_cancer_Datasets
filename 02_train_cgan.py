import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------
CSV_PATH = "master_dataset_clean.csv"  # The file you just created
IMG_SIZE = 64                    # 64x64 is stable and fast for GANs
BATCH_SIZE = 32
EPOCHS = 50                      # 30-50 is usually good for simple textures
LR = 0.0002
Z_DIM = 100                      # Size of the noise vector
DEVICE = "cpu"

print(f"Training on: {DEVICE}")

# ----------------------------
# 1. DATASET
# ----------------------------
class GANDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
        # Mapping labels to integers (0, 1, 2...)
        self.labels = sorted(self.df['label'].unique())
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        self.num_classes = len(self.labels)
        
        print(f"Classes: {self.label2idx}")
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Map [0,1] to [-1,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label_str = row['label']
        label_idx = self.label2idx[label_str]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label_idx
        except Exception as e:
            # Fallback for corrupt images
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label_idx

# ----------------------------
# 2. MODELS (Generator & Discriminator)
# ----------------------------
class Generator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(Z_DIM + num_classes, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh() # Output is [-1, 1]
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Conv2d(3 + 1, 64, 4, 2, 1), # +1 channel for label
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.adv_layer = nn.Sequential(nn.Linear(512*4*4, 1), nn.Sigmoid())

    def forward(self, img, labels):
        # Spatially tile the label embedding to match image size
        # This lets the discriminator "see" the label at every pixel
        label_emb = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_channel = label_emb.repeat(1, 1, IMG_SIZE, IMG_SIZE)
        
        # Simplified: Just fill a channel with the label index value normalized
        # This is lighter on VRAM than full embedding expansion
        label_channel = torch.zeros(img.shape[0], 1, IMG_SIZE, IMG_SIZE).to(img.device)
        for i in range(img.shape[0]):
            label_channel[i].fill_(labels[i] / 10.0) # Normalize roughly 0-1

        d_in = torch.cat((img, label_channel), 1)
        validity = self.model(d_in)
        return self.adv_layer(validity)

# ----------------------------
# 3. TRAINING LOOP
# ----------------------------
def main():
    # Load Data
    ds = GANDataset(CSV_PATH)
    # num_workers=0 is safer on Windows to avoid process crashing
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    
    generator = Generator(ds.num_classes).to(DEVICE)
    discriminator = Discriminator(ds.num_classes).to(DEVICE)
    
    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    adversarial_loss = nn.BCELoss()
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(dl):
            
            # Skip bad batches
            if imgs.shape[0] != BATCH_SIZE: continue

            real_imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            valid = torch.ones(BATCH_SIZE, 1).to(DEVICE)
            fake = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
            
            # -----------------
            #  Train Generator
            # -----------------
            opt_g.zero_grad()
            
            z = torch.randn(BATCH_SIZE, Z_DIM).to(DEVICE)
            gen_labels = torch.randint(0, ds.num_classes, (BATCH_SIZE,)).to(DEVICE)
            
            gen_imgs = generator(z, gen_labels)
            
            # Loss: Generator wants D to think these images are Valid (1)
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            opt_g.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            
            # Real Loss
            real_pred = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid)
            
            # Fake Loss
            fake_pred = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            
        print(f"[Epoch {epoch}/{EPOCHS}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # Save check point every 10 epochs
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")

    # Final Save
    torch.save(generator.state_dict(), "generator_final.pth")
    print("Training Complete! Saved 'generator_final.pth'")

if __name__ == "__main__":
    main()