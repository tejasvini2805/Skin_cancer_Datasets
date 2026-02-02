import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os

# --- CONFIG ---
CSV_PATH = "master_dataset_clean.csv"
MODEL_NAME = "model_efficientnet"
IMG_SIZE = 224
BATCH_SIZE = 24 # Reduced batch size slightly for EfficientNet
EPOCHS = 25
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SkinDataset(Dataset):
    def __init__(self, df, transform=None, label2idx=None):
        self.df = df
        self.transform = transform
        self.label2idx = label2idx
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_idx = self.label2idx[row['label']]
        try: img = Image.open(row['path']).convert('RGB')
        except: img = Image.new('RGB', (224, 224))
        if self.transform: img = self.transform(img)
        return img, label_idx

def main():
    print(f"Training {MODEL_NAME} on {DEVICE}...")
    if not os.path.exists(CSV_PATH): return print("CSV Missing!")
    
    df = pd.read_csv(CSV_PATH)
    labels = sorted(df['label'].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    weights = [1.0/train_df['label'].value_counts()[l] for l in train_df['label']]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_dl = DataLoader(SkinDataset(train_df, tfm, label2idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_dl = DataLoader(SkinDataset(val_df, tfm, label2idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- ARCHITECTURE OPTIMIZATION ---
    model = models.efficientnet_b0(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
    
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(DEVICE)), targets.to(DEVICE))
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for inputs, targets in val_dl:
                out = model(inputs.to(DEVICE))
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets.to(DEVICE)).sum().item()
        
        acc = correct/total
        print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print(f"Done. Best Acc: {best_acc:.4f}")
    # Append results to a text file so you don't lose the score
    with open("final_results_log.txt", "a") as f:
        f.write(f"{MODEL_NAME}: {best_acc*100:.2f}%\n")

if __name__ == "__main__": main()