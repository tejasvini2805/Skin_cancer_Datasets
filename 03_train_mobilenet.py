import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
import time

# --- CONFIGURATION (OPTIMIZED FOR RTX A4000) ---
CSV_PATH = "master_dataset_clean.csv"
MODEL_NAME = "model_mobilenet"
CHECKPOINT_FILE = "checkpoint_mobilenet.pth"
IMG_SIZE = 224
BATCH_SIZE = 128        # Optimized for 16GB VRAM
EPOCHS = 25
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4         # Parallel loading

# --- DATASET CLASS ---
class SkinDataset(Dataset):
    def __init__(self, df, transform=None, label2idx=None):
        self.df = df
        self.transform = transform
        self.label2idx = label2idx
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_idx = self.label2idx[row['label']]
        try: 
            img = Image.open(row['path']).convert('RGB')
        except: 
            img = Image.new('RGB', (224, 224))
        if self.transform: 
            img = self.transform(img)
        return img, label_idx

def main():
    print(f"Training {MODEL_NAME} on {DEVICE}...")
    if not os.path.exists(CSV_PATH): return print("Error: CSV Missing!")
    
    # 1. SETUP DATA
    print("Loading Data...")
    df = pd.read_csv(CSV_PATH)
    labels = sorted(df['label'].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    NUM_CLASSES = len(labels)
    
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Balancing Weights
    weights = [1.0/train_df['label'].value_counts()[l] for l in train_df['label']]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Fast Transforms
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # OPTIMIZED DATALOADERS
    train_dl = DataLoader(
        SkinDataset(train_df, tfm, label2idx), 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_dl = DataLoader(
        SkinDataset(val_df, tfm, label2idx), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    # 2. SETUP MODEL
    print("Initializing MobileNetV3...")
    model = models.mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    best_acc = 0.0

    # 3. RESUME FROM CHECKPOINT
    if os.path.exists(CHECKPOINT_FILE):
        print(f"--> Resuming from {CHECKPOINT_FILE}...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"--> Resuming at Epoch {start_epoch+1} (Best Acc: {best_acc*100:.2f}%)")
    else:
        print("--> No checkpoint found. Starting fresh.")

    # 4. TRAINING LOOP
    print("Starting Training Loop...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print update every 50 batches
            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_dl)} | Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for inputs, targets in val_dl:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                # FIXED LINE BELOW:
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        acc = correct/total
        epoch_time = time.time() - start_time
        print(f">>> Epoch {epoch+1} Done | Val Acc: {acc*100:.2f}% | Time: {epoch_time:.1f}s")
        
        # Save Best Model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
            print(f"*** SAVED NEW BEST MODEL ({acc*100:.2f}%) ***")
            
        # Save Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc
        }
        torch.save(checkpoint, CHECKPOINT_FILE)

    print(f"Final Best Accuracy: {best_acc*100:.2f}%")
    
    # Save log
    with open("final_results_log.txt", "a") as f:
        f.write(f"{MODEL_NAME}: {best_acc*100:.2f}%\n")
        
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

if __name__ == "__main__": 
    main()
