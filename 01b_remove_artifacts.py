import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
# Input: The file created by 01_merge_data.py
INPUT_CSV = "master_dataset.csv"  

# Output: Where to save cleaned data
OUTPUT_CSV = "master_dataset_clean.csv"
OUTPUT_DIR = r"D:\codes\datasets\processed_images"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def remove_hair(image):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. BlackHat transform - finds dark details (hair) on light background
    # Kernel size (17,17) is tuned for typical hair thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Thresholding to create a "Hair Mask"
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Inpainting - Fill the hair pixels with neighboring skin colors
    clean_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return clean_image

def process_dataset():
    print(f"Reading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found. Run 01_merge_data.py first.")
        return

    new_paths = []
    valid_indices = []

    print(f"Starting artifact removal on {len(df)} images...")
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning Images"):
        orig_path = row['path']
        img_id = row['image_id']
        
        # Read the image
        img = cv2.imread(orig_path)
        
        if img is None:
            # If image is missing/corrupt, skip it
            continue
            
        try:
            # Apply Hair Removal
            clean_img = remove_hair(img)
            
            # Save cleaned image to new folder
            new_filename = f"{img_id}_clean.jpg"
            save_path = os.path.join(OUTPUT_DIR, new_filename)
            cv2.imwrite(save_path, clean_img)
            
            # Store new path
            new_paths.append(save_path)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Failed to process {orig_path}: {e}")

    # Create new DataFrame with only successfully processed images
    clean_df = df.iloc[valid_indices].copy()
    clean_df['path'] = new_paths
    
    # Save the new CSV
    clean_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n---------------------------------------------------")
    print(" PROCESSING COMPLETE")
    print("---------------------------------------------------")
    print(f"Original Images: {len(df)}")
    print(f"Cleaned Images:  {len(clean_df)}")
    print(f"Saved CSV to:    {OUTPUT_CSV}")
    print(f"Images saved in: {OUTPUT_DIR}")
    print("---------------------------------------------------")
    print("NEXT STEP: Update your GAN script to use 'master_dataset_clean.csv'")

if __name__ == "__main__":
    process_dataset()