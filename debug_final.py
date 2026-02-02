import os
import pandas as pd

# CONFIG
DATASET_DIR = r"D:\codes\datasets"

def check_pad():
    print("\n--- DIAGNOSING PAD-UFES-20 ---")
    path = os.path.join(DATASET_DIR, "PAD-UFES20")
    csv_path = os.path.join(path, "metadata.csv")
    
    if not os.path.exists(csv_path):
        print("PAD metadata not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Columns found: {list(df.columns)}")
    
    # Check first row
    first_id = df.iloc[0]['img_id']
    print(f"First Image ID in CSV: '{first_id}'")
    
    # Check if file exists
    folder = os.path.join(path, "imgs_part_1")
    # Try EXACT match
    p1 = os.path.join(folder, first_id)
    if os.path.exists(p1):
        print(f"[SUCCESS] Found file exactly as named: {p1}")
    else:
        print(f"[FAIL] Could not find: {p1}")
        # Try adding .png
        p2 = os.path.join(folder, first_id + ".png")
        if os.path.exists(p2):
            print(f"[SUCCESS] Found file by adding .png: {p2}")
        else:
            print(f"[FAIL] Could not find: {p2}")
            print(f"-> Listing first 3 files in {folder}:")
            try:
                print(os.listdir(folder)[:3])
            except:
                print("Could not list folder.")

def check_isic():
    print("\n--- DIAGNOSING ISIC 2019 ---")
    path = os.path.join(DATASET_DIR, "ISIC2019")
    csv_path = os.path.join(path, "ISIC_2019_Training_GroundTruth.csv")
    
    if not os.path.exists(csv_path):
        print("ISIC CSV not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Columns found: {list(df.columns)}")
    
    # Check if 'MEL' exists
    if 'MEL' in df.columns:
        print("One-Hot columns (MEL, NV...) found.")
        # Check first image
        first_id = df.iloc[0]['image']
        print(f"First Image ID: '{first_id}'")
        
        img_folder = os.path.join(path, "ISIC_2019_Training_Input")
        p = os.path.join(img_folder, first_id + ".jpg")
        if os.path.exists(p):
            print(f"[SUCCESS] Found image: {p}")
        else:
            print(f"[FAIL] Image not found: {p}")
            print(f"-> Listing first 3 files in {img_folder}:")
            try:
                print(os.listdir(img_folder)[:3])
            except:
                print("Could not list folder.")
    else:
        print("One-Hot columns NOT found. Labels might be missing or named differently.")

if __name__ == "__main__":
    check_pad()
    check_isic()
    