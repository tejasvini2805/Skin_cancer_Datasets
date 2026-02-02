import os
import pandas as pd
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DATASET_DIR = r"D:\codes\datasets"  

# Standardized Labels we want
unified_class_map = {
    "mel": "MEL", "melanoma": "MEL",
    "nv": "NV", "nev": "NV",
    "bcc": "BCC",
    "akiec": "AKIEC", "ak": "AKIEC", "ack": "AKIEC",
    "bkl": "BKL", "sek": "BKL",
    "df": "DF",
    "vasc": "VASC",
    "scc": "SCC",
    "unk": "UNK"
}

# ----------------------------
# LOADERS
# ----------------------------
def load_isic2019_data(base_path, unified_map):
    print(f"Processing ISIC2019 in: {base_path}")
    
    # Locate Metadata
    possible_names = ['ISIC_2019_Training_GroundTruth.csv', 'metadata.csv']
    metadata_path = next((os.path.join(base_path, n) for n in possible_names if os.path.exists(os.path.join(base_path, n))), None)
    if not metadata_path: return []

    metadata_df = pd.read_csv(metadata_path)
    data = []

    # Locate Images
    images_dir = os.path.join(base_path, 'ISIC_2019_Training_Input')
    if not os.path.exists(images_dir): images_dir = os.path.join(base_path, 'images')

    # FIX: Correct One-Hot Columns based on your Debug Output
    # Your CSV has 'AK', not 'AKIEC'
    one_hot_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    
    # Check if this CSV is One-Hot encoded
    cols_present = [c for c in one_hot_cols if c in metadata_df.columns]
    is_one_hot = len(cols_present) == len(one_hot_cols)

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Loading ISIC"):
        image_id = row.get('image') or row.get('image_id')
        
        found_label = None
        if is_one_hot:
            for c in cols_present:
                if row[c] == 1.0:
                    # Map 'AK' to 'AKIEC' here if needed
                    raw_label = "AKIEC" if c == "AK" else c
                    found_label = raw_label
                    break
        else:
            label_raw = row.get('dx') or row.get('diagnosis')
            found_label = unified_map.get(str(label_raw).lower())

        if found_label and found_label != 'UNK':
            img_path = os.path.join(images_dir, image_id + '.jpg')
            if os.path.exists(img_path):
                data.append({'image_id': image_id, 'path': img_path, 'label': found_label, 'source': 'ISIC2019'})
            
    return data

def load_padufes20_data(base_path, unified_map):
    print(f"Processing PAD-UFES20 in: {base_path}")
    metadata_path = os.path.join(base_path, 'metadata.csv')
    if not os.path.exists(metadata_path): return []

    metadata_df = pd.read_csv(metadata_path)
    data = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Loading PAD"):
        image_id = str(row.get('img_id') or row.get('image_id'))
        label_raw = row.get('diagnostic') or row.get('dx')
        standardized_label = unified_map.get(str(label_raw).lower())
        
        if standardized_label and standardized_label != 'UNK':
            # FIX: Check if ID already has extension
            has_ext = image_id.lower().endswith(('.png', '.jpg'))
            
            found = False
            # FIX: Check ALL 3 folders for EVERY image
            for part in [1, 2, 3]:
                folder = os.path.join(base_path, f"imgs_part_{part}")
                
                # Case 1: ID has extension (e.g., PAT_... .png)
                p1 = os.path.join(folder, image_id)
                if os.path.exists(p1):
                    data.append({'image_id': image_id, 'path': p1, 'label': standardized_label, 'source': 'PAD'})
                    found = True; break
                
                # Case 2: ID needs extension
                if not has_ext:
                    for ext in ['.png', '.jpg']:
                        p2 = os.path.join(folder, image_id + ext)
                        if os.path.exists(p2):
                            data.append({'image_id': image_id, 'path': p2, 'label': standardized_label, 'source': 'PAD'})
                            found = True; break
                if found: break
                
    return data

def merge_datasets():
    print(f"Starting merge from: {DATASET_DIR}")
    
    # 1. Load Data
    isic_data = load_isic2019_data(os.path.join(DATASET_DIR, 'ISIC2019'), unified_class_map)
    pad_data = load_padufes20_data(os.path.join(DATASET_DIR, 'PAD-UFES20'), unified_class_map)
    
    # 2. Merge
    all_data = isic_data + pad_data
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("CRITICAL ERROR: No images found.")
        return

    # 3. Save
    df = df.drop_duplicates(subset=['image_id'], keep='first')
    print(f"\nSuccess! Merged {len(df)} images.")
    print(df['label'].value_counts())
    df.to_csv("master_dataset.csv", index=False)

if __name__ == "__main__":
    merge_datasets()