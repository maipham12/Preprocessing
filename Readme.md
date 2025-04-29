
## Project Overview

**Scope**: Data Cleaning, Feature Engineering, and Dataset Preparation for Modeling.

---

## 1. Dataset Overview
- **Images**: 10,407 training images labeled into 10 classes (9 diseases + 1 normal).
- **Metadata**: Each image has an associated `variety` and `age` (days).
- **Folder structure**:
  - `train_images/{label}/{image_id}`
  - `test_images/{image_id}`

---

## 2. Preprocessing Steps

### Step 1: Ingestion & Integrity Check
- Loaded `meta_train.csv`.
- Built correct `image_path` for each image.
- Verified:
  - No missing files.
  - No duplicate `image_id`s.

### Step 2: Exploratory Data Analysis (EDA)
- Analyzed:
  - Class distribution.
  - Age distribution (overall and per class).
  - Variety distribution.

### Step 3: Data Cleaning
- Detected no significant outliers in `age`.
- Grouped rare varieties (sample count < 300) into an "Other" category.

### Step 4: Feature Engineering
- Encoded:
  - `label` -> `label_id` (0–9).
  - `variety_grouped` -> one-hot columns.
- Standardized `age` into `age_scaled` (mean=0, std=1).

### Step 5: Train/Validation Split
- Performed stratified 80/20 split based on `label_id`.
- Saved:
  - `train_processed.csv`
  - `val_processed.csv`
  - `test_processed.csv`

### Step 6: Image Pipeline & DataLoader
- Defined `RiceDataset`:
  - Reads and transforms images.
  - Includes metadata features.
- Created PyTorch DataLoaders ready for modeling.

### Step 7: Final Sanity Check
- Confirmed:
  - No missing/duplicate entries.
  - All image paths valid.
  - One-hot columns binary.
  - `age_scaled` properly standardized.

---

## 3. Deliverables
- **Files**:
  - `train_processed.csv`
  - `val_processed.csv`
  - `test_processed.csv`
- **Folders**:
  - `train_images/`
  - `test_images/`
- **Dataset Class**:
  - `RiceDataset`
- **DataLoaders Ready**:
  - `train_loader`
  - `val_loader`
  - `test_loader`

---

# 📖 README for Modeling Team

## 1. Folder Structure

- `train_images/{label}/{image_id}`
- `test_images/{image_id}`
- `train_processed.csv`
- `val_processed.csv`
- `test_processed.csv`

## 2. CSV Details

- `train_processed.csv` and `val_processed.csv` include:
  - `image_id`
  - `image_path`
  - `label`, `label_id`
  - One-hot variety columns
  - `age_scaled`

- `test_processed.csv` includes:
  - `image_id`
  - `image_path`
  
(No labels provided for test.)

## 3. How to Load Data

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Load CSVs
train_df = pd.read_csv('train_processed.csv')
val_df = pd.read_csv('val_processed.csv')
test_df = pd.read_csv('test_processed.csv')

meta_cols = [c for c in train_df.columns if c.startswith('var_')] + ['age_scaled']

# Image transforms
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# Custom Dataset
class RiceDataset(Dataset):
    def __init__(self, df, transform, meta_cols):
        self.paths = df['image_path'].values
        self.labels = df['label_id'].values if 'label_id' in df else [-1]*len(df)
        self.meta = df[meta_cols].to_numpy(dtype='float32')
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.tf(img)
        meta = torch.from_numpy(self.meta[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, meta, label

# DataLoaders
train_loader = DataLoader(RiceDataset(train_df, train_tf, meta_cols),
                          batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(RiceDataset(val_df, val_tf, meta_cols),
                        batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(RiceDataset(test_df, val_tf, meta_cols),
                         batch_size=32, shuffle=False, num_workers=0)
```

## 4. Next Steps for Modeling
- Build CNN backbone for images.
- Build MLP head for metadata.
- Fuse outputs and classify into 10 classes.

