# Library imports
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd 
import plotly.express as px 
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from tqdm.notebook import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import timm
import pickle

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms (same as validation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class
class InferenceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Model definition
class Model_def:
    

# Load test dataset
# Importing the dataset and image directory
image_dir = "/cluster/CBIO/data1/hfeki/Datasets/dataset original/MIDOG25_Binary_Classification_Train_Set"
dataset_file = "/cluster/CBIO/data1/hfeki/Datasets/dataset original/MIDOG25_Atypical_Classification_Train_Set.csv"

# Set up data 
def load_data_from_csv(csv_path, images_folder, label_col='majority'):
    """
    Reads a CSV file that contains image filenames and a 'majority' column 
    indicating the label ('AMF' or 'NMF'). 
    Returns:
        images (list of str): Full paths to images
        labels (list of int): Numeric labels (0 for AMF -> Atypical, 1 for NMF -> Normal)
    """
    df = pd.read_csv(csv_path)
    
    # Map string labels to numeric
    label_map = {
        'AMF': 0,  # Atypical
        'NMF': 1   # Normal
    }

    images = []
    labels = []

    for _, row in df.iterrows():
        img_name = row['image_id']
        label_str = row[label_col]
        img_path = os.path.join(images_folder, img_name)
        if not os.path.isfile(img_path):
            continue
        
        images.append(img_path)
        labels.append(label_map[label_str])
    
    return images, labels

# Load data 
images, labels = load_data_from_csv(dataset_file, image_dir, label_col='majority')

# Split data into training and test split 
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Prepare dataset and loader
test_dataset = InferenceDataset(test_images, test_labels, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# Load models
num_folds = 5
model_paths = [f"/cluster/CBIO/home/hfeki/Midog25/MIDOG_2025_Guide/classification_results_all/classif_resnet50/MIDOG25_binary_classification_baseline_fold_{i + 1}_best.pth" for i in range(num_folds)]
models = []

for path in model_paths:
    model = BinaryEfficientNetV2M().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models.append(model)

# Inference
true_labels = np.array(labels)
fold_bal_accs, fold_aurocs = [], []
fold_probs_dict = {}

for i, model in enumerate(models):
    fold_probs = []

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Inference Fold {i + 1}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            fold_probs.extend(probs)

    fold_probs = np.array(fold_probs)
    fold_preds = (fold_probs > 0.5).astype(int)

    bal_acc = balanced_accuracy_score(true_labels, fold_preds)
    auroc = roc_auc_score(true_labels, fold_probs)

    fold_bal_accs.append(bal_acc)
    fold_aurocs.append(auroc)

    print(f"\nFold {i + 1} - Balanced Accuracy: {bal_acc:.4f}, AUROC: {auroc:.4f}")

    fold_probs_dict[f"fold_{i + 1}"] = {
        "probs": fold_probs,
        "preds": fold_preds,
        "true_labels": true_labels
    }

# Summary
mean_bal_acc = np.mean(fold_bal_accs)
std_bal_acc = np.std(fold_bal_accs)
mean_auroc = np.mean(fold_aurocs)
std_auroc = np.std(fold_aurocs)

print("\n--- Per-Fold Evaluation Summary (EfficientNetV2-M) ---")
print(f"Balanced Accuracy: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
print(f"AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")

# Save predictions
output_path = "ResNet50_predictions.pkl"
with open(output_path, "wb") as f:
    pickle.dump(fold_probs_dict, f)

print(f"\nSaved fold predictions and labels to: {output_path}")
