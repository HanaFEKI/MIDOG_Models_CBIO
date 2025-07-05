# Library imports
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
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
class ResNet50(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        logits = self.model(x).squeeze()
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).float()
        return logits, Y_prob, Y_hat

# Setup logging
experiment_dir = "/cluster/CBIO/home/hfeki/Midog25/MIDOG_2025_Guide/classification_results_all/ResNet50_Eval"
exp_dir = Path(experiment_dir)
exp_dir.mkdir(exist_ok=True, parents=True)

log_file = exp_dir / 'training.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
logger.info(f"Created experiment directory: {str(exp_dir)}")

# Load data from CSV
def load_data_from_csv(csv_path, images_folder, label_col='majority'):
    df = pd.read_csv(csv_path)
    label_map = {'AMF': 0, 'NMF': 1}
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(images_folder, row['image_id'])
        if os.path.isfile(img_path):
            images.append(img_path)
            labels.append(label_map[row[label_col]])

    return images, labels

# Paths
image_dir = "/cluster/CBIO/data1/hfeki/Datasets/dataset original/MIDOG25_Binary_Classification_Train_Set"
dataset_file = "/cluster/CBIO/data1/hfeki/Datasets/dataset original/MIDOG25_Atypical_Classification_Train_Set.csv"

# Load and split data
images, labels = load_data_from_csv(dataset_file, image_dir)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# DataLoader
test_dataset = InferenceDataset(test_images, test_labels, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# Load models
import timm
num_classes = 1
num_folds = 5
model_paths = [f"/cluster/CBIO/home/hfeki/Midog25/MIDOG_2025_Guide/classification_results_all/classif_resnet50/MIDOG25_binary_classification_baseline_fold_{i + 1}_best.pth" for i in range(num_folds)]

models = []
for i, path in enumerate(model_paths):
    base_model = timm.create_model('resnet50', pretrained=False)
    model = ResNet50(base_model, num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models.append(model)
    logger.info(f"Loaded model from: {path}")

# Inference
true_labels = np.array(test_labels)
fold_bal_accs, fold_aurocs = [], []
fold_probs_dict = {}

for i, model in enumerate(models):
    fold_probs = []

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Inference Fold {i + 1}"):
            images = images.to(device)
            _, probs, _ = model(images)
            fold_probs.extend(probs.cpu().numpy())

    fold_probs = np.array(fold_probs)
    fold_preds = (fold_probs > 0.5).astype(int)

    bal_acc = balanced_accuracy_score(true_labels, fold_preds)
    auroc = roc_auc_score(true_labels, fold_probs)

    fold_bal_accs.append(bal_acc)
    fold_aurocs.append(auroc)

    logger.info(f"Fold {i + 1} - Balanced Accuracy: {bal_acc:.4f}, AUROC: {auroc:.4f}")

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

logger.info("\n--- Per-Fold Evaluation Summary (ResNet50) ---")
logger.info(f"Balanced Accuracy: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
logger.info(f"AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")

# Save predictions
output_path = exp_dir / "ResNet50_predictions.pkl"
with open(output_path, "wb") as f:
    pickle.dump(fold_probs_dict, f)
logger.info(f"Saved fold predictions and labels to: {output_path}")
