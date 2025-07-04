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
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from tqdm.notebook import tqdm 

from utils.crop_classif_utils import MitosisClassifier, ClassificationDataset, MitosisTrainer


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


# Set up training configurations
num_epochs = 10
batch_size = 128
num_folds = 5
lr=1e-4
model_name = 'convnext_small'
weights = 'IMAGENET1K_V1'
experiment_dir = 'classification_results_convnext_crop60'


# Set up the trainer
trainer = MitosisTrainer(
    model_name=model_name,
    weights=weights,
    num_epochs=num_epochs,
    batch_size=batch_size,
    num_folds=num_folds,
    lr=lr,
    experiment_dir=experiment_dir
)

# Run the k-fold cross validation and evaluate on the test set
val_accuracies, test_accuracies = trainer.train_and_evaluate(
    train_images=train_images,
    train_labels=train_labels, 
    test_images=test_images,
    test_labels=test_labels
)
