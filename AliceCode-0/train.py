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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from tqdm.notebook import tqdm 

from utils.alice_utils import MitosisClassifier, ClassificationDataset, MitosisTrainer
import timm 

image_dir = "/cluster/CBIO/data1/hfeki/Datasets/dataset original/MIDOG25_Binary_Classification_Train_Set"
csv_path ="/cluster/CBIO/data1/hfeki/Datasets/dataset original/MIDOG25_Atypical_Classification_Train_Set.csv"

def load_csv_data(csv_path,image_dir,label_col='majority'):
    df=pd.read_csv(csv_path)
    label_map={'AMF':0,'NMF':1}
    labels, images_paths= [] , []
    for index,row in df.iterrows():
        img_path=os.path.join(image_dir,row['image_id'])
        if os.path.isfile(img_path):
            images_paths.append(img_path)
            label=label_map[row[label_col]]
            labels.append(label)
    return images_paths, labels

images_paths, labels=load_csv_data(csv_path,image_dir)

#train test split
train_images,val_images, train_labels, val_labels = train_test_split(images_paths, labels, test_size=0.2, random_state=42)


# Model 
checkpoint_path= Path("/cluster/CBIO/data1/ablondel1/cell_SSL_data/best_resnet/best.pth.tar")


# the state dictionary is a mapping of layer names to weights
def select_only_relevant_weights(state_dict):
    """ 
    Select only relevant weights for the model. Useful to load weights from a different model trained with moco.
    Keep only the backbone weights (here the layers of ResNet) — excludes the fc head, which is task-specific (Classification).
    Remove the "module.base_encoder." prefix to make the keys compatible with the timm resnet50 model.
    """
    for key in list(state_dict.keys()):
        if key.startswith('module.base_encoder') and not key.startswith('module.base_encoder.%s' % 'fc'):
            # remove prefix
            state_dict[key[len("module.base_encoder."):]] = state_dict[key]
        # delete renamed or unused keys
        del state_dict[key]
    return state_dict


# ResNet-50 encoder without classification head : num_classes=0 tells timm to remove the final classification layer
encoder = timm.create_model("resnet50", pretrained=False, num_classes=0)
checkpoint=torch.load(checkpoint_path, map_location="cpu", weights_only=True)
state_dict = checkpoint["state_dict"]
state_dict = select_only_relevant_weights(state_dict)
encoder.load_state_dict(state_dict, strict=True) #strict=True ensures every weight in the model matches — helps catch errors if something doesn’t fit.

# Set up training configurations
num_epochs = 10
batch_size = 128
num_folds = 5
lr=1e-4
experiment_dir = 'logs_and_weights'

# Set up the trainer
trainer= MitosisTrainer(encoder=encoder, experiment_dir=experiment_dir, num_epochs=num_epochs, batch_size=batch_size, lr=lr, num_folds=num_folds)

# Run the k-fold cross validation and evaluate on the test set
val_accuracies, test_accuracies, test_auc_roc_scores = trainer.train_and_evaluate(train_images=train_images,train_labels=train_labels, test_images=val_images,test_labels=val_labels)

