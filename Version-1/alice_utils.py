from typing import List, Tuple, Union, Optional

import albumentations as A # image augmentation library
import logging
import numpy as np 
import torchvision # torchvision for pre-trained models and transformations
import torch
import torch.nn as nn
import torch.optim as optim
import timm


from pathlib import Path 
from PIL import Image
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision.models import get_model
from tqdm.notebook import tqdm 

from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

mean = [0.485, 0.456, 0.406] # Imaginet
std  = [0.229, 0.224, 0.225] # Imaginet


# code provided by Alice and adopted in our case
class LinearBatchNorm(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, constant_size=True, dim_batch=None):
        super().__init__()
        if dim_batch is None:
            dim_batch = out_features
        self.cs=constant_size
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            self.get_norm(constant_size, dim_batch),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def get_norm(self, constant_size, out_features):
        if not constant_size:
            norm = nn.InstanceNorm1d(out_features)
        else:
            norm = nn.BatchNorm1d(out_features)
        return norm

    def forward(self, x):
        return self.block(x)


class MitosisClassifier(nn.Module):
    def __init__(self, encoder, classifier_hidden_dims=[512, 128], dropout=0.5, num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.feature_depth = 2048  # output of ResNet50
        self.classifier_hidden_dims = classifier_hidden_dims
        self.dropout = dropout
        self.num_classes = num_classes
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        layers = []
        prev_dim = self.feature_depth
        for hidden_dim in self.classifier_hidden_dims:
            layers.append(LinearBatchNorm(prev_dim, hidden_dim, self.dropout, constant_size=True))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.num_classes))  # last layer: linear only
        return nn.Sequential(*layers)


    def forward(self, x):
        features = self.encoder(x)              # shape: [B, 2048]
        if features.dim() == 4:
            features = features.mean([2, 3])  # GAP if needed
        logits = self.classifier(features).squeeze(1)  # [B]
        Y_prob = torch.sigmoid(logits)                  # shape: [B]
        Y_hat = (Y_prob > 0.5).float()                  # shape: [B]
        return logits, Y_prob, Y_hat                   



class ClassificationDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)

        return image, label




class MitosisTrainer:
    def __init__(self, encoder, experiment_dir: str, num_epochs: int=2, batch_size: int=128, lr: float=1e-4, num_folds: int=5):
        self.encoder=encoder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_folds = num_folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()
        self.classes = ["Atypical", "Normal"]
        self.experiment_dir = experiment_dir 


    @property
    def train_transform(self):
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(size=48),
            ])
        return transform


    @property
    def val_transform(self):
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(size=48),
    ])
        return transform



    def setup_experiment(self):
        """Set up experiment directory to save log files and model checkpoints"""
        self.exp_dir = Path(self.experiment_dir)
        self.exp_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.exp_dir / 'training.log'
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(self.log_file),
                            logging.StreamHandler()
                        ]) 
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created experiment directoy: {str(self.exp_dir)}.")


    def prepare_data_loaders(self, images, labels, transform, is_training=True):
        """
        Prepare DataLoader objects for training or evaluation.

        Args:
            images (list): List of image paths or data
            labels (list): List of corresponding labels
            transform: Transformations to apply to the images
            is_training (bool, optional): Whether this is for training (uses sampler) or evaluation.
                                        Defaults to True.

        Returns:
            DataLoader: PyTorch DataLoader object
        """
        dataset = ClassificationDataset(images, labels, transform=transform)

        if is_training:
            # Compute class weights for WeightedRandomSampler
            class_counts = [0, 0]
            for lbl in labels:
                class_counts[lbl] += 1

            class_weights = [1.0 / max(count, 1) for count in class_counts]
            sample_weights = [class_weights[lbl] for lbl in labels]

            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=8,
                pin_memory=True
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

        return loader


    def train_epoch(self, model, train_loader, optimizer):
        """
        Train the model for one epoch.

        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            optimizer: Optimizer for updating model parameters

        Returns:
            tuple: (average training loss, predictions, targets)
        """
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        train_probs=[]

        for images_batch, labels_batch in tqdm(train_loader, desc="Training"):
            images_batch, labels_batch = images_batch.to(self.device), labels_batch.to(self.device)

            optimizer.zero_grad()
            logits, Y_prob, Y_hat = model(images_batch)

            loss = self.criterion(logits, labels_batch.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(Y_hat.cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())
            train_probs.extend(Y_prob.detach().cpu().numpy())


        return train_loss / len(train_loader), train_preds, train_targets , train_probs


    def evaluate(self, model, data_loader, phase="Validation"):
        """
        Evaluate the model on the provided data.

        Args:
            model: The neural network model
            data_loader: DataLoader for evaluation data
            phase (str, optional): Phase name for progress bar. Defaults to "Validation".

        Returns:
            tuple: (average loss, predictions, targets)
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        val_probs=[]

        with torch.no_grad():
            for images_batch, labels_batch in tqdm(data_loader, desc=phase):
                images_batch, labels_batch = images_batch.to(self.device), labels_batch.to(self.device)

                logits, Y_prob, Y_hat = model(images_batch)
                loss = self.criterion(logits, labels_batch.float())

                total_loss += loss.item()
                all_preds.extend(Y_hat.cpu().numpy())
                all_targets.extend(labels_batch.cpu().numpy())
                val_probs.extend(Y_prob.detach().cpu().numpy())

        return total_loss / len(data_loader), all_preds, all_targets, val_probs


    def train_fold(self, fold, train_images, train_labels, val_images, val_labels):
        """
        Train and validate the model for one fold.

        Args:
            fold (int): Current fold number
            train_images (list): Training images for this fold
            train_labels (list): Training labels for this fold
            val_images (list): Validation images for this fold
            val_labels (list): Validation labels for this fold

        Returns:
            tuple: (best validation accuracy, path to best model checkpoint)
        """
        # Build DataLoaders
        train_loader = self.prepare_data_loaders(train_images, train_labels, self.train_transform, is_training=True)
        val_loader = self.prepare_data_loaders(val_images, val_labels, self.val_transform, is_training=False)

        # Initialize model, optimizer, scheduler
        model = MitosisClassifier(encoder=self.encoder,classifier_hidden_dims=[1024, 512, 128, 64],dropout=0.3,num_classes=1).to(self.device)
        for param in model.encoder.parameters(): #freezing the encoder and training the fc layer
            param.requires_grad = False
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-7)

        best_val_auc_roc = 0.0
        best_model_path = self.exp_dir / f'MIDOG25_binary_classification_baseline_fold{fold + 1}_best.pth'

        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_preds, train_targets, train_probs = self.train_epoch(model, train_loader, optimizer)
            train_balanced_accuracy = balanced_accuracy_score(train_targets, train_preds)
            train_auc_roc = roc_auc_score(train_targets, train_probs)
            # Validation
            val_loss, val_preds, val_targets, val_probs = self.evaluate(model, val_loader, "Validation")
            val_balanced_accuracy = balanced_accuracy_score(val_targets, val_preds)
            val_auc_roc=roc_auc_score(val_targets, val_probs)

            # Save best model
            if val_auc_roc > best_val_auc_roc:
                best_val_auc_roc = val_auc_roc
                torch.save(model.state_dict(), best_model_path)

            scheduler.step()

            # Logging
            self.logger.info(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Balanced Acc: {train_balanced_accuracy:.4f}, "
                f"Train Auc-ROC score: {train_auc_roc:.4f} | "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Balanced Acc: {val_balanced_accuracy:.4f}, "
                f"Val Auc-ROC score: {val_auc_roc:.4f} | "
            )

        return best_val_auc_roc, best_model_path


    def train_and_evaluate(self, train_images, train_labels, test_images=None, test_labels=None):
        """
        Perform k-fold cross-validation training and optional test set evaluation.

        Args:
            train_images (list): Complete set of training images
            train_labels (list): Complete set of training labels
            test_images (list, optional): Test set images. Defaults to None.
            test_labels (list, optional): Test set labels. Defaults to None.

        Returns:
            list: List of best validation accuracies for each fold
        """
        # Setup logging 
        self.setup_experiment()

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_accuracies = []
        best_model_paths = []

        # K-fold Cross Validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_images, train_labels)):
            self.logger.info(f"Starting Fold {fold + 1}")

            # Split data
            fold_train_images = [train_images[i] for i in train_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_images = [train_images[i] for i in val_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]

            # Train fold
            best_auc_roc, best_model_path = self.train_fold(
                fold,
                fold_train_images,
                fold_train_labels,
                fold_val_images,
                fold_val_labels
            )

            fold_accuracies.append(best_auc_roc)
            best_model_paths.append(best_model_path)
            self.logger.info(f"Fold {fold + 1} - Best Validation Auc-Roc score : {best_auc_roc:.4f}")

        self.logger.info(f"Average Validation Auc-Roc score : {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")


        # Test set evaluation
        if test_images is not None and test_labels is not None:
            self.logger.info("Evaluating on test set...")
            test_loader = self.prepare_data_loaders(test_images, test_labels, self.val_transform, is_training=False)

            # Evaluate each fold's best model on test set
            test_accuracies = []
            test_auc_roc_scores=[]
            for fold, model_path in enumerate(best_model_paths):
                model =MitosisClassifier(encoder=self.encoder, classifier_hidden_dims=[1024, 512, 128, 64],dropout=0.3,num_classes=1).to(self.device)
                model.load_state_dict(torch.load(model_path))

                test_loss, test_preds, test_targets, test_probs = self.evaluate(model, test_loader, "Test")
                test_balanced_accuracy = balanced_accuracy_score(test_targets, test_preds)
                test_accuracies.append(test_balanced_accuracy)
                test_auc_roc=roc_auc_score(test_targets,test_probs)
                test_auc_roc_scores.append(test_auc_roc)


                self.logger.info(
                    f"Fold {fold + 1} - Test Balanced Accuracy: {test_balanced_accuracy:.4f} - Test AUC-ROC score: {test_auc_roc:.4f}"
                    )

            self.logger.info(
                f"Average Test Balanced Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f} | "
                f"Average AUC-ROC score: {np.mean(test_auc_roc_scores):.4f} ± {np.std(test_auc_roc_scores):.4f}"
                )

            return fold_accuracies, test_accuracies,test_auc_roc_scores

        return fold_accuracies
