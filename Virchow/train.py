# Setup and Libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchivision import transformers
from torchvision.utils.data import Dataset, 
import timm
from tqdm as tqdm


from PIL import Image
import pandas as pd
import numpy as np
import logging

from sklearn.metric import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold



