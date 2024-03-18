# For systems
import os
import logging

# For external libraries
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Training related global variables
RANDOM_SEED = 413
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = ['target_class_kidney', 'target_class_lung', 'target_class_breast',
                  'target_class_hematopoietic_lymphoid', 'target_class_colon', 'target_class_prostate',
                  'target_class_ovary', 'target_class_skin', 'target_class_brain']

# Path definitions
CUR_FILE_PATH = os.path.abspath(__file__)
SRC_PATH = os.path.dirname(CUR_FILE_PATH)

# Define the dataset directory path
DATASET_DIR = os.path.join(os.path.dirname(SRC_PATH), 'datasets')
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

