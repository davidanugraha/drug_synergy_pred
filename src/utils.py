# For systems
import os

import torch

CUR_FILE_PATH = os.path.abspath(__file__)
SRC_PATH = os.path.dirname(CUR_FILE_PATH)

RANDOM_SEED = 413

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
