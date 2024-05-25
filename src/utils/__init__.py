from .logger import Logger
from .visualize import visualize
import numpy as np
import torch
import random
import os


def set_seed(seed_value=1234):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.manual_seed(seed_value) 
    torch.cuda.manual_seed(seed_value)