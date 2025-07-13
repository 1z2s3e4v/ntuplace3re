import os
import sys
import argparse

import numpy as np
from numpy import ndarray, random
from torch import Tensor, no_grad
from torch.nn import functional as F

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def ensure_positive(tensor: Tensor, alpha: float = 1) -> Tensor:
    out = F.elu(tensor, alpha=alpha) + alpha
    assert (out >= 0).all(), out[out < 0]
    return out

def flatten(data: Tensor, batch: bool = True) -> Tensor:
    if batch:
        return data.reshape(data.shape[0], -1)  # Use `view` will get error...QQ
    else:
        return data.view(-1)
    
class ArgxParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--aux', default=None, type=str, help='benchmark aux')
        self.parser.add_argument('--cfg', required=True, type=str, help='config json')
        self.parser.add_argument('--msg', required=None, type=str, help='features for this run')
