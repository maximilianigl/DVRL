import sys
import torch
import numpy as np


def init(cuda_, seed=None, device=None):
    # Random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Default Tensor
    global cuda
    if torch.cuda.is_available() and cuda_:
        cuda = True
        if device is not None:
            torch.cuda.device(device)
        if seed is not None:
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        cuda = False
        torch.set_default_tensor_type('torch.FloatTensor')
