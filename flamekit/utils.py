
import random
from pathlib import Path

import torch
import numpy as np


def to_mb(num_bytes:float):
    return num_bytes / (1000**2)  # Bytes to MB


def to_mib(num_bytes:float):
    return num_bytes / (1024**2)  # Bytes to MiB


def tensor_memory_size(tensor):
    # Get number of elements in the tensor
    num_elements = tensor.numel()
    # Get size of each element in bytes
    element_size = tensor.element_size()
    # Total memory in bytes
    total_bytes = num_elements * element_size
    return total_bytes


def seed_all(seed:int):
    """ 
    Seeds every random process in the 'random', 'numpy', and 'torch' libraries
    to reproduce each random operation when rerunning the same code.
    """
    random.seed(seed)
    np.random.seed(seed=seed)
    # For Torch CPU operations
    torch.manual_seed(seed)
    # For Torch GPU operations, a separate seed also needs to be set
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def set_up_reproducible_env(seed:int):
    """ 
    Seeds every random process using the seed_all() method and disables dynamic algorithm selection.

    To improve efficiency, some operations on a GPU are implemented stochastically
    or are selected dynamically, such as the type of convolutional algorithm used at any given moment.
    
    These behaviors are disabled after calling this function to ensure that
    all operations are deterministic on the GPU (if used) for reproducibility. This may result
    in a slight reduction in performance.
    """
    seed_all(seed)
    torch.backends.cudnn.deterministic = True # Use deterministic operations
    torch.backends.cudnn.benchmark = False # Disable dynamic algorithm selection


def get_next_experiment_path(base_path:'str | Path', exp_name='experiment', mkdir=True) -> Path:
    """
    Generate a path for the next experiment, ensuring it does not overwrite an existing one.

    Parameters:
    - base_path (str or Path): The base path where experiment directories will be created.
    - exp_name (str): The base name for the experiment directory. Defaults to 'experiment'.
    - mkdir (bool): If True, creates the experiment directory if it doesn't exist. Defaults to True.

    Returns:
    - Path: The path to the next available experiment directory.

    Example:
    >>> get_next_experiment_path('/path/to/experiments')
    Path('/path/to/experiments/experiment_1')
    """
    exp_num = 1
    
    def get_exp_path():
        return Path(base_path)/(exp_name+f"_{exp_num}")
    
    while get_exp_path().exists():
        exp_num += 1
        
    exp_path = get_exp_path()
    if mkdir: exp_path.mkdir(parents=True)
    return exp_path