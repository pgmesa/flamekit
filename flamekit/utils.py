
import random
from pathlib import Path

import torch
import numpy as np


def seed_all(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    # Torch CPU operations
    torch.manual_seed(seed)
    # Torch GPU operations have a separate seed we also have to set
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def setup_reproducible_env(seed:int):
    """ 
    Seed every random process in 'random', 'numpy' and 'torch' libraries
    to be able to reproduce each random operation when rerunning the same code.

    Additionally, in order to increase efficiency, some operations on a GPU are implemented stochastic
    or are selected on the fly, such as the type of convolutional algorithm to use in each moment.
    These behaviours are disabled after calling this function because we want to ensure that
    all operations are deterministic on GPU (if used) for reproducibility. This can have the side effect
    of reducing a little bit the performance.
    """
    seed_all(seed)
    torch.backends.cudnn.deterministic = True # Use deterministic operations
    torch.backends.cudnn.benchmark = False # Select best algorithms on the fly


def get_next_experiment_path(base_path:'str | Path', exp_name='experiment', mkdir=True) -> Path:
    exp_num = 1
    
    def get_exp_path():
        return Path(base_path)/(exp_name+f"_{exp_num}")
    
    while get_exp_path().exists():
        exp_num += 1
        
    exp_path = get_exp_path()
    if mkdir: exp_path.mkdir(parents=True)
    return exp_path