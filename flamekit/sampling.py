import torch
import numpy as np
import torch.utils.data as data

from typing import Iterator, Sequence


class CustomRandomSampler(data.Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator
        self.selected_indices_order = []

    def __iter__(self) -> Iterator[int]:
        self.selected_indices_order = []
        for i in torch.randperm(len(self.indices), generator=self.generator):
            ind = self.indices[i]
            self.selected_indices_order.append(ind)
            yield ind

    def __len__(self) -> int:
        return len(self.indices)
    

class CustomSequentialSampler(data.Sampler[int]):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        self.selected_indices_order = []
        for ind in self.indices:
            yield ind

    def __len__(self) -> int:
        return len(self.indices)
    

def split_dataset(dataset:data.Dataset, val_split=0.2, test_split=0.1, shuffle=True, test_indices=None) -> tuple: 
    
    def get_split_indices(data_size, test_split, val_split, test_indices):
        # Creating data indices for training and validation splits:
        indices = list(range(data_size))
        if test_indices is None:
            split = int(np.floor(test_split * data_size))
            if shuffle:
                np.random.shuffle(indices)
            train_val_indices, test_indices = indices[split:], indices[:split]
        else:
            for index in sorted(test_indices, reverse=True):
                indices.pop(index)
            if shuffle:
                np.random.shuffle(indices)
            train_val_indices = indices        
        
        val_split = int(np.floor(val_split * len(train_val_indices)))
        train_indices, val_indices = train_val_indices[val_split:], train_val_indices[:val_split]
        
        return train_indices, val_indices, test_indices

    train_indices, val_indices, test_indices = \
        get_split_indices(len(dataset), test_split, val_split, test_indices)

    # Creating PT data samplers and loaders:
    return train_indices, val_indices, test_indices