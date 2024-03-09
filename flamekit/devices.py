
from collections.abc import Iterable

import torch
import torch.nn as nn
from tabulate import tabulate


def print_cuda_available_devices(tabulate_fmt='pretty'):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_info = []
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_info.append([i, device_name])
        print("Available CUDA devices:")
        print(tabulate(device_info, headers=['Index', 'Device Name'], tablefmt=tabulate_fmt))
    else:
        print("No CUDA devices were found")
        

def select_device(index=0, cuda=True):
    if torch.cuda.is_available() and cuda:
        device = torch.device(f"cuda:{index}")
    else:
        device = torch.device("cpu")
    return device


def to_device(data, device:'str | torch.DeviceObjType') -> 'Iterable | torch.Tensor':
    """ Recursively send to device """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        elements = {}
        for val, element in data.items():
            element = to_device(element, device)
            elements[val] = element
        return elements
    elif isinstance(data, Iterable):
        elements = []
        for element in data:
            element = to_device(element, device)
            elements.append(element)
        return elements
    else:
        raise ValueError(f"Some elements in data are not either Iterables or Tensors '{type(data)}'")
    
    
class DataParallelWrapper(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)