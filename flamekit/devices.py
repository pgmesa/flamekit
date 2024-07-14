
from collections.abc import Iterable

from flamekit.utils import to_mib, to_mb

import torch
from tabulate import tabulate


def print_cuda_memory_stats(device=None, in_mib=True):
    if in_mib: unit = "MiB"; converter = to_mib
    else: unit = "MB"; converter = to_mb
    alloc_mem = converter(torch.cuda.memory_allocated(device))
    res_mem = converter(torch.cuda.memory_reserved(device))
    max_alloc_mem = converter(torch.cuda.max_memory_allocated(device))
    max_res_mem = converter(torch.cuda.max_memory_reserved(device))
    print("Allocated Memory:     ", round(alloc_mem, ndigits=2), unit)
    print("Reserved Memory:      ", round(res_mem, ndigits=2), unit)
    print("Max Memory Allocated: ", round(max_alloc_mem, ndigits=2), unit)
    print("Max Memory Reserved:  ", round(max_res_mem, ndigits=2), unit)


def print_cuda_available_devices(tabulate_fmt='pretty'):
    """ 
    Print available CUDA devices
    
    Args:
        tabulate_fmt (str): Format for printing the table. Defaults to 'pretty'
    """
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
        

def select_device(index=0, cuda=True) -> torch.device:
    """ Selects device """
    if torch.cuda.is_available() and cuda:
        device = torch.device(f"cuda:{index}")
    else:
        device = torch.device("cpu")
    return device


def to_device(data:'torch.Tensor | Iterable', device:'str | torch.DeviceObjType') -> 'Iterable | torch.Tensor':
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
    raise ValueError(f"Some elements in data are not either Iterables or Tensors '{type(data)}'")