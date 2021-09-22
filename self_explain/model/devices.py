import logging 
import torch
import multiprocessing 

def get_gpus(requested: int=None) -> int:
    """ Return number of GPUS to use based on requested and available devices
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = ", ".join([torch.cuda.get_device_name(i) for i in range(device_count)])
    else:
        device_count = 0
        device_names = []
    print(f"found {device_count} GPUs: {device_names}")
    # return device_count if None else minimum of requested and available GPUs
    return device_count if requested is None else min(device_count, requested)


def get_cpus(requested: int=None) -> int:
    """ Return number of CPUs available
    """
    device_count = multiprocessing.cpu_count()
    return device_count if requested is None else min(device_count, max(1, requested))

