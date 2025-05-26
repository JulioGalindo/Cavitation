# utils/device_utils.py
import os, torch

def configure_global():
    # MPS high-precision matmuls
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    # full CPU threading
    torch.set_num_threads(os.cpu_count())

def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():   return torch.device('cuda')
    return torch.device('cpu')
