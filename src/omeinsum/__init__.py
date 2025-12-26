import os

import torch

from .omeinsum import OMEinsum

CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
os.environ.setdefault("OMEINSUM_GPU_COUNT", str(GPU_COUNT))


def get_gpu_count() -> int:
    return GPU_COUNT

__all__ = ["OMEinsum", "CUDA_AVAILABLE", "GPU_COUNT", "get_gpu_count"]
