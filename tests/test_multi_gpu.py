import pytest
import torch
import opt_einsum
from omeinsum import OMEinsum


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
def test_multi_gpu_correctness():
    num_gpus = torch.cuda.device_count()

    # Choose 'b' dimension as a multiple of num_gpus * 4 to allow many chunks
    b = num_gpus * 4

    # Construct shapes (abcd, ce, df, ag -> ebfg)
    A = torch.randn(16, b, 12, 10, requires_grad=True, dtype=torch.float64)
    B = torch.randn(12, 6, requires_grad=True, dtype=torch.float64)
    C = torch.randn(10, 7, requires_grad=True, dtype=torch.float64)
    D = torch.randn(16, 9, requires_grad=True, dtype=torch.float64)

    equation = "abcd,ce,df,ag->ebfg"
    block_dim = 'b'

    # We want number of chunks divisible by num_gpus.
    # Set desired chunks = num_gpus * 2  => batch_size = b / (num_gpus * 2) = 2
    batch_size = 2

    # Sanity: ensure divisibility holds for this construction
    num_chunks = (b + batch_size - 1) // batch_size
    assert num_chunks % num_gpus == 0

    ref = opt_einsum.contract(equation, A, B, C, D)

    model = OMEinsum(equation, block_dim=block_dim, batch_size=batch_size, use_checkpoint=False)
    res = model(A, B, C, D)

    assert torch.allclose(ref.to(device=torch.device("cpu")), res.to(device=torch.device("cpu")), atol=1e-8)
