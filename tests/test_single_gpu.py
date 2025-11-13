import pytest
import torch
import opt_einsum
from omeinsum import OMEinsum


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() != 1, reason="requires exactly 1 CUDA device")
def test_single_gpu_correctness():
    device = torch.device("cuda:0")

    # Shapes
    A = torch.randn(16, 8, 12, 10, requires_grad=True, dtype=torch.float64, device=device)
    B = torch.randn(12, 6, requires_grad=True, dtype=torch.float64, device=device)
    C = torch.randn(10, 7, requires_grad=True, dtype=torch.float64, device=device)
    D = torch.randn(16, 9, requires_grad=True, dtype=torch.float64, device=device)

    equation = "abcd,ce,df,ag->ebfg"
    block_dim = 'b'
    batch_size = 3  # any positive; chunks % 1 == 0 always

    ref = opt_einsum.contract(equation, A, B, C, D)

    model = OMEinsum(equation, block_dim=block_dim, batch_size=batch_size, use_checkpoint=False)
    res = model(A, B, C, D)

    assert torch.allclose(ref, res, atol=1e-8)
