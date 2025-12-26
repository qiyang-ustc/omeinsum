import pytest
import torch
import opt_einsum
from omeinsum import OMEinsum


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_smoke():
    device = torch.device("cuda:0")
    A = torch.randn(8, 4, 6, 5, requires_grad=True, dtype=torch.float64, device=device)
    B = torch.randn(6, 3, requires_grad=True, dtype=torch.float64, device=device)
    C = torch.randn(5, 2, requires_grad=True, dtype=torch.float64, device=device)
    D = torch.randn(8, 7, requires_grad=True, dtype=torch.float64, device=device)

    equation = "abcd,ce,df,ag->ebfg"
    block_dim = "b"
    batch_size = 2

    ref = opt_einsum.contract(equation, A, B, C, D)

    model = OMEinsum(equation, block_dim=block_dim, batch_size=batch_size, use_checkpoint=False)
    res = model(A, B, C, D)

    assert torch.allclose(ref.cpu(), res.cpu(), atol=1e-8)
