import torch
import opt_einsum
from omeinsum import OMEinsum


def test_cpu_smoke():
    A = torch.randn(8, 4, 6, 5, requires_grad=True, dtype=torch.float64)
    B = torch.randn(6, 3, requires_grad=True, dtype=torch.float64)
    C = torch.randn(5, 2, requires_grad=True, dtype=torch.float64)
    D = torch.randn(8, 7, requires_grad=True, dtype=torch.float64)

    equation = "abcd,ce,df,ag->ebfg"
    block_dim = 'b'
    batch_size = 2

    ref = opt_einsum.contract(equation, A, B, C, D)

    model = OMEinsum(equation, block_dim=block_dim, batch_size=batch_size, use_checkpoint=False)
    res = model(A, B, C, D)

    assert torch.allclose(ref, res, atol=1e-8)
