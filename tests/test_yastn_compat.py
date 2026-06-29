import pytest
import torch

from omeinsum.compat.block_gemm import yastn_block_gemm_stats, yastn_transpose_dot_sum
from omeinsum.compat.scheduler import BlockGemmPolicy
from omeinsum.compat.yastn import enable_yastn_omeinsum

yastn = pytest.importorskip("yastn")
backend_torch = pytest.importorskip("yastn.backend.backend_torch")


def test_yastn_wrapper_overrides_only_transpose_dot_sum():
    backend = enable_yastn_omeinsum(backend_torch)

    assert backend.BACKEND_ID == backend_torch.BACKEND_ID
    assert backend.transpose_dot_sum != backend_torch.transpose_dot_sum
    assert backend.dot is backend_torch.dot
    assert backend.svd is backend_torch.svd


def test_block_gemm_accumulates_multiple_contributions():
    A0 = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    A1 = torch.arange(6, 12, dtype=torch.float64).reshape(2, 3)
    B0 = torch.arange(12, 18, dtype=torch.float64).reshape(3, 2)
    B1 = torch.arange(18, 24, dtype=torch.float64).reshape(3, 2)

    Adata = torch.cat([A0.reshape(-1), A1.reshape(-1)])
    Bdata = torch.cat([B0.reshape(-1), B1.reshape(-1)])
    Areshape = (((0, 6), (2, 3), 2, 3), ((6, 12), (2, 3), 2, 3))
    Breshape = (((0, 6), (3, 2), 3, 2), ((6, 12), (3, 2), 3, 2))
    meta_dot = [((0, 4), (2, 2), [(0, 0), (1, 1)])]

    run_stats: dict[str, int] = {}
    out = yastn_transpose_dot_sum(
        Adata,
        Bdata,
        meta_dot,
        Areshape,
        Breshape,
        (0, 1),
        (0, 1),
        4,
        policy=BlockGemmPolicy(batch_size=2, min_group=2, stats=run_stats),
    )

    assert torch.allclose(out.reshape(2, 2), A0 @ B0 + A1 @ B1)
    assert run_stats["yastn_compat_block_gemm_calls"] == 1
    assert run_stats["yastn_compat_block_gemm_contributions"] == 2
    assert run_stats["yastn_compat_block_gemm_unique_shapes"] == 1
    assert run_stats["yastn_compat_block_gemm_shape_group_size_2_4"] == 1
    assert run_stats["yastn_compat_block_gemm_batched_tasks"] == 2

    stats = yastn_block_gemm_stats(meta_dot, Areshape, Breshape)
    assert stats.num_contributions == 2
    assert stats.unique_shapes == 1
    assert stats.total_flops == 2 * 2 * 2 * 3 * 2


def test_block_gemm_batched_policy_matches_unbatched_backward():
    torch.manual_seed(11)
    Ablocks = [torch.randn(2, 3, dtype=torch.float64) for _ in range(4)]
    Bblocks = [torch.randn(3, 2, dtype=torch.float64) for _ in range(4)]
    Adata0 = torch.cat([block.reshape(-1) for block in Ablocks]).requires_grad_(True)
    Bdata0 = torch.cat([block.reshape(-1) for block in Bblocks]).requires_grad_(True)
    Adata1 = Adata0.detach().clone().requires_grad_(True)
    Bdata1 = Bdata0.detach().clone().requires_grad_(True)
    Areshape = tuple(((6 * idx, 6 * (idx + 1)), (2, 3), 2, 3) for idx in range(4))
    Breshape = tuple(((6 * idx, 6 * (idx + 1)), (3, 2), 3, 2) for idx in range(4))
    meta_dot = [
        ((0, 4), (2, 2), [(0, 0), (1, 1)]),
        ((4, 8), (2, 2), [(2, 2), (3, 3)]),
    ]

    out_ref = yastn_transpose_dot_sum(
        Adata0,
        Bdata0,
        meta_dot,
        Areshape,
        Breshape,
        (0, 1),
        (0, 1),
        8,
        policy=BlockGemmPolicy(batch_size=1, min_group=999),
    )
    out_new = yastn_transpose_dot_sum(
        Adata1,
        Bdata1,
        meta_dot,
        Areshape,
        Breshape,
        (0, 1),
        (0, 1),
        8,
        policy=BlockGemmPolicy(batch_size=2, min_group=2),
    )
    upstream = torch.randn_like(out_ref)
    grad_ref = torch.autograd.grad((out_ref * upstream).sum(), (Adata0, Bdata0))
    grad_new = torch.autograd.grad((out_new * upstream).sum(), (Adata1, Bdata1))

    assert torch.allclose(out_new, out_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(grad_new[0], grad_ref[0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(grad_new[1], grad_ref[1], atol=1e-12, rtol=1e-12)


def test_block_gemm_custom_no_stack_matches_unbatched_backward():
    torch.manual_seed(17)
    Ablocks = [torch.randn(2, 3, dtype=torch.float64) for _ in range(4)]
    Bblocks = [torch.randn(3, 2, dtype=torch.float64) for _ in range(4)]
    Adata0 = torch.cat([block.reshape(-1) for block in Ablocks]).requires_grad_(True)
    Bdata0 = torch.cat([block.reshape(-1) for block in Bblocks]).requires_grad_(True)
    Adata1 = Adata0.detach().clone().requires_grad_(True)
    Bdata1 = Bdata0.detach().clone().requires_grad_(True)
    Areshape = tuple(((6 * idx, 6 * (idx + 1)), (2, 3), 2, 3) for idx in range(4))
    Breshape = tuple(((6 * idx, 6 * (idx + 1)), (3, 2), 3, 2) for idx in range(4))
    meta_dot = [
        ((0, 4), (2, 2), [(0, 0), (1, 1)]),
        ((4, 8), (2, 2), [(2, 2), (3, 3)]),
    ]
    run_stats: dict[str, int] = {}

    out_ref = yastn_transpose_dot_sum(
        Adata0,
        Bdata0,
        meta_dot,
        Areshape,
        Breshape,
        (0, 1),
        (0, 1),
        8,
        policy=BlockGemmPolicy(batch_size=1, min_group=999),
    )
    out_new = yastn_transpose_dot_sum(
        Adata1,
        Bdata1,
        meta_dot,
        Areshape,
        Breshape,
        (0, 1),
        (0, 1),
        8,
        policy=BlockGemmPolicy(executor="custom_no_stack", stats=run_stats),
    )
    upstream = torch.randn_like(out_ref)
    grad_ref = torch.autograd.grad((out_ref * upstream).sum(), (Adata0, Bdata0))
    grad_new = torch.autograd.grad((out_new * upstream).sum(), (Adata1, Bdata1))

    assert torch.allclose(out_new, out_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(grad_new[0], grad_ref[0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(grad_new[1], grad_ref[1], atol=1e-12, rtol=1e-12)
    assert run_stats["yastn_compat_block_gemm_custom_no_stack_calls"] == 1
    assert run_stats["yastn_compat_block_gemm_custom_no_stack_backward_tasks"] == 4


def test_block_gemm_rejects_reserved_cpu_routing():
    with pytest.raises(NotImplementedError, match="mixed placement"):
        yastn_transpose_dot_sum(
            torch.zeros(1),
            torch.zeros(1),
            [],
            (),
            (),
            (),
            (),
            0,
            policy=BlockGemmPolicy(cpu_enabled=True),
        )


def test_yastn_dense_no_fusion_forward_backward_matches_torch_backend():
    cfg_ref = _config("none", backend_torch)
    cfg_new = _config("none", enable_yastn_omeinsum(backend_torch))

    A = yastn.rand(config=cfg_ref, s=(1, -1), D=(2, 3))
    B = yastn.rand(config=cfg_ref, s=(1, -1), D=(3, 4))

    _assert_tensordot_matches(A, B, cfg_ref, cfg_new, axes=(1, 0))


def test_yastn_u1_no_fusion_forward_backward_matches_torch_backend():
    cfg_ref = _config("U1", backend_torch)
    cfg_new = _config("U1", enable_yastn_omeinsum(backend_torch))

    leg = yastn.Leg(cfg_ref, s=1, t=(-1, 0, 1), D=(1, 2, 1))
    A = yastn.rand(config=cfg_ref, legs=(leg, leg.conj(), leg.conj()), n=0)
    B = yastn.rand(config=cfg_ref, legs=(leg, leg, leg), n=0)

    _assert_tensordot_matches(A, B, cfg_ref, cfg_new, axes=((1, 2), (0, 1)))


def _config(sym, backend):
    return yastn.make_config(
        backend=backend,
        sym=sym,
        default_dtype="float64",
        default_device="cpu",
        tensordot_policy="no_fusion",
    )


def _clone_with_config(tensor, config):
    data = tensor.data.detach().clone().requires_grad_(True)
    return tensor._replace(config=config, data=data)


def _assert_tensordot_matches(A, B, cfg_ref, cfg_new, axes):
    A_ref = _clone_with_config(A, cfg_ref)
    B_ref = _clone_with_config(B, cfg_ref)
    C_ref = yastn.tensordot(A_ref, B_ref, axes=axes)
    loss_ref = (C_ref.data * C_ref.data).sum()
    loss_ref.backward()

    A_new = _clone_with_config(A, cfg_new)
    B_new = _clone_with_config(B, cfg_new)
    C_new = yastn.tensordot(A_new, B_new, axes=axes)
    loss_new = (C_new.data * C_new.data).sum()
    loss_new.backward()

    assert C_new.struct == C_ref.struct
    assert torch.allclose(C_new.data, C_ref.data, atol=1e-12, rtol=1e-12)
    assert torch.allclose(A_new.data.grad, A_ref.data.grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(B_new.data.grad, B_ref.data.grad, atol=1e-12, rtol=1e-12)
