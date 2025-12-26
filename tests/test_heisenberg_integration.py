import pytest
import torch

from tests.heisenberg_omeinsum import mwe_main


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices",
)
def test_heisenberg_integration_multi_gpu():
    cfg = {
        "chi": 8,
        "d": 2,
        "D": 2,
        "seed": 0,
        "ADiter": 1,
        "warmup": 0,
        "warmupiter": 0,
        "maxoptiter": 1,
        "checkpoint": True,
        "use_omeinsum": True,
        "ome_batch_size": 2,
        "ctmrg_cpu_offload": True,
        "ome_checkpoint": True,
        "ome_use_reentrant": False,
    }

    energy = mwe_main(cfg)
    assert torch.isfinite(torch.tensor(energy))
