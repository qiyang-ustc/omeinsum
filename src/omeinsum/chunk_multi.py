from __future__ import annotations

from typing import Callable, Sequence

import torch

from .chunk_single import run_chunked_einsum


def _split_chunks_for_devices(
    chunks: Sequence[torch.Tensor],
    num_devices: int,
) -> list[list[torch.Tensor]]:
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")

    total = len(chunks)
    base = total // num_devices
    extra = total % num_devices
    out: list[list[torch.Tensor]] = []
    cursor = 0
    for dev in range(num_devices):
        count = base + (1 if dev < extra else 0)
        out.append(list(chunks[cursor : cursor + count]))
        cursor += count
    return out


def run_chunked_einsum_multi_device(
    einsum_fn: Callable[..., torch.Tensor],
    tensors: Sequence[torch.Tensor],
    tidx: int,
    chunks: Sequence[torch.Tensor],
    out_dim: int,
    use_checkpoint: bool,
    use_reentrant: bool,
    devices: Sequence[torch.device] | None = None,
) -> torch.Tensor:
    if not chunks:
        return einsum_fn(*tensors)

    if devices is None:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if len(devices) == 0:
        raise RuntimeError("No CUDA devices available for multi-GPU execution.")
    if len(devices) == 1:
        base_inputs = list(tensors)
        return run_chunked_einsum(
            einsum_fn,
            base_inputs,
            tidx,
            chunks,
            out_dim,
            use_checkpoint,
            use_reentrant,
        )

    chunks_per_device = _split_chunks_for_devices(chunks, len(devices))
    base_inputs_per_device: list[list[torch.Tensor | None]] = []
    moved_chunks_per_device: list[list[torch.Tensor]] = []

    for device, device_chunks in zip(devices, chunks_per_device):
        base_inputs: list[torch.Tensor | None] = []
        for idx, t in enumerate(tensors):
            if idx == tidx:
                base_inputs.append(None)
            else:
                base_inputs.append(t if t.device == device else t.to(device, non_blocking=True))
        base_inputs_per_device.append(base_inputs)

        moved_chunks: list[torch.Tensor] = []
        for chunk in device_chunks:
            moved_chunks.append(chunk if chunk.device == device else chunk.to(device, non_blocking=True))
        moved_chunks_per_device.append(moved_chunks)

    outputs: list[torch.Tensor] = []
    for device, base_inputs, device_chunks in zip(
        devices, base_inputs_per_device, moved_chunks_per_device
    ):
        if not device_chunks:
            continue
        device_outputs = run_chunked_einsum(
            einsum_fn,
            base_inputs,
            tidx,
            device_chunks,
            out_dim,
            use_checkpoint,
            use_reentrant,
        )
        outputs.append(device_outputs if device_outputs.device == device else device_outputs.to(device))

    output_device = tensors[tidx].device
    gathered = [
        out if out.device == output_device else out.to(output_device, non_blocking=True) for out in outputs
    ]
    return torch.cat(gathered, dim=out_dim)
