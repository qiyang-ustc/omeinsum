import torch
import torch.nn as nn
import opt_einsum


class OMEinsum(nn.Module):
    def __init__(self, equation: str, block_dim: str, batch_size: int, use_checkpoint: bool = False, device: torch.device = None):
        super().__init__()
        self.equation = equation  # ein string
        self.block_dim = block_dim  # which dim to slice?
        self.batch_size = batch_size
        self.use_checkpoint = use_checkpoint
        self.cpu_offload = False
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        # Parse equation
        if '->' not in equation:
            raise ValueError("Only explicit einsum equations with '->' are supported.")
        input_part, output_subscript = equation.split('->')
        self.input_subscripts = [s.strip() for s in input_part.split(',')]
        self.output_subscript = output_subscript.strip()

        # Find which tensor/dim to block
        found = False
        for tidx, sub in enumerate(self.input_subscripts):
            if block_dim in sub:
                self.tidx = tidx
                self.didx = sub.index(block_dim)
                found = True
                break
        assert found, f"block_dim '{block_dim}' not found in any input subscript."
        assert block_dim in self.output_subscript, \
            f"block_dim '{block_dim}' must appear in output for meaningful slicing."
        count = sum([sub.count(block_dim) for sub in self.input_subscripts])
        assert count == 1, f"block_dim '{block_dim}' is contracted (appears in multiple inputs), cannot block."
        self.out_dim = self.output_subscript.index(block_dim)

    def forward(self, *tensors):
        return self._forward(*tensors)

    def _forward_impl(self, *tensors):
        tensor_to_block = tensors[self.tidx]
        # Fast path for empty-length along the blocked dimension
        if tensor_to_block.size(self.didx) == 0:
            return opt_einsum.contract(self.equation, *tensors)

        chunks = torch.split(tensor_to_block, self.batch_size, dim=self.didx)

        # Choose flow by device topology
        if (not torch.cuda.is_available()) or torch.cuda.device_count() <= 1:
            return self.forward_single_device(chunks, tensors)
        else:
            return self.forward_multi_gpu(chunks, tensors)

    def forward_single_device(self, chunks, tensors):
        results = []
        for chunk in chunks:
            chunk_tensors = list(tensors)
            chunk_tensors[self.tidx] = chunk
            chunk_result = opt_einsum.contract(self.equation, *chunk_tensors)
            results.append(chunk_result)
        return torch.cat(results, dim=self.out_dim)

    def forward_multi_gpu(self, chunks, tensors):
        num_gpus = torch.cuda.device_count()
        # Decide where to gather outputs:
        # - If cpu_offload or all inputs are CPU: gather to CPU (to match CPU refs)
        # - Else if any input is CUDA: gather to that device (first CUDA input)
        # - Fallback: cuda:0
        all_inputs_cpu = True
        first_cuda_device = None
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                all_inputs_cpu = False
                if first_cuda_device is None:
                    first_cuda_device = t.device
        if self.cpu_offload or all_inputs_cpu:
            output_device = torch.device("cpu")
        else:
            output_device = first_cuda_device if first_cuda_device is not None else torch.device("cuda:0")

        # No strict requirement on divisibility; we round-robin assign chunks to devices.

        # one-time copy of invariant inputs to each device
        per_device_inputs = {}
        for dev in range(num_gpus):
            device = torch.device(f"cuda:{dev}")
            base_inputs = []
            for idx, t in enumerate(tensors):
                if idx == self.tidx:
                    base_inputs.append(None)
                else:
                    if isinstance(t, torch.Tensor):
                        base_inputs.append(t.to(device, non_blocking=True))
                    else:
                        base_inputs.append(t)
            per_device_inputs[dev] = base_inputs

        # Prefetch all chunks to their target devices before any compute starts
        per_device_chunks = {dev: [] for dev in range(num_gpus)}
        chunk_index_map = []  # preserves original order: (dev, local_idx)
        for i, chunk in enumerate(chunks):
            dev = i % num_gpus
            device = torch.device(f"cuda:{dev}")
            pref = chunk.to(device, non_blocking=True)
            per_device_chunks[dev].append(pref)
            chunk_index_map.append((dev, len(per_device_chunks[dev]) - 1))
        # Ensure all copies are done before launching compute to avoid inter-device copy blocking
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Launch all compute first (keep outputs on their compute device),
        # gather to the target device after all kernels are enqueued to improve overlap.
        results_on_devices = []
        for i, (dev, local_idx) in enumerate(chunk_index_map):
            device = torch.device(f"cuda:{dev}")
            chunk_on_device = per_device_chunks[dev][local_idx]
            inputs = list(per_device_inputs[dev])
            inputs[self.tidx] = chunk_on_device
            out = opt_einsum.contract(self.equation, *inputs)
            results_on_devices.append(out)

        # Now gather to the chosen output device
        if output_device.type == "cpu":
            moved = [t.to(output_device) if isinstance(t, torch.Tensor) and t.is_cuda else t for t in results_on_devices]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return torch.cat(moved, dim=self.out_dim)
        else:
            moved = []
            for t in results_on_devices:
                if t.device != output_device:
                    moved.append(t.to(output_device, non_blocking=True))
                else:
                    moved.append(t)
            torch.cuda.synchronize(output_device)
            return torch.cat(moved, dim=self.out_dim)

    def _cpu_save_memory_impl(self, *tensors):
        if self.cpu_offload and torch.is_grad_enabled():
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return self._forward_impl(*tensors)
        else:
            return self._forward_impl(*tensors)

    def _forward(self, *tensors):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._cpu_save_memory_impl, *tensors, use_reentrant=False)
        else:
            return self._forward_impl(*tensors)


