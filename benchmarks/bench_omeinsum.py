import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch


# Try normal import first; if not installed, add repo src to path
try:
    from omeinsum import OMEinsum
except ImportError:
    repo_root = Path(__file__) .resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
        from omeinsum import OMEinsum
    else:
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OMEinsum performance for custom einsum while sweeping chi."
    )
    parser.add_argument("--min-chi", type=int, default=512, help="Minimum chi.")
    parser.add_argument("--max-chi", type=int, default=1024, help="Maximum chi.")
    parser.add_argument("--step-chi", type=int, default=64, help="Step size for chi.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Fixed batch size along the blocked dim (here 'p').",
    )
    parser.add_argument("--iters", type=int, default=10, help="Timing iterations per chi.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per chi.")
    parser.add_argument(
        "--target-sec",
        type=float,
        default=0.0,
        help="If > 0, run enough iterations per chi to reach at least this many seconds; overrides --iters.",
    )
    parser.add_argument(
        "--input-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to allocate input tensors on. Compute device(s) are chosen automatically by OMEinsum.",
    )
    parser.add_argument("--D", type=int, default=10, help="Set shared D dimension (default 10).")
    parser.add_argument("--d", type=int, default=2, help="Set small d dimension (default 2).")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Tensor dtype for inputs.",
    )
    parser.add_argument("--use-checkpoint", action="store_true", help="Enable checkpointing in OMEinsum.")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload (effective only with grad).")
    parser.add_argument("--with-grad", action="store_true", help="Build autograd graph (no backward by default).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--plot", action="store_true", help="Save a PNG plot of time vs chi.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path for plot.")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path for raw results.")
    return parser.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def allocate_inputs(
    chi: int,
    D: int,
    d_small: int,
    device: torch.device,
    dtype: torch.dtype,
    with_grad: bool,
):
    # T: "ibfj" -> (chi, D, D, chi)
    T = torch.randn(chi, D, D, chi, requires_grad=with_grad, dtype=dtype, device=device)
    # v: used twice "iaep" and "jcgq" -> both (chi, D, D, chi)
    V = torch.randn(chi, D, D, chi, requires_grad=with_grad, dtype=dtype, device=device)
    # Mu: "xabcd" -> (d, D, D, D, D)
    Mu = torch.randn(d_small, D, D, D, D, requires_grad=with_grad, dtype=dtype, device=device)
    # Md: "xefgh" -> (d, D, D, D, D)
    Md = torch.randn(d_small, D, D, D, D, requires_grad=with_grad, dtype=dtype, device=device)
    return T, V, Mu, Md


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_for_chi(
    chi: int,
    batch_size: int,
    equation: str,
    block_dim: str,
    dims: Dict[str, int],
    dtype: torch.dtype,
    input_device: str,
    target_sec: float,
    iters: int,
    warmup: int,
    use_checkpoint: bool,
    cpu_offload: bool,
    with_grad: bool,
) -> Dict[str, Any]:
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Fixed batch size (no auto-scaling)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    alloc_device = torch.device("cuda:0") if (input_device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    T, V, Mu, Md = allocate_inputs(
        chi=chi,
        D=dims["D"],
        d_small=dims["d"],
        device=alloc_device,
        dtype=dtype,
        with_grad=with_grad,
    )

    model = OMEinsum(equation, block_dim=block_dim, batch_size=batch_size, use_checkpoint=use_checkpoint)
    model.cpu_offload = cpu_offload
    model.eval()

    # Warmup
    if not with_grad:
        ctx = torch.no_grad()
    else:
        ctx = torch.enable_grad()
    with ctx:
        for _ in range(max(0, warmup)):
            _ = model(T, V, Mu, Md, V)
            synchronize()

        # Timed
        avg_s: float
        effective_iters: int
        synchronize()
        if target_sec > 0:
            # Accumulate iterations until reaching target_sec, synchronizing each step for accurate timing
            start = time.perf_counter()
            cnt = 0
            while True:
                _ = model(T, V, Mu, Md, V)
                synchronize()
                cnt += 1
                elapsed = time.perf_counter() - start
                if elapsed >= target_sec:
                    avg_s = elapsed / cnt
                    effective_iters = cnt
                    break
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(T, V, Mu, Md, V)
            synchronize()
            t1 = time.perf_counter()
            avg_s = (t1 - t0) / max(1, iters)
            effective_iters = iters

    avg_ms = avg_s * 1000.0

    return {
        "chi": chi,
        "batch_size": batch_size,
        "avg_ms": avg_ms,
        "iters": effective_iters,
        "warmup": warmup,
        "dtype": str(dtype),
        "input_device": input_device,
        "num_gpus": device_count,
        "use_checkpoint": use_checkpoint,
        "cpu_offload": cpu_offload,
        "with_grad": with_grad,
    }


def maybe_save_plot(results: List[Dict[str, Any]], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[bench] matplotlib not available ({e}); skipping plot.")
        return

    bs = [r["chi"] for r in results]
    times = [r["avg_ms"] for r in results]
    num_gpus = results[0]["num_gpus"] if results else 0
    input_device = results[0]["input_device"] if results else "cpu"
    dtype = results[0]["dtype"].split(".")[-1] if results else "float32"

    plt.figure(figsize=(7, 4))
    plt.plot(bs, times, marker="o")
    plt.xlabel("chi")
    plt.ylabel("Average time per forward (ms)")
    plt.title(f"OMEinsum chi vs time — {num_gpus} GPUs, inputs={input_device}, dtype={dtype}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[bench] Saved plot to {out_path}")


def maybe_save_csv(results: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chi",
        "batch_size",
        "avg_ms",
        "iters",
        "warmup",
        "dtype",
        "input_device",
        "num_gpus",
        "use_checkpoint",
        "cpu_offload",
        "with_grad",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[bench] Saved CSV to {out_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # New equation and block dimension:
    # contract("ibfj,iaep,xabcd,xefgh,jcgq->pdhq", T, v, Mu, Md, v)
    # We choose to block along 'p' (appears in one input and in output).
    equation = "ibfj,iaep,xabcd,xefgh,jcgq->pdhq"
    block_dim = "p"
    # Base dims
    dims = dict(D=max(1, int(args.D)), d=max(1, int(args.d)))

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[bench] PyTorch {torch.__version__} | GPUs visible: {device_count} | input_device={args.input_device}")

    dtype = str_to_dtype(args.dtype)
    chi_values = list(range(args.min_chi, args.max_chi + 1, args.step_chi))

    results: List[Dict[str, Any]] = []
    for chi in chi_values:
        res = benchmark_for_chi(
            chi=chi,
            batch_size=args.batch_size,
            equation=equation,
            block_dim=block_dim,
            dims=dims,
            dtype=dtype,
            input_device=args.input_device,
            target_sec=args.target_sec,
            iters=args.iters,
            warmup=args.warmup,
            use_checkpoint=args.use_checkpoint,
            cpu_offload=args.cpu_offload,
            with_grad=args.with_grad,
        )
        print(f"[bench] chi={chi:4d} | batch_size={res['batch_size']:4d} | {res['avg_ms']:.3f} ms")
        results.append(res)

    # Outputs
    if args.csv:
        maybe_save_csv(results, Path(args.csv))

    if args.plot:
        default_name = f"bench_chi_vs_time_{results[0]['num_gpus']}g_{args.input_device}_{args.dtype}.png" if results else "bench.png"
        out_path = Path(args.out) if args.out else Path("benchmarks") / default_name
        maybe_save_plot(results, out_path)


if __name__ == "__main__":
    main()


