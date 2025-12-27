#!/usr/bin/env python3
import argparse
import csv
import time
from typing import List

import torch

from tests.kitaev_omeinsum import (
    SymmetrizedTensorKitaev,
    EnvironmentInitializer as KitaevEnvInit,
    PhysicalOperators,
    KitaevEnergy,
    QRCTMRG,
    normalize,
)
from tests.heisenberg_omeinsum import (
    C4Symmetrizer,
    EnvironmentInitializer as HeisEnvInit,
    HeisenbergEnergy,
    CTMRG,
    bulk_tensor,
)

try:
    from omeinsum import OMEinsum
except Exception:
    OMEinsum = None


def _parse_int_list(value: str) -> List[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _run_kitaev(
    *,
    chi: int,
    d: int,
    D: int,
    ADiter: int,
    warmup: int,
    seed: int,
    ome_batch_size: int,
    checkpoint: bool,
    ctmrg_cpu_offload: bool,
    use_omeinsum_update_r: bool,
    ome_checkpoint: bool,
    ome_use_reentrant: bool,
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    if OMEinsum is None:
        raise RuntimeError("OMEinsum not available; install omeinsum.")

    device = torch.device("cuda:0")
    torch.manual_seed(seed)

    symmetrizer = SymmetrizedTensorKitaev(d=d, device=device)
    initializer = KitaevEnvInit()
    operators = PhysicalOperators(d=d, device=device)
    ome_L = OMEinsum(
        "ibca,ijkm,xjbp,xy,ykcq->apqm",
        block_dim="a",
        batch_size=ome_batch_size,
        use_checkpoint=ome_checkpoint,
        use_reentrant=ome_use_reentrant,
    )
    energy_module = KitaevEnergy(operators, ome_L=ome_L)
    ctmrg_module = QRCTMRG(
        checkpoint=checkpoint,
        ctmrg_cpu_offload=ctmrg_cpu_offload,
        use_omeinsum_update_r=use_omeinsum_update_r,
        ome_batch_size=ome_batch_size,
        ome_checkpoint=ome_checkpoint,
        ome_use_reentrant=ome_use_reentrant,
    )

    P = (torch.randn(d, D, D, D, dtype=torch.complex128, device=device) * 0.1).requires_grad_(True)

    with torch.no_grad():
        M0 = normalize(symmetrizer(P))
        env = initializer(M0, chi)
        env = ctmrg_module(M0, env, warmup=0, ADiter=0)

    with torch.no_grad():
        M = normalize(symmetrizer(P))
        env2 = ctmrg_module(M, env, warmup=0, ADiter=ADiter)
        _ = energy_module(M, env2)
        torch.cuda.synchronize()

    start = time.perf_counter()
    if P.grad is not None:
        P.grad = None
    M = normalize(symmetrizer(P))
    env2 = ctmrg_module(M, env, warmup=0, ADiter=ADiter)
    E = energy_module(M, env2)
    E.backward()
    with torch.no_grad():
        M_now = normalize(symmetrizer(P))
        env = ctmrg_module(M_now, env, warmup=warmup, ADiter=0)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _run_heisenberg(
    *,
    chi: int,
    d: int,
    D: int,
    ADiter: int,
    warmup: int,
    seed: int,
    ome_batch_size: int,
    checkpoint: bool,
    ctmrg_cpu_offload: bool,
    use_omeinsum: bool,
    use_omeinsum_update_t: bool,
    ome_checkpoint: bool,
    ome_use_reentrant: bool,
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    if use_omeinsum and OMEinsum is None:
        raise RuntimeError("OMEinsum not available; install omeinsum.")

    device = torch.device("cuda:0")
    torch.manual_seed(seed)

    sym = C4Symmetrizer()
    initializer = HeisEnvInit()

    ome_rho_xy = None
    if use_omeinsum:
        ome_rho_xy = OMEinsum(
            "ibfj,iaep,xabcd,yefgh,jcgq->pdhqxy",
            block_dim="p",
            batch_size=ome_batch_size,
            use_checkpoint=ome_checkpoint,
            use_reentrant=ome_use_reentrant,
        )
    energy_fn = HeisenbergEnergy(ome_rho_xy=ome_rho_xy)
    ctmrg = CTMRG(
        checkpoint=checkpoint,
        ctmrg_cpu_offload=ctmrg_cpu_offload,
        use_omeinsum_update_t=use_omeinsum_update_t,
        ome_batch_size=ome_batch_size,
        ome_checkpoint=ome_checkpoint,
        ome_use_reentrant=ome_use_reentrant,
    )

    nparams = d * D * D * D * D
    params = (torch.rand(nparams, device=device, dtype=torch.float64) - 0.5).requires_grad_(True)

    with torch.no_grad():
        M0 = sym(bulk_tensor(params, d, D))
        env = initializer(M0, chi)
        env = ctmrg(M0, env, warmup=0, ADiter=0)

    with torch.no_grad():
        M = sym(bulk_tensor(params, d, D))
        env2 = ctmrg(M, env, warmup=0, ADiter=ADiter)
        _ = energy_fn(M, env2)
        torch.cuda.synchronize()

    start = time.perf_counter()
    if params.grad is not None:
        params.grad = None
    M = sym(bulk_tensor(params, d, D))
    env2 = ctmrg(M, env, warmup=0, ADiter=ADiter)
    E = energy_fn(M, env2)
    E.backward()
    with torch.no_grad():
        env = ctmrg(sym(bulk_tensor(params, d, D)), env, warmup=warmup, ADiter=0)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="OMEinsum benchmark runner (Heisenberg + Kitaev)")
    parser.add_argument("--model", choices=["heisenberg", "kitaev", "both"], default="both")
    parser.add_argument("--chis", type=str, default="968,1536")
    parser.add_argument("--batch-sizes", type=str, default="4,8")
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--D", type=int, default=12)
    parser.add_argument("--ADiter", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ctmrg-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-omeinsum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-omeinsum-update-t", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-omeinsum-update-r", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ome-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ome-use-reentrant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", type=str, default="bench_runs.csv")
    args = parser.parse_args()

    chis = _parse_int_list(args.chis)
    batch_sizes = _parse_int_list(args.batch_sizes)

    models = ["heisenberg", "kitaev"] if args.model == "both" else [args.model]
    results = []
    for model in models:
        for chi in chis:
            for batch_size in batch_sizes:
                try:
                    if model == "heisenberg":
                        elapsed = _run_heisenberg(
                            chi=chi,
                            d=args.d,
                            D=args.D,
                            ADiter=args.ADiter,
                            warmup=args.warmup,
                            seed=args.seed,
                            ome_batch_size=batch_size,
                            checkpoint=args.checkpoint,
                            ctmrg_cpu_offload=args.ctmrg_cpu_offload,
                            use_omeinsum=args.use_omeinsum,
                            use_omeinsum_update_t=args.use_omeinsum_update_t,
                            ome_checkpoint=args.ome_checkpoint,
                            ome_use_reentrant=args.ome_use_reentrant,
                        )
                    else:
                        elapsed = _run_kitaev(
                            chi=chi,
                            d=args.d,
                            D=args.D,
                            ADiter=args.ADiter,
                            warmup=args.warmup,
                            seed=args.seed,
                            ome_batch_size=batch_size,
                            checkpoint=args.checkpoint,
                            ctmrg_cpu_offload=args.ctmrg_cpu_offload,
                            use_omeinsum_update_r=args.use_omeinsum_update_r,
                            ome_checkpoint=args.ome_checkpoint,
                            ome_use_reentrant=args.ome_use_reentrant,
                        )
                    status = "ok"
                    error = ""
                except RuntimeError as exc:
                    msg = str(exc)
                    if "out of memory" in msg.lower():
                        status = "oom"
                    else:
                        status = "error"
                    elapsed = -1.0
                    error = msg.replace("\n", " ")

                results.append(
                    {
                        "model": model,
                        "chi": chi,
                        "batch_size": batch_size,
                        "elapsed_s": f"{elapsed:.2f}" if elapsed >= 0 else "",
                        "status": status,
                        "error": error,
                    }
                )
                torch.cuda.empty_cache()

    print("model,chi,batch_size,elapsed_s,status")
    for row in results:
        print(f"{row['model']},{row['chi']},{row['batch_size']},{row['elapsed_s']},{row['status']}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["model", "chi", "batch_size", "elapsed_s", "status", "error"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
