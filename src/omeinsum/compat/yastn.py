from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module, util
from types import ModuleType
from typing import Any, Iterator

from .block_gemm import yastn_transpose_dot_sum
from .scheduler import BlockGemmPolicy


def is_yastn_available() -> bool:
    return util.find_spec("yastn") is not None


def enable_yastn_omeinsum(
    backend_module: ModuleType | None = None,
    *,
    policy: BlockGemmPolicy | None = None,
) -> "_YastnBackendWrapper":
    """Return a YASTN backend wrapper using OMEinsum for block GEMM."""

    if backend_module is None:
        backend_module = import_module("yastn.backend.backend_torch")
    return _YastnBackendWrapper(backend_module, policy or BlockGemmPolicy())


@contextmanager
def patch_yastn_backend(
    backend_module: ModuleType | None = None,
    *,
    policy: BlockGemmPolicy | None = None,
) -> Iterator[ModuleType]:
    """Temporarily patch ``backend.transpose_dot_sum``.

    This is for experiments only.  The normal path is to use the wrapper
    returned by :func:`enable_yastn_omeinsum`.
    """

    wrapper = enable_yastn_omeinsum(backend_module, policy=policy)
    backend = wrapper._backend
    original = backend.transpose_dot_sum
    backend.transpose_dot_sum = wrapper.transpose_dot_sum
    try:
        yield backend
    finally:
        backend.transpose_dot_sum = original


class _YastnBackendWrapper:
    def __init__(self, backend: ModuleType, policy: BlockGemmPolicy) -> None:
        self._backend = backend
        self._policy = policy
        self._omeinsum_block_gemm_policy = policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    @property
    def block_gemm_policy(self) -> BlockGemmPolicy:
        return self._policy

    def transpose_dot_sum(self, *args: Any) -> Any:
        return yastn_transpose_dot_sum(*args, policy=self._policy)
