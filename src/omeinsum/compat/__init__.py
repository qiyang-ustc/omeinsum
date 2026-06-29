"""Optional compatibility layers for external tensor libraries."""

from .yastn import enable_yastn_omeinsum, is_yastn_available, patch_yastn_backend

__all__ = ["enable_yastn_omeinsum", "is_yastn_available", "patch_yastn_backend"]
