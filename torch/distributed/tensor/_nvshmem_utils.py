import os
from typing import Any

import torch

try:
    import nvshmem.core as nvshmem
except Exception:
    nvshmem = None  # type: ignore[assignment]


_DTENSOR_USE_NVSHMEM: bool | None = None


def _dtensor_use_nvshmem() -> bool:
    global _DTENSOR_USE_NVSHMEM
    if _DTENSOR_USE_NVSHMEM is None:
        _DTENSOR_USE_NVSHMEM = os.getenv("TORCH_DTENSOR_USE_NVSHMEM", "0") == "1"
    return _DTENSOR_USE_NVSHMEM


def should_use_nvshmem(device: torch.device) -> bool:
    return _dtensor_use_nvshmem() and nvshmem is not None and device.type == "cuda"


def nvshmem_tensor(
    size: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    if should_use_nvshmem(device):
        tensor = nvshmem.tensor(size, dtype=dtype)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor
    return torch.empty(size, dtype=dtype, device=device, requires_grad=requires_grad)


def nvshmem_tensor_like(tensor: torch.Tensor) -> torch.Tensor:
    return nvshmem_tensor(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        requires_grad=tensor.requires_grad,
    )


def copy_tensor_to_nvshmem(tensor: torch.Tensor) -> torch.Tensor:
    if not should_use_nvshmem(tensor.device):
        return tensor
    out = nvshmem.tensor(tensor.shape, dtype=tensor.dtype)
    out.copy_(tensor)
    if tensor.requires_grad:
        out.requires_grad_(True)
    return out
