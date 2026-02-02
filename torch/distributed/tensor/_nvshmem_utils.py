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
        tensor._nvshmem_alloc = True  # type: ignore[attr-defined]
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


def _slice_to_shape(base: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if base.shape == shape:
        return base
    if len(shape) == 0:
        return base
    slices = tuple(slice(0, dim) for dim in shape)
    return base[slices]


def nvshmem_symmetric_tensor(
    local_shape: torch.Size,
    symmetric_shape: torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    if should_use_nvshmem(device):
        base = nvshmem.tensor(symmetric_shape, dtype=dtype)
        base._nvshmem_alloc = True  # type: ignore[attr-defined]
        view = _slice_to_shape(base, local_shape)
        # Track the base allocation so callers can free the full NVSHMEM buffer.
        view._nvshmem_base = base  # type: ignore[attr-defined]
        if requires_grad:
            view.requires_grad_(True)
        return view
    return torch.empty(
        local_shape, dtype=dtype, device=device, requires_grad=requires_grad
    )


def copy_tensor_to_nvshmem(
    tensor: torch.Tensor, symmetric_shape: torch.Size | None = None
) -> torch.Tensor:
    if not should_use_nvshmem(tensor.device):
        return tensor
    if symmetric_shape is None or tuple(tensor.shape) == tuple(symmetric_shape):
        out = nvshmem.tensor(tensor.shape, dtype=tensor.dtype)
        out._nvshmem_alloc = True  # type: ignore[attr-defined]
        out.copy_(tensor)
        if tensor.requires_grad:
            out.requires_grad_(True)
        return out
    base = nvshmem.tensor(symmetric_shape, dtype=tensor.dtype)
    base._nvshmem_alloc = True  # type: ignore[attr-defined]
    view = _slice_to_shape(base, torch.Size(tensor.shape))
    # Track the base allocation so callers can free the full NVSHMEM buffer.
    view._nvshmem_base = base  # type: ignore[attr-defined]
    view.copy_(tensor)
    if tensor.requires_grad:
        view.requires_grad_(True)
    return view


def nvshmem_base(tensor: torch.Tensor) -> torch.Tensor:
    return getattr(tensor, "_nvshmem_base", tensor)


def free_nvshmem_tensor(tensor: torch.Tensor) -> None:
    if nvshmem is None:
        return
    base = nvshmem_base(tensor)
    if getattr(base, "_nvshmem_alloc", False):
        nvshmem.free_tensor(base)
