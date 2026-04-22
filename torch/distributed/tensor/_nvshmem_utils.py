import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import Placement

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


def _chunk_sizes(length: int, num_chunks: int) -> list[int]:
    if num_chunks <= 0:
        return [length]
    if length <= 0:
        return [0] * num_chunks
    chunk = (length + num_chunks - 1) // num_chunks
    sizes = []
    for i in range(num_chunks):
        start = i * chunk
        sizes.append(max(min(chunk, length - start), 0))
    return sizes


def _max_strided_local_size(length: int, num_chunks: int, split_factor: int) -> int:
    if length <= 0:
        return 0
    if split_factor <= 0:
        return 0
    first_sizes = _chunk_sizes(length, split_factor)
    per_rank = [0] * num_chunks
    for size in first_sizes:
        second_sizes = _chunk_sizes(size, num_chunks)
        for rank in range(num_chunks):
            per_rank[rank] += second_sizes[rank]
    return max(per_rank) if per_rank else 0


def compute_nvshmem_symmetric_shape(
    size: torch.Size, device_mesh: "DeviceMesh", placements: Sequence["Placement"]
) -> torch.Size | None:
    from torch.distributed.tensor.placement_types import _StridedShard, Shard

    if not device_mesh._is_current_rank_part_of_mesh():
        return None
    sym_shape = list(size)
    ndim = len(sym_shape)
    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, (Shard, _StridedShard)):
            continue
        dim = placement.dim
        if dim < 0:
            dim = dim + ndim
        if dim < 0 or dim >= ndim:
            return None
        curr = sym_shape[dim]
        if isinstance(placement, _StridedShard):
            sym_shape[dim] = _max_strided_local_size(
                curr, device_mesh.size(mesh_dim), placement.split_factor
            )
        else:
            sym_shape[dim] = (curr + device_mesh.size(mesh_dim) - 1) // device_mesh.size(
                mesh_dim
            )
    return torch.Size(sym_shape)


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
