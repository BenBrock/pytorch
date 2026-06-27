#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace c10d::ishmem_extension {

TORCH_API bool is_ishmem_available();

TORCH_API void ishmem_get_out(
    at::Tensor& dst,
    const c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory>& hdl,
    int64_t offset,
    int64_t size,
    int64_t peer);

} // namespace c10d::ishmem_extension
