#include <torch/csrc/distributed/c10d/symm_mem/ishmem_extension.hpp>

#include <dlfcn.h>

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <ishmemx.h>

#include <mutex>

namespace c10d::ishmem_extension {

bool is_ishmem_available() {
  static std::mutex mutex;
  static int is_available = -1;
  std::lock_guard<std::mutex> lock(mutex);

  if (is_available == -1) {
    void* handle = dlopen("libishmem_host.so", RTLD_LAZY);
    if (handle == nullptr) {
      is_available = 0;
    } else {
      dlclose(handle);
      is_available = 1;
    }
  }
  return is_available == 1;
}

void ishmem_get_out(
    at::Tensor& dst,
    const c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory>& hdl,
    int64_t offset,
    int64_t size,
    int64_t peer) {
  TORCH_CHECK(dst.is_xpu(), "symm_mem.get: expected an XPU tensor");
  TORCH_CHECK(
      dst.device() == hdl->get_device(),
      "symm_mem.get: dst must be on the same device as hdl");
  TORCH_CHECK(
      dst.is_contiguous(),
      "symm_mem.get: dst must be backed by contiguous memory");
  TORCH_CHECK(offset >= 0, "symm_mem.get: offset must be non-negative");
  TORCH_CHECK(size >= 0, "symm_mem.get: size must be non-negative");
  TORCH_CHECK(
      dst.numel() >= size,
      "symm_mem.get: dst must contain at least `size` elements");
  TORCH_CHECK(
      peer >= 0 && peer < hdl->get_world_size(), "symm_mem.get: invalid peer");

  auto element_size = static_cast<size_t>(dst.element_size());
  auto buffer_offset = hdl->get_offset();
  TORCH_CHECK(
      buffer_offset % element_size == 0,
      "symm_mem.get: handle offset is not element-aligned");
  auto buffer_size = hdl->get_buffer_size();
  TORCH_CHECK(
      buffer_offset <= buffer_size,
      "symm_mem.get: handle offset exceeds symmetric allocation");
  auto available_bytes = buffer_size - buffer_offset;
  auto requested_bytes = (offset + size) * element_size;
  TORCH_CHECK(
      requested_bytes <= available_bytes,
      "symm_mem.get: requested range exceeds symmetric allocation");

  c10::DeviceGuard guard(dst.device());
  auto* remote_base = static_cast<char*>(hdl->get_buffer_ptrs().at(peer));
  TORCH_CHECK(
      remote_base != nullptr,
      "symm_mem.get: peer is not directly accessible; ISHMEM get fallback is "
      "not implemented yet");
  auto* src = remote_base + buffer_offset + offset * element_size;
  auto* dst_ptr = static_cast<char*>(dst.mutable_data_ptr());
  auto& queue = c10::xpu::getCurrentXPUStream(dst.device().index()).queue();
  queue.memcpy(dst_ptr, src, size * element_size);
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ishmem_get_out", ishmem_get_out);
}

} // namespace c10d::ishmem_extension
