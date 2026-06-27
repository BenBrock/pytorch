#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <c10/core/DeviceGuard.h>
#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>

#include <ishmemx.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace c10d {
namespace symmetric_memory {

static StoreExchange storeExchange = StoreExchange("ISHMEMSymmetricMemory");

struct ISHMEMAllocation {
  void* ptr;
  size_t buffer_size;
  int device_idx;

  ISHMEMAllocation(void* ptr, size_t buffer_size, int device_idx)
      : ptr(ptr), buffer_size(buffer_size), device_idx(device_idx) {}

  ISHMEMAllocation(const ISHMEMAllocation&) = delete;
  ISHMEMAllocation& operator=(const ISHMEMAllocation&) = delete;
  ISHMEMAllocation(ISHMEMAllocation&&) = delete;
  ISHMEMAllocation& operator=(ISHMEMAllocation&&) = delete;

  ~ISHMEMAllocation() {
    if (is_finalizing()) {
      return;
    }
    c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_idx));
    ishmem_free(ptr);
  }
};

static std::mutex rank_map_mutex;
static std::unordered_map<std::string, std::vector<int>>
    rank_to_global_rank_map{};
static std::unordered_map<std::string, int*> rank_to_global_rank_dev_map{};

class ISHMEMPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  ISHMEMPeerAllocInfo(
      ISHMEMAllocation* allocation,
      const std::string& group_name)
      : base_ptr_(allocation->ptr),
        buffer_size_(allocation->buffer_size),
        device_idx_(allocation->device_idx) {
    c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_idx_));

    auto group = resolve_process_group(group_name);
    rank_ = group->getRank();
    world_size_ = group->getSize();
    auto store = group->getStore();

    {
      std::lock_guard<std::mutex> rank_map_lock(rank_map_mutex);
      auto it = rank_to_global_rank_map.find(group_name);
      if (it == rank_to_global_rank_map.end()) {
        auto global_group = resolve_process_group("0");
        auto global_rank = global_group->getRank();
        auto rank_to_global_rank =
            storeExchange.all_gather(store, rank_, world_size_, global_rank);
        it = rank_to_global_rank_map.emplace_hint(
            it, group_name, rank_to_global_rank);

        auto* rank_to_global_rank_dev =
            reinterpret_cast<int*>(c10::xpu::XPUCachingAllocator::raw_alloc(
                sizeof(int) * world_size_));
        auto& queue = c10::xpu::getCurrentXPUStream(device_idx_).queue();
        queue.memcpy(
                 rank_to_global_rank_dev,
                 rank_to_global_rank.data(),
                 sizeof(int) * world_size_)
            .wait();
        rank_to_global_rank_dev_map[group_name] = rank_to_global_rank_dev;
      }
      rank_to_global_rank_ = it->second;
    }

    world_within_direct_access_ = true;
    for (int r = 0; r < world_size_; ++r) {
      auto peer_ptr = ishmem_ptr(base_ptr_, rank_to_global_rank_[r]);
      buffers_.push_back(peer_ptr);
      if (peer_ptr == nullptr) {
        world_within_direct_access_ = false;
      }
    }

    signal_pad_ptr_ = ishmem_malloc(get_signal_pad_size());
    TORCH_CHECK(signal_pad_ptr_ != nullptr, "ishmem_malloc failed");
    auto& queue = c10::xpu::getCurrentXPUStream(device_idx_).queue();
    queue.memset(signal_pad_ptr_, 0, get_signal_pad_size()).wait();

    for (int r = 0; r < world_size_; ++r) {
      signal_pads_.push_back(
          ishmem_ptr(signal_pad_ptr_, rank_to_global_rank_[r]));
    }

    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));

    queue.memcpy(buffers_dev_, buffers_.data(), arr_size).wait();
    queue.memcpy(signal_pads_dev_, signal_pads_.data(), arr_size).wait();
  }

  ~ISHMEMPeerAllocInfo() override {
    if (is_finalizing()) {
      return;
    }
    c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_idx_));
    if (buffers_dev_ != nullptr) {
      c10::xpu::XPUCachingAllocator::raw_delete(buffers_dev_);
    }
    if (signal_pads_dev_ != nullptr) {
      c10::xpu::XPUCachingAllocator::raw_delete(signal_pads_dev_);
    }
    if (signal_pad_ptr_ != nullptr) {
      ishmem_free(signal_pad_ptr_);
    }
  }

 private:
  void* base_ptr_;
  size_t buffer_size_;
  int device_idx_;
  int rank_;
  int world_size_;
  std::vector<int> rank_to_global_rank_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_{nullptr};
  void** signal_pads_dev_{nullptr};
  void* signal_pad_ptr_{nullptr};
  bool world_within_direct_access_;

  friend class ISHMEMSymmetricMemory;
};

class ISHMEMSymmetricMemory : public SymmetricMemory {
 public:
  ISHMEMSymmetricMemory(
      ISHMEMAllocation* allocation,
      const std::string& group_name)
      : device_idx_(allocation->device_idx), group_name_(group_name) {
    pai_ = c10::make_intrusive<ISHMEMPeerAllocInfo>(allocation, group_name);
  }

  ISHMEMSymmetricMemory(const ISHMEMSymmetricMemory& other) = delete;

  ISHMEMSymmetricMemory(const ISHMEMSymmetricMemory& other, size_t offset)
      : device_idx_(other.device_idx_),
        group_name_(other.group_name_),
        pai_(other.pai_),
        offset_(offset) {}

  std::vector<void*> get_buffer_ptrs() override {
    return pai_->buffers_;
  }

  std::vector<void*> get_signal_pad_ptrs() override {
    return pai_->signal_pads_;
  }

  void** get_buffer_ptrs_dev() override {
    return pai_->buffers_dev_;
  }

  void** get_signal_pad_ptrs_dev() override {
    return pai_->signal_pads_dev_;
  }

  size_t get_buffer_size() override {
    return pai_->buffer_size_;
  }

  bool has_multicast_support() override {
    return false;
  }

  void* get_multicast_ptr() override {
    return nullptr;
  }

  size_t get_offset() override {
    return offset_;
  }

  void barrier(int /* channel */, size_t timeout_ms) override {
    auto group = resolve_process_group(group_name_);
    BarrierOptions opts;
    opts.device = c10::Device(c10::DeviceType::XPU, device_idx_);
    if (timeout_ms > 0) {
      opts.timeout = std::chrono::milliseconds(timeout_ms);
    }
    auto work = group->barrier(opts);
    work->wait(opts.timeout);
  }

  void put_signal(int /* dst_rank */, int /* channel */, size_t /* timeout_ms */)
      override {
    TORCH_CHECK(false, "ISHMEMSymmetricMemory::put_signal is not implemented");
  }

  void wait_signal(int /* src_rank */, int /* channel */, size_t /* timeout_ms */)
      override {
    TORCH_CHECK(false, "ISHMEMSymmetricMemory::wait_signal is not implemented");
  }

  int get_rank() override {
    return pai_->rank_;
  }

  int get_world_size() override {
    return pai_->world_size_;
  }

  c10::Device get_device() override {
    return c10::Device(c10::DeviceType::XPU, device_idx_);
  }

  const std::vector<int>& get_rank_to_global_rank() override {
    return pai_->rank_to_global_rank_;
  }

  int* get_rank_to_global_rank_dev() override {
    std::lock_guard<std::mutex> lock(rank_map_mutex);
    auto it = rank_to_global_rank_dev_map.find(group_name_);
    TORCH_CHECK(
        it != rank_to_global_rank_dev_map.end(),
        "Group name not found in rank_to_global_rank_dev_map");
    return it->second;
  }

  bool world_within_direct_access() override {
    return pai_->world_within_direct_access_;
  }

 private:
  int device_idx_;
  std::string group_name_;
  c10::intrusive_ptr<ISHMEMPeerAllocInfo> pai_;
  size_t offset_{0};
};

static void initialize_ishmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size,
    int device_idx) {
  static std::mutex init_mutex;
  std::lock_guard<std::mutex> lock(init_mutex);

  int initialized = 0;
  ishmemx_query_initialized(&initialized);
  if (initialized != 0) {
    return;
  }

  c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_idx));

  ishmemx_uniqueid_t unique_id;
  int err = ishmemx_get_uniqueid(&unique_id);
  TORCH_CHECK(err == 0, "ishmemx_get_uniqueid failed with error code ", err);
  auto unique_ids =
      storeExchange.all_gather(store, rank, world_size, unique_id);
  for (int r = 1; r < world_size; ++r) {
    TORCH_CHECK(
        std::memcmp(&unique_ids[0], &unique_ids[r], sizeof(unique_id)) == 0,
        "ISHMEM unique IDs differ across ranks");
  }

  ishmemx_attr_t attr;
  attr.use_uid = true;
  attr.nranks = world_size;
  attr.rank = rank;
  attr.uid = &unique_ids[0];
  attr.device_idx = device_idx;
  attr.gpu = true;
  ishmemx_init_attr(&attr);

  ishmemx_query_initialized(&initialized);
  TORCH_CHECK(initialized != 0, "ishmemx_init_attr did not initialize ISHMEM");
}

class ISHMEMSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name == std::nullopt,
        "ISHMEMSymmetricMemoryAllocator::alloc must not be called with a "
        "group_name");
    c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_idx));

    auto group = resolve_process_group("0");
    initialize_ishmem_with_store(
        group->getStore(), group->getRank(), group->getSize(), device_idx);

    auto ptr = ishmem_malloc(size);
    TORCH_CHECK(ptr != nullptr || size == 0, "ishmem_malloc failed");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.try_emplace(
          ptr, std::make_unique<ISHMEMAllocation>(ptr, size, device_idx));
    }
    return ptr;
  }

  void free(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.erase(ptr);
  }

  size_t get_alloc_size(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    TORCH_CHECK(
        it != allocations_.end(),
        ptr,
        " is not allocated with ISHMEMSymmetricMemoryAllocator");
    return it->second->buffer_size;
  }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(group_name.has_value());
    std::lock_guard<std::mutex> lock(mutex_);
    {
      auto it = symm_mems_.find(SymmMemKey{ptr, *group_name});
      if (it != symm_mems_.end()) {
        return it->second;
      }
    }

    auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
    auto alloc_it = std::find_if(
        allocations_.begin(), allocations_.end(), [&](const auto& pair) {
          auto& allocation = pair.second;
          auto base_ptr = reinterpret_cast<uintptr_t>(allocation->ptr);
          return ptr_int >= base_ptr &&
              ptr_int < base_ptr + allocation->buffer_size;
        });
    TORCH_CHECK(
        alloc_it != allocations_.end(),
        "Pointer not within any SymmetricMemory allocation, "
        "is the tensor allocated from SymmetricMemory?");

    auto& allocation = alloc_it->second;
    auto it = symm_mems_.find(SymmMemKey{allocation->ptr, *group_name});
    c10::intrusive_ptr<ISHMEMSymmetricMemory> symm_mem;
    if (it != symm_mems_.end()) {
      symm_mem = it->second;
    } else {
      symm_mem = c10::make_intrusive<ISHMEMSymmetricMemory>(
          allocation.get(), *group_name);
    }
    symm_mems_[SymmMemKey{allocation->ptr, *group_name}] = symm_mem;

    if (ptr == allocation->ptr) {
      return symm_mem;
    }
    return c10::make_intrusive<ISHMEMSymmetricMemory>(
        *symm_mem,
        reinterpret_cast<uintptr_t>(ptr) -
            reinterpret_cast<uintptr_t>(allocation->ptr));
  }

  bool has_multicast_support(int /* device_idx */) override {
    return false;
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::XPU;
  }

  std::string name() override {
    return "ISHMEM";
  }

  bool has_allocation(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
    auto alloc_it = std::find_if(
        allocations_.begin(), allocations_.end(), [&](const auto& pair) {
          auto base_ptr = reinterpret_cast<uintptr_t>(pair.second->ptr);
          return ptr_int >= base_ptr &&
              ptr_int < base_ptr + pair.second->buffer_size;
        });
    return alloc_it != allocations_.end();
  }

 private:
  std::mutex mutex_;
  std::unordered_map<void*, std::unique_ptr<ISHMEMAllocation>> allocations_;
  ska::flat_hash_map<
      SymmMemKey,
      c10::intrusive_ptr<ISHMEMSymmetricMemory>,
      SymmMemKeyHash>
      symm_mems_;
};

struct RegisterISHMEMSymmetricMemoryAllocator {
  RegisterISHMEMSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<ISHMEMSymmetricMemoryAllocator>();
    register_availability("ISHMEM", allocator);
  }
};

static RegisterISHMEMSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
