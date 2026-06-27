# Owner(s): ["oncall: distributed"]

# To run (one process per local GPU is spawned by the harness):
#   python test/distributed/test_ishmem.py
#
# Requires a PyTorch XPU build with USE_ISHMEM=1 and the Intel SHMEM runtime
# available at load time (see AGENTS.md for the libfabric/SOS environment).

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_XPU,
)


# Decorator
def requires_ishmem():
    # NOTE: PLATFORM_SUPPORTS_SYMM_MEM is CUDA-only, so we gate on XPU + the
    # build/runtime availability of Intel SHMEM instead.
    return skip_but_pass_in_sandcastle_if(
        not TEST_XPU or not symm_mem.is_ishmem_available(),
        "test_ishmem requires an XPU build with Intel SHMEM available, skipping tests",
    )


device_type = "xpu"
device_module = torch.get_device_module(device_type)


@instantiate_parametrized_tests
@requires_ishmem()
class ISHMEMSymmetricMemoryTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "xccl"

    def _init_device(self) -> None:
        device_module.set_device(self.device)
        # Select Intel SHMEM as the SymmMem backend for XPU.
        symm_mem.set_backend("ISHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def test_alloc(self) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        # Allocation + rendezvous of a throwaway tensor must not leave dangling
        # state that breaks a subsequent allocation.
        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    def test_alloc_without_device_context(self) -> None:
        # Allocation should pick up the device from the tensor factory, without
        # a prior set_device call in this test.
        symm_mem.set_backend("ISHMEM")
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        self.assertEqual(out.device, self.device)
        symm_mem.rendezvous(out, group=group_name)

    def test_rendezvous_metadata(self) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        t = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(t, group=group_name)

        self.assertEqual(hdl.rank, self.rank)
        self.assertEqual(hdl.world_size, self.world_size)
        self.assertGreaterEqual(hdl.buffer_size, numel * t.element_size())
        # One peer buffer pointer per rank; all non-null on a single node where
        # peers are directly accessible.
        self.assertEqual(len(hdl.buffer_ptrs), self.world_size)
        for ptr in hdl.buffer_ptrs:
            self.assertNotEqual(ptr, 0)

    def test_handle_offset(self) -> None:
        """rendezvous keys on the storage base, so a view shares the base handle."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        base = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl_base = symm_mem.rendezvous(base, group=group_name)
        self.assertEqual(hdl_base.offset, 0)

        # rendezvous uses tensor.storage().data_ptr(), which is identical for a
        # view and its parent, so the view resolves to the same allocation base
        # with offset 0. (Non-zero handle offsets require a MemPool packing
        # multiple tensors into one allocation, which is not covered here.)
        view = base[numel // 2 :]
        hdl_view = symm_mem.rendezvous(view, group=group_name)
        self.assertEqual(hdl_view.offset, 0)

    def test_multicast_ptr(self) -> None:
        """Intel SHMEM does not expose multicast; the pointer must be null."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        tensor = symm_mem.empty(1, device=self.device)
        handle = symm_mem.rendezvous(tensor, group_name)
        self.assertEqual(handle.multicast_ptr, 0)

    def test_get_remote_tensor(self) -> None:
        """Write into a peer's symmetric tensor via direct-access mapping."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        # src holds my rank; y is the symmetric destination peers write into.
        x = torch.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        y = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        hdl_y = symm_mem.rendezvous(y, group=group_name)

        peer = (self.rank + 1) % self.world_size  # shifting pattern
        y_remote = hdl_y.get_remote_tensor(peer, y.size(), y.dtype)
        y_remote.copy_(x)
        device_module.synchronize()
        dist.barrier()

        # y should now hold the rank of whoever wrote into us (rank - 1).
        expected = torch.empty(numel, dtype=dtype, device=self.device).fill_(
            (self.rank - 1) % self.world_size
        )
        self.assertEqual(y, expected)

    def test_get_remote_tensors(self) -> None:
        """get_remote_tensors returns one directly-readable tensor per peer."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        my_tensor = symm_mem.empty(1, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(my_tensor, group=group_name)
        remote_tensors = torch.ops.symm_mem.get_remote_tensors(my_tensor, group_name)
        dist.barrier()

        for peer, tensor in enumerate(remote_tensors):
            self.assertEqual(tensor, peer)

    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int32])
    def test_get(self, dtype) -> None:
        """One-sided get via ishmem_get_out: full buffer + offset + views."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        numel = 1024

        # Full-buffer get from a peer's allocation.
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(src, group=group_name)
        dist.barrier()

        if self.rank == 0:
            peer = 1
            dst = torch.empty_like(src)
            symm_mem.get(dst, hdl, peer=peer)
            device_module.synchronize()
            self.assertEqual(dst, torch.ones_like(dst))

        dist.barrier()

        # Offset get: copy a sub-region of the peer's allocation.
        src_base = symm_mem.empty(2 * numel, dtype=dtype, device=self.device)
        src_base.copy_(
            torch.arange(2 * numel, dtype=dtype, device=self.device)
            + self.rank * 2 * numel
        )
        hdl = symm_mem.rendezvous(src_base, group=group_name)
        dist.barrier()

        if self.rank == 0:
            offset = numel // 2
            dst = torch.empty(numel, dtype=dtype, device=self.device)
            symm_mem.get(dst, hdl, peer=1, offset=offset)
            device_module.synchronize()
            expected = (
                torch.arange(offset, offset + numel, dtype=dtype, device=self.device)
                + 2 * numel
            )
            self.assertEqual(dst, expected)

            # Filling a sub-region via a view leaves the rest of dst untouched.
            larger_dst = torch.full((numel + 1,), -1, dtype=dtype, device=self.device)
            symm_mem.get(larger_dst[:numel], hdl, peer=1, offset=offset)
            device_module.synchronize()
            self.assertEqual(larger_dst[:numel], expected)
            self.assertEqual(larger_dst[numel], torch.tensor(-1, dtype=dtype))

        dist.barrier()

    def test_get_input_validation(self) -> None:
        """The Python-side argument checks in symm_mem.get reject bad inputs."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(src, group=group_name)
        dist.barrier()

        if self.rank == 0:
            noncontig_dst = torch.empty(2 * numel, dtype=dtype, device=self.device)[::2]
            with self.assertRaisesRegex(ValueError, "contiguous"):
                symm_mem.get(noncontig_dst, hdl, peer=1)

            with self.assertRaisesRegex(ValueError, "non-negative"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=-1,
                )

            with self.assertRaisesRegex(ValueError, "invalid peer"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=hdl.world_size,
                )

            with self.assertRaisesRegex(ValueError, "exceeds"):
                symm_mem.get(
                    torch.empty(1, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=hdl.buffer_size // src.element_size(),
                )

        dist.barrier()

    def test_barrier(self) -> None:
        """Handle-level barrier is backed by the process group barrier."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        tensor = symm_mem.empty(16, device=self.device)
        hdl = symm_mem.rendezvous(tensor, group=group_name)
        hdl.barrier()

    def test_signal_ops_unimplemented(self) -> None:
        """put_signal / wait_signal are not implemented for the ISHMEM backend."""
        self._init_device()
        group_name = dist.group.WORLD.group_name

        tensor = symm_mem.empty(16, device=self.device)
        hdl = symm_mem.rendezvous(tensor, group=group_name)
        with self.assertRaises(RuntimeError):
            hdl.put_signal(dst_rank=(self.rank + 1) % self.world_size)
        with self.assertRaises(RuntimeError):
            hdl.wait_signal(src_rank=(self.rank - 1) % self.world_size)


if __name__ == "__main__":
    run_tests()
