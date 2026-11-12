"""tennis_tracker/cuda_kernels.py: compiling, caching and launching a kernel, and its failures."""
import pytest
import torch

from tennis_tracker import cuda_kernels
from tennis_tracker.cuda_kernels import KernelUnavailable, load_kernel

ADD = r"""
extern "C" __global__ void add_value(const unsigned char* a, int* out, int n, int value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (int)a[i] + value;
}
"""


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device("cuda")


def test_a_kernel_compiles_once_and_then_loads_from_the_cache(tmp_path, monkeypatch):
    device = _cuda_or_skip()
    try:
        kernel = load_kernel(ADD, "add_value", device, tmp_path)
    except KernelUnavailable as error:
        pytest.skip(f"no NVRTC here: {error}")
    cached = list(tmp_path.glob("add_value-sm*.cubin"))
    assert len(cached) == 1
    a = torch.arange(300, dtype=torch.int32).to(torch.uint8).to(device)
    out = torch.empty(300, dtype=torch.int32, device=device)
    kernel((2, 1, 1), (256, 1, 1), [a, out, 300, 7])
    assert torch.equal(out.cpu(), a.cpu().to(torch.int32) + 7)

    def no_compile(*args):
        raise AssertionError("the cached cubin should have been loaded")

    monkeypatch.setattr(cuda_kernels, "_compile", no_compile)
    again = load_kernel(ADD, "add_value", device, tmp_path)
    out.zero_()
    again((2, 1, 1), (256, 1, 1), [a, out, 300, 1], stream=torch.cuda.current_stream(device))
    assert torch.equal(out.cpu(), a.cpu().to(torch.int32) + 1)
    with pytest.raises(TypeError, match="flag"):
        again((1, 1, 1), (1, 1, 1), [a, out, True, 1])


def test_a_source_that_does_not_compile_is_unavailable():
    device = _cuda_or_skip()
    try:
        cuda_kernels._nvrtc_version()
    except KernelUnavailable as error:
        pytest.skip(f"no NVRTC here: {error}")
    with pytest.raises(KernelUnavailable, match="could not compile"):
        load_kernel('extern "C" __global__ void broken( { }', "broken", device)


def test_a_kernel_needs_a_cuda_device():
    with pytest.raises(KernelUnavailable, match="CUDA device"):
        load_kernel(ADD, "add_value", torch.device("cpu"))
