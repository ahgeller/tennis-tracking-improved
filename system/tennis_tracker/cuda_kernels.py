"""Hand-written CUDA kernels, compiled at run time by NVRTC and launched through the CUDA driver.

Some GPU work is a long chain of small elementwise steps that PyTorch runs as one kernel each,
every one reading and writing its whole tensor; one kernel that does the chain per element reads
the input once and writes the output once.  PyTorch's own run-time compilers do not fit: the
jiterator takes elementwise expressions only (no gathers) and compiles each kernel again in every
process (about 0.5 s a kernel, ``audit/records/opt_pass1.md``), and
``torch.cuda._compile_kernel`` needs a CUDA toolkit's headers (``CUDA_HOME``).

``load_kernel`` compiles a self-contained source (no ``#include``) with the NVRTC library PyTorch
ships beside its CUDA runtime, for the device's own architecture (a cubin, so the driver has
nothing left to compile), and keeps the cubin in a cache directory keyed on the source, the
architecture and the NVRTC version: a cached kernel loads in milliseconds.  Only the stable C
APIs are used (``nvrtc*`` and the driver's ``cuModuleLoadData`` / ``cuLaunchKernel``), through
ctypes.  Anything that fails (no NVRTC, a driver too old for the architecture, a compile error)
raises ``KernelUnavailable``, and the caller keeps its PyTorch path.
"""
from __future__ import annotations

import ctypes
import hashlib
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

# A kernel argument: a tensor (its device address), an int (a C ``int``) or a ctypes value.
KernelArg = Union[torch.Tensor, int, ctypes._SimpleCData]

_LOCK = threading.Lock()
_LIBRARIES: Dict[str, Any] = {}


class KernelUnavailable(RuntimeError):
    """The kernel cannot be compiled or loaded here; the caller falls back to its PyTorch path."""


def _library(kind: str) -> Any:
    """The NVRTC library PyTorch ships (``nvrtc``) or the CUDA driver (``cuda``), loaded once."""
    with _LOCK:
        if kind not in _LIBRARIES:
            major = str(torch.version.cuda or "").split(".")[0]
            if kind == "nvrtc":
                names = [f"nvrtc64_{major}0_0.dll"] if sys.platform == "win32" else [f"libnvrtc.so.{major}", "libnvrtc.so"]
            else:
                names = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1"]
            error: Optional[OSError] = None
            for name in names:
                try:
                    _LIBRARIES[kind] = ctypes.CDLL(name)
                    break
                except OSError as failure:
                    error = failure
            else:
                raise KernelUnavailable(f"cannot load the {kind} library ({error})")
        return _LIBRARIES[kind]


def _nvrtc_version() -> Tuple[int, int]:
    major, minor = ctypes.c_int(), ctypes.c_int()
    if _library("nvrtc").nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
        raise KernelUnavailable("nvrtcVersion failed")
    return major.value, minor.value


def _compile(source: str, name: str, arch: str) -> bytes:
    """``source`` compiled by NVRTC to a cubin for ``sm_<arch>``."""
    nvrtc = _library("nvrtc")
    program = ctypes.c_void_p()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(program), source.encode(), f"{name}.cu".encode(), 0, None, None) != 0:
        raise KernelUnavailable("nvrtcCreateProgram failed")
    try:
        options = [f"--gpu-architecture=sm_{arch}".encode()]
        result = nvrtc.nvrtcCompileProgram(program, len(options), (ctypes.c_char_p * len(options))(*options))
        if result != 0:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise KernelUnavailable(f"NVRTC could not compile {name}: {log.value.decode(errors='replace').strip()}")
        size = ctypes.c_size_t()
        if nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)) != 0 or size.value == 0:
            raise KernelUnavailable(f"NVRTC gave no cubin for {name}")
        cubin = ctypes.create_string_buffer(size.value)
        if nvrtc.nvrtcGetCUBIN(program, cubin) != 0:
            raise KernelUnavailable(f"NVRTC gave no cubin for {name}")
        return cubin.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def _check(result: int, what: str) -> None:
    if result != 0:
        message = ctypes.c_char_p()
        _library("cuda").cuGetErrorString(result, ctypes.byref(message))
        text = message.value.decode() if message.value else f"error {result}"
        raise KernelUnavailable(f"{what}: {text}")


class Kernel:
    """One loaded kernel; calling it launches it on a stream (the current one by default)."""

    def __init__(self, name: str, image: bytes, device: torch.device) -> None:
        self.name = name
        self.device = device
        cuda = _library("cuda")
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        with torch.cuda.device(device):
            torch.cuda.current_stream(device).synchronize()      # the device's context is current on this thread
            _check(cuda.cuModuleLoadData(ctypes.byref(self._module), image), f"loading {name}")
            _check(cuda.cuModuleGetFunction(ctypes.byref(self._function), self._module, name.encode()),
                   f"finding {name}")

    def __call__(self, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: Sequence[KernelArg],
                 stream: Optional[torch.cuda.Stream] = None) -> None:
        values: List[ctypes._SimpleCData] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.device != self.device:
                    raise ValueError(f"{self.name}: a tensor argument is on {arg.device}, not {self.device}")
                values.append(ctypes.c_void_p(arg.data_ptr()))
            elif isinstance(arg, bool):
                raise TypeError(f"{self.name}: pass a flag as an int")
            elif isinstance(arg, int):
                values.append(ctypes.c_int(arg))
            else:
                values.append(arg)
        pointers = (ctypes.c_void_p * len(values))(*[ctypes.addressof(value) for value in values])
        launch_stream = stream if stream is not None else torch.cuda.current_stream(self.device)
        _check(_library("cuda").cuLaunchKernel(self._function, *grid, *block, 0,
                                               ctypes.c_void_p(launch_stream.cuda_stream), pointers, None),
               f"launching {self.name}")


def load_kernel(source: str, name: str, device: torch.device, cache_dir: Optional[Path] = None) -> Kernel:
    """The ``extern "C"`` kernel ``name`` from ``source`` on ``device``, compiled once per source,
    architecture and NVRTC version and kept in ``cache_dir`` (no cache without one); raises
    ``KernelUnavailable`` when it cannot be compiled or loaded."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise KernelUnavailable("a CUDA kernel needs a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        props = torch.cuda.get_device_properties(device)
        arch = f"{props.major}{props.minor}"
        version = _nvrtc_version()
        key = hashlib.sha256(f"{name}\n{arch}\n{version}\n{source}".encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"{name}-sm{arch}-{key}.cubin" if cache_dir is not None else None
        image: Optional[bytes] = None
        if path is not None and path.is_file():
            image = path.read_bytes()
        if image is None:
            image = _compile(source, name, arch)
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
                    temporary.write_bytes(image)
                    temporary.replace(path)
                except OSError:
                    pass                      # an unwritable cache only costs the next run a compile
        return Kernel(name, image, device)
    except (OSError, AttributeError) as error:
        raise KernelUnavailable(str(error)) from error
