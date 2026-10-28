#!/usr/bin/env python3
"""Generate CUDA_GPU_Architecture_Project.ipynb from this script + the sources in kernels/.

Why a generator? The CUDA sources live in kernels/*.cu (so they are readable and
diff-able on GitHub) AND are embedded in the notebook as %%writefile cells (so the
notebook is self-contained when opened directly in Colab). Generating the notebook
from one place guarantees the two copies never drift apart.

Usage (from the repository root):
    python tools/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kernels"
OUT = ROOT / "CUDA_GPU_Architecture_Project.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


def writefile(name: str) -> None:
    """Embed kernels/<name> as a %%writefile cell."""
    src = (KERNELS / name).read_text()
    code(f"%%writefile kernels/{name}\n{src}")


# =============================================================================
# Title & orientation
# =============================================================================
md(r"""
# CUDA Lab: From GPU Architecture to Kernel Optimization

An end-to-end, **runnable** walk through GPU computing in Google Colab:

> hardware discovery → execution model → custom CUDA C++ kernels → memory behaviour →
> parallel algorithms → benchmarking → optimization (tiling, fusion) → a small ML workload

**Scope note.** A notebook cannot model a GPU at the transistor level. Instead, every
architecture concept here is tied to code you run and a number you measure on the GPU
Colab gives you. Nothing in this notebook hard-codes a GPU model or a benchmark result:
all device facts are queried and all numbers are produced by *your* run.

### How to run
1. `Runtime → Change runtime type → GPU` (T4 or whatever is offered).
2. `Runtime → Run all`. The full notebook takes a few minutes on a T4.
3. Every section degrades gracefully: with no GPU, GPU cells print *skipped*; with no
   `nvcc`, custom-kernel cells fall back to clearly labelled PyTorch code.

### Status labels used throughout
| Label | Meaning |
|---|---|
| `executed` | implemented **and** ran in this runtime |
| `executed (fallback: PyTorch, not custom CUDA)` | the custom kernel could not be built, so a PyTorch equivalent ran instead |
| `skipped: …` | implemented but **not executed** in this runtime (reason given) |
| 📘 *Conceptual* | explanation only, no code claims |
""")

md(r"""
### Project flow

```
 Part A  System discovery        what GPU do I have? (queried, not assumed)
    │
 Part B  Mental model             grid → blocks → warps → SMs → memory hierarchy
    │
 Part C  PyTorch on the GPU       transfers, async execution, correct timing
    │
 Part D  First custom kernel      vector_add.cu  → nvcc → .so → ctypes → verified
    │
 Part E  Launch geometry          how (blockIdx, threadIdx) map to data
    │
 Part F  Memory experiments       coalesced vs strided, naive vs tiled transpose
    │
 Part G  Reduction                atomics vs reduction tree, float non-associativity
    │
 Part H  Matrix multiplication    naive vs shared-memory tiled vs cuBLAS (PyTorch)
    │
 Part I  Kernel fusion            relu(a·x+b): 3 kernels vs 2 vs 1
    │
 Part J  Benchmarking/profiling   methodology, noise, transfer vs kernel time
    │
 Part K  Small ML workload        MLP training CPU vs GPU + torch.profiler kernel list
    │
 Part L  Results & interpretation table, plot, CSV, what the numbers mean
```

**How custom kernels are wired in.** Each `kernels/*.cu` file is compiled by `nvcc` into a
shared library and called from Python through `ctypes`. Python hands the kernel raw device
pointers from PyTorch tensors plus PyTorch's current CUDA stream, so custom kernels and
PyTorch kernels run on the same stream and are timed with the same CUDA events.

```
 PyTorch tensor (GPU memory) ──data_ptr()──►  ctypes  ──►  launch_xxx(...)  in build/xxx.so
                                                               │  kernel<<<grid, block, smem, stream>>>
 torch.cuda.Event ── record ──────────── same CUDA stream ─────┘
```
""")

# =============================================================================
# Setup
# =============================================================================
md(r"""
---
## 0 · Setup: imports, seeds, folders, result recorder

Run this first. Every later cell depends on the names defined here
(`DEVICE`, `HAS_CUDA`, `RESULTS`, `record_result`, …).
""")

code(r'''
import ctypes
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Shared chart style: one hue, recessive grid, text in neutral inks (not the series colour).
INK, INK_2, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else (platform.processor() or "CPU")

# Make PyTorch float32 matmuls use full FP32 (not TF32) so comparisons with our FP32
# kernels are apples-to-apples. (TF32 is discussed in the extensions section.)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PROJECT_DIR = Path.cwd()
KERNEL_DIR = PROJECT_DIR / "kernels"
BUILD_DIR = PROJECT_DIR / "build"
RESULTS_DIR = PROJECT_DIR / "results"
for d in (KERNEL_DIR, BUILD_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

# Status vocabulary (see the table at the top of the notebook).
EXECUTED = "executed"
FALLBACK = "executed (fallback: PyTorch, not custom CUDA)"
SKIP_NO_GPU = "skipped: no CUDA GPU in this runtime"

RESULTS: list[dict] = []   # one row per measured experiment; Part L turns this into a table


def record_result(experiment, implementation, *, size, dtype="float32", median_ms=None,
                  metric=None, unit=None, correct=None, status=EXECUTED, device=None, notes=""):
    """Append one measured (never invented) result row."""
    RESULTS.append(dict(
        experiment=experiment, implementation=implementation,
        device=device or (DEVICE_NAME if HAS_CUDA else "CPU"),
        size=size, dtype=dtype,
        median_ms=None if median_ms is None else round(median_ms, 4),
        metric=None if metric is None else round(metric, 2), unit=unit,
        correct=correct, status=status, notes=notes,
    ))


print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA available: {HAS_CUDA}")
print(f"Working directory: {PROJECT_DIR}")
''')

# =============================================================================
# Part A
# =============================================================================
md(r"""
---
## Part A · GPU system discovery

**1. Concept.** Before measuring anything, record *what* you are measuring on. Colab assigns
different GPUs (T4, L4, A100, …) across sessions, and every number later in this notebook
depends on the device, driver, and software versions.

**2. Why it matters.** A benchmark without its environment is not reproducible. "My kernel
hits 250 GB/s" means nothing until you know the GPU's peak bandwidth.

**3. Implementation.** We query PyTorch (`torch.cuda.get_device_properties`), `nvidia-smi`
and `nvcc` if they exist. Each fact is labelled with where it came from.
""")

code(r'''
def run_cmd(cmd: list[str]) -> str | None:
    """Run a shell tool if it exists; return stdout or None. Never raises for a missing tool."""
    if shutil.which(cmd[0]) is None:
        return None
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else f"(exit {proc.returncode}) {proc.stderr.strip()}"


def find_nvcc() -> str | None:
    for candidate in (shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


NVCC = find_nvcc()
info = [
    ("Python", platform.python_version(), "platform"),
    ("OS", platform.platform(), "platform"),
    ("PyTorch", torch.__version__, "torch"),
    ("PyTorch built with CUDA", torch.version.cuda or "n/a (CPU build)", "torch.version.cuda"),
    ("cuDNN", str(torch.backends.cudnn.version()) if HAS_CUDA else "n/a", "torch.backends.cudnn"),
    ("CUDA available", str(HAS_CUDA), "torch.cuda.is_available()"),
]

if HAS_CUDA:
    props = torch.cuda.get_device_properties(0)
    def prop(name, fmt=str):
        value = getattr(props, name, None)          # some fields only exist in newer PyTorch
        return fmt(value) if value is not None else "not exposed by this PyTorch"
    info += [
        ("GPU name", props.name, "device properties"),
        ("Compute capability", f"{props.major}.{props.minor}", "device properties"),
        ("Total memory (GiB)", f"{props.total_memory / 2**30:.2f}", "device properties"),
        ("Streaming multiprocessors (SMs)", str(props.multi_processor_count), "device properties"),
        ("Warp size", prop("warp_size"), "device properties"),
        ("Max threads per SM", prop("max_threads_per_multi_processor"), "device properties"),
        ("Registers per SM", prop("regs_per_multiprocessor"), "device properties"),
        ("L2 cache (MiB)", prop("L2_cache_size", lambda v: f"{v / 2**20:.1f}"), "device properties"),
    ]
    smi = run_cmd(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem,power.limit",
                   "--format=csv,noheader"])
    info.append(("Driver, max SM MHz, max mem MHz, power limit", smi or "nvidia-smi not found", "nvidia-smi"))
else:
    info.append(("GPU", "none assigned — see troubleshooting below", "torch"))

nvcc_out = run_cmd([NVCC, "--version"]) if NVCC else None
info.append(("nvcc", nvcc_out.splitlines()[-1] if nvcc_out else "not found (custom kernels will use fallbacks)", "nvcc --version"))

ENV_INFO = pd.DataFrame(info, columns=["Fact", "Value", "Source"])
(RESULTS_DIR / "environment.json").write_text(json.dumps({k: v for k, v, _ in info}, indent=2))
ENV_INFO
''')

md(r"""
**4. Correctness check / what is *not* visible.** Everything above was queried. Things the
notebook **cannot** see directly:

| Not directly visible | Why / how you would find it |
|---|---|
| CUDA cores per SM | Not reported by the driver API; it comes from the architecture whitepaper for the compute capability. |
| Clock speed *during* your kernel | GPUs boost and throttle with temperature/power; `nvidia-smi` shows only a snapshot. |
| Peak DRAM bandwidth | Derived from memory clock × bus width × data rate (datasheet). We measure *achieved* bandwidth instead. |
| L1/shared-memory split, cache policies | Architecture- and driver-dependent; partly configurable per kernel. |
| Which physical SM ran which block | Decided by the hardware scheduler at runtime. |

**6. How to interpret.** Note your GPU name and compute capability: Part L interprets results
relative to this device. Look up its datasheet peak memory bandwidth now; you will compare against it.

**7. Common mistakes.** Assuming you got a T4; assuming the PyTorch CUDA version equals the
`nvcc` version (they are separate toolchains and often differ in Colab).

**8. Mini exercise.** How many warps can be resident on one SM of your GPU at once?
*Hint:* divide two numbers from the table.
<details><summary>Solution</summary>

`Max threads per SM / Warp size` — e.g. 1024 / 32 = 32 warps on a T4 (compute capability 7.5).
Your value may differ; use your table.
</details>

**Troubleshooting — "CUDA available: False".** `Runtime → Change runtime type → GPU`, then
`Runtime → Restart session` and run again. If Colab says no GPU is available (quota), the
notebook still runs: GPU cells will print `skipped`.
""")

# =============================================================================
# Part B
# =============================================================================
md(r"""
---
## Part B · Mental model of GPU architecture  📘 *Conceptual (simplified model)*

### From one line of Python to transistors

```
 Python (host, CPU)        y = torch.relu(x)      or      lib.launch_vector_add(...)
        │
        ▼
 Framework / CUDA runtime  chooses a kernel, packs arguments, enqueues it on a STREAM
        │                  (returns to Python immediately — the launch is asynchronous)
        ▼
 Kernel launch             kernel<<< grid , block >>>
        │
        ▼
 GRID  ┌──────────┬──────────┬──────────┬───── ... ──┐   independent THREAD BLOCKS
       │ block 0  │ block 1  │ block 2  │            │   (any order, any SM)
       └────┬─────┴──────────┴──────────┴────────────┘
            │  block scheduler assigns whole blocks to SMs that have free resources
            ▼
 SM (Streaming Multiprocessor) ──────────────────────────────────────────────┐
 │  warp schedulers ── each cycle pick a READY warp and issue its instruction │
 │  WARP = 32 threads executing the same instruction (SIMT)                    │
 │  ┌ register file (private per thread, fastest) ────────────────────────┐   │
 │  ├ shared memory / L1 (per SM, on-chip, shared by a block) ────────────┤   │
 │  └ execution units: FP32/INT "CUDA cores", SFUs, load/store, (Tensor Cores)  │
 └──────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
                         L2 cache (shared by all SMs)
                                    ▼
                    Global memory: device DRAM (GDDR/HBM)   ◄──PCIe──►  host RAM
```

| Level | Scope | Latency (order of magnitude, architecture-dependent) | Programmer controls? |
|---|---|---|---|
| Registers | one thread | ~1 cycle | indirectly (compiler allocates) |
| Shared memory | one block | tens of cycles | yes (`__shared__`) |
| L1 / L2 cache | SM / whole GPU | tens / hundreds of cycles | mostly no |
| Global (DRAM) | whole GPU + host-visible | hundreds of cycles | yes (`cudaMalloc`, tensors) |
| Host RAM over PCIe | system | microseconds | yes (copies) |

### Key ideas (introductory but accurate)
- **A CUDA thread is a programming abstraction.** Launching 16 million threads does not mean
  16 million cores. Threads are grouped into **warps of 32** that share one instruction stream;
  an SM holds many warps and interleaves them on its execution units.
- **Latency hiding.** A global-memory load takes hundreds of cycles. Instead of waiting, the warp
  scheduler switches to another warp that is ready. GPUs need *lots* of resident warps to stay busy.
- **Occupancy** = resident warps / maximum warps per SM. It is limited by registers per thread,
  shared memory per block, and block size. Higher occupancy helps hide latency, but it is a
  means, not a goal — a kernel with lower occupancy and more work per thread can be faster.
- **Divergence.** If threads of one warp take different branches of an `if`, the warp executes
  both paths with some threads masked off. Divergence across *different* warps costs nothing extra.
- **Synchronization.** `__syncthreads()` is a barrier for the threads of **one block**.
  There is no cheap barrier across blocks inside a kernel — that is why the reduction in Part G
  needs two kernel launches.
- **Coalescing.** When the 32 threads of a warp access consecutive addresses, the hardware merges
  them into a few wide memory transactions. Scattered addresses waste bandwidth (Part F).

**Labelled simplifications.** Exact numbers of schedulers, cores per SM, cache sizes and whether
L1 and shared memory share storage all vary by architecture. Since Volta, threads in a warp also
have independent program counters, so "lock-step" is a mental model, not a guarantee to rely on.

**Checkpoint — predict.** A block of 100 threads: how many warps does it occupy, and how many
thread slots are wasted?
<details><summary>Answer</summary>

`ceil(100/32) = 4` warps = 128 slots, so 28 slots idle. Prefer block sizes that are multiples of 32.
</details>
""")

# =============================================================================
# Shared benchmarking utilities
# =============================================================================
md(r"""
---
## Shared utilities: correctness checks and a benchmark helper

Used by every experiment below. The methodology is explained in detail in **Part J**; in short:

* **Warm-up** runs first (lazy initialization, cuBLAS handles, caches, clock ramp-up).
* On GPU, each iteration is bracketed by **CUDA events** recorded on the same stream as the work,
  then we wait on the end event. This measures GPU execution, not just Python's launch time.
* On CPU, `time.perf_counter()` around a synchronous call.
* We report the **median** (robust to outliers) and keep min/max/stdev.

Formulas:
* Effective bandwidth (GB/s) = bytes read + bytes written / time / 10⁹
* Throughput (GFLOP/s) = floating-point operations / time / 10⁹  (matmul: `2·M·N·K` FLOPs)
""")

code(r'''
@dataclass
class BenchStats:
    median_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    iters: int
    timer: str

    def __str__(self):
        return (f"median {self.median_ms:.4f} ms | min {self.min_ms:.4f} | max {self.max_ms:.4f} | "
                f"stdev {self.stdev_ms:.4f} | n={self.iters} ({self.timer})")


def benchmark(fn, *, warmup: int = 10, iters: int = 50, device: torch.device | None = None) -> BenchStats:
    """Time fn() with warm-up and repeated trials. fn must not return anything needed later."""
    device = device or DEVICE
    for _ in range(warmup):
        fn()
    times = []
    if device.type == "cuda":
        torch.cuda.synchronize()                       # nothing from before leaks into the first trial
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()                             # recorded on the current stream
            fn()
            end.record()
            end.synchronize()                          # wait for the GPU to reach `end`
            times.append(start.elapsed_time(end))      # milliseconds between the two events
        timer = "cuda events"
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
        timer = "perf_counter"
    return BenchStats(statistics.median(times), statistics.fmean(times),
                      statistics.stdev(times) if len(times) > 1 else 0.0,
                      min(times), max(times), iters, timer)


def gb_per_s(bytes_moved: float, ms: float) -> float:
    return bytes_moved / (ms * 1e-3) / 1e9


def gflop_per_s(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e9


def check_close(label: str, actual: torch.Tensor, expected: torch.Tensor, *, rtol=1e-5, atol=1e-6) -> bool:
    """Compare against a trusted reference with explicit tolerances and print the worst error."""
    if actual.shape != expected.shape:
        print(f"  ✗ {label}: shape {tuple(actual.shape)} != expected {tuple(expected.shape)}")
        return False
    a = actual.detach().to("cpu", torch.float64)
    e = expected.detach().to("cpu", torch.float64)
    ok = torch.allclose(a, e, rtol=rtol, atol=atol)
    max_err = (a - e).abs().max().item() if a.numel() else 0.0
    print(f"  {'✓' if ok else '✗'} {label}: max |error| = {max_err:.3e}  (rtol={rtol}, atol={atol})")
    return ok


print("benchmark(), check_close(), gb_per_s(), gflop_per_s() ready")
''')

# =============================================================================
# Part C
# =============================================================================
md(r"""
---
## Part C · First GPU workload with PyTorch

**1. Concept.** A PyTorch tensor lives on one device. `a.to("cuda")` copies it from host RAM to
GPU memory over PCIe (**H2D**); `a_gpu + b_gpu` launches a GPU kernel; `.cpu()` copies back (**D2H**).
Kernel launches are **asynchronous**: Python continues as soon as the work is queued.

**2. Why it matters.** Timing GPU code with `time.time()` and no synchronization measures only
how long it took to *queue* the work. And for cheap operations like addition, the PCIe transfer
often costs far more than the computation.

**3. Implementation** — vector addition `c = a + b` on `N = 2²⁴` float32 values (64 MiB per array).
""")

code(r'''
N_VEC = 1 << 24                     # 16,777,216 elements; reduce if you hit out-of-memory
a_cpu = torch.rand(N_VEC, dtype=torch.float32)
b_cpu = torch.rand(N_VEC, dtype=torch.float32)
c_ref = a_cpu + b_cpu               # trusted CPU reference

if HAS_CUDA:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    a_gpu = a_cpu.to(DEVICE)        # H2D copy
    b_gpu = b_cpu.to(DEVICE)
    torch.cuda.synchronize()
    h2d_ms = (time.perf_counter() - t0) * 1e3

    c_gpu = a_gpu + b_gpu           # GPU kernel (asynchronous)

    t0 = time.perf_counter()
    c_back = c_gpu.cpu()            # D2H copy (synchronizes: it must wait for the kernel)
    d2h_ms = (time.perf_counter() - t0) * 1e3

    print(f"tensor devices: a_gpu={a_gpu.device}, c_gpu={c_gpu.device}, c_back={c_back.device}")
    print(f"H2D of 2 × {N_VEC * 4 / 2**20:.0f} MiB: {h2d_ms:.2f} ms  |  D2H of 1 array (+ kernel wait): {d2h_ms:.2f} ms")
    print("4. Correctness check:")
    vec_add_torch_ok = check_close("PyTorch GPU add vs CPU reference", c_back, c_ref, rtol=0, atol=0)
else:
    print(SKIP_NO_GPU, "— CPU reference computed only.")
''')

md(r"""
### Asynchronous execution: why naïve timing lies

**Checkpoint — predict:** will the un-synchronized timing below be larger or smaller than the
synchronized one?
""")

code(r'''
if HAS_CUDA:
    big_a = torch.rand(1 << 26, device=DEVICE)   # 256 MiB, enough work to make the effect obvious
    big_b = torch.rand(1 << 26, device=DEVICE)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        big_c = big_a * big_b + big_a
    naive_ms = (time.perf_counter() - t0) * 1e3          # measures queuing only

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        big_c = big_a * big_b + big_a
    torch.cuda.synchronize()                              # wait for the GPU to finish
    synced_ms = (time.perf_counter() - t0) * 1e3

    print(f"20 iterations without synchronize: {naive_ms:8.2f} ms   (time to *enqueue* kernels)")
    print(f"20 iterations with synchronize   : {synced_ms:8.2f} ms   (time for the GPU to *do* the work)")
    del big_a, big_b, big_c
    torch.cuda.empty_cache()
else:
    print(SKIP_NO_GPU)
''')

md(r"""
**5. Benchmark** — kernel-only time (data already on the device) vs. the CPU, plus the cost of moving data.
""")

code(r'''
bytes_vec_add = 3 * N_VEC * 4                         # read a, read b, write c (float32 = 4 bytes)

c_out_cpu = torch.empty_like(a_cpu)
s_cpu = benchmark(lambda: torch.add(a_cpu, b_cpu, out=c_out_cpu), warmup=3, iters=10, device=torch.device("cpu"))
print("CPU torch.add      :", s_cpu, f"→ {gb_per_s(bytes_vec_add, s_cpu.median_ms):.1f} GB/s")
record_result("vector_add", "PyTorch CPU", size=f"N={N_VEC}", median_ms=s_cpu.median_ms,
              metric=gb_per_s(bytes_vec_add, s_cpu.median_ms), unit="GB/s", correct=True,
              device=platform.processor() or "CPU", notes="data in host RAM")

if HAS_CUDA:
    c_out_gpu = torch.empty_like(a_gpu)
    s_gpu = benchmark(lambda: torch.add(a_gpu, b_gpu, out=c_out_gpu))
    print("GPU torch.add      :", s_gpu, f"→ {gb_per_s(bytes_vec_add, s_gpu.median_ms):.1f} GB/s")
    record_result("vector_add", "PyTorch GPU", size=f"N={N_VEC}", median_ms=s_gpu.median_ms,
                  metric=gb_per_s(bytes_vec_add, s_gpu.median_ms), unit="GB/s", correct=vec_add_torch_ok,
                  notes="kernel only; inputs already on device")

    s_h2d = benchmark(lambda: a_cpu.to(DEVICE), warmup=3, iters=10)
    print("H2D copy (pageable):", s_h2d, f"→ {gb_per_s(N_VEC * 4, s_h2d.median_ms):.1f} GB/s")
    record_result("host_to_device_copy", "pageable memory", size=f"N={N_VEC}", median_ms=s_h2d.median_ms,
                  metric=gb_per_s(N_VEC * 4, s_h2d.median_ms), unit="GB/s", notes="PCIe transfer, 1 array")
    print(f"\nMoving ONE input array to the GPU takes ~{s_h2d.median_ms / s_gpu.median_ms:.0f}× "
          "longer than the add kernel itself.")
else:
    print(SKIP_NO_GPU)
''')

md(r"""
**6. How to interpret.** Vector add does 1 FLOP per 12 bytes — it is **memory-bound**, so GB/s
is the right metric. Compare the GPU number with your GPU's datasheet bandwidth (e.g. T4 ≈ 320 GB/s
peak); a well-behaved elementwise kernel typically reaches a large fraction of it. The H2D number
is limited by PCIe and is usually an order of magnitude lower.

**7. Common mistakes.** Timing without synchronization; including the H2D copy in a "GPU is X×
faster" claim without saying so; benchmarking a single call (includes one-time setup).

**8. Mini exercise.** Re-run the benchmark with `N_VEC = 1 << 12`. Does the GPU still beat the CPU?
*Hint:* think about fixed per-launch overhead (~microseconds).
<details><summary>Solution</summary>

Usually the GPU advantage shrinks or disappears: at 4096 elements the work is tiny and the
launch overhead dominates. GPUs are throughput machines; they need enough parallel work.
</details>
""")

# =============================================================================
# Part D
# =============================================================================
md(r"""
---
## Part D · Your first custom CUDA C++ kernel

**1. Concept.** We write `vector_add_kernel` ourselves, compile it with `nvcc`, and launch it on
PyTorch-allocated memory. This is genuinely *your* CUDA code running on the GPU — distinct from
PyTorch's built-in kernels and from Python orchestration code.

**2. Why it matters.** Every GPU library is built from kernels like this one. The index
calculation and bounds check are the foundation of everything that follows.

**3. Implementation.** First, the shared header, then the kernel. The `%%writefile` cells save the
sources into `kernels/` (identical to the files in the GitHub repo).
""")

writefile("common.cuh")
writefile("vector_add.cu")

md(r"""
**Line-by-line: the parts that matter**

| Code | What it does |
|---|---|
| `__global__` | Marks a kernel: runs on the device, launched from the host, returns `void`. |
| `const float* __restrict__ a` | Pointer into GPU global memory; `__restrict__` promises no aliasing so the compiler can optimize loads. |
| `blockIdx.x * blockDim.x + threadIdx.x` | Global thread index: block number × threads per block + position inside the block. |
| `static_cast<long long>(...)` | Do the multiply in 64-bit so indexing > 2³¹ elements cannot overflow. |
| `if (i < n)` | Bounds check — the last block usually has threads with no element to process. |
| `(n + tpb - 1) / tpb` | Ceiling division: the number of blocks needed to cover `n`. |
| `<<<blocks, tpb, 0, stream>>>` | Launch configuration: grid size, block size, dynamic shared memory bytes, stream. |
| `cudaGetLastError()` | Reports *launch* errors (bad configuration, no kernel image for this GPU). Errors during execution surface at the next synchronization. |

### Build the kernel library (optional: needs `nvcc` + GPU)
The build helper compiles for your GPU's exact compute capability (`-arch=sm_XY`), and falls back to
`-arch=native` and then to `nvcc`'s default (PTX that the driver JIT-compiles) if your `nvcc` is
older than your GPU.
""")

code(r'''
CUSTOM_CUDA_AVAILABLE = HAS_CUDA and NVCC is not None
CUSTOM_CUDA_REASON = ("" if CUSTOM_CUDA_AVAILABLE else
                      "no CUDA GPU in this runtime" if not HAS_CUDA else "nvcc not found")
_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def build_cuda_library(name: str) -> ctypes.CDLL:
    """Compile kernels/<name>.cu into build/<name>.so and load it. Raises RuntimeError with nvcc's message."""
    src, out = KERNEL_DIR / f"{name}.cu", BUILD_DIR / f"{name}.so"
    major, minor = torch.cuda.get_device_capability()
    errors = []
    for arch_flags in ([f"-arch=sm_{major}{minor}"], ["-arch=native"], []):
        cmd = [NVCC, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", *arch_flags,
               "-I", str(KERNEL_DIR), str(src), "-o", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            lib = ctypes.CDLL(str(out))
            lib.lab_error_string.argtypes = [ctypes.c_int]
            lib.lab_error_string.restype = ctypes.c_char_p
            print(f"[nvcc] built {out.name} with {' '.join(arch_flags) or '(nvcc default arch)'}")
            return lib
        errors.append(f"$ {' '.join(cmd)}\n{proc.stderr.strip()}")
    raise RuntimeError("nvcc failed for every arch option:\n\n" + "\n\n".join(errors))


def get_cuda_lib(name: str) -> ctypes.CDLL | None:
    """Return the compiled library, or None (with a loud explanation) if custom CUDA is unavailable."""
    if not CUSTOM_CUDA_AVAILABLE:
        print(f"⚠ custom CUDA kernel '{name}' NOT built: {CUSTOM_CUDA_REASON}. A labelled fallback will run instead.")
        return None
    if name not in _LIB_CACHE:
        try:
            _LIB_CACHE[name] = build_cuda_library(name)
        except (RuntimeError, OSError) as err:   # report, do not hide: the message is printed in full
            print(f"⚠ BUILD FAILED for '{name}'. Falling back for this section.\n{err}")
            return None
    return _LIB_CACHE[name]


# --- ctypes glue -------------------------------------------------------------------------
VOIDP, I32, I64, F32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_float


def bind(lib: ctypes.CDLL, fn_name: str, *argtypes):
    """Declare a launcher's C signature so ctypes converts arguments correctly."""
    fn = getattr(lib, fn_name)
    fn.argtypes = list(argtypes)
    fn.restype = ctypes.c_int
    return fn


def dptr(t: torch.Tensor, dtype: torch.dtype = torch.float32) -> ctypes.c_void_p:
    """Raw device pointer of a contiguous CUDA tensor, with checks that catch the classic bugs."""
    if not t.is_cuda:
        raise ValueError("tensor must be on the GPU")
    if not t.is_contiguous():
        raise ValueError("tensor must be contiguous (kernels assume dense row-major memory)")
    if t.dtype != dtype:
        raise TypeError(f"expected {dtype}, got {t.dtype}")
    return ctypes.c_void_p(t.data_ptr())


def stream() -> ctypes.c_void_p:
    """PyTorch's current CUDA stream, so our kernels are ordered with PyTorch's work."""
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def check_launch(lib: ctypes.CDLL, code_: int, what: str) -> None:
    if code_ != 0:
        raise RuntimeError(f"{what} failed: {lib.lab_error_string(code_).decode()} (cudaError {code_})")


vec_lib = get_cuda_lib("vector_add")
if vec_lib is not None:
    launch_vector_add = bind(vec_lib, "launch_vector_add", VOIDP, VOIDP, VOIDP, I64, I32, VOIDP)
print("custom CUDA available:", vec_lib is not None)
''')

md(r"""
**4. Correctness check** — edge cases: empty input, fewer elements than one block, a length that is
not a multiple of the block size, and large arrays. Change `THREADS_PER_BLOCK` and re-run.
""")

code(r'''
THREADS_PER_BLOCK = 256   # try 32, 128, 256, 512, 1024 (must be 1..1024)


def custom_vector_add(a: torch.Tensor, b: torch.Tensor, threads_per_block: int = THREADS_PER_BLOCK) -> torch.Tensor:
    assert a.shape == b.shape and a.ndim == 1, "expects two 1-D tensors of equal length"
    out = torch.empty_like(a)
    rc = launch_vector_add(dptr(a), dptr(b), dptr(out), a.numel(), threads_per_block, stream())
    check_launch(vec_lib, rc, "launch_vector_add")
    return out


edge_sizes = [0, 1, THREADS_PER_BLOCK - 1, THREADS_PER_BLOCK, 1000, (1 << 20) + 3, N_VEC]
vec_add_custom_ok = None
if vec_lib is not None:
    results_ok = []
    for n in edge_sizes:
        x = torch.rand(n, device=DEVICE)
        y = torch.rand(n, device=DEVICE)
        out = custom_vector_add(x, y)
        torch.cuda.synchronize()       # surfaces asynchronous execution errors here, not later
        results_ok.append(check_close(f"n={n:>9,}", out, x + y, rtol=0, atol=0))
    vec_add_custom_ok = all(results_ok)
    print("ALL EDGE CASES PASS" if vec_add_custom_ok else "SOME EDGE CASES FAILED")
elif HAS_CUDA:
    print("FALLBACK (PyTorch, not custom CUDA): torch.add on the GPU vs a CPU reference, same edge cases.")
    fallback_ok = []
    for n in edge_sizes:
        x_cpu, y_cpu = torch.rand(n), torch.rand(n)
        fallback_ok.append(check_close(f"n={n:>9,}", (x_cpu.to(DEVICE) + y_cpu.to(DEVICE)).cpu(), x_cpu + y_cpu, rtol=0, atol=0))
    vec_add_custom_ok = None   # the custom kernel itself was NOT tested
else:
    print(SKIP_NO_GPU)
''')

md(r"""
**5. Benchmark** — custom kernel vs. PyTorch's built-in add, same inputs, same output buffer pattern.
""")

code(r'''
if vec_lib is not None:
    out_custom = torch.empty_like(a_gpu)
    s_custom = benchmark(lambda: check_launch(vec_lib, launch_vector_add(
        dptr(a_gpu), dptr(b_gpu), dptr(out_custom), N_VEC, THREADS_PER_BLOCK, stream()), "vector_add"))
    print(f"custom vector_add (tpb={THREADS_PER_BLOCK}):", s_custom,
          f"→ {gb_per_s(bytes_vec_add, s_custom.median_ms):.1f} GB/s")
    print(f"PyTorch add (from Part C)        : median {s_gpu.median_ms:.4f} ms "
          f"→ {gb_per_s(bytes_vec_add, s_gpu.median_ms):.1f} GB/s")
    record_result("vector_add", "custom CUDA", size=f"N={N_VEC}", median_ms=s_custom.median_ms,
                  metric=gb_per_s(bytes_vec_add, s_custom.median_ms), unit="GB/s",
                  correct=vec_add_custom_ok, notes=f"threads_per_block={THREADS_PER_BLOCK}")

    print("\nBlock-size sweep (same kernel):")
    for tpb in (32, 64, 128, 256, 512, 1024):
        s = benchmark(lambda: launch_vector_add(dptr(a_gpu), dptr(b_gpu), dptr(out_custom), N_VEC, tpb, stream()),
                      warmup=5, iters=20)
        print(f"  tpb={tpb:5d}: {s.median_ms:.4f} ms  {gb_per_s(bytes_vec_add, s.median_ms):7.1f} GB/s")
elif HAS_CUDA:
    print(f"Custom kernel not available ({CUSTOM_CUDA_REASON}); PyTorch GPU result from Part C stands alone.")
    record_result("vector_add", "custom CUDA", size=f"N={N_VEC}", status=f"skipped: {CUSTOM_CUDA_REASON}")
else:
    print(SKIP_NO_GPU)
''')

md(r"""
**6. How to interpret.** For a memory-bound kernel, a simple custom kernel should land close to
PyTorch's add: both are limited by DRAM bandwidth, not by cleverness. Very small blocks (32) can be
slower because an SM can host only a limited number of *blocks*, capping resident warps.

**7. Common mistakes.** Forgetting the bounds check (silent memory corruption or illegal address);
computing `blocks = n / tpb` (drops the tail); 32-bit index overflow on huge arrays; not
synchronizing before checking results; passing a non-contiguous tensor's pointer.

**8. Mini exercise.** Delete `if (i < n)` mentally. Which edge-case sizes above would misbehave?
*Hint:* which sizes are not multiples of the block size?
<details><summary>Solution</summary>

Every size not divisible by `THREADS_PER_BLOCK` (1, 255, 1000, 2²⁰+3): the extra threads would read
and write past the end of the arrays. It may even *appear* to work, which is what makes it dangerous.
`n=0` launches nothing because the launcher returns early.
</details>

**Troubleshooting.**
* `nvcc not found` → the runtime has no CUDA toolkit; fallbacks run. On Colab GPU runtimes it is normally at `/usr/local/cuda/bin/nvcc`.
* `no kernel image is available for execution on the device` → `nvcc` is older than your GPU's architecture; the helper retries with PTX, which the driver JIT-compiles.
* `the provided PTX was compiled with an unsupported toolchain` → `nvcc` is *newer* than the driver supports. Use the PyTorch fallback path, or a runtime with a newer driver.
""")

# =============================================================================
# Part E
# =============================================================================
md(r"""
---
## Part E · Thread indexing and launch geometry

**1. Concept.** A launch is a grid of blocks of threads. Each thread derives *which data it owns*
from `blockIdx`, `blockDim`, and `threadIdx`. The kernel below does no math: each thread writes
down its own identity so you can see the mapping.

**Checkpoint — predict before running:** with `THREADS_PER_BLOCK_E = 256`, which global index does
**block 3, thread 17** handle? Which warp (within its block) is it in?
<details><summary>Hint</summary>`global = block × threads_per_block + thread`; warp = `thread // 32`.</details>
<details><summary>Solution</summary>`3 × 256 + 17 = 785`, warp `17 // 32 = 0`.</details>
""")

writefile("launch_geometry.cu")

code(r'''
ARRAY_LENGTH = 1000         # change me
THREADS_PER_BLOCK_E = 256   # change me (1..1024)

num_blocks = math.ceil(ARRAY_LENGTH / THREADS_PER_BLOCK_E)
idle = num_blocks * THREADS_PER_BLOCK_E - ARRAY_LENGTH
print(f"length={ARRAY_LENGTH}, threads/block={THREADS_PER_BLOCK_E} → blocks = ceil({ARRAY_LENGTH}/{THREADS_PER_BLOCK_E}) = {num_blocks}")
print(f"threads launched = {num_blocks * THREADS_PER_BLOCK_E}, idle (fail the bounds check) = {idle}")

geo_lib = get_cuda_lib("launch_geometry")
if geo_lib is not None:
    launch_record_1d = bind(geo_lib, "launch_record_1d", VOIDP, VOIDP, VOIDP, VOIDP, I32, I32, VOIDP)
    bufs = [torch.full((ARRAY_LENGTH,), -1, dtype=torch.int32, device=DEVICE) for _ in range(4)]
    check_launch(geo_lib, launch_record_1d(*(dptr(t, torch.int32) for t in bufs), ARRAY_LENGTH,
                                           THREADS_PER_BLOCK_E, stream()), "record_1d")
    torch.cuda.synchronize()
    geo = pd.DataFrame({k: v.cpu().numpy() for k, v in zip(["blockIdx.x", "threadIdx.x", "global_idx", "warp_in_block"], bufs)})
    source = "custom CUDA kernel (executed)"
else:
    idx = np.arange(ARRAY_LENGTH)   # pure-Python model of the same arithmetic, NOT a kernel
    geo = pd.DataFrame({"blockIdx.x": idx // THREADS_PER_BLOCK_E, "threadIdx.x": idx % THREADS_PER_BLOCK_E,
                        "global_idx": idx, "warp_in_block": (idx % THREADS_PER_BLOCK_E) // 32})
    source = "Python simulation of the index arithmetic (no kernel ran)"

formula_ok = bool((geo["global_idx"] == geo["blockIdx.x"] * THREADS_PER_BLOCK_E + geo["threadIdx.x"]).all())
covered_once = sorted(geo["global_idx"].tolist()) == list(range(ARRAY_LENGTH))
print(f"source: {source}")
print(f"global = block*tpb + thread for every element: {formula_ok};  every element written exactly once: {covered_once}")
display(pd.concat([geo.head(3), geo.iloc[[THREADS_PER_BLOCK_E - 1, THREADS_PER_BLOCK_E]] if ARRAY_LENGTH > THREADS_PER_BLOCK_E else geo.head(0), geo.tail(3)]))
''')

md(r"""
### Two-dimensional indexing (matrices)

For a `rows × cols` row-major matrix, launch a 2-D grid of 2-D blocks:

```
col = blockIdx.x * blockDim.x + threadIdx.x      // x ↔ columns  (consecutive threads → consecutive addresses)
row = blockIdx.y * blockDim.y + threadIdx.y      // y ↔ rows
if (row < rows && col < cols)  idx = row * cols + col
```

Mapping **x to columns** matters: threads `threadIdx.x = 0..31` of one warp then touch 32 adjacent
floats — a coalesced access. The cell below shows which block owns each element of a 5×7 matrix
with 4×4 blocks (note the partially-filled blocks at the edges).
""")

code(r'''
ROWS_2D, COLS_2D, BX, BY = 5, 7, 4, 4
if geo_lib is not None:
    launch_record_2d = bind(geo_lib, "launch_record_2d", VOIDP, VOIDP, I32, I32, I32, I32, VOIDP)
    lin = torch.full((ROWS_2D * COLS_2D,), -1, dtype=torch.int32, device=DEVICE)
    blk = torch.full_like(lin, -1)
    check_launch(geo_lib, launch_record_2d(dptr(lin, torch.int32), dptr(blk, torch.int32),
                                           ROWS_2D, COLS_2D, BX, BY, stream()), "record_2d")
    torch.cuda.synchronize()
    block_map = blk.view(ROWS_2D, COLS_2D).cpu().numpy()
    print("custom CUDA kernel (executed): linear block id owning each element")
else:
    r, c = np.meshgrid(np.arange(ROWS_2D), np.arange(COLS_2D), indexing="ij")
    grid_x = math.ceil(COLS_2D / BX)
    block_map = (r // BY) * grid_x + (c // BX)
    print("Python simulation (no kernel ran): linear block id owning each element")
print(f"grid = ({math.ceil(COLS_2D / BX)}, {math.ceil(ROWS_2D / BY)}) blocks of ({BX}, {BY}) threads")
print(block_map)
''')

md(r"""
**7. Common mistakes.** Swapping x/y (rows on x) — still correct, but uncoalesced; using
`blockDim.x * blockDim.y > 1024`; forgetting that both dimensions need bounds checks.

**8. Mini exercise.** For a 1000×1000 matrix with 16×16 blocks, how many blocks are launched, and how
many threads idle? *Hint:* `ceil(1000/16)` per dimension.
<details><summary>Solution</summary>

`63 × 63 = 3969` blocks = 1,016,064 threads for 1,000,000 elements → 16,064 idle threads.
</details>
""")

# =============================================================================
# Part F
# =============================================================================
md(r"""
---
## Part F · Memory hierarchy experiments

**1. Concept.** Global memory is accessed in aligned segments (32-byte sectors grouped into
128-byte cache lines on recent NVIDIA GPUs; details vary). If a warp's 32 threads read 32
*consecutive* floats, that is 128 bytes — a handful of sectors. If they read floats `stride` apart,
the same warp may touch 32 different segments and the hardware fetches far more bytes than it uses.

**2. Why it matters.** Most real kernels are memory-bound. Access pattern is often the single
largest performance factor you control.

**3. Implementation — designed to be a fair comparison.**
* `copy_coalesced`: thread `i` reads `in[i]`.
* `gather_strided`: thread `i` reads a **permutation** of the input in which consecutive threads are
  `stride` elements apart. Both kernels read every element once and write every element once (same
  bytes, same thread count, same launch overhead); writes are coalesced in both. Only the read pattern differs.
* Arrays are 64 MiB — far larger than L2 — so we measure DRAM behaviour, not cache hits.
* Then a classic: **matrix transpose**, naive vs. **shared-memory tiled**.
""")

writefile("memory_access.cu")

code(r'''
N_MEM = 1 << 24                       # 64 MiB of float32; must be divisible by every stride below
STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
TPB_MEM = 256
bytes_copy = 2 * N_MEM * 4            # read once + write once

mem_lib = get_cuda_lib("memory_access")
stride_sweep = []                      # (stride, GB/s) for the plot below

if HAS_CUDA:
    src = torch.rand(N_MEM, device=DEVICE)
    dst = torch.empty_like(src)
    if mem_lib is not None:
        launch_copy = bind(mem_lib, "launch_copy_coalesced", VOIDP, VOIDP, I64, I32, VOIDP)
        launch_gather = bind(mem_lib, "launch_gather_strided", VOIDP, VOIDP, I64, I64, I32, VOIDP)

        # 4. Correctness: expected result of the strided gather is a transpose of the (rows x stride) view.
        print("4. Correctness:")
        check_launch(mem_lib, launch_copy(dptr(src), dptr(dst), N_MEM, TPB_MEM, stream()), "copy")
        torch.cuda.synchronize()
        coalesced_ok = check_close("coalesced copy", dst, src, rtol=0, atol=0)
        strided_ok = True
        for s in STRIDES:
            check_launch(mem_lib, launch_gather(dptr(src), dptr(dst), N_MEM, s, TPB_MEM, stream()), "gather")
            torch.cuda.synchronize()
            expected = src.view(N_MEM // s, s).t().reshape(-1)
            strided_ok &= check_close(f"strided gather, stride={s:>3}", dst, expected, rtol=0, atol=0)

        # 5. Benchmark
        print("\n5. Benchmark (same bytes moved in every row):")
        st = benchmark(lambda: launch_copy(dptr(src), dptr(dst), N_MEM, TPB_MEM, stream()))
        print(f"  coalesced copy      : {st.median_ms:.4f} ms  {gb_per_s(bytes_copy, st.median_ms):7.1f} GB/s")
        record_result("memory_access", "coalesced copy (custom)", size=f"N={N_MEM}", median_ms=st.median_ms,
                      metric=gb_per_s(bytes_copy, st.median_ms), unit="GB/s", correct=coalesced_ok)
        for s in STRIDES:
            st = benchmark(lambda: launch_gather(dptr(src), dptr(dst), N_MEM, s, TPB_MEM, stream()), iters=30)
            bw = gb_per_s(bytes_copy, st.median_ms)
            stride_sweep.append((s, bw, st.median_ms))
            print(f"  strided, stride={s:>3} : {st.median_ms:.4f} ms  {bw:7.1f} GB/s")
        worst = min(stride_sweep, key=lambda t: t[1])
        record_result("memory_access", f"strided gather s={worst[0]} (custom)", size=f"N={N_MEM}",
                      median_ms=worst[2], metric=worst[1], unit="GB/s",
                      correct=strided_ok, notes="slowest stride in sweep")
    else:
        print("FALLBACK (PyTorch, not custom CUDA): framework-level experiment with clone() vs a strided transpose-copy.")
        for s in STRIDES:
            st = benchmark(lambda: src.view(N_MEM // s, s).t().contiguous(), iters=30)
            stride_sweep.append((s, gb_per_s(bytes_copy, st.median_ms), st.median_ms))
            print(f"  PyTorch view(-1,{s}).t().contiguous(): {st.median_ms:.4f} ms  {stride_sweep[-1][1]:7.1f} GB/s")
        record_result("memory_access", "strided transpose-copy (PyTorch)", size=f"N={N_MEM}",
                      metric=min(b for _, b in stride_sweep), unit="GB/s", status=FALLBACK)
else:
    print(SKIP_NO_GPU)
''')

code(r'''
if stride_sweep:
    xs, ys, _ = zip(*stride_sweep)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(xs, ys, color=SERIES, linewidth=2, marker="o", markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("stride between consecutive threads' reads (elements)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title(f"Read pattern vs. achieved bandwidth — {DEVICE_NAME}")
    ax.annotate(f"{ys[0]:.0f} GB/s", (xs[0], ys[0]), textcoords="offset points", xytext=(6, 6), color=INK_2)
    ax.annotate(f"{ys[-1]:.0f} GB/s", (xs[-1], ys[-1]), textcoords="offset points", xytext=(-10, 8), color=INK_2, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stride_sweep.png", dpi=150)
    plt.show()
else:
    print("No sweep data (no GPU).")
''')

md(r"""
### Matrix transpose: naive vs shared-memory tiled

Transposing reads rows and writes columns, so a naive kernel cannot make **both** the read and the
write coalesced. The tiled kernel loads a 32×32 tile into **shared memory** with coalesced reads,
synchronizes the block (`__syncthreads()`), and writes the tile back transposed with coalesced writes.
The tile has 33 columns (one padding column) to avoid shared-memory **bank conflicts**.
""")

code(r'''
TR_ROWS, TR_COLS = 4096, 4096         # 64 MiB; reduce to 2048 if memory is tight
bytes_transpose = 2 * TR_ROWS * TR_COLS * 4

if HAS_CUDA:
    mat = torch.rand(TR_ROWS, TR_COLS, device=DEVICE)
    mat_t = torch.empty(TR_COLS, TR_ROWS, device=DEVICE)
    s_torch_t = benchmark(lambda: mat.t().contiguous())
    print(f"PyTorch .t().contiguous(): {s_torch_t.median_ms:.4f} ms  {gb_per_s(bytes_transpose, s_torch_t.median_ms):7.1f} GB/s")
    record_result("transpose", "PyTorch .t().contiguous()", size=f"{TR_ROWS}x{TR_COLS}", median_ms=s_torch_t.median_ms,
                  metric=gb_per_s(bytes_transpose, s_torch_t.median_ms), unit="GB/s", correct=True)

    if mem_lib is not None:
        tn = bind(mem_lib, "launch_transpose_naive", VOIDP, VOIDP, I32, I32, VOIDP)
        tt = bind(mem_lib, "launch_transpose_tiled", VOIDP, VOIDP, I32, I32, VOIDP)
        print("4. Correctness (odd shapes exercise the edge-tile bounds checks):")
        transpose_ok = {"naive": True, "tiled": True}
        for r, c in [(1, 1), (31, 33), (100, 257), (1024, 512), (TR_ROWS, TR_COLS)]:
            m = torch.rand(r, c, device=DEVICE)
            for label, fn in (("naive", tn), ("tiled", tt)):
                o = torch.full((c, r), float("nan"), device=DEVICE)   # NaN-filled: any unwritten element fails
                check_launch(mem_lib, fn(dptr(m), dptr(o), r, c, stream()), f"transpose_{label}")
                torch.cuda.synchronize()
                transpose_ok[label] &= check_close(f"{label:5} {r}x{c}", o, m.t(), rtol=0, atol=0)

        for label, fn in (("naive", tn), ("tiled", tt)):
            st = benchmark(lambda: fn(dptr(mat), dptr(mat_t), TR_ROWS, TR_COLS, stream()))
            print(f"custom {label:5} transpose : {st.median_ms:.4f} ms  {gb_per_s(bytes_transpose, st.median_ms):7.1f} GB/s")
            record_result("transpose", f"{label} (custom)", size=f"{TR_ROWS}x{TR_COLS}", median_ms=st.median_ms,
                          metric=gb_per_s(bytes_transpose, st.median_ms), unit="GB/s", correct=transpose_ok[label])
    else:
        print(f"Custom transpose kernels not run ({CUSTOM_CUDA_REASON}); only PyTorch's transpose was measured.")
    del mat, mat_t
else:
    print(SKIP_NO_GPU)
''')

md(r"""
**6. How to interpret.** Expect bandwidth to fall as the stride grows until each thread of a warp
hits a different segment, then flatten. Expect the tiled transpose to beat the naive one and to
approach the coalesced-copy bandwidth. The *size* of these gaps depends on the GPU, its caches,
and the compiler — this notebook does not promise a particular speedup; read your own numbers.
Small strides may look almost free because neighbouring warps reuse the same cache lines from L2.

**7. Common mistakes.** Comparing kernels that move different numbers of bytes; using arrays small
enough to fit in L2 (you would measure cache, not DRAM); forgetting `__syncthreads()` between writing
and reading shared memory (a race that gives wrong answers only sometimes).

**8. Mini exercise.** Remove the `+ 1` padding in `tile[TILE][TILE + 1]`. Predict: correctness or
speed affected? *Hint:* shared memory has 32 banks, each 4 bytes wide.
<details><summary>Solution</summary>

Still correct, usually slower: reading a tile *column* makes all 32 threads hit addresses 32 floats
apart — the same bank — so the accesses serialize (a 32-way bank conflict). Padding shifts each row
by one bank. Try it: edit `kernels/memory_access.cu`, delete `build/memory_access.so`, restart and re-run.
</details>
""")

# =============================================================================
# Part G
# =============================================================================
md(r"""
---
## Part G · Parallel reduction

**1. Concept.** Summing `n` numbers looks serial, but addition is (mathematically) associative, so
it can be done as a **tree**: pairs, then pairs of pairs… `log₂ n` levels deep.

* **Naive GPU approach:** every thread `atomicAdd`s into one global variable. Correct, but every
  update to the same address is serialized — effectively one long queue.
* **Tree reduction (two stages):** each thread first sums many elements in a register (grid-stride
  loop), each block reduces its threads' values in **shared memory** (halving the active threads each
  step, with `__syncthreads()` between steps) and writes one **partial sum**; a second launch reduces
  the partials. Two launches are needed because blocks cannot synchronize with each other.

**2. Why it matters.** Reductions are everywhere: losses, norms, softmax denominators, statistics.

**Floating-point caveat.** In floating point `(a + b) + c ≠ a + (b + c)` in general. Different
parallel orders give slightly different answers, so we validate against a **float64** reference with a
tolerance, never with `==`. Run the demonstration first:
""")

code(r'''
x = torch.tensor([1e8, 1.0, -1e8, 1.0], dtype=torch.float32)
print("(1e8 + 1) - 1e8 + 1 in float32 :", ((x[0] + x[1]) + x[2]) + x[3])
print("(1e8 - 1e8) + 1 + 1 in float32 :", ((x[0] + x[2]) + x[1]) + x[3])

vals = torch.rand(1 << 20, generator=torch.Generator().manual_seed(SEED))
fwd, rev = vals.sum().item(), vals.flip(0).sum().item()
seq = float(np.cumsum(vals.numpy(), dtype=np.float32)[-1])     # strict left-to-right float32
print(f"\nsum of 2^20 rand() values — float64 reference : {vals.double().sum().item():.6f}")
print(f"  torch float32 (blocked/pairwise)            : {fwd:.6f}")
print(f"  torch float32, reversed order               : {rev:.6f}")
print(f"  strict sequential float32 (np.cumsum)       : {seq:.6f}   ← error grows with the running total")
''')

writefile("reduction.cu")

code(r'''
N_RED = 1 << 24
TPB_RED = 256                                              # power of two (required by the tree kernel)
data = torch.rand(N_RED, generator=torch.Generator().manual_seed(SEED))
ref64 = data.double().sum().item()                         # trusted reference (CPU, float64)


def rel_err(v: float) -> float:
    return abs(v - ref64) / abs(ref64)


s_cpu_red = benchmark(lambda: data.sum(), warmup=3, iters=10, device=torch.device("cpu"))
record_result("reduction_sum", "PyTorch CPU", size=f"N={N_RED}", median_ms=s_cpu_red.median_ms,
              metric=gb_per_s(N_RED * 4, s_cpu_red.median_ms), unit="GB/s",
              correct=rel_err(data.sum().item()) < 1e-5, device=platform.processor() or "CPU")
print(f"float64 reference: {ref64:.4f}")
print(f"PyTorch CPU float32 sum: rel. error {rel_err(data.sum().item()):.2e} | {s_cpu_red}")

red_lib = get_cuda_lib("reduction")
if HAS_CUDA:
    d = data.to(DEVICE)
    s = benchmark(lambda: d.sum())
    torch_red = d.sum().item()
    print(f"PyTorch GPU sum        : rel. error {rel_err(torch_red):.2e} | {s}")
    record_result("reduction_sum", "PyTorch GPU", size=f"N={N_RED}", median_ms=s.median_ms,
                  metric=gb_per_s(N_RED * 4, s.median_ms), unit="GB/s", correct=rel_err(torch_red) < 1e-5)

    if red_lib is not None:
        launch_atomic = bind(red_lib, "launch_reduce_atomic", VOIDP, VOIDP, I64, I32, VOIDP)
        launch_tree = bind(red_lib, "launch_reduce_tree", VOIDP, VOIDP, VOIDP, I64, I32, I32, VOIDP)

        def tree_sum(t: torch.Tensor, max_blocks: int = 1024) -> torch.Tensor:
            nb = max(1, min(max_blocks, math.ceil(t.numel() / TPB_RED)))
            partials = torch.empty(nb, device=DEVICE)
            out = torch.empty(1, device=DEVICE)
            check_launch(red_lib, launch_tree(dptr(t), dptr(partials), dptr(out), t.numel(), nb, TPB_RED, stream()), "reduce_tree")
            return out

        def atomic_sum(t: torch.Tensor) -> torch.Tensor:
            out = torch.empty(1, device=DEVICE)
            check_launch(red_lib, launch_atomic(dptr(t), dptr(out), t.numel(), TPB_RED, stream()), "reduce_atomic")
            return out

        print("\n4. Correctness on edge sizes (vs float64, rtol 1e-5 tree / 1e-3 atomic):")
        ok_tree, ok_atomic = True, True
        for n in [0, 1, TPB_RED - 1, TPB_RED + 1, 100_003, N_RED]:
            t = d[:n].contiguous()
            ref = t.double().sum().reshape(1)
            ok_tree &= check_close(f"tree   n={n:>9,}", tree_sum(t), ref, rtol=1e-5, atol=1e-5)
            ok_atomic &= check_close(f"atomic n={n:>9,}", atomic_sum(t), ref, rtol=1e-3, atol=1e-5)

        tree_red, atomic_red = tree_sum(d).item(), atomic_sum(d).item()
        print(f"\nrelative error at N={N_RED:,}: tree {rel_err(tree_red):.2e}  |  atomic {rel_err(atomic_red):.2e}")

        # Pre-allocate buffers so we time only the kernels.
        nb = min(1024, math.ceil(N_RED / TPB_RED))
        partials, out1 = torch.empty(nb, device=DEVICE), torch.empty(1, device=DEVICE)
        s_tree = benchmark(lambda: launch_tree(dptr(d), dptr(partials), dptr(out1), N_RED, nb, TPB_RED, stream()))
        s_atomic = benchmark(lambda: launch_atomic(dptr(d), dptr(out1), N_RED, TPB_RED, stream()), warmup=2, iters=10)
        for label, st, ok in (("tree (custom)", s_tree, ok_tree), ("atomicAdd (custom)", s_atomic, ok_atomic)):
            print(f"{label:20}: {st}  → {gb_per_s(N_RED * 4, st.median_ms):.1f} GB/s")
            record_result("reduction_sum", label, size=f"N={N_RED}", median_ms=st.median_ms,
                          metric=gb_per_s(N_RED * 4, st.median_ms), unit="GB/s", correct=ok)
    else:
        print(f"Custom reductions not run ({CUSTOM_CUDA_REASON}). The PyTorch GPU sum above is the framework baseline.")
    del d
else:
    print(SKIP_NO_GPU)
''')

md(r"""
**6. How to interpret.** The tree should be orders of magnitude faster than the atomic version and
competitive with `torch.sum`. It is also typically *more accurate* than a sequential sum, because
each partial sum stays small (pairwise summation). The atomic version's result can change from run
to run — the order of atomic updates is not deterministic.

**7. Common mistakes.** Missing `__syncthreads()` inside the tree loop; a non-power-of-two block
size with the halving loop (silently skips elements); comparing float sums with `==`; timing
including the allocation of the partials buffer.

**8. Mini exercise.** Why can't stage 1 simply write the final sum directly, without stage 2?
*Hint:* what does `__syncthreads()` synchronize?
<details><summary>Solution</summary>

`__syncthreads()` only synchronizes threads **within one block**. Blocks may run in any order (or not
concurrently at all), so no block can know when the others have finished. Options: a second launch
(used here), atomics on the partials, or cooperative groups grid sync (advanced).
Further optimization to try: replace the last 5 tree steps with warp shuffles (`__shfl_down_sync`).
</details>
""")

# =============================================================================
# Part H
# =============================================================================
md(r"""
---
## Part H · Matrix multiplication

**1. Concept.** `C = A·B` with `A: M×K`, `B: K×N` performs `2·M·N·K` FLOPs on `(MK + KN + MN)`
values. **Arithmetic intensity** (FLOPs per byte) grows with matrix size, so — unlike vector add —
matmul can be **compute-bound** if data is reused well.

* **Naive kernel:** one thread per output, reading a full row of A and column of B from global
  memory. Each input value is re-fetched by many threads.
* **Tiled kernel:** a 16×16 block cooperatively loads a 16×16 tile of A and of B into shared memory,
  then each thread does 16 multiply-adds from shared memory. Global traffic drops ~16×.
* **PyTorch (`torch.matmul`)** calls cuBLAS: years of tuning (register tiling, vectorized loads,
  double buffering, Tensor Cores for lower precision). **We do not expect to beat it.**

**2. Why it matters.** Matmul dominates deep-learning compute (linear layers, attention, convolutions).
""")

writefile("tiled_matmul.cu")

code(r'''
mm_lib = get_cuda_lib("tiled_matmul")
MM_SHAPES_TEST = [(1, 1, 1), (17, 33, 9), (64, 64, 64), (100, 37, 250), (255, 129, 513)]
MM_N = 1024        # benchmark size (M = N = K); 2048 also fits comfortably, 4096+ may be slow for the naive kernel

if HAS_CUDA and mm_lib is not None:
    mm_naive = bind(mm_lib, "launch_matmul_naive", VOIDP, VOIDP, VOIDP, I32, I32, I32, VOIDP)
    mm_tiled = bind(mm_lib, "launch_matmul_tiled", VOIDP, VOIDP, VOIDP, I32, I32, I32, VOIDP)

    def custom_matmul(fn, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"inner dimensions differ: {K} vs {K2}"
        C = torch.full((M, N), float("nan"), device=DEVICE)
        check_launch(mm_lib, fn(dptr(A), dptr(B), dptr(C), M, N, K, stream()), "matmul")
        return C

    print("4. Correctness vs float64 CPU reference (tolerance scales with K: float32 accumulation error grows with length):")
    mm_ok = {"naive": True, "tiled": True}
    for M, N, K in MM_SHAPES_TEST:
        A = torch.randn(M, K, device=DEVICE)
        B = torch.randn(K, N, device=DEVICE)
        ref = A.double().cpu() @ B.double().cpu()
        for label, fn in (("naive", mm_naive), ("tiled", mm_tiled)):
            out = custom_matmul(fn, A, B)
            torch.cuda.synchronize()
            mm_ok[label] &= check_close(f"{label} {M}x{K} @ {K}x{N}", out, ref, rtol=1e-4, atol=1e-5 * K)
elif HAS_CUDA:
    print(f"Custom matmul kernels not run ({CUSTOM_CUDA_REASON}).")
else:
    print(SKIP_NO_GPU)
''')

code(r'''
flops = 2 * MM_N ** 3
A_cpu = torch.randn(MM_N, MM_N)
B_cpu = torch.randn(MM_N, MM_N)
s = benchmark(lambda: A_cpu @ B_cpu, warmup=2, iters=5, device=torch.device("cpu"))
print(f"CPU torch.matmul      {MM_N}³: {s.median_ms:9.3f} ms  {gflop_per_s(flops, s.median_ms):9.1f} GFLOP/s")
record_result("matmul", "PyTorch CPU", size=f"{MM_N}^3", median_ms=s.median_ms,
              metric=gflop_per_s(flops, s.median_ms), unit="GFLOP/s", correct=True, device=platform.processor() or "CPU")

if HAS_CUDA:
    A, B = A_cpu.to(DEVICE), B_cpu.to(DEVICE)
    C = torch.empty(MM_N, MM_N, device=DEVICE)
    ref_gpu = A @ B
    ok_torch = check_close("PyTorch GPU vs CPU", ref_gpu, A_cpu @ B_cpu, rtol=1e-3, atol=1e-3)
    s = benchmark(lambda: torch.matmul(A, B, out=C))
    print(f"PyTorch GPU (cuBLAS)  {MM_N}³: {s.median_ms:9.3f} ms  {gflop_per_s(flops, s.median_ms):9.1f} GFLOP/s")
    record_result("matmul", "PyTorch GPU (cuBLAS, FP32)", size=f"{MM_N}^3", median_ms=s.median_ms,
                  metric=gflop_per_s(flops, s.median_ms), unit="GFLOP/s", correct=ok_torch, notes="TF32 disabled")
    if mm_lib is not None:
        for label, fn, iters in (("naive", mm_naive, 10), ("tiled", mm_tiled, 20)):
            s = benchmark(lambda: fn(dptr(A), dptr(B), dptr(C), MM_N, MM_N, MM_N, stream()), warmup=3, iters=iters)
            print(f"custom {label:5}          {MM_N}³: {s.median_ms:9.3f} ms  {gflop_per_s(flops, s.median_ms):9.1f} GFLOP/s")
            record_result("matmul", f"{label} (custom)", size=f"{MM_N}^3", median_ms=s.median_ms,
                          metric=gflop_per_s(flops, s.median_ms), unit="GFLOP/s", correct=mm_ok[label])
    del A, B, C, ref_gpu
''')

md(r"""
**6. How to interpret.** Tiled should beat naive (fewer global loads per FLOP). cuBLAS should beat
both, often by a wide margin — it reuses data in *registers* as well as shared memory, uses wide
vector loads, and overlaps loads with compute. If your tiled kernel is *not* faster than naive on
your GPU, that is a real result: large L1/L2 caches on newer GPUs already capture much of the
naive kernel's reuse. Report what you measured.

**7. Common mistakes.** Forgetting the second `__syncthreads()` (next tile overwrites shared memory
while others still read it); exact-equality checks on float results; enabling TF32 in PyTorch and
then comparing against an FP32 kernel; benchmarking 8192³ in Colab (slow, can exhaust memory).

**8. Mini exercise.** Estimate arithmetic intensity for `N = 1024` assuming each matrix crosses DRAM
exactly once. *Hint:* FLOPs / (3 · N² · 4 bytes).
<details><summary>Solution</summary>

`2·1024³ / (3·1024²·4) ≈ 171 FLOP/byte` — far above the ~10–60 FLOP/byte at which typical GPUs
become compute-bound. The naive kernel wastes that potential by re-reading inputs from memory.
</details>
""")

# =============================================================================
# Part I
# =============================================================================
md(r"""
---
## Part I · Kernel fusion: `y = relu(α·x + β)`

**1. Concept.** Written as separate operations, each step is its own kernel that reads its input
from DRAM and writes its output back. **Fusion** does all steps in one kernel, keeping the
intermediate values in registers.

| Implementation | Kernels launched | DRAM traffic (float32, N elements) |
|---|---|---|
| PyTorch `torch.relu(alpha * x + beta)` | 3 (`mul`, `add`, `relu`) | read+write ×3 = **24·N bytes** |
| Custom unfused (`scale_shift` → `relu`) | 2 | **16·N bytes** |
| Custom fused | 1 | **8·N bytes** |

**2. Why it matters.** For memory-bound chains, fewer passes over memory ≈ proportionally less time.
This is what `torch.compile`, XLA, TensorRT and Triton do automatically.
""")

writefile("fused_elementwise.cu")

code(r'''
N_FUSE = 1 << 24
ALPHA, BETA = 1.5, -0.25
TPB_FUSE = 256
fuse_lib = get_cuda_lib("fused_elementwise")

if HAS_CUDA:
    xf = torch.randn(N_FUSE, device=DEVICE)
    y_ref = torch.relu(ALPHA * xf + BETA)
    yf, tmp = torch.empty_like(xf), torch.empty_like(xf)

    rows = [("PyTorch unfused (3 kernels)", lambda: torch.relu(ALPHA * xf + BETA), 24, True, EXECUTED)]
    if fuse_lib is not None:
        l_unf = bind(fuse_lib, "launch_unfused", VOIDP, VOIDP, VOIDP, F32, F32, I64, I32, VOIDP)
        l_fus = bind(fuse_lib, "launch_fused", VOIDP, VOIDP, F32, F32, I64, I32, VOIDP)
        print("4. Correctness (fmaf rounds once, PyTorch rounds twice → allow 1-ulp-level differences):")
        ok_unf, ok_fus = True, True
        for n in [0, 1, 1000, N_FUSE]:
            xs = xf[:n].contiguous()
            y1, t1, y2 = torch.empty_like(xs), torch.empty_like(xs), torch.empty_like(xs)
            check_launch(fuse_lib, l_unf(dptr(xs), dptr(t1), dptr(y1), ALPHA, BETA, n, TPB_FUSE, stream()), "unfused")
            check_launch(fuse_lib, l_fus(dptr(xs), dptr(y2), ALPHA, BETA, n, TPB_FUSE, stream()), "fused")
            torch.cuda.synchronize()
            r = torch.relu(ALPHA * xs + BETA)
            ok_unf &= check_close(f"unfused n={n:>9,}", y1, r, rtol=1e-6, atol=1e-6)
            ok_fus &= check_close(f"fused   n={n:>9,}", y2, r, rtol=1e-6, atol=1e-6)
        rows += [
            ("custom unfused (2 kernels)", lambda: l_unf(dptr(xf), dptr(tmp), dptr(yf), ALPHA, BETA, N_FUSE, TPB_FUSE, stream()), 16, ok_unf, EXECUTED),
            ("custom fused (1 kernel)", lambda: l_fus(dptr(xf), dptr(yf), ALPHA, BETA, N_FUSE, TPB_FUSE, stream()), 8, ok_fus, EXECUTED),
        ]
    else:
        print(f"Custom fused/unfused kernels not run ({CUSTOM_CUDA_REASON}). Only the PyTorch chain is measured.")

    print("\n5. Benchmark. 'min-traffic GB/s' divides the SAME 8·N bytes by each time (apples-to-apples speed);")
    print("   'actual-traffic GB/s' uses each implementation's real traffic (how hard memory is working).")
    for label, fn, bytes_per_elem, ok, status in rows:
        st = benchmark(fn)
        min_bw = gb_per_s(8 * N_FUSE, st.median_ms)
        act_bw = gb_per_s(bytes_per_elem * N_FUSE, st.median_ms)
        print(f"  {label:28}: {st.median_ms:.4f} ms | min-traffic {min_bw:7.1f} GB/s | actual-traffic {act_bw:7.1f} GB/s")
        record_result("fusion relu(a*x+b)", label, size=f"N={N_FUSE}", median_ms=st.median_ms, metric=min_bw,
                      unit="GB/s", correct=ok, status=status,
                      notes=f"metric uses minimum traffic 8N bytes; actual traffic {bytes_per_elem}N bytes")
    del xf, yf, tmp, y_ref
else:
    print(SKIP_NO_GPU)
''')

md(r"""
**6. How to interpret.** If all three implementations reach similar *actual-traffic* bandwidth, then
their times should be roughly proportional to their traffic (≈ 3 : 2 : 1). That is the signature of
memory-bound work: fusion wins by removing memory passes, not by doing math faster.

**When fusion does *not* help (or hurts):**
* Compute-bound ops (big matmuls): memory traffic was not the bottleneck.
* Fusing too much into one kernel raises **register pressure** → fewer resident warps
  (lower **occupancy**) or register **spills** to local memory, which is slow.
* Fusing an op that needs a global view (a reduction, a matmul) with elementwise work may
  force a worse parallel decomposition.
* Loss of reuse: a library kernel for one step may be better tuned than your fused version.

**7. Common mistakes.** Measuring tiny tensors (launch overhead dominates and makes fusion look even
better than it would at scale); forgetting that `torch.compile` may already fuse this for you.

**8. Mini exercise.** Predict the speedup of fused over PyTorch-unfused from the traffic table alone.
*Hint:* bytes ratio. <details><summary>Solution</summary>~24/8 = 3×, if all versions are bandwidth-bound. Compare with your measurement.</details>
""")

# =============================================================================
# Part J
# =============================================================================
md(r"""
---
## Part J · Profiling and benchmarking methodology

**1. Concept.** Every number in this notebook came from `benchmark()` (defined after Part B):

| Practice | Why |
|---|---|
| warm-up iterations | first calls pay for lazy CUDA context setup, cuBLAS handles, memory-allocator growth, JIT, GPU clock ramp-up |
| CUDA events on the work's stream | measures GPU time between two points in the stream, independent of Python overhead |
| `end.synchronize()` before reading | events are asynchronous too |
| many iterations, report **median** | robust against outliers (other processes, clock changes) |
| state size, dtype, what is included | a number is only meaningful with its conditions |

**Latency vs throughput.** Latency is the time for one operation; throughput is operations (or bytes,
or FLOPs) per second. GPUs trade latency for throughput: a single small kernel is slow relative to
its work, many large ones are fast in aggregate.

**Sources of noise.** GPU boost clocks and thermal/power throttling; Colab sharing the host; cache
state (we use buffers larger than L2); the compiler; first-run JIT (PTX → SASS).

Below: (a) what warm-up changes, (b) transfer time vs kernel time, (c) which profilers exist here,
(d) a **real** `torch.profiler` trace.
""")

code(r'''
if HAS_CUDA:
    # (a) First call vs steady state, on a fresh shape to trigger one-time costs.
    xa = torch.randn(3001, 3001, device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); _ = xa @ xa; torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3
    steady = benchmark(lambda: xa @ xa, warmup=5, iters=20)
    print(f"(a) first matmul call: {first_ms:.2f} ms   |   steady state: {steady}")
    del xa

    # (b) End-to-end (copy in, compute, copy out) vs kernel-only for vector add.
    def end_to_end():
        ag, bg = a_cpu.to(DEVICE), b_cpu.to(DEVICE)
        return (ag + bg).cpu()
    e2e = benchmark(end_to_end, warmup=2, iters=5)
    print(f"(b) vector add end-to-end incl. transfers: {e2e.median_ms:.2f} ms  vs  kernel only: {s_gpu.median_ms:.3f} ms "
          f"→ the kernel is {100 * s_gpu.median_ms / e2e.median_ms:.1f}% of the end-to-end time")
    record_result("vector_add", "PyTorch GPU incl. H2D+D2H", size=f"N={N_VEC}", median_ms=e2e.median_ms,
                  metric=gb_per_s(bytes_vec_add, e2e.median_ms), unit="GB/s", correct=True,
                  notes="end-to-end: transfers dominate")
else:
    print(SKIP_NO_GPU)

# (c) Which profilers exist in this runtime? (We never invent their output.)
for tool in ("nsys", "ncu", "nvprof"):
    print(f"(c) {tool:6}: {'found at ' + shutil.which(tool) if shutil.which(tool) else 'not available in this runtime'}")
''')

code(r'''
# (d) torch.profiler: real kernel-level trace of the 3-op PyTorch chain from Part I.
from torch.profiler import ProfilerActivity, profile

activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if HAS_CUDA else [])
xp = torch.randn(1 << 22, device=DEVICE)
for _ in range(3):
    torch.relu(1.5 * xp - 0.25)                     # warm-up outside the trace
with profile(activities=activities) as prof:
    for _ in range(5):
        torch.relu(1.5 * xp - 0.25)
    if HAS_CUDA:
        torch.cuda.synchronize()
sort_key = "cuda_time_total" if HAS_CUDA else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
print("Each Python-level op (mul, sub, relu) appears with its own GPU kernel — three passes over memory." if HAS_CUDA
      else "CPU-only trace: GPU kernel rows appear when run on a GPU runtime.")
del xp
''')

md(r"""
**6. How to interpret.** The profiler table is the ground truth for "how many kernels ran and how long
each took". Nsight Systems (`nsys`, timeline) and Nsight Compute (`ncu`, per-kernel hardware counters
such as achieved occupancy, DRAM throughput, bank conflicts) give much deeper detail; they are usually
not installed or not permitted in Colab, so this notebook relies on event timing + `torch.profiler`.

**7. Common mistakes.** Reporting the mean of a noisy distribution; comparing a CPU number that
excludes transfers with a GPU number that includes them (or vice versa) without saying so; reading
GB/s or GFLOP/s without the formula used.

**8. Mini exercise.** Re-run any benchmark three times. How much does the median move?
*Hint:* that spread is your noise floor; differences smaller than it are not findings.
""")

# =============================================================================
# Part K
# =============================================================================
md(r"""
---
## Part K · A small machine-learning workload

**1. Concept.** A neural network forward/backward pass is a sequence of tensor ops — mostly matmuls
(`Linear`), elementwise ops (activations), and reductions (loss, gradients). PyTorch dispatches each
to a pre-built GPU kernel (cuBLAS, cuDNN, or PyTorch's own). **None of these are kernels you wrote**;
the earlier parts show what such kernels look like inside.

**3. Implementation.** A 3-layer MLP classifies synthetic data (a random "teacher" network labels
random inputs), trained with the same seed on CPU and on GPU.
""")

code(r'''
import torch.nn as nn


def make_dataset(n=65_536, d_in=64, n_classes=10, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g)
    W1, W2 = torch.randn(d_in, 128, generator=g), torch.randn(128, n_classes, generator=g)
    y = (torch.tanh(X @ W1) @ W2).argmax(dim=1)          # labels from a fixed random teacher
    return X, y


def make_model(d_in=64, hidden=512, n_classes=10):
    torch.manual_seed(SEED)                              # identical initial weights on every device
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_classes))


def train(device: torch.device, epochs=5, batch_size=1024, lr=1e-3):
    X, y = make_dataset()
    X, y = X.to(device), y.to(device)                    # whole dataset fits in memory: copy once
    model = make_model().to(device)
    assert next(model.parameters()).device.type == device.type, "model is not on the expected device"
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    losses, epoch_ms = [], []
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0], generator=gen).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        running = torch.zeros((), device=device)
        for i in range(0, X.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.detach()                     # stay on device: no per-step sync
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_ms.append((time.perf_counter() - t0) * 1e3)
        losses.append(running.item() / math.ceil(X.shape[0] / batch_size))
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean().item()
    return losses, epoch_ms, acc


runs = {"CPU": train(torch.device("cpu"))}
if HAS_CUDA:
    runs["GPU"] = train(DEVICE)

for name, (losses, ms, acc) in runs.items():
    steady = statistics.median(ms[1:]) if len(ms) > 1 else ms[0]   # epoch 0 includes warm-up
    print(f"{name}: loss per epoch {[round(l, 4) for l in losses]} | train acc {acc:.3f} | "
          f"median epoch {steady:.1f} ms (first epoch {ms[0]:.1f} ms)")
    record_result("MLP training (64→512→512→10)", f"PyTorch {name}", size="65,536 samples, bs=1024",
                  median_ms=steady, metric=65_536 / (steady / 1e3), unit="samples/s",
                  correct=losses[-1] < losses[0], device=DEVICE_NAME if name == "GPU" else (platform.processor() or "CPU"),
                  notes="'correct' = loss decreased")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for (name, (losses, _, _)), color in zip(runs.items(), [SERIES, "#eb6834"]):
    ax.plot(range(1, len(losses) + 1), losses, color=color, linewidth=2, marker="o", markersize=7, label=name)
    ax.annotate(name, (len(losses), losses[-1]), textcoords="offset points", xytext=(8, 0), va="center", color=INK_2)
ax.set_xlabel("epoch"); ax.set_ylabel("mean training loss"); ax.set_title("Training loss decreases on every device")
ax.legend(frameon=False)
fig.tight_layout(); plt.show()
''')

md(r"""
**6. How to interpret.** CPU and GPU losses should track closely (same seed, same data) but not be
bit-identical: different kernels sum in different orders (Part G!). The GPU speedup depends heavily
on batch size: with tiny batches each step launches dozens of small kernels and launch overhead
dominates. Try `batch_size=64` and `batch_size=8192`.

**7. Common mistakes.** Calling `.item()` every step (forces a CPU–GPU sync each time); leaving data on
the CPU and moving each batch (the transfer becomes the bottleneck); timing the first epoch.

**8. Mini exercise.** Wrap the forward pass in `torch.autocast("cuda", dtype=torch.float16)` (with a
`GradScaler`). Predict the effect on speed on a GPU with Tensor Cores.
<details><summary>Solution</summary>Matmuls switch to FP16 Tensor-Core kernels — often a large speedup for
the Linear layers; elementwise ops gain less (they are memory-bound, but FP16 halves their bytes).</details>
""")

# =============================================================================
# Part L
# =============================================================================
md(r"""
---
## Part L · Results and interpretation

Everything below is generated from the `RESULTS` rows recorded by **this run**.
The table and plot are saved to `results/` (download them, or copy to Google Drive, before the
runtime disconnects).
""")

code(r'''
results_df = pd.DataFrame(RESULTS)
results_df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
print(f"{len(results_df)} rows → {RESULTS_DIR / 'benchmark_results.csv'}")
pd.set_option("display.max_colwidth", 60)
results_df
''')

code(r'''
bw = results_df[(results_df["unit"] == "GB/s") & results_df["metric"].notna()]
fl = results_df[(results_df["unit"] == "GFLOP/s") & results_df["metric"].notna()]
panels = [(df, unit) for df, unit in ((bw, "GB/s"), (fl, "GFLOP/s")) if len(df)]

if panels:
    heights = [max(2.0, 0.38 * len(df) + 1.0) for df, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, sum(heights)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (df, unit) in zip(axes[:, 0], panels):
        labels = (df["experiment"] + " · " + df["implementation"]).tolist()[::-1]
        values = df["metric"].tolist()[::-1]
        ax.barh(labels, values, color=SERIES, height=0.6)
        ax.grid(axis="y", visible=False)
        for y_pos, v in enumerate(values):
            ax.annotate(f"{v:,.0f}", (v, y_pos), textcoords="offset points", xytext=(4, 0), va="center", color=INK_2, fontsize=9)
        ax.set_xlim(0, max(values) * 1.15)
        ax.set_xlabel(unit)
        ax.set_title("Effective memory bandwidth (higher is better)" if unit == "GB/s"
                     else "Matrix-multiply throughput, FP32 (higher is better)")
    fig.suptitle(f"Measured on {DEVICE_NAME}", x=0.01, ha="left", color=INK_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "benchmark_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved {RESULTS_DIR / 'benchmark_plot.png'}")
else:
    print("No throughput results to plot (CPU-only runtime with no measurements).")
''')

md(r"""
### Automatic interpretation of *your* numbers
The cell below compares related rows and writes sentences only for comparisons that were actually
measured in this run.
""")

code(r'''
def metric_of(experiment: str, implementation: str):
    row = results_df[(results_df["experiment"] == experiment) & (results_df["implementation"] == implementation)]
    return None if row.empty or pd.isna(row["metric"].iloc[0]) else float(row["metric"].iloc[0])


def compare(exp, a, b, unit, what):
    va, vb = metric_of(exp, a), metric_of(exp, b)
    if va is None or vb is None:
        return f"• {what}: not measured in this run."
    return f"• {what}: {a} = {va:,.1f} {unit}, {b} = {vb:,.1f} {unit} → ratio {va / vb:.2f}×."


lines = [
    compare("vector_add", "PyTorch GPU", "PyTorch CPU", "GB/s", "GPU vs CPU for a memory-bound op (data already resident)"),
    compare("vector_add", "custom CUDA", "PyTorch GPU", "GB/s", "Your vector-add kernel vs PyTorch's"),
    compare("vector_add", "PyTorch GPU", "PyTorch GPU incl. H2D+D2H", "GB/s", "Kernel-only vs including PCIe transfers"),
    compare("transpose", "tiled (custom)", "naive (custom)", "GB/s", "Shared-memory tiling vs naive transpose"),
    compare("reduction_sum", "tree (custom)", "atomicAdd (custom)", "GB/s", "Reduction tree vs single-address atomics"),
    compare("matmul", "tiled (custom)", "naive (custom)", "GFLOP/s", "Tiled vs naive matmul"),
    compare("matmul", "PyTorch GPU (cuBLAS, FP32)", "tiled (custom)", "GFLOP/s", "cuBLAS vs your tiled matmul"),
    compare("fusion relu(a*x+b)", "custom fused (1 kernel)", "PyTorch unfused (3 kernels)", "GB/s", "Fused vs unfused chain (same-bytes metric)"),
]
cc = metric_of("memory_access", "coalesced copy (custom)")
if cc is not None:
    lines.append(f"• Best measured streaming bandwidth: {cc:,.1f} GB/s (coalesced copy). Compare with your GPU's datasheet peak.")
incorrect = results_df[results_df["correct"] == False]  # noqa: E712  (explicit False, not None)
lines.append("• Correctness: every measured implementation passed its check." if incorrect.empty
             else f"• Correctness FAILURES: {incorrect[['experiment', 'implementation']].values.tolist()}")
skipped = results_df[results_df["status"].str.startswith("skipped")]
if not skipped.empty:
    lines.append(f"• Not executed in this runtime: {skipped[['experiment', 'implementation', 'status']].values.tolist()}")
print("\n".join(lines))
''')

md(r"""
### Why your results may differ from someone else's
* **Different GPU**: Colab may assign a T4, L4, A100… with very different bandwidth and FLOP/s.
* **Clocks**: boost/throttling with temperature and power limits; shared hosts.
* **Software**: PyTorch/cuBLAS versions choose different kernels; `nvcc` version changes code generation.
* **Sizes**: small inputs are launch-overhead-bound; inputs smaller than L2 measure cache, not DRAM.

### What I learned  ✍️ *(fill in from your own run)*
* The ratio between my GPU's measured streaming bandwidth and its datasheet peak was ___.
* Coalescing mattered by ___× on this GPU; shared-memory tiling recovered ___% of copy bandwidth for transpose.
* My tiled matmul reached ___% of cuBLAS — the remaining gap comes from ___.
* Fusion gave ___× for the elementwise chain, consistent / inconsistent with the 3 : 2 : 1 traffic model because ___.
* End-to-end, PCIe transfers were ___% of the vector-add time, so for small workloads the GPU ___.

### Limitations
* Event timing measures whole kernels; it cannot attribute time to instructions, stalls, or cache misses.
* Custom kernels are deliberately simple teaching versions (no vectorized `float4` loads, no register tiling, no warp shuffles).
* All experiments use one GPU, one stream, FP32.

### Next steps / extensions
| Topic | What to try |
|---|---|
| **CUDA streams** | Overlap H2D copies of chunk *i+1* with compute on chunk *i* (`torch.cuda.Stream`, `non_blocking=True`). |
| **Pinned memory** | `a_cpu.pin_memory()` → faster, asynchronous H2D copies; re-measure Part C's transfer bandwidth. |
| **Occupancy analysis** | `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; compile with `-Xptxas -v` to see registers/shared memory per kernel. |
| **Warp-level primitives** | Finish the reduction with `__shfl_down_sync`; compare. |
| **Tensor Cores / mixed precision** | FP16/BF16 with `torch.autocast`; TF32 (`torch.backends.cuda.matmul.allow_tf32 = True`); WMMA in CUDA C++. |
| **Triton** | Write the fused kernel in Python with `triton.jit` and compare against the CUDA version. |
| **CUTLASS** | Templates for high-performance GEMM — see how real tiling hierarchies (block → warp → thread) look. |
| **`torch.compile`** | Let the compiler fuse Part I's chain and check the profiler for how many kernels remain. |
| **Nsight Systems / Compute** | On a machine where they are available: timeline (`nsys profile`) and hardware counters (`ncu --set full`). |
""")

md(r"""
---
## Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `CUDA available: False` | No GPU runtime. `Runtime → Change runtime type → GPU`, restart, re-run. If none is available, CPU paths still run. |
| `nvcc not found` | No CUDA toolkit in this image; custom-kernel cells run labelled PyTorch fallbacks. |
| `BUILD FAILED … unsupported gpu architecture` | `nvcc` older than your GPU. The helper already retries with `-arch=native` and PTX; read the printed `nvcc` error. |
| `no kernel image is available for execution` | Same root cause; delete `build/` and re-run Part D's build cell. |
| `PTX was compiled with an unsupported toolchain` | `nvcc` newer than the driver. Use the PyTorch fallback, or a runtime with a newer driver. |
| `CUDA error: an illegal memory access was encountered` | A kernel wrote out of bounds. The CUDA context is now unusable: `Runtime → Restart session`. Check bounds checks and tensor sizes. |
| `CUDA out of memory` | Reduce `N_VEC`, `N_MEM`, `TR_ROWS/TR_COLS`, `MM_N`; `del` tensors and call `torch.cuda.empty_cache()`; restart. |
| Kernel edits seem ignored | `ctypes` cannot reload a loaded `.so` in the same process. Delete `build/*.so` and restart the session. |
| Unexpectedly slow / noisy numbers | Another workload on the host, thermal throttling, too few iterations, or tiny inputs. Re-run; compare medians; increase sizes. |
| `ImportError` for pandas/matplotlib (outside Colab) | `pip install torch numpy pandas matplotlib` (see `requirements.txt`). |
""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
nbformat.validate(nb)
OUT.write_text(nbformat.writes(nb))
print(f"wrote {OUT.relative_to(ROOT)} with {len(cells)} cells")
