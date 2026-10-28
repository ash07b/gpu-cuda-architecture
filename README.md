# CUDA Lab: From GPU Architecture to Kernel Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ash07b/cuda-gpu-architecture-lab/blob/main/CUDA_GPU_Architecture_Project.ipynb)

An end-to-end, runnable GPU computing lab in a single Google Colab notebook. It ties each GPU architecture concept to **CUDA C++ kernels I wrote**, a **correctness test**, and a **measurement taken on whatever GPU the runtime provides**.

```
hardware discovery → execution model → custom CUDA kernels → memory behaviour
  → parallel algorithms → benchmarking → optimization (tiling, fusion) → small ML workload
```

The notebook doesn't hard-code a GPU model or a benchmark number. It queries every device fact at runtime and generates every result from the current session.

---

## What's inside

| Part | Topic | Custom CUDA kernel(s) | Compared against |
|---|---|---|---|
| A | GPU system discovery: SMs, compute capability, memory, versions | — | — |
| B | Mental model: grid → blocks → warps → SMs → memory hierarchy, occupancy, divergence | — | — |
| C | PyTorch on the GPU: H2D/D2H transfers, asynchronous execution, correct timing | — | CPU |
| D | First kernel: `nvcc` → `.so` → `ctypes`, with edge-case tests and a block-size sweep | `vector_add` | `torch.add` |
| E | Launch geometry: 1-D/2-D indexing, idle threads, warps | `record_1d`, `record_2d` | index formula |
| F | Memory: coalesced vs strided reads, naive vs shared-memory tiled transpose | `copy_coalesced`, `gather_strided`, `transpose_naive`, `transpose_tiled` | `tensor.t().contiguous()` |
| G | Reduction: atomics vs two-stage tree, floating-point non-associativity | `reduce_atomic`, `reduce_tree` | `torch.sum`, float64 reference |
| H | Matrix multiply: naive vs 16×16 shared-memory tiling, arithmetic intensity | `matmul_naive`, `matmul_tiled` | cuBLAS (`torch.matmul`), CPU |
| I | Kernel fusion: `relu(α·x + β)` as 3 vs 2 vs 1 kernels | `scale_shift` + `relu`, `fused_scale_shift_relu` | PyTorch op chain |
| J | Benchmark methodology: warm-up, CUDA events, noise, transfer vs kernel time, `torch.profiler` | — | — |
| K | Small ML workload: MLP training on synthetic data, CPU vs GPU | — | — |
| L | Results table, plots, automatic interpretation, limitations, extensions | — | — |

Each major experiment follows the same structure: **concept → why it matters → implementation → correctness check → benchmark → interpretation → common mistakes → mini exercise** (the hint comes before the solution).

## How the custom kernels are wired in

```
PyTorch tensor (GPU memory) ──data_ptr()──► ctypes ──► launch_xxx(...) in build/xxx.so
                                                            │ kernel<<<grid, block, smem, stream>>>
torch.cuda.Event ── record ─────────── same CUDA stream ────┘
```

* `nvcc` compiles each `kernels/*.cu` file into a shared library for the GPU's exact compute capability (`-arch=sm_XY`). If that fails it retries with `-arch=native`, then with PTX that the driver JIT-compiles.
* Python passes raw device pointers from PyTorch tensors plus PyTorch's **current CUDA stream**. Custom kernels and PyTorch kernels therefore run in order on the same stream, and the same CUDA events time both.
* Every launcher returns a `cudaError_t`. Python raises a readable error for any non-zero code, and correctness tests call `torch.cuda.synchronize()` so that asynchronous execution errors also surface.

## Run it

1. Click **Open in Colab** above.
2. Go to **Runtime → Change runtime type → GPU** (a T4 is fine).
3. Click **Runtime → Run all**. On a T4 the full run takes a few minutes.
4. Download `results/benchmark_results.csv` and `results/*.png` from the Colab file browser before the runtime disconnects.

The notebook still runs when parts of the stack are missing:

| Runtime | What happens |
|---|---|
| GPU + `nvcc` (standard Colab GPU runtime) | Everything runs, including the custom CUDA C++ kernels. |
| GPU, no `nvcc` | Custom-kernel cells print a warning and run **clearly labelled PyTorch fallbacks**. The fallbacks are never presented as custom kernels. |
| CPU only | GPU cells print `skipped: no CUDA GPU`. CPU references, the reduction precision demo, the MLP and the results table still run. |

To run it locally on a machine with an NVIDIA GPU and the CUDA toolkit:

```bash
pip install -r requirements.txt
jupyter notebook CUDA_GPU_Architecture_Project.ipynb
```

## Repository layout

```
cuda-gpu-architecture-lab/
├── CUDA_GPU_Architecture_Project.ipynb   # the lab (Parts A–L)
├── kernels/
│   ├── common.cuh                        # export macro + error-string helper
│   ├── vector_add.cu                     # Part D
│   ├── launch_geometry.cu                # Part E
│   ├── memory_access.cu                  # Part F: coalesced/strided, naive/tiled transpose
│   ├── reduction.cu                      # Part G: atomic vs tree reduction
│   ├── tiled_matmul.cu                   # Part H: naive vs shared-memory tiled GEMM
│   └── fused_elementwise.cu              # Part I: unfused vs fused relu(a*x+b)
├── results/                              # written by the notebook at runtime
├── tools/
│   └── build_notebook.py                 # regenerates the notebook from kernels/ + cell sources
├── requirements.txt
└── README.md
```

`kernels/*.cu` is the single source of truth for the CUDA code. The notebook embeds the same files as `%%writefile` cells so that it works when opened directly in Colab. After editing a kernel, regenerate the notebook so the two copies stay identical:

```bash
pip install nbformat
python tools/build_notebook.py
```

## Results

This repository deliberately contains **no pre-filled benchmark numbers**. Running the notebook writes:

* `results/environment.json`: GPU name, compute capability, SM count, memory, driver/PyTorch/CUDA/`nvcc` versions
* `results/benchmark_results.csv`: one row per experiment with latency, GB/s or GFLOP/s, dtype, size, correctness status and execution status
* `results/benchmark_plot.png`: effective bandwidth and matmul throughput, side by side
* `results/stride_sweep.png`: bandwidth versus read stride

To publish your own run, commit those files and add a short summary here (GPU model, best streaming bandwidth versus datasheet peak, tiled versus naive matmul, fused versus unfused speedup).

## Methodology at a glance

* **Timing:** warm-up iterations, then CUDA events recorded on the work's stream, with the **median** of many trials reported. CPU timings use `time.perf_counter`.
* **Bandwidth:** `(bytes read + bytes written) / time`. **Throughput:** `2·M·N·K / time` for matmul.
* **Fair comparisons:** each benchmark states whether transfers are included, plus the tensor size and dtype. TF32 is disabled so that cuBLAS is compared against FP32 kernels in FP32.
* **Correctness:** every custom kernel is checked against PyTorch or a float64 CPU reference, on edge-case sizes as well as large ones. Exact equality is used where the arithmetic is exact (copies, adds), and tolerances where the summation order differs (reductions, matmul).

## Limitations and extensions

The kernels are deliberately simple teaching versions: no vectorized loads, register tiling or warp shuffles. Every experiment uses FP32 on a single stream. Colab doesn't normally provide Nsight, so the notebook relies on event timing and `torch.profiler`. Suggested next steps, covered in Part L, include CUDA streams and pinned memory, occupancy analysis (`-Xptxas -v`), warp-shuffle reductions, Tensor Cores and mixed precision, Triton, CUTLASS, `torch.compile`, and Nsight Systems/Compute.

## License

MIT. See [LICENSE](LICENSE).
