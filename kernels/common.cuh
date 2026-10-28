// common.cuh — shared helpers for every kernel library in this lab.
//
// Each .cu file in kernels/ is compiled into its own shared library (.so) with
//   nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_XY kernels/<name>.cu -o build/<name>.so
// and loaded from Python with ctypes. Python passes raw device pointers that
// come from PyTorch tensors (tensor.data_ptr()), plus the CUDA stream PyTorch
// is currently using, so our kernels and PyTorch's kernels are ordered on the
// same stream and can be timed with the same CUDA events.
#pragma once

#include <cuda_runtime.h>

// extern "C" disables C++ name mangling so ctypes can find symbols by name.
#define LAB_EXPORT extern "C"

// Every launcher returns an int that is really a cudaError_t (0 == cudaSuccess).
// cudaGetLastError() reports launch-configuration errors (bad grid/block size,
// too much shared memory, missing kernel image for this GPU, ...). Errors that
// happen *while* the kernel runs (e.g. illegal memory access) are asynchronous
// and only surface at the next synchronizing call, which is why the notebook
// calls torch.cuda.synchronize() after launches in its correctness tests.
#define LAB_RETURN_LAUNCH_STATUS() return static_cast<int>(cudaGetLastError())

// Lets Python turn an error code into a readable message.
LAB_EXPORT const char* lab_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

// Largest grid size allowed in the x dimension (2^31 - 1 on all GPUs this lab targets).
#define LAB_MAX_GRID_X 2147483647LL
