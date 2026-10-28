// vector_add.cu — the "hello world" of CUDA: out[i] = a[i] + b[i].
#include "common.cuh"

// __global__ : this function runs on the GPU and is launched from the host.
// __restrict__: promises the compiler the three arrays do not overlap, which
//               lets it use read-only/cached loads more aggressively.
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out,
                                  long long n) {
    // Global index = (which block am I in) * (threads per block) + (my slot in the block).
    // Cast to 64-bit *before* multiplying so very large arrays cannot overflow int32.
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Bounds check: the last block is usually only partly needed because
    // n is rarely an exact multiple of blockDim.x. Extra threads simply do nothing.
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// Host-side launcher called from Python via ctypes.
LAB_EXPORT int launch_vector_add(const float* a, const float* b, float* out,
                                 long long n, int threads_per_block,
                                 cudaStream_t stream) {
    if (n == 0) return static_cast<int>(cudaSuccess);           // empty input: nothing to launch
    if (threads_per_block <= 0 || threads_per_block > 1024)      // hardware limit on block size
        return static_cast<int>(cudaErrorInvalidValue);

    // Ceiling division: enough blocks to cover every element.
    long long blocks = (n + threads_per_block - 1) / threads_per_block;
    if (blocks > LAB_MAX_GRID_X) return static_cast<int>(cudaErrorInvalidValue);

    // <<<grid, block, dynamic_shared_bytes, stream>>>
    vector_add_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0, stream>>>(a, b, out, n);
    LAB_RETURN_LAUNCH_STATUS();
}
