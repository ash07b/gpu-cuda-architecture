// reduction.cu — summing an array: atomics vs. a two-stage shared-memory reduction tree.
#include "common.cuh"

// ---------------------------------------------------------------------------
// Naive: every thread does atomicAdd on ONE global float.
// Correct (atomics make each read-modify-write indivisible) but every update to
// the same address is serialized by the memory system, so this scales poorly.
// ---------------------------------------------------------------------------
__global__ void reduce_atomic_kernel(const float* __restrict__ in, float* out, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);
}

LAB_EXPORT int launch_reduce_atomic(const float* in, float* out, long long n,
                                    int threads_per_block, cudaStream_t stream) {
    // The accumulator must start at zero; the memset is ordered on the same stream.
    cudaError_t err = cudaMemsetAsync(out, 0, sizeof(float), stream);
    if (err != cudaSuccess) return static_cast<int>(err);
    if (n == 0) return static_cast<int>(cudaSuccess);
    long long blocks = (n + threads_per_block - 1) / threads_per_block;
    if (blocks > LAB_MAX_GRID_X) return static_cast<int>(cudaErrorInvalidValue);
    reduce_atomic_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0, stream>>>(in, out, n);
    LAB_RETURN_LAUNCH_STATUS();
}

// ---------------------------------------------------------------------------
// Tree reduction.
// Stage 1: `num_blocks` blocks. Each thread first sums many elements with a
//          grid-stride loop (cheap, private register accumulator), then the
//          block combines its threads' sums in shared memory as a binary tree
//          (log2(blockDim) steps) and writes ONE partial sum.
// Stage 2: the same kernel with a single block reduces the partial sums.
// blockDim.x must be a power of two for the halving loop below.
// ---------------------------------------------------------------------------
__global__ void reduce_tree_kernel(const float* __restrict__ in, float* __restrict__ out,
                                   long long n) {
    extern __shared__ float sdata[];              // size given at launch (blockDim.x floats)
    unsigned int tid = threadIdx.x;

    // Grid-stride loop: thread t of block b handles elements t+b*B, t+b*B+G, t+b*B+2G, ...
    float acc = 0.0f;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + tid; i < n; i += stride)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();                              // all partials visible before the tree starts

    // Halve the number of active threads each step: 256 -> 128 -> ... -> 1.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();                          // every step must finish before the next reads
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

static bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }

// partials must hold at least num_blocks floats; out holds 1 float.
LAB_EXPORT int launch_reduce_tree(const float* in, float* partials, float* out, long long n,
                                  int num_blocks, int threads_per_block, cudaStream_t stream) {
    if (!is_power_of_two(threads_per_block) || threads_per_block > 1024 || num_blocks <= 0 ||
        num_blocks > 65535)
        return static_cast<int>(cudaErrorInvalidValue);
    size_t shmem = static_cast<size_t>(threads_per_block) * sizeof(float);

    reduce_tree_kernel<<<num_blocks, threads_per_block, shmem, stream>>>(in, partials, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);

    // Stage 2: one block reduces num_blocks partials (an empty input simply yields 0).
    reduce_tree_kernel<<<1, threads_per_block, shmem, stream>>>(partials, out, num_blocks);
    LAB_RETURN_LAUNCH_STATUS();
}
