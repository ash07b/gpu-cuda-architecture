// memory_access.cu — experiments on how *access pattern* changes memory performance.
#include "common.cuh"

// ---------------------------------------------------------------------------
// 1) Coalesced copy vs strided gather.
//    Both kernels read every input element exactly once and write every output
//    element exactly once (same bytes moved). Only the *read pattern* differs.
// ---------------------------------------------------------------------------

// Thread i reads in[i]: the 32 threads of a warp read 32 consecutive floats
// (128 contiguous bytes), which the hardware services with few memory transactions.
__global__ void copy_coalesced_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// View the input as a (rows x stride) row-major matrix and read it column by column.
// Consecutive threads read addresses `stride` floats apart, so one warp touches
// up to 32 different memory segments. Requires n % stride == 0 (checked by launcher),
// which makes the index mapping a permutation: still one read per element.
__global__ void gather_strided_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      long long n, long long stride) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        long long rows = n / stride;
        long long r = i % rows;            // consecutive i -> consecutive rows ...
        long long c = i / rows;
        out[i] = in[r * stride + c];       // ... which are `stride` elements apart in memory
    }
}

LAB_EXPORT int launch_copy_coalesced(const float* in, float* out, long long n,
                                     int threads_per_block, cudaStream_t stream) {
    if (n == 0) return static_cast<int>(cudaSuccess);
    long long blocks = (n + threads_per_block - 1) / threads_per_block;
    if (blocks > LAB_MAX_GRID_X) return static_cast<int>(cudaErrorInvalidValue);
    copy_coalesced_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0, stream>>>(in, out, n);
    LAB_RETURN_LAUNCH_STATUS();
}

LAB_EXPORT int launch_gather_strided(const float* in, float* out, long long n, long long stride,
                                     int threads_per_block, cudaStream_t stream) {
    if (n == 0) return static_cast<int>(cudaSuccess);
    if (stride <= 0 || n % stride != 0) return static_cast<int>(cudaErrorInvalidValue);
    long long blocks = (n + threads_per_block - 1) / threads_per_block;
    if (blocks > LAB_MAX_GRID_X) return static_cast<int>(cudaErrorInvalidValue);
    gather_strided_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0, stream>>>(in, out, n, stride);
    LAB_RETURN_LAUNCH_STATUS();
}

// ---------------------------------------------------------------------------
// 2) Matrix transpose: naive vs shared-memory tiled.
//    in  : rows x cols (row-major)
//    out : cols x rows (row-major)
//    Block = TILE x BLOCK_ROWS threads; each thread handles TILE/BLOCK_ROWS elements.
// ---------------------------------------------------------------------------
#define TILE 32
#define BLOCK_ROWS 8

// Reads are coalesced (consecutive threadIdx.x -> consecutive columns of `in`),
// but writes are strided by `rows` because column c of `in` becomes row c of `out`.
__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out,
                                       int rows, int cols) {
    int c  = blockIdx.x * TILE + threadIdx.x;
    int r0 = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        int r = r0 + j;
        if (r < rows && c < cols)
            out[static_cast<size_t>(c) * rows + r] = in[static_cast<size_t>(r) * cols + c];
    }
}

// Stage a TILE x TILE tile in on-chip shared memory, then write it out transposed.
// Both the global read and the global write are now coalesced; the "transpose"
// happens inside shared memory, which tolerates scattered access far better.
__global__ void transpose_tiled_kernel(const float* __restrict__ in, float* __restrict__ out,
                                       int rows, int cols) {
    // +1 column of padding: without it, threads reading a *column* of the tile
    // would all hit the same shared-memory bank (bank conflicts) and serialize.
    __shared__ float tile[TILE][TILE + 1];

    int c = blockIdx.x * TILE + threadIdx.x;
    int r = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (r + j < rows && c < cols)
            tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(r + j) * cols + c];
    }

    // Every thread must finish writing the tile before anyone reads it back.
    __syncthreads();

    // Swap block coordinates: this block's tile lands at the mirrored position in `out`.
    int out_c = blockIdx.y * TILE + threadIdx.x;   // column in out  (< rows)
    int out_r = blockIdx.x * TILE + threadIdx.y;   // row in out     (< cols)
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (out_r + j < cols && out_c < rows)
            out[static_cast<size_t>(out_r + j) * rows + out_c] = tile[threadIdx.x][threadIdx.y + j];
    }
}

static int launch_transpose_common(bool tiled, const float* in, float* out, int rows, int cols,
                                   cudaStream_t stream) {
    if (rows == 0 || cols == 0) return static_cast<int>(cudaSuccess);
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    if (tiled) transpose_tiled_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
    else       transpose_naive_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
    LAB_RETURN_LAUNCH_STATUS();
}

LAB_EXPORT int launch_transpose_naive(const float* in, float* out, int rows, int cols,
                                      cudaStream_t stream) {
    return launch_transpose_common(false, in, out, rows, cols, stream);
}

LAB_EXPORT int launch_transpose_tiled(const float* in, float* out, int rows, int cols,
                                      cudaStream_t stream) {
    return launch_transpose_common(true, in, out, rows, cols, stream);
}
