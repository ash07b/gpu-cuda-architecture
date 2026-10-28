// tiled_matmul.cu — C (M x N) = A (M x K) @ B (K x N), all row-major float32.
#include "common.cuh"

#define MM_TILE 16   // 16 x 16 = 256 threads per block; each tile is 1 KiB of shared memory

// Naive: one thread per output element, reading a full row of A and column of B
// straight from global memory. Each A/B element is re-read from DRAM/L2 by many threads.
__global__ void matmul_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}

// Tiled: the block cooperatively loads a MM_TILE x MM_TILE tile of A and of B into
// shared memory, every thread then reuses those 2*MM_TILE values MM_TILE times.
// Global-memory traffic drops by roughly a factor of MM_TILE.
__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * MM_TILE + ty;
    int col = blockIdx.x * MM_TILE + tx;
    float acc = 0.0f;                                   // lives in a register

    int num_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * MM_TILE + tx;                   // column of A this thread loads
        int b_row = t * MM_TILE + ty;                   // row of B this thread loads
        // Out-of-range loads become 0 so edge tiles contribute nothing (handles any M, N, K).
        As[ty][tx] = (row < M && a_col < K) ? A[static_cast<size_t>(row) * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[static_cast<size_t>(b_row) * N + col] : 0.0f;
        __syncthreads();                                // tile fully loaded before use

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                                // done reading before next overwrite
    }

    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
}

static int launch_matmul_common(bool tiled, const float* A, const float* B, float* C,
                                int M, int N, int K, cudaStream_t stream) {
    if (M == 0 || N == 0) return static_cast<int>(cudaSuccess);
    dim3 block(MM_TILE, MM_TILE);
    dim3 grid((N + MM_TILE - 1) / MM_TILE, (M + MM_TILE - 1) / MM_TILE);
    if (tiled) matmul_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    else       matmul_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    LAB_RETURN_LAUNCH_STATUS();
}

LAB_EXPORT int launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K,
                                   cudaStream_t stream) {
    return launch_matmul_common(false, A, B, C, M, N, K, stream);
}

LAB_EXPORT int launch_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                                   cudaStream_t stream) {
    return launch_matmul_common(true, A, B, C, M, N, K, stream);
}
