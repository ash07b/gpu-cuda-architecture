// launch_geometry.cu — kernels that do no math; they record *who* each thread is,
// so the notebook can show the mapping between (blockIdx, threadIdx) and data indices.
#include "common.cuh"

// 1-D: each in-bounds thread writes its block id, thread id, global id and warp id.
__global__ void record_1d_kernel(int* block_ids, int* thread_ids, int* global_ids,
                                 int* warp_ids, int n) {
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    if (global < n) {                       // threads past the end must not write
        block_ids[global]  = blockIdx.x;
        thread_ids[global] = threadIdx.x;
        global_ids[global] = global;
        warp_ids[global]   = threadIdx.x / warpSize;   // warp index *within* the block
    }
}

LAB_EXPORT int launch_record_1d(int* block_ids, int* thread_ids, int* global_ids,
                                int* warp_ids, int n, int threads_per_block,
                                cudaStream_t stream) {
    if (n == 0) return static_cast<int>(cudaSuccess);
    if (threads_per_block <= 0 || threads_per_block > 1024)
        return static_cast<int>(cudaErrorInvalidValue);
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    record_1d_kernel<<<blocks, threads_per_block, 0, stream>>>(block_ids, thread_ids,
                                                                global_ids, warp_ids, n);
    LAB_RETURN_LAUNCH_STATUS();
}

// 2-D: a (rows x cols) row-major matrix. x covers columns, y covers rows, so that
// neighbouring threads in a warp (consecutive threadIdx.x) touch neighbouring
// columns — i.e. neighbouring addresses — which is what makes access coalesced.
__global__ void record_2d_kernel(int* linear_ids, int* block_linear_ids, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int linear = row * cols + col;                         // row-major flattening
        linear_ids[linear]       = linear;
        block_linear_ids[linear] = blockIdx.y * gridDim.x + blockIdx.x;
    }
}

LAB_EXPORT int launch_record_2d(int* linear_ids, int* block_linear_ids, int rows, int cols,
                                int block_x, int block_y, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return static_cast<int>(cudaSuccess);
    if (block_x <= 0 || block_y <= 0 || block_x * block_y > 1024)
        return static_cast<int>(cudaErrorInvalidValue);
    dim3 block(block_x, block_y);
    dim3 grid((cols + block_x - 1) / block_x, (rows + block_y - 1) / block_y);
    record_2d_kernel<<<grid, block, 0, stream>>>(linear_ids, block_linear_ids, rows, cols);
    LAB_RETURN_LAUNCH_STATUS();
}
