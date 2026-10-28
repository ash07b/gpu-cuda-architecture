// fused_elementwise.cu — y = relu(alpha * x + beta), unfused (2 kernels) vs fused (1 kernel).
#include "common.cuh"

// Unfused step 1: tmp = alpha * x + beta   (reads x, writes tmp)
__global__ void scale_shift_kernel(const float* __restrict__ x, float* __restrict__ tmp,
                                   float alpha, float beta, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) tmp[i] = fmaf(alpha, x[i], beta);    // fused multiply-add instruction
}

// Unfused step 2: y = relu(tmp)            (reads tmp, writes y)
__global__ void relu_kernel(const float* __restrict__ tmp, float* __restrict__ y, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(tmp[i], 0.0f);
}

// Fused: the intermediate never leaves a register -> half the global memory traffic
// and one kernel launch instead of two.
__global__ void fused_scale_shift_relu_kernel(const float* __restrict__ x, float* __restrict__ y,
                                              float alpha, float beta, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = fmaf(alpha, x[i], beta);
        y[i] = fmaxf(v, 0.0f);
    }
}

static long long blocks_for(long long n, int tpb) { return (n + tpb - 1) / tpb; }

LAB_EXPORT int launch_unfused(const float* x, float* tmp, float* y, float alpha, float beta,
                              long long n, int threads_per_block, cudaStream_t stream) {
    if (n == 0) return static_cast<int>(cudaSuccess);
    long long blocks = blocks_for(n, threads_per_block);
    if (blocks > LAB_MAX_GRID_X) return static_cast<int>(cudaErrorInvalidValue);
    unsigned int g = static_cast<unsigned int>(blocks);
    scale_shift_kernel<<<g, threads_per_block, 0, stream>>>(x, tmp, alpha, beta, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
    relu_kernel<<<g, threads_per_block, 0, stream>>>(tmp, y, n);
    LAB_RETURN_LAUNCH_STATUS();
}

LAB_EXPORT int launch_fused(const float* x, float* y, float alpha, float beta,
                            long long n, int threads_per_block, cudaStream_t stream) {
    if (n == 0) return static_cast<int>(cudaSuccess);
    long long blocks = blocks_for(n, threads_per_block);
    if (blocks > LAB_MAX_GRID_X) return static_cast<int>(cudaErrorInvalidValue);
    fused_scale_shift_relu_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0, stream>>>(
        x, y, alpha, beta, n);
    LAB_RETURN_LAUNCH_STATUS();
}
