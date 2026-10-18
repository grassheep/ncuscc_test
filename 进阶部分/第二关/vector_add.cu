#include <cuda_runtime.h>

extern "C" __global__ void vector_add_kernel(const int* A, const int* B,
                                             int* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
