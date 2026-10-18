#include <cuda_runtime.h>

extern "C" __global__ void matrix_add_kernel(const int* A, const int* B,
                                             int* C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}
