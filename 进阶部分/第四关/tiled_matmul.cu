#include <cuda_runtime.h>

#define TILE 16

extern "C" __global__ void tiled_matmul_kernel(const int* __restrict__ A,
                                               const int* __restrict__ B,
                                               int*       __restrict__ C,
                                               const int M, const int K, const int N)
{
    __shared__ int As[TILE][TILE];
    __shared__ int Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    int sum = 0;

    const int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void solve(const int* A, const int* B, int* C, int M, int K, int N)
{
    const dim3 block(TILE, TILE);
    const dim3 grid((unsigned)((N + TILE - 1) / TILE),
                    (unsigned)((M + TILE - 1) / TILE));

    tiled_matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);

    cudaDeviceSynchronize();
    cudaGetLastError();
}
