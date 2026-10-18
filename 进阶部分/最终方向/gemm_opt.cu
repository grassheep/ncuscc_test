#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16

#define TM 4
#define TN 4

#define THREADS_X (BN / TN)
#define THREADS_Y (BM / TM)

#define NTHREADS  (THREADS_X * THREADS_Y)
#define A_ELEMS   (BM * BK)
#define B_ELEMS   (BK * BN)
#define A_LOADS   (A_ELEMS / NTHREADS)
#define B_LOADS   (B_ELEMS / NTHREADS)

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  const int M, const int K, const int N)
{
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const int rowBase = blockIdx.y * BM + ty * TM;
    const int colBase = blockIdx.x * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int numTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numTiles; ++t) {
        const int kBase = t * BK;

        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int idx = tid + i * NTHREADS;
            const int m   = idx / BK;
            const int k   = idx % BK;
            const int gm  = blockIdx.y * BM + m;
            const int gk  = kBase + k;
            As[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int idx = tid + i * NTHREADS;
            const int k   = idx / BN;
            const int n   = idx % BN;
            const int gk  = kBase + k;
            const int gn  = blockIdx.x * BN + n;
            Bs[k][n] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float aReg[TM];
            float bReg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) aReg[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += aReg[i] * bReg[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + i;
        if (gr < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gc = colBase + j;
                if (gc < N) {
                    C[gr * N + gc] = acc[i][j];
                }
            }
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int K, int N)
{
    if (M <= 0 || K <= 0 || N <= 0) return;

    const dim3 block(THREADS_X, THREADS_Y);

    const dim3 grid((unsigned)((N + BN - 1) / BN),
                    (unsigned)((M + BM - 1) / BM));

    gemm_tiled_kernel<<<grid, block>>>(A, B, C, M, K, N);

    cudaDeviceSynchronize();
    cudaGetLastError();
}
