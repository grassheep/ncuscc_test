/* test_matrix_add.cu —— A3 矩阵加法的本地测试（★不要提交）
 *
 * 编译运行（需要真 NVIDIA GPU）：
 *     nvcc -O2 -o /tmp/t_matrix_add test_matrix_add.cu && /tmp/t_matrix_add
 *
 * 只编这一个文件：它已把 matrix_add.cu 原样包含进来（不要两个文件一起传，会重复定义）。
 *
 * 设计要点：M=7、N=37 —— 两个方向都不是 16 的整数倍，
 *           平台会开 1×3 个 block（16×16 线程），绝大多数线程都落在矩阵外，
 *           正好检验 if (row < M && col < N) 与行主序下标 row*N+col。
 */

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "matrix_add.cu"     /* 复用 matrix_add_kernel */

int main()
{
    const int M = 7, N = 37;             /* 行 7、列 37，均不整除 16 */
    const int total = M * N;

    std::vector<int> hA(total), hB(total), hC(total, -1);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            hA[r * N + c] = r * 100 + c;         /* 值里带行列信息，出错便于定位 */
            hB[r * N + c] = 1000 - (r * N + c);
        }
    }

    int *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, total * sizeof(int));
    cudaMalloc(&dB, total * sizeof(int));
    cudaMalloc(&dC, total * sizeof(int));
    cudaMemcpy(dA, hA.data(), total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), total * sizeof(int), cudaMemcpyHostToDevice);

    const dim3 block(16, 16);
    const dim3 grid((N + 15) / 16, (M + 15) / 16);   /* x→列，y→行 */
    matrix_add_kernel<<<grid, block>>>(dA, dB, dC, M, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC.data(), dC, total * sizeof(int), cudaMemcpyDeviceToHost);

    int bad = 0, firstBad = -1;
    for (int k = 0; k < total; ++k) {
        if (hC[k] != hA[k] + hB[k]) {
            if (firstBad < 0) firstBad = k;
            ++bad;
        }
    }

    if (bad == 0)
        printf("PASS  M=%d N=%d  grid=(%d,%d)\n", M, N, grid.x, grid.y);
    else
        printf("FAIL  bad=%d/%d  首个错误下标=%d (row=%d, col=%d)\n",
               bad, total, firstBad, firstBad / N, firstBad % N);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad ? 2 : 0;
}
