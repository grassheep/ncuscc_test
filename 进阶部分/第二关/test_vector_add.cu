/* test_vector_add.cu —— A2 向量加法的本地测试（★不要提交）
 *
 * 编译运行（需要真 NVIDIA GPU）：
 *     nvcc -O2 -o /tmp/t_vector_add test_vector_add.cu && /tmp/t_vector_add
 *
 * 只编这一个文件：它已把 vector_add.cu 原样包含进来（不要两个文件一起传，会重复定义）。
 *
 * 设计要点：N 取 1000 —— 不是 256 的整数倍，必然产生 24 个越界线程，
 *           正好检验 vector_add_kernel 里的 if (idx < N) 有没有写。
 */

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "vector_add.cu"     /* 复用 vector_add_kernel */

int main()
{
    const int N = 1000;      /* 256 * 3.9 → 平台会开 4 个 block，最后 24 个线程越界 */

    std::vector<int> hA(N), hB(N), hC(N, -1);
    for (int i = 0; i < N; ++i) { hA[i] = i; hB[i] = i * 2; }

    int *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, N * sizeof(int));
    cudaMalloc(&dB, N * sizeof(int));
    cudaMalloc(&dC, N * sizeof(int));
    cudaMemcpy(dA, hA.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    vector_add_kernel<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC.data(), dC, N * sizeof(int), cudaMemcpyDeviceToHost);

    int bad = 0, firstBad = -1;
    for (int i = 0; i < N; ++i) {
        if (hC[i] != hA[i] + hB[i]) {
            if (firstBad < 0) firstBad = i;
            ++bad;
        }
    }

    if (bad == 0) printf("PASS  N=%d  blocks=%d\n", N, (N + 255) / 256);
    else          printf("FAIL  bad=%d/%d  首个错误下标=%d\n", bad, N, firstBad);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad ? 2 : 0;
}
