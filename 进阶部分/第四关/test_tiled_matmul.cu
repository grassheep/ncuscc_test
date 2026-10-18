/* test_tiled_matmul.cu —— 本地自测用（需要真 NVIDIA GPU）
 * ★ 不要提交这个文件；提交的是 tiled_matmul.cu
 *
 * 编译运行（只编这一个文件即可，它已包含 tiled_matmul.cu）：
 *   nvcc -O2 -o /tmp/test_tiled test_tiled_matmul.cu
 *   /tmp/test_tiled
 *
 * 它把 tiled_matmul.cu 的内容原样包含进来，加一个 main 做 CPU 对拍。
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tiled_matmul.cu"     /* 复用 solve() 与 kernel */

int main()
{
    /* 覆盖各种 tile 边界：不整除、刚整除、1、超小、最大 256 */
    const int sizes[][3] = {
        {1, 1, 1},   {2, 3, 4},   {16, 16, 16},
        {17, 15, 16}, {1, 256, 1}, {256, 1, 256},
        {255, 17, 33}, {33, 255, 17}, {256, 256, 256},
        {31, 32, 33}, {63, 65, 127}, {200, 200, 200}
    };
    const int ncase = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int c = 0; c < ncase; ++c) {
        const int M = sizes[c][0], K = sizes[c][1], N = sizes[c][2];

        std::vector<int> hA((size_t)M * K), hB((size_t)K * N);
        std::vector<int> hC((size_t)M * N, 0), ref((size_t)M * N, 0);
        for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int)(rand() % 2001) - 1000;
        for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int)(rand() % 2001) - 1000;

        /* CPU 参考实现 */
        for (int r = 0; r < M; ++r)
            for (int cc = 0; cc < N; ++cc) {
                int s = 0;
                for (int t = 0; t < K; ++t) s += hA[r * K + t] * hB[t * N + cc];
                ref[r * N + cc] = s;
            }

        int *dA = 0, *dB = 0, *dC = 0;
        if (cudaMalloc(&dA, hA.size() * 4) != cudaSuccess ||
            cudaMalloc(&dB, hB.size() * 4) != cudaSuccess ||
            cudaMalloc(&dC, hC.size() * 4) != cudaSuccess) {
            printf("cudaMalloc failed\n");
            return 1;
        }
        cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(dC, 0, hC.size() * 4);

        solve(dA, dB, dC, M, K, N);          /* ← 被测入口 */

        cudaError_t e = cudaDeviceSynchronize();
        cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost);

        int bad = 0;
        for (size_t i = 0; i < hC.size(); ++i) if (hC[i] != ref[i]) ++bad;

        printf("M=%-4d K=%-4d N=%-4d  %s  (mismatch=%d)  cuda=%s\n",
               M, K, N, bad == 0 ? "PASS" : "FAIL", bad, cudaGetErrorString(e));

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    printf("\n全部用例如无 FAIL 即通过。\n");
    return 0;
}
