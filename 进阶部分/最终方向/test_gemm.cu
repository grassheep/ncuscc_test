/* ============================================================================
 * test_gemm.cu —— 本地自测程序（**不要提交**）
 *
 * 用途：在自己有 GPU 的机器上，验证 gemm_opt.cu 的正确性并测耗时。
 *
 * 编译运行：
 *     nvcc -O2 -o test_gemm test_gemm.cu gemm_opt.cu
 *     ./test_gemm
 *
 * 它做的事：
 *   1) 用 CPU 三重循环算出参考结果
 *   2) 调 solve() 让 GPU 算一遍
 *   3) 按题目误差规则逐元素比对 abs(a-e) <= 1e-3 + 1e-3*abs(e)
 *   4) 用 CUDA event 测 kernel 耗时（warmup 后再测，取中位数）
 *
 * 覆盖的尺寸里特意混了「不整除 tile」的（17/33/65/127/255/511），
 * 那是最容易暴露边界 bug 的地方。
 *
 * 退出码：0 = 全部通过   1 = 有 FAIL   2 = 没有显卡（跳过，不算失败）   3 = kernel 运行出错
 * ==========================================================================*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

/* gemm_opt.cu 里的入口 */
extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int K, int N);

static const int WARMUP  = 10;
static const int REPEAT  = 50;

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Case { int M, K, N; };

int main() {
    /* ---- 前置检查：没有显卡就别往下跑了 ----------------------------------
     * 无显卡时 cudaMalloc / cudaMemcpy / kernel 启动**全部静默失败**，
     * hC 会保持初始的 0，于是每个元素都变成 |0 - e|，
     * 占容差恒等于 1000*|e|/(1+|e|)（饱和在 1000 附近）——
     * 那 16 行 FAIL 是假象，会把人吓一跳。所以先探一下有没有卡。 */
    int ndev = 0;
    cudaError_t derr = cudaGetDeviceCount(&ndev);
    if (derr != cudaSuccess || ndev <= 0) {
        printf("[SKIP] 本机没有可用的 NVIDIA 显卡，跳过运行阶段。\n");
        printf("       cudaGetDeviceCount -> %s (count = %d)\n",
               cudaGetErrorString(derr), ndev);
        printf("\n");
        printf("       这不是代码错误：能走到这里，说明 **编译和链接都已成功**，\n");
        printf("       extern \"C\" void solve(...) 的符号对得上，\n");
        printf("       只是 kernel / cudaMalloc / cudaMemcpy 没有被真正执行。\n");
        printf("\n");
        printf("       正确性验证请改用：python3 verify_gemm.py\n");
        return 2;                       /* 2 = 跳过（无显卡），区别于 0=通过 / 1=失败 */
    }

    /* ---- 打印设备信息（写 report 要用）---- */
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU              : %s\n", prop.name);
    printf("SM count         : %d\n", prop.multiProcessorCount);
    printf("global mem       : %.1f MiB\n", prop.totalGlobalMem / 1048576.0);
    printf("shared / block   : %zu B\n", prop.sharedMemPerBlock);
    printf("warmup=%d repeat=%d\n\n", WARMUP, REPEAT);

    std::vector<Case> cases = {
        {1, 1, 1},   {2, 3, 4},     {15, 16, 17}, {16, 16, 16}, {17, 15, 16},
        {31, 32, 33}, {63, 64, 65}, {64, 64, 64}, {65, 63, 64},
        {127, 129, 131}, {255, 32, 64}, {33, 511, 17}, {255, 255, 255},
        {128, 128, 128}, {256, 256, 256}, {512, 512, 512},
    };

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int failed = 0;
    double ratio512 = 0.0;

    for (const Case& c : cases) {
        const int M = c.M, K = c.K, N = c.N;
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N, 0.f);
        for (auto& x : hA) x = dist(gen);
        for (auto& x : hB) x = dist(gen);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMalloc(&dA, sizeof(float) * M * K);
        cudaMalloc(&dB, sizeof(float) * K * N);
        cudaMalloc(&dC, sizeof(float) * M * N);
        cudaMemcpy(dA, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

        /* ---- 正确性 ---- */
        solve(dA, dB, dC, M, K, N);
        cudaError_t lerr = cudaDeviceSynchronize();
        if (lerr != cudaSuccess) {
            printf("M=%-4d K=%-4d N=%-4d  [ERROR] kernel 执行失败：%s\n",
                   M, K, N, cudaGetErrorString(lerr));
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            return 3;                   /* 3 = kernel 运行期错误 */
        }
        cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        double maxerr = 0.0, maxratio = 0.0;
        for (int r = 0; r < M; ++r) {
            for (int c2 = 0; c2 < N; ++c2) {
                double e = 0.0;
                for (int t = 0; t < K; ++t)
                    e += (double)hA[(size_t)r * K + t] * (double)hB[(size_t)t * N + c2];
                double a = hC[(size_t)r * N + c2];
                if (std::isnan(a) || std::isinf(a)) { maxratio = 1e9; continue; }
                double d = std::fabs(a - e);
                maxerr = std::max(maxerr, d);
                maxratio = std::max(maxratio, d / (1e-3 + 1e-3 * std::fabs(e)));
            }
        }
        bool ok = (maxratio <= 1.0);
        if (!ok) ++failed;
        if (M == 512) ratio512 = maxratio;

        /* ---- 计时：CUDA event 只包住 solve() ---- */
        for (int i = 0; i < WARMUP; ++i) solve(dA, dB, dC, M, K, N);
        cudaDeviceSynchronize();

        std::vector<double> times;
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        for (int i = 0; i < REPEAT; ++i) {
            cudaEventRecord(ev0);
            solve(dA, dB, dC, M, K, N);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            times.push_back(ms * 1000.0);   /* 转成微秒 */
        }
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        double med = median(times);
        double gflops = 2.0 * M * K * N / (med * 1e-6) / 1e9;

        printf("M=%-4d K=%-4d N=%-4d  %s  最大误差=%.3e  占容差=%.3e  "
               "中位耗时=%9.2f us  %7.2f GFLOP/s\n",
               M, K, N, ok ? "PASS" : "FAIL", maxerr, maxratio, med, gflops);
        fflush(stdout);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    printf("\n%s\n", failed == 0 ? "全部通过 ✔" : "存在失败用例 ✘");
    printf("512^3 占容差比例 = %.3e （<=1 即合规）\n", ratio512);
    return failed == 0 ? 0 : 1;
}
