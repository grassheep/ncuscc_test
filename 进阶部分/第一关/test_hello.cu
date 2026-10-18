// test_hello.cu —— 本地自测用（★ 不要提交这个文件，只提交 hello_cuda.cu）
// 作用：补上平台会做的两件事 —— 启动 kernel(1 block, 4 threads) + 同步 + 检查错误
#include <cuda_runtime.h>
#include <cstdio>

// ===== 下面这部分必须和提交版完全一致 =====
extern "C" __global__ void hello_kernel() {
    printf("hello cuda %d\n", threadIdx.x);
}
// =========================================

int main() {
    // <<<grid, block>>> = 1 个 block、4 个线程（平台就是这么启动的）
    hello_kernel<<<1, 4>>>();

    // 等 GPU 跑完，顺便检查有没有出错
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
