#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void hello_kernel() {
    printf("hello cuda %d\n", threadIdx.x);
}
