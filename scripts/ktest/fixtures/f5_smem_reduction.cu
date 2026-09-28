// KTEST-05 / F-5 planted falsifier: a shared-memory tree reduction.
// Built normally it is race-free; built with -DNO_BARRIER the __syncthreads() between tree
// levels is removed, and compute-sanitizer racecheck MUST report hazards. If racecheck stays
// at 0 on the NO_BARRIER build, the L5 gate cannot see a missing barrier and is theater.
#include <cstdio>
#include <cuda_runtime.h>

#define N 256

__global__ void block_sum(const float *in, float *out) {
    __shared__ float s[N];
    unsigned t = threadIdx.x;
    s[t] = in[t];
    __syncthreads();
    for (unsigned stride = N / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
#ifndef NO_BARRIER
        __syncthreads();
#endif
    }
    if (t == 0) *out = s[0];
}

int main() {
    float h[N], *din, *dout, r = 0.f;
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    if (cudaMalloc(&din, sizeof h) || cudaMalloc(&dout, sizeof r)) return 2;
    cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
    block_sum<<<1, N>>>(din, dout);
    if (cudaDeviceSynchronize() != cudaSuccess) return 2;
    cudaMemcpy(&r, dout, sizeof r, cudaMemcpyDeviceToHost);
    printf("sum=%g\n", r);
    return 0;
}
