// KTEST-05: filter-artifact fixture. `compute-sanitizer --tool initcheck --kernel-name regex=rope` on this
// reports 32 uninitialized reads (the producer kernel is excluded, so its writes are untracked); unfiltered = 0.
#include <cstdio>
__global__ void producer_gemv(float *b){ b[threadIdx.x] = threadIdx.x; }
__global__ void consumer_rope(const float *b, float *o){ o[threadIdx.x] = b[threadIdx.x] * 2.f; }
int main(){ float *b,*o,h=0; cudaMalloc(&b,128); cudaMalloc(&o,128);
 producer_gemv<<<1,32>>>(b); consumer_rope<<<1,32>>>(b,o); cudaMemcpy(&h,o+3,4,cudaMemcpyDeviceToHost); printf("o3=%g\n",h); return 0; }
