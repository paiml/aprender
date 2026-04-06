/* cuBLAS GEMM benchmark for cgp compete.
 * Compile: nvcc -O3 -o gemm_cublas benchmarks/gemm_cublas.cu -lcublas
 * Run: ./gemm_cublas [size]
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv) {
    int size = 1024;
    if (argc > 1) size = atoi(argv[1]);
    int n = size;

    /* Allocate host */
    float *h_a = (float *)malloc(n * n * sizeof(float));
    float *h_b = (float *)malloc(n * n * sizeof(float));
    srand(42);
    for (int i = 0; i < n * n; i++) {
        h_a[i] = (float)rand() / RAND_MAX - 0.5f;
        h_b[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    /* Allocate device */
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));
    cudaMemcpy(d_a, h_a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
    }
    cudaDeviceSynchronize();

    /* Benchmark: min-of-20 */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float best_ms = 1e9;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) best_ms = ms;
    }

    double gflops = 2.0 * n * n * n / (best_ms / 1000.0) / 1e9;
    printf("cuBLAS FP32 SGEMM (%dx%dx%d): %.3f ms (%.1f GFLOPS)\n", n, n, n, best_ms, gflops);

    /* FP16 test */
    /* (Would need half-precision conversion — skip for now) */

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b);
    return 0;
}
